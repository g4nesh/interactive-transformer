import { createModelController } from "./model.js";
import { TransformerScene } from "./scene.js";

const DEFAULT_PROMPT = "Hey there! Hope you like this fun little interactive model!";
const BASE_PLAYBACK_DELAY_MS = 380;

export function createApp(root) {
  const model = createModelController();

  root.innerHTML = `
    <div class="layout">
      <aside class="sidebar">
        <div class="sidebar-head">
          <p class="eyebrow">Interactive Transformer</p>
          <h1>Autoregressive Text Generation</h1>
        </div>

        <label class="field">
          <span>Prompt</span>
          <textarea id="prompt-input" rows="8" spellcheck="false">${DEFAULT_PROMPT}</textarea>
        </label>

        <div class="actions">
          <button id="generate-button" class="primary-button">Generate</button>
          <p id="status-line" class="status-line">Preparing model loader...</p>
        </div>

        <section class="output-panel">
          <p class="output-label">Autoregressive output</p>
          <div id="output-text" class="output-text"></div>
        </section>

        <footer class="sidebar-footer">
          <p>Made by Luxen LLC</p>
          <a href="https://x.com/ganstlr" target="_blank" rel="noreferrer">x.com/ganstlr</a>
        </footer>
      </aside>

      <section class="viewer-shell">
        <div id="scene-host" class="scene-host"></div>
      </section>
    </div>
  `;

  const promptInput = root.querySelector("#prompt-input");
  const generateButton = root.querySelector("#generate-button");
  const statusLine = root.querySelector("#status-line");
  const outputText = root.querySelector("#output-text");
  const sceneHost = root.querySelector("#scene-host");

  const scene = new TransformerScene(sceneHost);
  const debugState = {
    prompt: DEFAULT_PROMPT,
    status: "",
    output: "",
    busy: false,
    outputTokenCount: 0,
    queryToken: "",
    topPredictions: [],
  };

  let generationToken = 0;
  let playbackQueue = [];
  let playbackRunning = false;
  let pendingDone = null;
  let displayedOutput = createEmptyOutput();

  async function boot() {
    setStatus("Loading model architecture...");
    const loaded = await model.ensureLoaded(handleProgress);
    scene.setArchitecture(describeArchitecture(loaded));
    setStatus(`${loaded.modelId.split("/").pop()} ready. Starting a sample decode...`);
    await runGeneration(DEFAULT_PROMPT);
  }

  async function runGeneration(prompt) {
    const turnId = ++generationToken;
    const cleanPrompt = prompt.trim() || DEFAULT_PROMPT;
    debugState.prompt = cleanPrompt;

    resetPlaybackState();
    generateButton.disabled = true;
    outputText.classList.add("is-streaming");
    outputText.textContent = "";
    scene.setBusy(true);
    scene.setStepActivity(null);
    debugState.busy = true;
    debugState.queryToken = "";
    debugState.topPredictions = [];
    syncDebugState();
    scene.setOutputTokens([], [], { activeIndex: -1 });

    try {
      await model.generate(cleanPrompt, {
        onProgress: handleProgress,
        onStatus: setStatus,
        onReady: ({ input, architecture }) => {
          if (turnId !== generationToken) {
            return;
          }

          scene.setArchitecture(architecture);
          scene.setPromptTokens(input.tokens, input.ids);
          scene.setOutputTokens([], [], { activeIndex: -1 });
          scene.setStepActivity(null);
          setStatus(`Running ${architecture.label} autoregressive decode...`);
        },
        onStep: ({ step }) => {
          if (turnId !== generationToken) {
            return;
          }

          enqueueStep(turnId, step);
        },
        onDone: (payload) => {
          if (turnId !== generationToken) {
            return;
          }

          pendingDone = payload;
          maybeFinalize(turnId);
        },
      });

      maybeFinalize(turnId);
    } catch (error) {
      if (turnId === generationToken) {
        handleFailure(error);
      }
    }
  }

  function enqueueStep(turnId, step) {
    playbackQueue.push(step);
    if (!playbackRunning) {
      playbackLoop(turnId).catch((error) => {
        if (turnId === generationToken) {
          handleFailure(error);
        }
      });
    }
  }

  async function playbackLoop(turnId) {
    playbackRunning = true;

    while (playbackQueue.length && turnId === generationToken) {
      const next = playbackQueue.shift();
      const nextIndex = displayedOutput.tokens.length + 1;
      scene.setStepActivity(next.activity);
      debugState.queryToken = next.activity.queryToken;
      debugState.topPredictions = next.topPredictions;
      syncDebugState();
      const pulseDuration = scene.enqueueTokenPulse(next.token, nextIndex);

      setStatus(`Evaluating token ${nextIndex} from ${truncateToken(next.activity.queryToken)}...`);
      await wait(resolvePreRevealDelay(next.piece, pulseDuration));

      displayedOutput.ids.push(next.id);
      displayedOutput.tokens.push(next.token);
      displayedOutput.pieces.push(next.piece);
      displayedOutput.text = await model.decodeTokens(displayedOutput.ids);

      outputText.textContent = displayedOutput.text;
      debugState.output = displayedOutput.text;
      debugState.outputTokenCount = displayedOutput.tokens.length;
      syncDebugState();
      scene.setOutputTokens(displayedOutput.tokens, displayedOutput.ids, {
        activeIndex: displayedOutput.tokens.length - 1,
      });
      setStatus(`Selected token ${displayedOutput.tokens.length}: ${truncateToken(next.token)}`);

      await wait(resolvePostRevealDelay(next.piece, pulseDuration));
    }

    playbackRunning = false;
    maybeFinalize(turnId);
  }

  function maybeFinalize(turnId) {
    if (turnId !== generationToken || playbackRunning || playbackQueue.length || !pendingDone) {
      return;
    }

    if (displayedOutput.tokens.length < pendingDone.output.tokens.length) {
      return;
    }

    outputText.textContent = pendingDone.continuation || displayedOutput.text || "(no continuation returned)";
    outputText.classList.remove("is-streaming");
    debugState.output = outputText.textContent;
    debugState.busy = false;
    debugState.outputTokenCount = displayedOutput.tokens.length;
    debugState.topPredictions = pendingDone.lastActivity?.topPredictions ?? [];
    debugState.queryToken = pendingDone.lastActivity?.queryToken ?? "";
    syncDebugState();
    scene.setOutputTokens(displayedOutput.tokens, displayedOutput.ids, {
      activeIndex: displayedOutput.tokens.length - 1,
    });
    scene.setStepActivity(pendingDone.lastActivity ?? null);
    scene.setBusy(false);
    generateButton.disabled = false;
    setStatus(`${pendingDone.architecture.label} decoded ${displayedOutput.tokens.length} new tokens.`);
    pendingDone = null;
  }

  function resetPlaybackState() {
    playbackQueue = [];
    playbackRunning = false;
    pendingDone = null;
    displayedOutput = createEmptyOutput();
  }

  function handleProgress(info) {
    if (info?.label) {
      setStatus(info.label);
    }
  }

  function handleFailure(error) {
    console.error(error);
    setStatus("Model loading or generation failed. Open devtools for details.");
    outputText.classList.remove("is-streaming");
    outputText.textContent = "The browser model failed to run.";
    debugState.output = outputText.textContent;
    debugState.busy = false;
    debugState.topPredictions = [];
    debugState.queryToken = "";
    syncDebugState();
    generateButton.disabled = false;
    scene.setStepActivity(null);
    scene.setBusy(false);
  }

  function setStatus(text) {
    statusLine.textContent = text;
    debugState.status = text;
    syncDebugState();
  }

  generateButton.addEventListener("click", () => {
    runGeneration(promptInput.value).catch((error) => {
      handleFailure(error);
    });
  });

  promptInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      runGeneration(promptInput.value).catch((error) => {
        handleFailure(error);
      });
    }
  });

  syncDebugState();
  boot().catch((error) => {
    handleFailure(error);
  });

  function syncDebugState() {
    const payload = {
      ...debugState,
      selectedStage: scene.getDebugState().selectedId,
    };
    window.__transformerViewerState = payload;
    window.render_game_to_text = () => JSON.stringify(payload);
    window.advanceTime = () => { };
  }
}

function createEmptyOutput() {
  return {
    ids: [],
    tokens: [],
    pieces: [],
    text: "",
  };
}

function resolvePlaybackDelay(piece) {
  const compact = piece.replace(/\s+/g, "");
  let delay = BASE_PLAYBACK_DELAY_MS + compact.length * 34;

  if (/[,.]/.test(piece)) {
    delay += 130;
  }
  if (/[!?;:]/.test(piece)) {
    delay += 210;
  }
  if (/[.]/.test(piece)) {
    delay += 260;
  }
  if (/\n/.test(piece)) {
    delay += 220;
  }

  return Math.min(940, delay);
}

function wait(duration) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, duration);
  });
}

function resolvePreRevealDelay(piece, pulseDuration) {
  const revealDelay = Math.max(620, pulseDuration * 0.68);
  return Math.min(revealDelay, resolvePlaybackDelay(piece) * 0.72);
}

function resolvePostRevealDelay(piece, pulseDuration) {
  const total = Math.max(resolvePlaybackDelay(piece), pulseDuration * 0.84);
  return Math.max(140, total - resolvePreRevealDelay(piece, pulseDuration));
}

function truncateToken(token) {
  const clean = token.replaceAll("�", "byte");
  return clean.length > 16 ? `${clean.slice(0, 15)}...` : clean;
}

function describeArchitecture(loaded) {
  return {
    modelId: loaded.modelId,
    label: loaded.modelId.split("/").pop(),
    layers: Number(loaded.config.n_layer ?? loaded.config.num_hidden_layers ?? 6),
    heads: Number(loaded.config.n_head ?? loaded.config.num_attention_heads ?? 12),
    hiddenSize: Number(loaded.config.n_embd ?? loaded.config.hidden_size ?? 768),
    contextLength: Number(loaded.config.n_positions ?? loaded.config.max_position_embeddings ?? 1024),
    vocabSize: Number(loaded.config.vocab_size ?? 50257),
    activation: loaded.config.activation_function ?? loaded.config.hidden_act ?? "gelu",
    tokenizerKind: "Byte-level BPE",
  };
}
