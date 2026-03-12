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
          <div class="sidebar-footer-links">
            <a
              href="https://x.com/ganstlr"
              target="_blank"
              rel="noreferrer"
              aria-label="X profile"
              title="X profile"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M18.901 2H21.98l-6.726 7.686L23.167 22h-6.193l-4.85-7.297L5.74 22H2.659l7.194-8.224L.833 2h6.35l4.384 6.704L18.901 2Zm-1.083 18.13h1.706L6.26 3.774H4.43l13.388 16.357Z"
                  fill="currentColor"
                />
              </svg>
            </a>
            <a
              href="https://github.com/g4nesh/interactive-transformer"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 .5C5.648.5.5 5.648.5 12a11.5 11.5 0 0 0 7.86 10.918c.575.105.785-.25.785-.555 0-.273-.01-1-.016-1.962-3.197.695-3.872-1.54-3.872-1.54-.523-1.328-1.277-1.682-1.277-1.682-1.044-.713.079-.699.079-.699 1.154.081 1.761 1.186 1.761 1.186 1.026 1.758 2.692 1.25 3.348.956.104-.743.402-1.25.732-1.537-2.552-.29-5.236-1.276-5.236-5.68 0-1.254.448-2.28 1.183-3.083-.12-.289-.513-1.457.111-3.038 0 0 .965-.309 3.162 1.178A10.97 10.97 0 0 1 12 6.04c.975.005 1.958.132 2.876.388 2.195-1.487 3.158-1.178 3.158-1.178.626 1.58.233 2.749.114 3.038.737.803 1.18 1.83 1.18 3.083 0 4.415-2.688 5.386-5.248 5.67.414.357.783 1.061.783 2.139 0 1.545-.014 2.79-.014 3.17 0 .308.207.666.79.553A11.502 11.502 0 0 0 23.5 12C23.5 5.648 18.352.5 12 .5Z"
                  fill="currentColor"
                />
              </svg>
            </a>
          </div>
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
