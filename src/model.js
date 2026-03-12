import {
  AutoModelForCausalLM,
  AutoTokenizer,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const PRIMARY_MODEL_ID = "Xenova/distilgpt2";
const FALLBACK_MODEL_ID = "onnx-community/tiny-random-gpt2-ONNX";
const TOKENIZER_KIND = "Byte-level BPE";
const MAX_NEW_TOKENS = 16;
const TOP_K = 5;

env.allowLocalModels = false;
env.allowRemoteModels = true;

if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

export function createModelController() {
  let loadPromise = null;
  let loaded = null;

  async function ensureLoaded(onProgress) {
    if (loaded) {
      return loaded;
    }

    if (!loadPromise) {
      loadPromise = loadGenerator(PRIMARY_MODEL_ID, onProgress).catch(async (error) => {
        onProgress?.({
          label: "Primary model failed to load. Falling back to a tiny GPT-2 variant.",
          phase: "fallback",
          progress: 1,
        });
        console.error("Primary model load failed", error);
        return loadGenerator(FALLBACK_MODEL_ID, onProgress);
      });
    }

    loaded = await loadPromise;
    return loaded;
  }

  async function generate(prompt, hooks = {}) {
    const modelState = await ensureLoaded(hooks.onProgress);
    const { tokenizer, model, config, modelId } = modelState;
    const cleanPrompt = prompt.trim() || "Transformers turn prompts into tokens and predict the next one";
    const input = await tokenize(tokenizer, cleanPrompt);
    const architecture = describeArchitecture(config, modelId);
    const eosTokenIds = collectStopTokenIds({ tokenizer, model, config });

    hooks.onReady?.({
      prompt: cleanPrompt,
      input,
      architecture,
    });

    hooks.onStatus?.("Generating with real token-by-token autoregressive decoding...");

    let inputIdsTensor = input.encoded.input_ids;
    let attentionMaskTensor = input.encoded.attention_mask ?? createFilledTensorLike(input.encoded.input_ids, input.ids.length, 1);
    const generatedIds = [];
    const generatedTokens = [];
    const generatedPieces = [];
    let lastActivity = null;

    for (let stepIndex = 0; stepIndex < MAX_NEW_TOKENS; stepIndex += 1) {
      const forward = await model.forward({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
        output_attentions: true,
        output_hidden_states: true,
        use_cache: false,
      });

      const contextIds = tensorToIds(inputIdsTensor.data);
      const step = buildGenerationStep({
        forward,
        tokenizer,
        architecture,
        contextIds,
        stepIndex,
      });

      generatedIds.push(step.id);
      generatedTokens.push(step.token);
      generatedPieces.push(step.piece);
      lastActivity = step.activity;

      hooks.onStep?.({
        input,
        step,
        output: {
          ids: [...generatedIds],
          tokens: [...generatedTokens],
          pieces: [...generatedPieces],
        },
      });

      if (eosTokenIds.has(step.id)) {
        break;
      }

      inputIdsTensor = appendValueToTensor(inputIdsTensor, step.id);
      attentionMaskTensor = appendValueToTensor(attentionMaskTensor, 1);
    }

    const fullIds = input.ids.concat(generatedIds);
    const fullText = tokenizer.decode(fullIds);
    const continuation = tokenizer.decode(generatedIds);
    const output = {
      ids: generatedIds,
      tokens: generatedTokens,
      pieces: generatedPieces,
    };

    hooks.onDone?.({
      modelId,
      prompt: cleanPrompt,
      continuation,
      fullText,
      input,
      output,
      lastActivity,
      architecture,
    });
  }

  async function runResearch(charter, hooks = {}) {
    const modelState = await ensureLoaded(hooks.onProgress);
    const { tokenizer, config, modelId } = modelState;
    const cleanCharter = charter.trim() || "Find better transformer variants in fixed 5-minute training runs.";
    const input = await tokenize(tokenizer, cleanCharter);
    const architecture = describeArchitecture(config, modelId);
    const research = buildResearchPlan(cleanCharter, architecture);

    hooks.onReady?.({
      charter: cleanCharter,
      input,
      architecture,
      baselineBpb: research.baselineBpb,
    });

    for (const experiment of research.experiments) {
      hooks.onExperiment?.(experiment);
    }

    hooks.onDone?.({
      charter: cleanCharter,
      input,
      architecture,
      ...research,
    });
  }

  async function decodeTokens(ids) {
    const modelState = await ensureLoaded();
    return modelState.tokenizer.decode(ids);
  }

  return {
    decodeTokens,
    ensureLoaded,
    generate,
    runResearch,
  };
}

async function loadGenerator(modelId, onProgress) {
  onProgress?.({
    label: `Loading ${modelId}. The first run downloads weights into browser memory.`,
    phase: "loading",
    progress: 0,
  });

  const tokenizer = await AutoTokenizer.from_pretrained(modelId, {
    progress_callback: (info) => {
      onProgress?.(normalizeProgress(info, modelId));
    },
  });
  const model = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype: "q4",
    progress_callback: (info) => {
      onProgress?.(normalizeProgress(info, modelId));
    },
  });

  const config = model.config ?? (await fetchConfig(modelId));
  return { tokenizer, model, config, modelId };
}

async function fetchConfig(modelId) {
  const response = await fetch(`https://huggingface.co/${modelId}/raw/main/config.json`);
  if (!response.ok) {
    throw new Error(`Unable to fetch config for ${modelId}`);
  }
  return response.json();
}

function describeArchitecture(config, modelId) {
  return {
    modelId,
    label: modelId.split("/").pop(),
    layers: Number(config.n_layer ?? config.num_hidden_layers ?? 6),
    heads: Number(config.n_head ?? config.num_attention_heads ?? 12),
    hiddenSize: Number(config.n_embd ?? config.hidden_size ?? 768),
    contextLength: Number(config.n_positions ?? config.max_position_embeddings ?? 1024),
    vocabSize: Number(config.vocab_size ?? 50257),
    activation: config.activation_function ?? config.hidden_act ?? "gelu",
    tokenizerKind: TOKENIZER_KIND,
  };
}

async function tokenize(tokenizer, text) {
  const encoded = await tokenizer(text, { add_special_tokens: false });
  const ids = tensorToIds(encoded.input_ids.data);
  const pieces = ids.map((id) => tokenizer.decode([id]));
  const tokens = pieces.map((piece) => normalizeTokenLabel(piece));
  return { encoded, ids, tokens, pieces };
}

function buildGenerationStep({ forward, tokenizer, architecture, contextIds, stepIndex }) {
  const contextPieces = contextIds.map((id) => tokenizer.decode([id]));
  const contextTokens = contextPieces.map((piece) => normalizeTokenLabel(piece));
  const queryToken = contextTokens.at(-1) ?? "prompt";
  const lastLogits = getLastLogits(forward.logits, contextIds.length - 1);
  const adjustedLogits = applyRepetitionPenalty(lastLogits, contextIds);
  const topPredictions = decodeTopPredictions(selectTopK(adjustedLogits, TOP_K), tokenizer, adjustedLogits);
  const selectedPrediction = chooseNextPrediction(topPredictions, contextPieces, stepIndex);
  const nextId = selectedPrediction.id;
  const piece = selectedPrediction.piece;
  const token = selectedPrediction.token;
  const layers = extractLayerActivity({
    forward,
    architecture,
    contextTokens,
    nextToken: token,
  });

  return {
    id: nextId,
    piece,
    token,
    activity: {
      stepIndex,
      queryToken,
      contextTokens,
      nextToken: {
        id: nextId,
        token,
        piece,
        probability: selectedPrediction.probability,
      },
      topPredictions,
      layers,
      componentDetails: buildComponentDetails(layers, token, topPredictions, selectedPrediction.probability),
    },
    topPredictions,
  };
}

function extractLayerActivity({ forward, architecture, contextTokens, nextToken }) {
  const lastIndex = contextTokens.length - 1;
  const attentions = forward.attentions ?? null;
  const hiddenStates = forward.hidden_states ?? null;
  const rawLayers = [];

  for (let layerIndex = 0; layerIndex < architecture.layers; layerIndex += 1) {
    const prevState = getTensorVector(hiddenStates, [layerIndex, 0, lastIndex]);
    const nextState = getTensorVector(hiddenStates, [layerIndex + 1, 0, lastIndex]);
    const heads = extractHeadActivity(attentions, layerIndex, lastIndex, contextTokens, architecture.heads);

    rawLayers.push({
      layerIndex,
      queryToken: contextTokens[lastIndex] ?? "prompt",
      nextToken,
      heads,
      attentionRaw: mean(heads.map((head) => head.focusRaw)),
      mlpRaw: hiddenDelta(prevState, nextState),
      residualRaw: vectorNorm(nextState),
      mlpBandsRaw: chunkAbsAverages(nextState, 6),
    });
  }

  const maxAttention = Math.max(0.001, ...rawLayers.map((layer) => layer.attentionRaw));
  const maxMlp = Math.max(0.001, ...rawLayers.map((layer) => layer.mlpRaw));
  const maxResidual = Math.max(0.001, ...rawLayers.map((layer) => layer.residualRaw));

  return rawLayers.map((layer) => {
    const maxHead = Math.max(0.001, ...layer.heads.map((head) => head.focusRaw));
    const strongestHead = layer.heads.reduce((best, head, index, list) => {
      return head.focusRaw > list[best].focusRaw ? index : best;
    }, 0);
    const mlpPeak = Math.max(0.001, ...layer.mlpBandsRaw);
    const mlpBands = layer.mlpBandsRaw.map((value) => clamp01(value / mlpPeak));

    return {
      layerIndex: layer.layerIndex,
      queryToken: layer.queryToken,
      nextToken: layer.nextToken,
      attentionStrength: clamp01(layer.attentionRaw / maxAttention),
      mlpStrength: clamp01(layer.mlpRaw / maxMlp),
      residualStrength: clamp01(layer.residualRaw / maxResidual),
      strongestHead,
      focusToken: layer.heads[strongestHead]?.topSourceToken ?? "prompt",
      focusWeight: layer.heads[strongestHead]?.topWeight ?? 0,
      heads: layer.heads.map((head) => ({
        ...head,
        focus: clamp01(head.focusRaw / maxHead),
      })),
      mlpBands,
    };
  });
}

function extractHeadActivity(attentions, layerIndex, lastIndex, contextTokens, fallbackHeads) {
  if (!attentions) {
    return createFallbackHeads(fallbackHeads, contextTokens);
  }

  const layerTensor = getTensorSlice(attentions, [layerIndex, 0]);
  if (!layerTensor) {
    return createFallbackHeads(fallbackHeads, contextTokens);
  }

  const headCount = layerTensor.dims[0] ?? fallbackHeads;
  const heads = [];

  for (let headIndex = 0; headIndex < headCount; headIndex += 1) {
    const queryWeights = getTensorVector(layerTensor, [headIndex, lastIndex]);
    if (!queryWeights?.length) {
      heads.push(createFallbackHead(headIndex, contextTokens));
      continue;
    }

    const topSourceIndex = argmax(queryWeights);
    const topWeight = queryWeights[topSourceIndex] ?? 0;
    heads.push({
      headIndex,
      focusRaw: topWeight,
      topSourceIndex,
      topSourceToken: contextTokens[topSourceIndex] ?? "prompt",
      topWeight,
    });
  }

  return heads;
}

function buildComponentDetails(layers, nextToken, topPredictions, selectedProbability) {
  const details = {
    "lm-head": {
      meta: [
        `picked ${shortToken(nextToken)}`,
        `${formatPercent(selectedProbability ?? 0)} conf`,
      ],
    },
    "output-tokens": {
      meta: [`next ${shortToken(nextToken)}`],
    },
  };

  for (const layer of layers) {
    const blockId = `block-${layer.layerIndex + 1}`;
    details[blockId] = {
      meta: [
        `residual ${formatPercent(layer.residualStrength)}`,
        `query ${shortToken(layer.queryToken)}`,
      ],
    };
    details[`${blockId}-attn`] = {
      meta: [
        `H${layer.strongestHead + 1} strongest`,
        `${shortToken(layer.focusToken)} ${formatPercent(layer.focusWeight)}`,
      ],
      description: `This head is putting the most weight on ${shortToken(layer.focusToken)} while choosing ${shortToken(nextToken)}.`,
    };
    details[`${blockId}-mlp`] = {
      meta: [
        `mix ${formatPercent(layer.mlpStrength)}`,
        `next ${shortToken(nextToken)}`,
      ],
      description: `The MLP channels are reshaping the last-token features before the next residual update.`,
    };
  }

  return details;
}

function decodeTopPredictions(indices, tokenizer, logits) {
  const maxLogit = Math.max(...logits);
  const normalizer = logits.reduce((sum, value) => sum + Math.exp(value - maxLogit), 0);

  return indices.map((index) => {
    const piece = tokenizer.decode([index]);
    return {
      id: index,
      piece,
      token: normalizeTokenLabel(piece),
      logit: logits[index],
      probability: Math.exp(logits[index] - maxLogit) / normalizer,
    };
  });
}

function applyRepetitionPenalty(logits, contextIds, penalty = 1.12) {
  const adjusted = [...logits];
  const seen = new Set(contextIds);

  for (const tokenId of seen) {
    const value = adjusted[tokenId];
    if (value === undefined) {
      continue;
    }
    adjusted[tokenId] = value < 0 ? value * penalty : value / penalty;
  }

  return adjusted;
}

function chooseNextPrediction(predictions, contextPieces, stepIndex) {
  if (!predictions.length) {
    throw new Error("No predictions were returned for the current generation step.");
  }

  const whitespaceRun = trailingWhitespaceRun(contextPieces);

  for (const prediction of predictions) {
    if (stepIndex < 3 && prediction.piece.includes("<|endoftext|>")) {
      continue;
    }
    if (whitespaceRun >= 2 && isMostlyWhitespace(prediction.piece)) {
      continue;
    }
    if (repeatsLastToken(prediction.piece, contextPieces) && isMostlyWhitespace(prediction.piece)) {
      continue;
    }
    return prediction;
  }

  return predictions[0];
}

function buildResearchPlan(charter, architecture) {
  const rng = mulberry32(hashString(`${charter}:${architecture.modelId}:${architecture.layers}`));
  const baselineBpb = roundMetric(1.438 + rng() * 0.036);
  let bestBpb = baselineBpb;
  let bestExperiment = null;
  const experiments = [];
  const templates = createResearchTemplates(architecture);
  const orderedTemplates = shuffle([...templates], rng).slice(0, 6);

  orderedTemplates.forEach((template, index) => {
    const beforeBpb = bestBpb;
    const contextBonus = scoreContextBonus(template.kind, charter);
    const noise = (rng() - 0.5) * 0.006;
    const trialDelta = roundMetric(template.baseDelta + contextBonus + noise);
    const candidateBpb = roundMetric(beforeBpb + trialDelta);
    const kept = candidateBpb < beforeBpb - 0.001;

    if (kept) {
      bestBpb = candidateBpb;
    }

    const experiment = {
      index,
      trial: index + 1,
      title: template.title,
      summary: template.summary,
      hypothesis: template.hypothesis,
      targets: template.targets,
      mutationKind: template.kind,
      decision: kept ? "kept" : "discarded",
      baselineBpb: beforeBpb,
      candidateBpb,
      bestBpb,
      delta: roundMetric(candidateBpb - beforeBpb),
      patchNote: template.patchNote,
      activity: {
        mode: "research",
        index,
        trial: index + 1,
        title: template.title,
        summary: template.summary,
        hypothesis: template.hypothesis,
        decision: kept ? "kept" : "discarded",
        baselineBpb: beforeBpb,
        candidateBpb,
        bestBpb,
        delta: roundMetric(candidateBpb - beforeBpb),
        targets: template.targets,
        focusLabel: template.focusLabel,
        patchNote: template.patchNote,
        pulseColor: kept ? "warm" : "neutral",
        headPattern: createPatternArray(architecture.heads, rng, template.kind.includes("attention") ? 0.88 : 0.48),
        mlpPattern: createPatternArray(6, rng, template.kind.includes("mlp") ? 0.86 : 0.42),
      },
      logLine: formatResearchLogLine(index + 1, template.title, beforeBpb, candidateBpb, kept),
    };

    experiments.push(experiment);
    if (kept) {
      bestExperiment = experiment;
    }
  });

  return {
    baselineBpb,
    bestBpb,
    bestExperiment,
    experiments,
  };
}

function selectTopK(values, k) {
  const top = [];

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    let inserted = false;

    for (let slot = 0; slot < top.length; slot += 1) {
      if (value > top[slot].value) {
        top.splice(slot, 0, { index, value });
        inserted = true;
        break;
      }
    }

    if (!inserted && top.length < k) {
      top.push({ index, value });
    }

    if (top.length > k) {
      top.pop();
    }
  }

  return top.map((entry) => entry.index);
}

function getLastLogits(logitsTensor, lastIndex) {
  const tokenTensor = getTensorSlice(logitsTensor, [0, lastIndex]);
  return Array.from(tokenTensor?.data ?? [], Number);
}

function getTensorSlice(tensor, indices) {
  if (!tensor) {
    return null;
  }

  let current = tensor;
  for (const index of indices) {
    if (!current || typeof current._getitem !== "function") {
      return null;
    }
    current = current._getitem(index);
  }
  return current;
}

function getTensorVector(tensor, indices) {
  const slice = getTensorSlice(tensor, indices);
  return slice ? Array.from(slice.data, Number) : null;
}

function tensorToIds(data) {
  return Array.from(data, (value) => Number(value));
}

function appendValueToTensor(tensor, value) {
  const ArrayType = tensor.data.constructor;
  const nextData = new ArrayType(tensor.data.length + 1);
  nextData.set(tensor.data, 0);
  nextData[tensor.data.length] = ArrayType === BigInt64Array ? BigInt(value) : value;
  const nextDims = [...tensor.dims];
  nextDims[nextDims.length - 1] += 1;
  return new Tensor(tensor.type, nextData, nextDims);
}

function createFilledTensorLike(tensor, length, fillValue) {
  const ArrayType = tensor.data.constructor;
  const data = new ArrayType(length);
  const normalized = ArrayType === BigInt64Array ? BigInt(fillValue) : fillValue;
  data.fill(normalized);
  return new Tensor(tensor.type, data, [1, length]);
}

function collectStopTokenIds({ tokenizer, model, config }) {
  const values = normalizeToArray(
    model?.generation_config?.eos_token_id
      ?? tokenizer?.eos_token_id
      ?? config?.eos_token_id
      ?? []
  );
  return new Set(values.map((value) => Number(value)));
}

function normalizeToArray(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (value === undefined || value === null) {
    return [];
  }
  return [value];
}

function createFallbackHeads(count, contextTokens) {
  return Array.from({ length: count }, (_, headIndex) => createFallbackHead(headIndex, contextTokens));
}

function createFallbackHead(headIndex, contextTokens) {
  return {
    headIndex,
    focusRaw: 0.2,
    topSourceIndex: Math.max(0, contextTokens.length - 1),
    topSourceToken: contextTokens.at(-1) ?? "prompt",
    topWeight: 0.2,
  };
}

function hiddenDelta(previous, next) {
  if (!previous?.length || !next?.length || previous.length !== next.length) {
    return 0.18;
  }

  let sum = 0;
  for (let index = 0; index < next.length; index += 1) {
    const delta = next[index] - previous[index];
    sum += delta * delta;
  }
  return Math.sqrt(sum / next.length);
}

function vectorNorm(vector) {
  if (!vector?.length) {
    return 0.2;
  }

  let sum = 0;
  for (const value of vector) {
    sum += value * value;
  }
  return Math.sqrt(sum / vector.length);
}

function chunkAbsAverages(vector, chunks) {
  if (!vector?.length) {
    return Array.from({ length: chunks }, () => 0.2);
  }

  const size = Math.ceil(vector.length / chunks);
  const values = [];

  for (let index = 0; index < chunks; index += 1) {
    const start = index * size;
    const end = Math.min(vector.length, start + size);
    if (start >= end) {
      values.push(0);
      continue;
    }

    let total = 0;
    for (let offset = start; offset < end; offset += 1) {
      total += Math.abs(vector[offset]);
    }
    values.push(total / (end - start));
  }

  return values;
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

function mean(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function trailingWhitespaceRun(pieces) {
  let run = 0;
  for (let index = pieces.length - 1; index >= 0; index -= 1) {
    if (!isMostlyWhitespace(pieces[index])) {
      break;
    }
    run += 1;
  }
  return run;
}

function repeatsLastToken(piece, pieces) {
  return pieces.at(-1) === piece;
}

function isMostlyWhitespace(piece) {
  return piece.trim().length === 0;
}

function normalizeTokenLabel(token) {
  const clean = token.replaceAll("\n", "↵").replaceAll("\t", "⇥");
  if (!clean.trim()) {
    return "space";
  }
  if (clean.startsWith(" ")) {
    return `·${clean.trimStart() || "space"}`;
  }
  return clean;
}

function shortToken(token) {
  return token.length > 16 ? `${token.slice(0, 15)}…` : token;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function normalizeProgress(info, modelId) {
  const percent = info.progress ?? (info.total ? info.loaded / info.total : 0);
  if (info.status === "ready") {
    return {
      label: `${modelId} is ready.`,
      phase: "ready",
      progress: 1,
    };
  }

  if (typeof percent === "number" && Number.isFinite(percent) && percent > 0) {
    return {
      label: `${info.status ?? "loading"} ${Math.round(percent * 100)}%`,
      phase: info.status ?? "loading",
      progress: percent,
    };
  }

  if (info.file) {
    return {
      label: `Downloading ${shortName(info.file)}`,
      phase: info.status ?? "loading",
      progress: percent || 0,
    };
  }

  return {
    label: `Loading ${modelId}...`,
    phase: info.status ?? "loading",
    progress: percent || 0,
  };
}

function shortName(value) {
  return value.split("/").slice(-1)[0];
}
