import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const COLORS = {
  background: "#2a4763",
  stage: "#08121c",
  panel: "#101d29",
  line: "#4f647a",
  lineSoft: "#24384a",
  shell: "#0b1621",
  token: "#eff6ff",
  tokenSoft: "#d8e6f8",
  cool: "#7bd0ff",
  coolSoft: "#2b6484",
  warm: "#ffbd78",
  warmSoft: "#7b5530",
  neutral: "#b6c8dc",
  glow: "#d8ecff",
};

export class TransformerScene {
  constructor(host) {
    this.host = host;
    this.components = new Map();
    this.activityStrengths = new Map();
    this.stageAnchors = new Map();
    this.connections = [];
    this.connectionLookup = new Map();
    this.interactiveMeshes = [];
    this.pulses = [];
    this.pointer = new THREE.Vector2();
    this.raycaster = new THREE.Raycaster();
    this.hoveredId = null;
    this.selectedId = null;
    this.route = [];
    this.stepActivity = null;
    this.isBusy = false;
    this.outputActiveIndex = -1;
    this.defaultCameraPosition = new THREE.Vector3(13.5, 9.2, 21.6);
    this.defaultCameraTarget = new THREE.Vector3(2.5, -2.2, 2.1);
    this.tempVector = new THREE.Vector3();
    this.tempVectorB = new THREE.Vector3();
    this.pointerDown = null;
    this.dragDistance = 0;

    this.root = document.createElement("div");
    this.root.className = "viewer-root";
    this.host.append(this.root);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.24;
    this.renderer.domElement.style.cursor = "grab";
    this.root.append(this.renderer.domElement);

    this.overlay = document.createElement("div");
    this.overlay.className = "viewer-overlay";
    this.overlay.innerHTML = `
      <div class="viewer-tip">Drag to orbit. Right-drag to pan. Double-click to reset.</div>
      <div class="viewer-predictions">
        <p class="viewer-card-kicker">Next-token logits</p>
        <p class="viewer-prediction-step">Waiting for the first decode step...</p>
        <div class="viewer-prediction-list"></div>
      </div>
      <div class="viewer-card">
        <p class="viewer-card-kicker">Transformer stage</p>
        <h3 class="viewer-card-title">Waiting for model</h3>
        <p class="viewer-card-body">The scene will rebuild once the tokenizer and model config are loaded.</p>
        <div class="viewer-card-meta"></div>
      </div>
    `;
    this.root.append(this.overlay);

    this.infoTitle = this.overlay.querySelector(".viewer-card-title");
    this.infoBody = this.overlay.querySelector(".viewer-card-body");
    this.infoMeta = this.overlay.querySelector(".viewer-card-meta");
    this.predictionStep = this.overlay.querySelector(".viewer-prediction-step");
    this.predictionList = this.overlay.querySelector(".viewer-prediction-list");

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(COLORS.background, 0.015);

    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    this.camera.position.copy(this.defaultCameraPosition);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 48;
    this.controls.minPolarAngle = 0.12;
    this.controls.maxPolarAngle = Math.PI - 0.12;
    this.controls.rotateSpeed = 0.88;
    this.controls.zoomSpeed = 0.92;
    this.controls.panSpeed = 0.72;
    this.controls.target.copy(this.defaultCameraTarget);

    this.architectureGroup = new THREE.Group();
    this.architectureGroup.rotation.x = -0.79;
    this.architectureGroup.rotation.y = -0.34;
    this.architectureGroup.rotation.z = -0.08;
    this.architectureGroup.position.set(-0.9, -0.2, 0.2);
    this.architectureGroup.scale.setScalar(0.86);
    this.scene.add(this.architectureGroup);

    this.addLighting();
    this.addAtmosphere();

    this.boundPointerMove = this.handlePointerMove.bind(this);
    this.boundPointerDown = this.handlePointerDown.bind(this);
    this.boundPointerUp = this.handlePointerUp.bind(this);
    this.boundPointerClick = this.handlePointerClick.bind(this);
    this.boundResetCamera = this.resetCamera.bind(this);
    this.boundResize = this.resize.bind(this);

    this.renderer.domElement.addEventListener("pointerdown", this.boundPointerDown);
    this.renderer.domElement.addEventListener("pointerup", this.boundPointerUp);
    this.renderer.domElement.addEventListener("pointermove", this.boundPointerMove);
    this.renderer.domElement.addEventListener("click", this.boundPointerClick);
    this.renderer.domElement.addEventListener("dblclick", this.boundResetCamera);
    window.addEventListener("resize", this.boundResize);

    this.resize();
    this.renderPredictionPanel(null);
    this.animate = this.animate.bind(this);
    this.animate();
  }

  setArchitecture(architecture) {
    this.architecture = architecture;
    this.clearArchitecture();

    this.components.clear();
    this.stageAnchors.clear();
    this.connections = [];
    this.connectionLookup.clear();
    this.interactiveMeshes.length = 0;
    this.pulses = [];
    this.outputActiveIndex = -1;

    this.connectionGroup = new THREE.Group();
    this.pulseGroup = new THREE.Group();
    this.architectureGroup.add(this.connectionGroup);
    this.architectureGroup.add(this.pulseGroup);

    this.tokenLane = this.createTokenLane({
      id: "prompt-tokens",
      title: "Tokenized prompt",
      description: `${architecture.tokenizerKind} pieces produced from the raw text prompt.`,
      label: "prompt pieces",
      x: -0.45,
      y: 7.0,
      z: -0.08,
      accent: COLORS.tokenSoft,
    });

    this.createPlate({
      id: "tokenizer",
      title: `${architecture.tokenizerKind} tokenizer`,
      description: "Maps raw text into subword pieces before any neural layers run.",
      label: "tokenizer",
      position: [0.1, 5.55, 0],
      size: [11.0, 0.3, 0.95],
      accent: COLORS.tokenSoft,
      meta: [`${architecture.vocabSize.toLocaleString()} vocab`, "subword encoding"],
    });

    this.idLane = this.createTokenLane({
      id: "token-ids",
      title: "Token ids",
      description: "Each prompt piece becomes an integer index in the vocabulary table.",
      label: "token ids",
      x: -0.2,
      y: 4.35,
      z: 0.05,
      accent: COLORS.cool,
    });

    this.createPlate({
      id: "embeddings",
      title: "Token + position embeddings",
      description: "Token ids are projected into dense vectors and combined with position information before the first decoder block.",
      label: "embeddings",
      position: [0.85, 2.35, 0.18],
      size: [12.4, 0.52, 4.2],
      accent: COLORS.cool,
      meta: [`${architecture.hiddenSize} hidden`, `${architecture.contextLength} context`],
    });

    this.layerStack = new THREE.Group();
    this.architectureGroup.add(this.layerStack);

    const route = ["prompt-tokens", "tokenizer", "token-ids", "embeddings"];
    for (let index = 0; index < architecture.layers; index += 1) {
      route.push(...this.createTransformerBlock(index, architecture));
    }

    this.createPlate({
      id: "final-norm",
      title: "Final layer norm",
      description: "Normalizes the residual stream before projection into vocabulary logits.",
      label: "final norm",
      position: [3.2, -3.05, 2.35],
      size: [10.2, 0.24, 1.18],
      accent: COLORS.neutral,
      meta: ["residual cleanup"],
    });
    route.push("final-norm");

    this.createPlate({
      id: "lm-head",
      title: "LM head",
      description: "Projects the final hidden state into vocabulary logits so the next token can be chosen.",
      label: "lm head",
      position: [3.55, -4.25, 2.68],
      size: [10.0, 0.34, 1.52],
      accent: COLORS.warm,
      meta: [`${architecture.vocabSize.toLocaleString()} logits`],
    });
    route.push("lm-head");

    this.outputLane = this.createTokenLane({
      id: "output-tokens",
      title: "Generated continuation",
      description: "The selected token is appended to the context and the whole stack runs again autoregressively.",
      label: "output tokens",
      x: 3.55,
      y: -6.2,
      z: 3.02,
      accent: COLORS.warm,
    });
    route.push("output-tokens");

    this.createConnections(route);
    this.route = route;
    this.resetCamera();
    this.select("tokenizer", false);
    this.setStepActivity(null);
  }

  setPromptTokens(tokens, ids) {
    if (!this.tokenLane || !this.idLane) {
      return;
    }

    this.renderLane(this.tokenLane, tokens, {
      mode: "token",
      strategy: "compact",
      accent: COLORS.tokenSoft,
      activeIndex: -1,
    });
    this.renderLane(this.idLane, ids.map((id) => String(id)), {
      mode: "id",
      strategy: "compact",
      accent: COLORS.cool,
      activeIndex: -1,
    });
  }

  setOutputTokens(tokens, ids, options = {}) {
    if (!this.outputLane) {
      return;
    }

    this.outputActiveIndex = options.activeIndex ?? (tokens.length ? tokens.length - 1 : -1);

    if (!tokens.length) {
      this.renderLane(this.outputLane, ["..."], {
        mode: "token",
        strategy: "tail",
        accent: COLORS.warm,
        placeholder: true,
        activeIndex: -1,
      });
      return;
    }

    this.renderLane(this.outputLane, tokens, {
      mode: "token",
      strategy: "tail",
      accent: COLORS.warm,
      activeIndex: this.outputActiveIndex,
    });
  }

  enqueueTokenPulse(token, index = this.outputActiveIndex + 1) {
    if (!this.route.length || !this.pulseGroup) {
      return 0;
    }

    const color = new THREE.Color(index % 2 === 0 ? COLORS.cool : COLORS.warm);
    const duration = 1550 + Math.min(260, this.route.length * 14);
    const group = new THREE.Group();
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 18, 18),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 1.8,
        roughness: 0.1,
        metalness: 0.12,
        transparent: true,
        opacity: 0.95,
      })
    );
    const aura = new THREE.Mesh(
      new THREE.SphereGeometry(0.24, 18, 18),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.2,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    group.add(aura, core);
    this.pulseGroup.add(group);

    this.pulses.push({
      group,
      core,
      aura,
      startedAt: performance.now(),
      duration,
      route: [...this.route],
      color,
      token,
    });

    return duration;
  }

  setBusy(isBusy) {
    this.isBusy = isBusy;
  }

  setStepActivity(activity) {
    this.stepActivity = activity;
    this.activityStrengths.clear();
    this.resetRuntimeActivity();
    this.renderPredictionPanel(activity);

    if (!activity) {
      this.refreshSelectionDetails();
      return;
    }

    for (const layer of activity.layers) {
      const blockId = `block-${layer.layerIndex + 1}`;
      const block = this.components.get(blockId);
      if (block) {
        block.runtimeMeta = [
          `residual ${formatPercent(layer.residualStrength)}`,
          `query ${shortLabel(layer.queryToken)}`,
        ];
        block.activityStrength = 0.18 + layer.residualStrength * 0.68;
        this.activityStrengths.set(blockId, block.activityStrength);
      }

      const attnId = `${blockId}-attn`;
      const attn = this.components.get(attnId);
      if (attn) {
        attn.runtimeMeta = [
          `H${layer.strongestHead + 1} strongest`,
          `${shortLabel(layer.focusToken)} ${formatPercent(layer.focusWeight)}`,
        ];
        attn.runtimeDescription = `Real attention weights show the strongest head focusing on ${shortLabel(layer.focusToken)} while the model evaluates ${shortLabel(activity.nextToken.token)}.`;
        attn.activityStrength = 0.18 + layer.attentionStrength * 0.78;
        this.activityStrengths.set(attnId, attn.activityStrength);
        this.applyActivityBars(attn.activityBars, layer.heads.map((head) => head.focus));
      }

      const mlpId = `${blockId}-mlp`;
      const mlp = this.components.get(mlpId);
      if (mlp) {
        mlp.runtimeMeta = [
          `mix ${formatPercent(layer.mlpStrength)}`,
          `next ${shortLabel(activity.nextToken.token)}`,
        ];
        mlp.runtimeDescription = "These bars represent the current feature-channel magnitude for the last token after the block update.";
        mlp.activityStrength = 0.16 + layer.mlpStrength * 0.82;
        this.activityStrengths.set(mlpId, mlp.activityStrength);
        this.applyActivityBars(mlp.activityBars, layer.mlpBands);
      }
    }

    const finalNorm = this.components.get("final-norm");
    if (finalNorm) {
      finalNorm.runtimeMeta = [`query ${shortLabel(activity.queryToken)}`];
      finalNorm.activityStrength = 0.24 + (activity.layers.at(-1)?.residualStrength ?? 0.2) * 0.62;
      this.activityStrengths.set("final-norm", finalNorm.activityStrength);
    }

    const lmHead = this.components.get("lm-head");
    if (lmHead) {
      lmHead.runtimeMeta = [
        `picked ${shortLabel(activity.nextToken.token)}`,
        `${formatPercent(activity.nextToken.probability)} conf`,
      ];
      lmHead.runtimeDescription = "The LM head is projecting the final hidden state into vocabulary logits; the ranked list shows the exact top-k candidates for this step.";
      lmHead.activityStrength = 0.28 + activity.nextToken.probability * 0.92;
      this.activityStrengths.set("lm-head", lmHead.activityStrength);
    }

    const outputTokens = this.components.get("output-tokens");
    if (outputTokens) {
      outputTokens.runtimeMeta = [`next ${shortLabel(activity.nextToken.token)}`];
      outputTokens.activityStrength = 0.24 + activity.nextToken.probability * 0.66;
      this.activityStrengths.set("output-tokens", outputTokens.activityStrength);
    }

    this.refreshSelectionDetails();
  }

  getDebugState() {
    return {
      hoveredId: this.hoveredId,
      selectedId: this.selectedId,
      outputActiveIndex: this.outputActiveIndex,
      busy: this.isBusy,
      topPrediction: this.stepActivity?.topPredictions?.[0]?.token ?? null,
    };
  }

  updateSelectionDetails(title, description, meta = []) {
    this.infoTitle.textContent = title;
    this.infoBody.textContent = description;
    this.infoMeta.innerHTML = meta.map((value) => `<span>${value}</span>`).join("");
  }

  refreshSelectionDetails() {
    const target = this.components.get(this.selectedId);
    if (!target) {
      return;
    }

    const description = target.runtimeDescription ?? target.description;
    const meta = [...(target.meta ?? [])];
    if (target.runtimeMeta?.length) {
      meta.unshift(...target.runtimeMeta);
    }
    this.updateSelectionDetails(target.title, description, meta.slice(0, 5));
  }

  destroy() {
    window.removeEventListener("resize", this.boundResize);
    this.renderer.domElement.removeEventListener("pointerdown", this.boundPointerDown);
    this.renderer.domElement.removeEventListener("pointerup", this.boundPointerUp);
    this.renderer.domElement.removeEventListener("pointermove", this.boundPointerMove);
    this.renderer.domElement.removeEventListener("click", this.boundPointerClick);
    this.renderer.domElement.removeEventListener("dblclick", this.boundResetCamera);
    this.controls.dispose();
    this.renderer.dispose();
    this.host.innerHTML = "";
  }

  addLighting() {
    const hemi = new THREE.HemisphereLight("#b3d7ff", "#081018", 1.25);
    const key = new THREE.DirectionalLight("#f5f7ff", 1.28);
    key.position.set(8, 11, 9);
    const fill = new THREE.PointLight("#8abfff", 1.3, 28);
    fill.position.set(-9, 2, 10);
    const warm = new THREE.PointLight("#ffb36c", 0.94, 22);
    warm.position.set(7, -5, 8);

    this.scene.add(hemi, key, fill, warm);
  }

  addAtmosphere() {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(38, 38),
      new THREE.MeshBasicMaterial({
        color: "#102030",
        transparent: true,
        opacity: 0.08,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(2.2, -10.8, 1.2);

    const backdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(34, 22),
      new THREE.MeshBasicMaterial({
        color: "#40607d",
        transparent: true,
        opacity: 0.04,
      })
    );
    backdrop.position.set(4.8, 0.2, -9.5);
    backdrop.rotation.y = -0.18;

    const warmHalo = new THREE.Mesh(
      new THREE.PlaneGeometry(13, 9),
      new THREE.MeshBasicMaterial({
        color: "#ffb36c",
        transparent: true,
        opacity: 0.026,
      })
    );
    warmHalo.position.set(6.4, -4.2, 6.2);
    warmHalo.rotation.y = -0.58;

    const coolHalo = new THREE.Mesh(
      new THREE.PlaneGeometry(16, 10),
      new THREE.MeshBasicMaterial({
        color: COLORS.cool,
        transparent: true,
        opacity: 0.024,
      })
    );
    coolHalo.position.set(-5.4, 2.8, 5.6);
    coolHalo.rotation.y = 0.36;

    this.scene.add(floor, backdrop, warmHalo, coolHalo);
  }

  createTokenLane({ id, title, description, label, x, y, z, accent }) {
    const lane = new THREE.Group();
    lane.position.set(x, y, z);
    this.architectureGroup.add(lane);

    const plate = this.createInteractiveMesh(
      {
        id,
        title,
        description,
        meta: [],
        accent,
      },
      new THREE.BoxGeometry(11.6, 0.24, 0.82),
      [0, -0.2, 0],
      COLORS.stage
    );
    lane.add(plate.group);

    const guide = new THREE.Mesh(
      new THREE.BoxGeometry(11.1, 0.05, 0.12),
      new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 0.28,
        transparent: true,
        opacity: 0.18,
        roughness: 0.25,
        metalness: 0.12,
      })
    );
    guide.position.set(0, -0.03, 0.34);
    lane.add(guide);

    const underGlow = new THREE.Mesh(
      new THREE.BoxGeometry(10.8, 0.02, 0.7),
      new THREE.MeshBasicMaterial({
        color: accent,
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
      })
    );
    underGlow.position.set(0, -0.06, 0.04);
    lane.add(underGlow);

    const labelSprite = createTextSprite(label, {
      size: 82,
      scale: 1.8,
      opacity: 0.84,
      y: 0.62,
    });
    lane.add(labelSprite);

    lane.userData.labels = [];
    lane.userData.boxes = [];
    lane.userData.materials = [];

    this.stageAnchors.set(id, {
      in: new THREE.Vector3(x - 5.2, y - 0.18, z),
      out: new THREE.Vector3(x + 5.2, y - 0.18, z),
    });

    return lane;
  }

  renderLane(lane, values, options) {
    lane.userData.labels.forEach((entry) => lane.remove(entry));
    lane.userData.boxes.forEach((entry) => lane.remove(entry));
    lane.userData.labels = [];
    lane.userData.boxes = [];
    lane.userData.materials = [];

    const { values: visibleValues, activeDisplayIndex } = clipLaneValues(
      values,
      8,
      options.activeIndex,
      options.strategy
    );
    const spacing = 1.0;
    const startX = -((visibleValues.length - 1) * spacing) / 2;
    const accentColor = new THREE.Color(options.accent);

    visibleValues.forEach((value, index) => {
      const isActive = index === activeDisplayIndex;
      const fillColor = new THREE.Color(options.placeholder ? COLORS.panel : COLORS.stage).lerp(
        accentColor,
        isActive ? 0.18 : 0.04
      );
      const material = new THREE.MeshStandardMaterial({
        color: fillColor,
        emissive: isActive ? accentColor : fillColor.clone(),
        emissiveIntensity: isActive ? 1.5 : 0.24,
        roughness: 0.28,
        metalness: 0.08,
        transparent: true,
        opacity: options.placeholder ? 0.38 : 0.96,
      });

      const box = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.18, 0.76), material);
      box.position.set(startX + index * spacing, 0.05, 0);
      box.scale.set(isActive ? 1.15 : 1, isActive ? 1.08 : 1, isActive ? 1.12 : 1);
      lane.add(box);
      lane.userData.boxes.push(box);
      lane.userData.materials.push(material);

      const label = createTextSprite(formatLaneLabel(value, options.mode), {
        size: options.mode === "id" ? 80 : 72,
        scale: options.mode === "id" ? 0.92 : 1.04,
        y: 0.82,
        opacity: isActive ? 1 : 0.84,
      });
      label.position.x = box.position.x;
      lane.add(label);
      lane.userData.labels.push(label);
    });
  }

  createTransformerBlock(index, architecture) {
    const layerNumber = index + 1;
    const blockId = `block-${layerNumber}`;
    const baseY = 1.1 - index * 1.06;
    const baseX = 1.28 + index * 0.44;
    const baseZ = 0.76 + index * 0.38;
    const group = new THREE.Group();
    group.position.set(baseX, baseY, baseZ);
    this.layerStack.add(group);

    const shell = this.createInteractiveMesh(
      {
        id: blockId,
        title: `Transformer block ${layerNumber}`,
        description: `Layer ${layerNumber} repeats pre-norm, masked self-attention, residual mixing, pre-norm, MLP, and another residual update.`,
        meta: [`${architecture.heads} heads`, `${architecture.activation} MLP`],
        accent: COLORS.neutral,
      },
      new THREE.BoxGeometry(12.8, 0.3, 4.94),
      [0, 0, 0],
      COLORS.shell
    );
    group.add(shell.group);

    const shadow = new THREE.Mesh(
      new THREE.BoxGeometry(12.9, 0.04, 5.0),
      new THREE.MeshBasicMaterial({
        color: "#03070d",
        transparent: true,
        opacity: 0.16,
      })
    );
    shadow.position.set(0.06, -0.12, 0.08);
    group.add(shadow);

    const deck = new THREE.Mesh(
      new THREE.BoxGeometry(12.2, 0.03, 4.38),
      new THREE.MeshBasicMaterial({
        color: COLORS.lineSoft,
        transparent: true,
        opacity: 0.32,
      })
    );
    deck.position.set(0, 0.08, 0);
    group.add(deck);

    const shellHalo = new THREE.Mesh(
      new THREE.BoxGeometry(12.7, 0.04, 4.7),
      new THREE.MeshBasicMaterial({
        color: COLORS.neutral,
        transparent: true,
        opacity: 0.05,
        blending: THREE.AdditiveBlending,
      })
    );
    shellHalo.position.set(0, 0.04, 0);
    group.add(shellHalo);

    const attnId = `${blockId}-attn`;
    const mlpId = `${blockId}-mlp`;
    const attn = this.createInteractiveMesh(
      {
        id: attnId,
        title: `Block ${layerNumber} masked attention`,
        description: `Each token mixes information from earlier tokens only. This block uses ${architecture.heads} parallel attention heads with a causal mask.`,
        meta: [`${architecture.heads} heads`, "causal mask"],
        accent: COLORS.cool,
      },
      new THREE.BoxGeometry(4.2, 0.22, 4.24),
      [-1.95, 0.22, 0],
      COLORS.coolSoft
    );
    group.add(attn.group);

    const mlp = this.createInteractiveMesh(
      {
        id: mlpId,
        title: `Block ${layerNumber} MLP`,
        description: `The feed-forward sublayer expands and compresses each token independently using ${architecture.activation}.`,
        meta: [architecture.activation, "position-wise"],
        accent: COLORS.warm,
      },
      new THREE.BoxGeometry(4.55, 0.22, 4.24),
      [2.4, 0.22, 0],
      COLORS.warmSoft
    );
    group.add(mlp.group);

    const headActivity = this.createActivityBars({
      count: architecture.heads,
      color: COLORS.cool,
      centerX: -1.95,
      centerZ: 0,
      columns: 4,
      spacingX: 0.76,
      spacingZ: 0.82,
      baseY: 0.6,
      size: [0.38, 0.16, 0.26],
    });
    group.add(headActivity.group);
    this.components.get(attnId).activityBars = headActivity.bars;

    const mlpActivity = this.createActivityBars({
      count: 6,
      color: COLORS.warm,
      centerX: 2.4,
      centerZ: 0,
      columns: 3,
      spacingX: 0.82,
      spacingZ: 1.0,
      baseY: 0.62,
      size: [0.44, 0.2, 0.32],
    });
    group.add(mlpActivity.group);
    this.components.get(mlpId).activityBars = mlpActivity.bars;

    this.addBlockDecor(group, -1.95, COLORS.cool, "heads");
    this.addBlockDecor(group, 2.4, COLORS.warm, "mlp");
    this.addBlockFlowLines(group);

    const normA = createThinPlate(0.9, 3.92, "#cad6e6");
    normA.position.set(-4.72, 0.19, 0);
    group.add(normA);

    const normB = createThinPlate(0.9, 3.92, "#cad6e6");
    normB.position.set(0.2, 0.19, 0);
    group.add(normB);

    const residualRail = this.createInlineRail(
      [
        new THREE.Vector3(-6.2, 0.22, 0),
        new THREE.Vector3(-4.2, 0.22, 0),
        new THREE.Vector3(0.05, 0.22, 0),
        new THREE.Vector3(4.9, 0.22, 0),
        new THREE.Vector3(6.2, 0.22, 0),
      ],
      COLORS.neutral,
      0.28,
      0.028
    );
    group.add(residualRail);

    const blockLabel = createTextSprite(`block ${layerNumber}`, {
      size: 70,
      scale: 1.12,
      opacity: 0.78,
      y: 0.72,
    });
    blockLabel.position.set(-5.85, 0.54, 2.42);
    group.add(blockLabel);

    this.stageAnchors.set(blockId, {
      in: new THREE.Vector3(baseX - 6.1, baseY + 0.18, baseZ),
      out: new THREE.Vector3(baseX + 6.1, baseY + 0.18, baseZ),
    });
    this.stageAnchors.set(attnId, {
      in: new THREE.Vector3(baseX - 4.05, baseY + 0.18, baseZ),
      out: new THREE.Vector3(baseX + 0.1, baseY + 0.18, baseZ),
    });
    this.stageAnchors.set(mlpId, {
      in: new THREE.Vector3(baseX + 0.42, baseY + 0.18, baseZ),
      out: new THREE.Vector3(baseX + 4.72, baseY + 0.18, baseZ),
    });

    return [attnId, mlpId];
  }

  addBlockDecor(group, centerX, color, mode) {
    const accent = new THREE.Color(color);
    if (mode === "heads") {
      for (let index = 0; index < 4; index += 1) {
        const rib = new THREE.Mesh(
          new THREE.BoxGeometry(2.8, 0.05, 0.28),
          new THREE.MeshStandardMaterial({
            color: accent,
            emissive: accent,
            emissiveIntensity: 0.34,
            transparent: true,
            opacity: 0.24,
            roughness: 0.15,
            metalness: 0.08,
          })
        );
        rib.position.set(centerX, 0.36, -0.96 + index * 0.64);
        group.add(rib);
      }

      for (let index = 0; index < 5; index += 1) {
        const startZ = -1.45 + index * 0.7;
        const endZ = -0.95 + index * 0.34;
        const attentionLine = this.createInlineRail(
          [
            new THREE.Vector3(centerX - 1.5, 0.34, startZ),
            new THREE.Vector3(centerX - 0.3, 0.42, startZ * 0.52),
            new THREE.Vector3(centerX + 1.45, 0.34, endZ),
          ],
          color,
          0.28,
          0.018
        );
        group.add(attentionLine);
      }
      return;
    }

    for (let index = 0; index < 5; index += 1) {
      const height = 0.16 + (index % 2 === 0 ? 0.12 : 0.22);
      const column = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, height, 2.9),
        new THREE.MeshStandardMaterial({
          color: accent,
          emissive: accent,
          emissiveIntensity: 0.28,
          transparent: true,
          opacity: 0.2,
          roughness: 0.18,
          metalness: 0.06,
        })
      );
      column.position.set(centerX - 1.15 + index * 0.58, 0.3 + height * 0.26, 0);
      group.add(column);
    }

    const bridge = this.createInlineRail(
      [
        new THREE.Vector3(centerX - 1.55, 0.44, -1.18),
        new THREE.Vector3(centerX - 0.2, 0.52, -0.18),
        new THREE.Vector3(centerX + 1.42, 0.44, 1.02),
      ],
      color,
      0.26,
      0.018
    );
    group.add(bridge);
  }

  addBlockFlowLines(group) {
    const entry = this.createInlineRail(
      [
        new THREE.Vector3(-4.68, 0.29, 0),
        new THREE.Vector3(-3.55, 0.4, 0),
        new THREE.Vector3(-2.65, 0.4, 0),
      ],
      COLORS.cool,
      0.34,
      0.022
    );
    const attentionToNorm = this.createInlineRail(
      [
        new THREE.Vector3(-1.18, 0.38, 0),
        new THREE.Vector3(-0.44, 0.46, 0),
        new THREE.Vector3(0.12, 0.38, 0),
      ],
      COLORS.cool,
      0.28,
      0.018
    );
    const normToMlp = this.createInlineRail(
      [
        new THREE.Vector3(0.48, 0.29, 0),
        new THREE.Vector3(1.36, 0.4, 0),
        new THREE.Vector3(2.05, 0.4, 0),
      ],
      COLORS.warm,
      0.32,
      0.022
    );
    const mlpToExit = this.createInlineRail(
      [
        new THREE.Vector3(4.74, 0.39, 0),
        new THREE.Vector3(5.36, 0.44, 0),
        new THREE.Vector3(6.12, 0.3, 0),
      ],
      COLORS.warm,
      0.3,
      0.02
    );

    group.add(entry, attentionToNorm, normToMlp, mlpToExit);
  }

  createPlate({ id, title, description, label, position, size, accent, meta }) {
    const plate = this.createInteractiveMesh(
      {
        id,
        title,
        description,
        meta,
        accent,
      },
      new THREE.BoxGeometry(...size),
      position,
      COLORS.stage
    );
    this.architectureGroup.add(plate.group);

    const accentStrip = new THREE.Mesh(
      new THREE.BoxGeometry(size[0] * 0.84, 0.05, 0.14),
      new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 0.42,
        transparent: true,
        opacity: 0.2,
        roughness: 0.14,
        metalness: 0.08,
      })
    );
    accentStrip.position.set(position[0], position[1] + size[1] * 0.62, position[2] + size[2] * 0.36);
    this.architectureGroup.add(accentStrip);

    const accentGlow = new THREE.Mesh(
      new THREE.BoxGeometry(size[0] * 0.8, 0.02, size[2] * 0.8),
      new THREE.MeshBasicMaterial({
        color: accent,
        transparent: true,
        opacity: 0.045,
        blending: THREE.AdditiveBlending,
      })
    );
    accentGlow.position.set(position[0], position[1] + size[1] * 0.22, position[2]);
    this.architectureGroup.add(accentGlow);

    const labelSprite = createTextSprite(label, {
      size: 88,
      scale: 1.46,
      opacity: 0.82,
      y: 0.56,
    });
    labelSprite.position.set(position[0], position[1] + size[1] * 0.5 + 0.48, position[2] + size[2] * 0.5);
    this.architectureGroup.add(labelSprite);

    this.stageAnchors.set(id, {
      in: new THREE.Vector3(position[0] - size[0] * 0.48, position[1] + size[1] * 0.08, position[2]),
      out: new THREE.Vector3(position[0] + size[0] * 0.48, position[1] + size[1] * 0.08, position[2]),
    });
  }

  createInteractiveMesh(meta, geometry, position, color) {
    const group = new THREE.Group();
    group.position.set(...position);

    const accentColor = new THREE.Color(meta.accent ?? COLORS.neutral);
    const baseColor = new THREE.Color(color);
    const material = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: accentColor.clone().multiplyScalar(0.08),
      emissiveIntensity: 0.35,
      roughness: 0.22,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(geometry, material);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({
        color: accentColor,
        transparent: true,
        opacity: 0.22,
      })
    );
    const glow = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: accentColor,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        depthWrite: false,
      })
    );
    glow.scale.set(1.05, 1.2, 1.08);

    mesh.userData.componentId = meta.id;
    edges.userData.componentId = meta.id;

    group.add(glow, mesh, edges);

    this.components.set(meta.id, {
      ...meta,
      group,
      mesh,
      edges,
      glow,
      material,
      baseColor,
      accentColor,
      runtimeMeta: [],
      runtimeDescription: null,
      activityStrength: 0,
      activityBars: [],
    });
    this.interactiveMeshes.push(mesh);

    return { group, mesh, edges, glow };
  }

  createConnections(route) {
    for (let index = 0; index < route.length - 1; index += 1) {
      const from = route[index];
      const to = route[index + 1];
      const fromAnchor = this.stageAnchors.get(from);
      const toAnchor = this.stageAnchors.get(to);
      if (!fromAnchor || !toAnchor) {
        continue;
      }

      const accent = pickConnectionColor(from, to);
      const connector = this.createConnectionMesh(from, to, fromAnchor.out, toAnchor.in, accent, index);
      this.connections.push(connector);
      this.connectionLookup.set(connector.id, connector);
    }
  }

  createConnectionMesh(from, to, start, end, accent, index) {
    const points = buildConnectorPoints(start, end, index);
    const curve = new THREE.CatmullRomCurve3(points);
    const geometry = new THREE.TubeGeometry(curve, 56, 0.07, 12, false);
    const tube = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: COLORS.lineSoft,
        emissive: accent,
        transparent: true,
        opacity: 0.28,
        roughness: 0.22,
        metalness: 0.06,
      })
    );
    const glow = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 56, 0.12, 12, false),
      new THREE.MeshBasicMaterial({
        color: accent,
        transparent: true,
        opacity: 0.07,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );

    const wire = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
      new THREE.LineBasicMaterial({
        color: accent,
        transparent: true,
        opacity: 0.56,
      })
    );

    const nodeMaterial = new THREE.MeshStandardMaterial({
      color: accent,
      emissive: accent,
      emissiveIntensity: 0.34,
      roughness: 0.18,
      metalness: 0.04,
      transparent: true,
      opacity: 0.5,
    });
    const startNode = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 14), nodeMaterial.clone());
    startNode.position.copy(start);
    const endNode = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 14), nodeMaterial.clone());
    endNode.position.copy(end);

    this.connectionGroup.add(glow, tube, wire, startNode, endNode);

    return {
      id: `${from}->${to}`,
      from,
      to,
      curve,
      glow,
      tube,
      wire,
      startNode,
      endNode,
      baseColor: new THREE.Color(COLORS.lineSoft),
      accentColor: new THREE.Color(accent),
    };
  }

  createInlineRail(points, color, opacity, radius = 0.018) {
    const curve = new THREE.CatmullRomCurve3(points);
    const group = new THREE.Group();
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 18, radius, 10, false),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.22,
        transparent: true,
        opacity: Math.min(0.48, opacity * 0.58),
        roughness: 0.18,
        metalness: 0.04,
      })
    );
    const wire = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
      })
    );
    group.add(tube, wire);
    return group;
  }

  createActivityBars({ count, color, centerX, centerZ, columns, spacingX, spacingZ, baseY, size }) {
    const group = new THREE.Group();
    const bars = [];
    const rows = Math.ceil(count / columns);
    const width = (Math.min(columns, count) - 1) * spacingX;
    const depth = (rows - 1) * spacingZ;

    for (let index = 0; index < count; index += 1) {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.18,
        transparent: true,
        opacity: 0.22,
        roughness: 0.16,
        metalness: 0.05,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
      mesh.position.set(
        centerX - width / 2 + column * spacingX,
        baseY,
        centerZ - depth / 2 + row * spacingZ
      );
      mesh.userData.baseY = baseY;
      mesh.scale.y = 0.35;
      group.add(mesh);
      bars.push({ mesh, material, baseY });
    }

    return { group, bars };
  }

  applyActivityBars(bars, values) {
    if (!bars?.length) {
      return;
    }

    bars.forEach((bar, index) => {
      const strength = clamp(values?.[index] ?? 0.05, 0.05, 1);
      const scaleY = 0.35 + strength * 2.8;
      bar.mesh.scale.y = scaleY;
      bar.mesh.position.y = bar.baseY + (scaleY - 1) * 0.08;
      bar.material.opacity = 0.18 + strength * 0.56;
      bar.material.emissiveIntensity = 0.12 + strength * 1.15;
    });
  }

  resetRuntimeActivity() {
    this.components.forEach((component) => {
      component.runtimeMeta = [];
      component.runtimeDescription = null;
      component.activityStrength = 0;
      this.applyActivityBars(component.activityBars, []);
    });
  }

  renderPredictionPanel(activity) {
    if (!activity) {
      this.predictionStep.textContent = "Waiting for the first decode step...";
      this.predictionList.innerHTML = "";
      return;
    }

    this.predictionStep.textContent = `Step ${activity.stepIndex + 1} · query ${shortLabel(activity.queryToken)}`;
    this.predictionList.innerHTML = activity.topPredictions
      .map((prediction, index) => {
        const width = Math.max(8, prediction.probability * 100);
        return `
          <div class="prediction-row${prediction.id === activity.nextToken.id ? " is-active" : ""}">
            <div class="prediction-head">
              <span class="prediction-rank">${index + 1}</span>
              <span class="prediction-token">${escapeHtml(shortLabel(prediction.token))}</span>
              <span class="prediction-prob">${formatPercent(prediction.probability)}</span>
            </div>
            <div class="prediction-bar">
              <span style="width:${width}%"></span>
            </div>
          </div>
        `;
      })
      .join("");
  }

  clearArchitecture() {
    while (this.architectureGroup.children.length) {
      const child = this.architectureGroup.children[0];
      this.architectureGroup.remove(child);
      disposeHierarchy(child);
    }
  }

  select(id, shouldFocus = false) {
    const target = this.components.get(id);
    if (!target) {
      return;
    }

    this.selectedId = id;
    this.refreshSelectionDetails();
    if (shouldFocus) {
      this.focusOnComponent(target.group);
    }
  }

  animate(time = performance.now()) {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.updateHighlights(time);
    this.renderer.render(this.scene, this.camera);
  }

  updateHighlights(time) {
    const pulseStageStrength = new Map();
    const pulseConnectionStrength = new Map();
    this.updatePulses(time, pulseStageStrength, pulseConnectionStrength);

    const busyBreath = this.isBusy ? 0.08 + (Math.sin(time * 0.006) + 1) * 0.035 : 0;

    this.components.forEach((component, id) => {
      let strength = 0.12 + busyBreath;

      if (id === this.selectedId) {
        strength = Math.max(strength, 0.62);
      }
      if (id === this.hoveredId) {
        strength = Math.max(strength, 0.48);
      }
      if (component.activityStrength) {
        strength = Math.max(strength, component.activityStrength);
      }
      if (pulseStageStrength.has(id)) {
        strength = Math.max(strength, pulseStageStrength.get(id));
      }

      if (isParentOfPulse(id, pulseStageStrength)) {
        strength = Math.max(strength, 0.28);
      }

      component.material.color.copy(component.baseColor).lerp(component.accentColor, strength * 0.52);
      component.material.emissive.copy(component.accentColor).multiplyScalar(0.12 + strength * 1.05);
      component.material.emissiveIntensity = 0.36 + strength * 1.02;
      component.edges.material.opacity = 0.18 + strength * 0.74;
      component.glow.material.opacity = 0.05 + strength * 0.28;
    });

    this.connections.forEach((connection) => {
      let strength = busyBreath * 0.8;

      if (isRelatedStage(this.selectedId, connection.from) || isRelatedStage(this.selectedId, connection.to)) {
        strength = Math.max(strength, 0.34);
      }
      if (isRelatedStage(this.hoveredId, connection.from) || isRelatedStage(this.hoveredId, connection.to)) {
        strength = Math.max(strength, 0.2);
      }
      if (pulseConnectionStrength.has(connection.id)) {
        strength = Math.max(strength, pulseConnectionStrength.get(connection.id));
      }

      connection.tube.material.color.copy(connection.baseColor).lerp(connection.accentColor, strength * 0.58);
      connection.tube.material.emissive.copy(connection.accentColor).multiplyScalar(0.08 + strength * 0.95);
      connection.tube.material.opacity = 0.18 + strength * 0.36;
      connection.glow.material.opacity = 0.04 + strength * 0.24;
      connection.wire.material.opacity = 0.24 + strength * 0.76;
      [connection.startNode, connection.endNode].forEach((node) => {
        node.material.opacity = 0.28 + strength * 0.5;
        node.material.emissive.copy(connection.accentColor).multiplyScalar(0.12 + strength * 0.95);
      });
    });
  }

  updatePulses(time, pulseStageStrength, pulseConnectionStrength) {
    const activePulses = [];

    for (const pulse of this.pulses) {
      const progress = (time - pulse.startedAt) / pulse.duration;
      if (progress >= 1) {
        disposeHierarchy(pulse.group);
        this.pulseGroup.remove(pulse.group);
        continue;
      }

      const scaled = progress * (pulse.route.length - 1);
      const segmentIndex = Math.min(Math.floor(scaled), pulse.route.length - 2);
      const local = scaled - segmentIndex;
      const from = pulse.route[segmentIndex];
      const to = pulse.route[segmentIndex + 1];
      const connectionId = `${from}->${to}`;
      const connector = this.connectionLookup.get(connectionId);

      if (connector) {
        pulse.group.position.copy(connector.curve.getPointAt(local));
        const scale = 0.8 + Math.sin(local * Math.PI) * 0.55;
        pulse.core.scale.setScalar(scale);
        pulse.aura.scale.setScalar(scale * 2.35);
        pulse.core.material.opacity = progress > 0.84 ? 1 - (progress - 0.84) / 0.16 : 0.95;
        pulse.aura.material.opacity = progress > 0.84 ? (1 - (progress - 0.84) / 0.16) * 0.16 : 0.16;
      }

      pulseStageStrength.set(from, Math.max(pulseStageStrength.get(from) ?? 0, 0.2 + (1 - local) * 0.2));
      pulseStageStrength.set(to, Math.max(pulseStageStrength.get(to) ?? 0, 0.42 + Math.sin(local * Math.PI) * 0.4));
      pulseConnectionStrength.set(connectionId, Math.max(pulseConnectionStrength.get(connectionId) ?? 0, 0.55 + Math.sin(local * Math.PI) * 0.45));

      activePulses.push(pulse);
    }

    this.pulses = activePulses;
  }

  handlePointerMove(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    if (this.pointerDown) {
      const dx = event.clientX - this.pointerDown.x;
      const dy = event.clientY - this.pointerDown.y;
      this.dragDistance = Math.max(this.dragDistance, Math.hypot(dx, dy));
    }

    this.raycaster.setFromCamera(this.pointer, this.camera);
    const [hit] = this.raycaster.intersectObjects(this.interactiveMeshes, false);
    this.hoveredId = hit?.object?.userData?.componentId ?? null;
    this.renderer.domElement.style.cursor = this.pointerDown
      ? "grabbing"
      : this.hoveredId
        ? "pointer"
        : "grab";
  }

  handlePointerDown(event) {
    this.pointerDown = { x: event.clientX, y: event.clientY };
    this.dragDistance = 0;
    this.renderer.domElement.style.cursor = "grabbing";
  }

  handlePointerUp() {
    this.pointerDown = null;
    this.renderer.domElement.style.cursor = this.hoveredId ? "pointer" : "grab";
  }

  handlePointerClick() {
    if (this.dragDistance > 6) {
      this.dragDistance = 0;
      return;
    }
    if (this.hoveredId) {
      this.select(this.hoveredId, false);
    }
  }

  resetCamera() {
    this.camera.position.copy(this.defaultCameraPosition);
    this.controls.target.copy(this.defaultCameraTarget);
    this.controls.update();
  }

  resize() {
    const width = this.host.clientWidth || window.innerWidth;
    const height = this.host.clientHeight || window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  focusOnComponent(group) {
    this.architectureGroup.updateMatrixWorld(true);
    group.getWorldPosition(this.tempVector);
    this.controls.target.lerp(this.tempVector, 0.35);
    this.controls.update();
  }
}

function createTextSprite(text, options = {}) {
  const size = options.size ?? 72;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 1024;
  canvas.height = 256;

  context.font = `600 ${size}px "IBM Plex Mono", monospace`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = `rgba(245, 248, 255, ${options.opacity ?? 1})`;
  context.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    sizeAttenuation: true,
  });

  const sprite = new THREE.Sprite(material);
  sprite.scale.set(options.scale ?? 1, (options.scale ?? 1) * 0.28, 1);
  sprite.position.y = options.y ?? 0;
  return sprite;
}

function createThinPlate(width, depth, color) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.05, depth),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.08,
      transparent: true,
      opacity: 0.18,
      roughness: 0.14,
      metalness: 0.06,
    })
  );
}

function clipLaneValues(values, maxItems, activeIndex, strategy = "compact") {
  if (values.length <= maxItems) {
    return { values, activeDisplayIndex: activeIndex };
  }

  if (strategy === "tail") {
    const tail = values.slice(-(maxItems - 1));
    return {
      values: ["..."].concat(tail),
      activeDisplayIndex: activeIndex >= values.length - tail.length ? tail.length : -1,
    };
  }

  const headCount = Math.ceil((maxItems - 1) / 2);
  const tailCount = maxItems - headCount - 1;
  return {
    values: [...values.slice(0, headCount), "...", ...values.slice(-tailCount)],
    activeDisplayIndex: -1,
  };
}

function formatLaneLabel(value, mode) {
  if (value === "...") {
    return value;
  }

  if (mode === "id") {
    return value.length > 6 ? `${value.slice(0, 5)}...` : value;
  }

  return value.length > 11 ? `${value.slice(0, 10)}...` : value;
}

function buildConnectorPoints(start, end, index) {
  const mid = start.clone().lerp(end, 0.5);
  const delta = end.clone().sub(start);
  const arcLift = 0.68 + Math.min(1.2, delta.length() * 0.052);
  mid.y += arcLift;
  mid.z += 0.62 + (index % 2 === 0 ? 0.28 : 0.18);

  return [
    start,
    start.clone().lerp(mid, 0.38),
    mid,
    mid.clone().lerp(end, 0.62),
    end,
  ];
}

function pickConnectionColor(from, to) {
  if (to.includes("attn") || from.includes("attn")) {
    return COLORS.cool;
  }
  if (to.includes("mlp") || from.includes("mlp") || to === "lm-head") {
    return COLORS.warm;
  }
  if (to === "token-ids" || to === "embeddings") {
    return COLORS.cool;
  }
  if (to === "output-tokens") {
    return COLORS.warm;
  }
  return COLORS.neutral;
}

function isParentOfPulse(id, pulseStageStrength) {
  if (!/^block-\d+$/.test(id)) {
    return false;
  }

  for (const stageId of pulseStageStrength.keys()) {
    if (typeof stageId === "string" && stageId.startsWith(`${id}-`)) {
      return true;
    }
  }

  return false;
}

function isRelatedStage(componentId, stageId) {
  if (!componentId || !stageId) {
    return false;
  }

  if (componentId === stageId) {
    return true;
  }

  if (/^block-\d+$/.test(componentId) && stageId.startsWith(`${componentId}-`)) {
    return true;
  }

  if (/^block-\d+$/.test(stageId) && componentId.startsWith(`${stageId}-`)) {
    return true;
  }

  return false;
}

function shortLabel(value) {
  if (!value) {
    return "token";
  }
  return value.length > 16 ? `${value.slice(0, 15)}…` : value;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function disposeHierarchy(object) {
  object.traverse((child) => {
    if (child.material?.map) {
      child.material.map.dispose();
    }
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((entry) => entry.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}
