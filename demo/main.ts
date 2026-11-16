import { CameraHelper } from "three";
import type { PerspectiveCamera } from "three";
import { MESHLET_FILE_EXTENSION, serializeMeshlets, validateMeshletData } from "../src/index.js";
import type { MeshletData } from "../src/index.js";
import { buildInWorker } from "./builder.js";
import { Content } from "./content.js";
import type { DemoApi } from "./demo-api.js";
import { createGui } from "./gui.js";
import type { ToggleState } from "./gui.js";
import { renderHud, setStatus } from "./hud.js";
import { loadModel } from "./models.js";
import { readParams, writeParams } from "./params.js";
import type { ModelName } from "./params.js";
import { createRenderer } from "./renderer.js";
import { Stage } from "./stage.js";

const params = readParams();
const toggles: ToggleState = { wireframe: false, shadows: true, freezeLod: false, compare: false };
let fps = 0;

const demo: DemoApi = (window.__demo = {
  ready: false,
  error: null,
  renderer: params.renderer,
  backend: "",
  frames: 0,
  mesh: null,
  stats: () => ({
    ...(demo.mesh?.stats ?? { instances: 0, clusters: 0, draws: 0, triangles: 0, slots: 0, levelInstances: 0, levelVisible: 0, levelDraws: 0, lastUpdateMs: 0, lastSelected: 0 }),
    fullTriangles: fullTriangles(),
    fps,
  }),
});

function fail(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  demo.error ??= message;
  setStatus(`Error: ${message}`, true);
  console.error(error);
}
window.addEventListener("error", (event) => fail(event.error ?? event.message));
window.addEventListener("unhandledrejection", (event) => fail(event.reason));

const { renderer, backend, drawCalls } = await createRenderer(params.renderer);
demo.backend = backend;
document.body.prepend(renderer.domElement);

const stage = new Stage();
stage.attach(renderer.domElement);

const content = new Content(stage.scene, {
  grid: params.grid,
  errorThreshold: params.threshold,
  debugColors: params.debug,
  wireframe: toggles.wireframe,
  shadows: toggles.shadows,
  lodCamera: null,
  compare: toggles.compare,
});


let data: MeshletData | null = null;
let droppedFile: File | null = null;
let generation = 0;
let freezeHelper: CameraHelper | null = null;

function modelLabel(model: ModelName): string {
  return model === "file" ? (droppedFile?.name ?? "file") : model;
}

function fullTriangles(): number {
  return content.triangles * (demo.mesh?.count ?? 0);
}

/** Loads the model, builds its hierarchy in a worker and swaps it in. A newer request supersedes an older one. */
async function showModel(model: ModelName): Promise<void> {
  const id = ++generation;
  const label = modelLabel(model);
  demo.ready = false;
  demo.error = null;
  setStatus(`Loading ${label} ...`);
  try {
    const geometry = await loadModel(model, droppedFile);
    if (id !== generation) return geometry.dispose();
    const triangles = geometry.getIndex()!.count / 3;
    setStatus(`Building ${label} (${triangles.toLocaleString("en-US")} triangles) ...`);

    const result = await buildInWorker(geometry, (level, clusters) => {
      if (id === generation) setStatus(`Building ${label}: level ${level}, ${clusters.toLocaleString("en-US")} clusters`);
    });
    if (id !== generation) return geometry.dispose();

    const problems = validateMeshletData(result.data);
    if (problems.length > 0) console.warn(`validateMeshletData found ${problems.length} problem(s):`, problems);

    data = result.data;
    content.setModel(geometry, result.data);
    demo.mesh = content.mesh;
    stage.fit(content.half, content.height);
    setStatus(
      `${label}: ${triangles.toLocaleString("en-US")} triangles -> ${data.clusterCount.toLocaleString("en-US")} clusters, ` +
        `${data.levelCount} levels, built in ${(result.ms / 1000).toFixed(2)} s` +
        (problems.length > 0 ? ` (${problems.length} validation problems, see console)` : ""),
    );
  } catch (error) {
    if (id === generation) fail(error);
  }
}

function setFrozen(on: boolean): void {
  if (freezeHelper) {
    stage.scene.remove(freezeHelper);
    freezeHelper.dispose();
    freezeHelper = null;
  }
  let lodCamera: PerspectiveCamera | null = null;
  if (on) {
    lodCamera = stage.camera.clone();
    lodCamera.far = Math.min(lodCamera.far, stage.camera.position.length());
    lodCamera.updateProjectionMatrix();
    freezeHelper = new CameraHelper(lodCamera);
    stage.scene.add(freezeHelper);
  }
  content.settings.lodCamera = lodCamera;
  content.applySettings();
}

function syncToggles(): void {
  const s = content.settings;
  s.wireframe = toggles.wireframe;
  s.shadows = toggles.shadows;
  s.compare = toggles.compare;
  stage.sun.castShadow = toggles.shadows;
  content.applySettings();
}

function download(): void {
  if (!data) return;
  const url = URL.createObjectURL(new Blob([serializeMeshlets(data)], { type: "application/octet-stream" }));
  const name = modelLabel(params.model).replace(/\.[^.]+$/, "");
  const link = Object.assign(document.createElement("a"), { href: url, download: name + MESHLET_FILE_EXTENSION });
  link.click();
  URL.revokeObjectURL(url);
}

function change(apply: () => void): void {
  apply();
  writeParams(params);
}

const gui = createGui(params, toggles, {
  renderer: (kind) => {
    const q = new URLSearchParams(location.search);
    q.set("renderer", kind);
    location.search = q.toString();
  },
  model: (model) => change(() => void showModel(model)),
  grid: (grid) =>
    change(() => {
      content.setGrid(grid);
      stage.fit(content.half, content.height);
    }),
  threshold: (px) =>
    change(() => {
      content.settings.errorThreshold = px;
      content.applySettings();
    }),
  debug: (mode) =>
    change(() => {
      content.settings.debugColors = mode;
      content.applySettings();
    }),
  toggles: syncToggles,
  freeze: setFrozen,
  download,
});

// drag and drop of .glb / .gltf / .obj
window.addEventListener("dragover", (event) => {
  event.preventDefault();
  document.body.classList.add("dragging");
});
window.addEventListener("dragleave", () => document.body.classList.remove("dragging"));
window.addEventListener("drop", (event) => {
  event.preventDefault();
  document.body.classList.remove("dragging");
  const file = event.dataTransfer?.files[0];
  if (!file) return;
  droppedFile = file;
  gui.showDroppedFile(file.name);
  change(() => void showModel("file"));
});

function resize(): void {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  stage.resize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", resize);
resize();

let frames = 0;
let lastHud = performance.now();

function frame(): void {
  stage.controls?.update();
  renderer.render(stage.scene, stage.camera);
  demo.frames++;
  frames++;
  const mesh = demo.mesh;
  if (!demo.ready && mesh && mesh.stats.clusters > 0) demo.ready = true;

  const now = performance.now();
  if (now - lastHud >= 250) {
    fps = (frames * 1000) / (now - lastHud);
    frames = 0;
    lastHud = now;
    const s = demo.stats();
    renderHud({
      renderer: demo.backend,
      fps,
      instances: s.instances,
      clusters: s.clusters,
      triangles: toggles.compare ? s.fullTriangles : s.triangles,
      fullTriangles: s.fullTriangles,
      draws: s.draws,
      levelInstances: s.levelInstances,
      levelVisible: s.levelVisible,
      levelDraws: s.levelDraws,
      slots: s.slots,
      drawCapacity: mesh?.drawCapacity ?? 0,
      updateMs: s.lastUpdateMs,
      drawCalls: drawCalls(),
      comparing: toggles.compare,
    });
  }
}

writeParams(params);
void showModel(params.model);
renderer.setAnimationLoop(frame);
