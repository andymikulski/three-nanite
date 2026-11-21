import {
  ACESFilmicToneMapping,
  CanvasTexture,
  Color,
  DataTexture,
  DirectionalLight,
  EquirectangularReflectionMapping,
  FogExp2,
  Group,
  HemisphereLight,
  InstancedMesh,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SRGBColorSpace,
  Vector3,
} from "three";
import type { Material, Matrix4, WebGLRenderer } from "three";
import GUI from "lil-gui";
import { MeshletBatch, MeshletMesh } from "../../src/index.js";
import type { MeshletData } from "../../src/index.js";
import { colorMapFromMaterial } from "../../src/three/colorMap.js";
import { createRenderer } from "../renderer.js";
import { bake, clearBakeCache, endBake } from "./bake.js";
import { Controls } from "./controls.js";
import type { CameraMode } from "./controls.js";
import { createMaterials, loadModel, MODELS } from "./kit.js";
import type { KitModel, KitPart } from "./kit.js";
import { buildImpostors } from "./impostors.js";
import type { ImpostorField } from "./impostors.js";
import { scatter } from "./scatter.js";
import { createTerrain, heightAt, SIZE, WATER } from "./terrain.js";

// ---------------------------------------------------------------------------
// settings (kept in localStorage; nothing is switched through the address)
// ---------------------------------------------------------------------------

interface Settings {
  draw: "nanite" | "instanced" | "impostors";
  renderer: "webgl" | "webgpu";
  errorThreshold: number;
  colors: "none" | "lod";
  freeze: boolean;
  shadows: boolean;
  density: number;
  camera: CameraMode;
}
const SETTINGS_KEY = "three-nanite-valley";
const settings: Settings = {
  draw: "nanite",
  renderer: "webgl",
  errorThreshold: 2,
  colors: "none",
  freeze: false,
  shadows: true,
  density: 1,
  camera: "tour",
  ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") as Partial<Settings>),
};
// (settings saved before the default became 2 px: one threshold for everything, leaves included)
if (localStorage.getItem(SETTINGS_KEY + ":v") !== "2") {
  settings.errorThreshold = 2;
  localStorage.setItem(SETTINGS_KEY + ":v", "2");
}
delete (settings as unknown as { foliageError?: number }).foliageError;
const save = () => localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));

const loading = document.getElementById("loading")!;
const hud = document.getElementById("hud")!;
const say = (text: string) => {
  loading.textContent = text;
  return new Promise((r) => requestAnimationFrame(r));
};

// ---------------------------------------------------------------------------
// renderer, scene, light, sky
// ---------------------------------------------------------------------------

const handle = await createRenderer(settings.renderer);
const { renderer } = handle;
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const camera = new PerspectiveCamera(60, innerWidth / innerHeight, 0.2, 4000);

const HORIZON = new Color().setHex(0xcfe3ee, SRGBColorSpace);
scene.fog = new FogExp2(HORIZON.getHex(), 0.00055);
{
  // a painted sky: deep blue overhead, pale at the horizon, warm towards the sun
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const g = canvas.getContext("2d")!;
  const v = g.createLinearGradient(0, 0, 0, 512);
  v.addColorStop(0, "#3d8fd6");
  v.addColorStop(0.42, "#9ccbec");
  v.addColorStop(0.5, "#d9ebf2");
  v.addColorStop(1, "#d9ebf2");
  g.fillStyle = v;
  g.fillRect(0, 0, 1024, 512);
  const sun = g.createRadialGradient(840, 200, 0, 840, 200, 260);
  sun.addColorStop(0, "rgba(255, 236, 190, 0.9)");
  sun.addColorStop(1, "rgba(255, 236, 190, 0)");
  g.fillStyle = sun;
  g.fillRect(0, 0, 1024, 512);
  const sky = new CanvasTexture(canvas);
  sky.mapping = EquirectangularReflectionMapping;
  sky.colorSpace = SRGBColorSpace;
  scene.background = sky;
}

const hemi = new HemisphereLight(0xcfe8ff, 0x5a6b2a, 1.1);
scene.add(hemi);
const sun = new DirectionalLight(0xfff0d0, 3.2);
const SUN_DIRECTION = new Vector3(-0.55, 0.62, -0.56).normalize();
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const SHADOW_RANGE = 130;
Object.assign(sun.shadow.camera, { left: -SHADOW_RANGE, right: SHADOW_RANGE, top: SHADOW_RANGE, bottom: -SHADOW_RANGE, near: 1, far: 2000 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.6;
scene.add(sun, sun.target);

const water = new Mesh(
  new PlaneGeometry(SIZE, SIZE).rotateX(-Math.PI / 2),
  new MeshStandardMaterial({ color: 0x3f86a8, roughness: 0.12, metalness: 0.1, transparent: true, opacity: 0.82 }),
);
water.position.y = WATER;
water.receiveShadow = true;
scene.add(water);

window.addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------------------
// the kit, its levels of detail, the terrain
// ---------------------------------------------------------------------------

const t0 = performance.now();
const models = new Map<string, KitModel>();
{
  let done = 0;
  const queue = [...MODELS];
  await Promise.all(
    Array.from({ length: 8 }, async () => {
      for (let name = queue.shift(); name; name = queue.shift()) {
        models.set(name, await loadModel(name));
        if (++done % 10 === 0) await say(`loading models ${done} / ${MODELS.length}`);
      }
    }),
  );
}
await say("loading textures");
const materials = await createMaterials();

const terrainGeometry = createTerrain();
const terrainMaterial = new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 });

/** Every part of every model, with its hierarchy. */
interface Part extends KitPart {
  data: MeshletData;
}
const parts: Part[] = [];
let terrainData: MeshletData;
{
  let built = 0;
  let done = 0;
  const all = [...models.values()].flatMap((m) => m.parts);
  const total = all.length + 1;
  const progress = () => say(`building levels of detail ${done} / ${total}${built ? ` (${built} built, the rest from the cache)` : ""}`);
  await progress();
  const terrain = bake("terrain-1", terrainGeometry, { attributes: ["normal", "color"], vertexColors: true }).then((r) => {
    built += r.built ? 1 : 0;
    done++;
    terrainData = r.data;
  });
  await Promise.all([
    terrain,
    ...all.map(async (part) => {
      const material = materials.get(part.material)!;
      const attributes = part.geometry.getAttribute("color") ? ["normal", "uv", "color"] : ["normal", "uv"];
      // as buildMeshletsFromGeometry would choose from the material (the workers get its colours, not
      // the material): every material here is opaque or a cut-out that keeps its coverage
      // (prepareCutoutTexture), so loose blades and cards keep theirs at every level too
      const dilateBorders = true;
      const r = await bake(`${part.model}/${part.material}`, part.geometry, { attributes, colorMap: colorMapFromMaterial(material), dilateBorders });
      parts.push({ ...part, data: r.data });
      built += r.built ? 1 : 0;
      done++;
      if (done % 8 === 0) await progress();
    }),
  ]);
  endBake();
}
const loadMs = performance.now() - t0;

// ---------------------------------------------------------------------------
// the two ways of drawing it
// ---------------------------------------------------------------------------

/** Materials whose copies are too small and many to be worth a shadow. */
const NO_SHADOW = new Set(["Grass", "Grass_Wheat", "Grass_Wispy", "Grass_Wide", "Flowers", "Leaves"]);

const naniteGroup = new Group();
const instancedGroup = new Group();
const impostorGroup = new Group();
scene.add(naniteGroup, instancedGroup, impostorGroup);
let impostors: ImpostorField | null = null;
let impostorsBuilding: Promise<void> | null = null;
let batches: MeshletBatch[] = [];
let terrainMesh: MeshletMesh;
let instanceCount = 0;

function partsByMaterial(): Map<string, Part[]> {
  const out = new Map<string, Part[]>();
  for (const p of parts) {
    let list = out.get(p.material);
    if (!list) out.set(p.material, (list = []));
    list.push(p);
  }
  // (a stable order, whatever order the bake finished in)
  for (const list of out.values()) list.sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
  return out;
}

let placements: Map<string, Matrix4[]>;

function populate(): void {
  for (const b of batches) {
    naniteGroup.remove(b);
    b.dispose();
  }
  batches = [];
  for (const o of [...instancedGroup.children]) {
    instancedGroup.remove(o);
    (o as InstancedMesh).dispose?.();
  }
  dropImpostors();
  placements = scatter(settings.density);
  instanceCount = 0;
  for (const list of placements.values()) instanceCount += list.length;

  for (const [name, list] of partsByMaterial()) {
    const used = list.filter((p) => placements.has(p.model));
    if (used.length === 0) continue;
    const count = used.reduce((n, p) => n + placements.get(p.model)!.length, 0);
    const batch = new MeshletBatch(
      used.map((p) => p.data),
      materials.get(name),
      { maxInstances: count },
    );
    used.forEach((p, k) => {
      for (const m of placements.get(p.model)!) batch.addInstance(k, m);
    });
    batch.name = name;
    batch.castShadow = !NO_SHADOW.has(name);
    batch.receiveShadow = true;
    batches.push(batch);
    naniteGroup.add(batch);
  }
  if (settings.draw === "instanced") buildInstanced();
  if (settings.draw === "impostors") void buildImpostorField();
  applySettings();
}

/** The comparison: one InstancedMesh per model and material at full detail, and the full terrain. */
function buildInstanced(): void {
  if (instancedGroup.children.length > 0) return;
  for (const p of parts) {
    const list = placements.get(p.model);
    if (!list) continue;
    const mesh = new InstancedMesh(p.geometry, materials.get(p.material) as Material, list.length);
    list.forEach((m, i) => mesh.setMatrixAt(i, m));
    mesh.castShadow = !NO_SHADOW.has(p.material);
    mesh.receiveShadow = true;
    mesh.computeBoundingSphere();
    instancedGroup.add(mesh);
  }
  const terrain = new Mesh(terrainGeometry, terrainMaterial);
  terrain.castShadow = terrain.receiveShadow = true;
  instancedGroup.add(terrain);
}

terrainMesh = new MeshletMesh(terrainData!, terrainMaterial);
terrainMesh.castShadow = terrainMesh.receiveShadow = true;
naniteGroup.add(terrainMesh);

/** The impostor comparison: baked the first time it is asked for (it takes a while), then kept. */
function buildImpostorField(): Promise<void> {
  if (impostors || impostorsBuilding) return impostorsBuilding ?? Promise.resolve();
  impostorsBuilding = (async () => {
    loading.classList.remove("done");
    await say("baking impostors");
    let field: ImpostorField;
    try {
      field = await buildImpostors(renderer as WebGLRenderer, [...models.values()], materials as Map<string, Material>, placements, say);
    } catch (error) {
      // back to three-nanite, with what went wrong in the readout
      note = "impostors: " + (error instanceof Error ? error.message : String(error));
      console.error(error);
      impostorsBuilding = null;
      loading.classList.add("done");
      settings.draw = "nanite";
      applySettings();
      return;
    }
    // the terrain as the plain mesh, as in the instanced comparison
    const terrain = new Mesh(terrainGeometry, terrainMaterial);
    terrain.castShadow = terrain.receiveShadow = true;
    field.group.add(terrain);
    impostorGroup.add(field.group);
    impostors = field;
    impostorsBuilding = null;
    loading.classList.add("done");
  })();
  return impostorsBuilding;
}

function dropImpostors(): void {
  if (!impostors) return;
  impostorGroup.remove(impostors.group);
  impostors.dispose();
  impostors = null;
}

const frozen = new PerspectiveCamera();

const whites = new Map<Material, MeshStandardMaterial>();
/** `material` in white: no colour map or vertex colours, the same cut-out (its map's alpha as an alpha map). */
function whiteVersion(material: MeshStandardMaterial): MeshStandardMaterial {
  let white = whites.get(material);
  if (white) return white;
  white = new MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0, side: material.side });
  const map = material.map as DataTexture | null;
  if (material.alphaTest > 0 && map && map.isDataTexture && map.mipmaps.length > 0) {
    // (an alpha map is read from green: every level's alpha copied to its colour)
    const levels = (map.mipmaps as unknown as { data: Uint8Array; width: number; height: number }[]).map(({ data, width, height }) => {
      const out = new Uint8Array(data.length);
      for (let i = 0; i < data.length; i += 4) out[i] = out[i + 1] = out[i + 2] = out[i + 3] = data[i + 3];
      return { data: out, width, height };
    });
    const alpha = new DataTexture(levels[0].data, levels[0].width, levels[0].height);
    alpha.mipmaps = levels as unknown as typeof alpha.mipmaps;
    alpha.generateMipmaps = false;
    alpha.minFilter = map.minFilter;
    alpha.magFilter = map.magFilter;
    alpha.flipY = map.flipY;
    alpha.needsUpdate = true;
    white.alphaMap = alpha;
    white.alphaTest = material.alphaTest;
    white.alphaToCoverage = material.alphaToCoverage;
  }
  whites.set(material, white);
  return white;
}

function applySettings(): void {
  if (settings.draw === "impostors" && (renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer === true) {
    // (three-impostor is for WebGLRenderer only)
    settings.draw = "nanite";
    note = "impostors need the WebGL renderer";
  }
  const nanite = settings.draw === "nanite";
  if (settings.draw === "instanced") buildInstanced();
  if (settings.draw === "impostors") void buildImpostorField();
  naniteGroup.visible = nanite;
  instancedGroup.visible = settings.draw === "instanced";
  impostorGroup.visible = settings.draw === "impostors";
  if (settings.freeze) {
    frozen.copy(camera);
    frozen.updateMatrixWorld(true);
  }
  const lodView = settings.colors === "lod";
  for (const o of [...batches, terrainMesh]) {
    o.errorThreshold = settings.errorThreshold;
    o.debugColors = settings.colors;
    o.lodCamera = settings.freeze ? frozen : null;
  }
  // the level colours on plain white, so they read as they are (cut-outs keep their shape)
  for (const b of batches) {
    const want = lodView ? whiteVersion(materials.get(b.name)!) : materials.get(b.name)!;
    if (b.material !== want) b.material = want;
  }
  const terrainWant = lodView ? whiteVersion(terrainMaterial) : terrainMaterial;
  if (terrainMesh.material !== terrainWant) terrainMesh.material = terrainWant;
  renderer.shadowMap.enabled = settings.shadows;
  sun.castShadow = settings.shadows;
  save();
}

// ---------------------------------------------------------------------------
// camera, panel
// ---------------------------------------------------------------------------

const controls = new Controls(camera, renderer.domElement);
// the overlook at the head of the trail
controls.setView(new Vector3(470, heightAt(470, 560) + 14, 560), new Vector3(150, 20, 210));
if (settings.camera === "tour") controls.startTour();
else controls.mode = settings.camera;

const gui = new GUI({ title: "three-nanite · the valley" });
gui
  .add(settings, "draw", { "three-nanite (MeshletBatch)": "nanite", "plain InstancedMesh, full detail": "instanced", "impostors (three-impostor, WebGL)": "impostors" })
  .name("draw as")
  .listen()
  .onChange(applySettings);
gui.add(settings, "renderer", { WebGL: "webgl", WebGPU: "webgpu" }).name("renderer (reloads)").onChange(() => {
  save();
  location.reload();
});
gui.add(settings, "camera", { "tour along the trail": "tour", fly: "fly", walk: "walk" }).name("camera").listen().onChange((mode: CameraMode) => {
  if (mode === "tour") controls.startTour();
  else controls.mode = mode;
  save();
});
const lod = gui.addFolder("levels of detail");
lod.add(settings, "errorThreshold", 0.25, 8, 0.25).name("error (pixels)").onChange(applySettings);
lod.add(settings, "colors", { none: "none", "by level": "lod" }).name("colours").onChange(applySettings);
lod.add(settings, "freeze").name("freeze levels here").onChange(applySettings);
const world = gui.addFolder("world");
world.add(settings, "shadows").onChange(applySettings);
world.add(settings, "density", { "a quarter": 0.25, half: 0.5, designed: 1, double: 2, "four times": 4 }).name("how much grows").onChange(() => {
  save();
  populate();
});
world.add(
  {
    rebuild: async () => {
      await clearBakeCache();
      location.reload();
    },
  },
  "rebuild",
).name("rebuild levels (clears the cache)");

controls.onLeaveTour = () => {
  settings.camera = "fly";
  save();
};

populate();
loading.classList.add("done");

// ---------------------------------------------------------------------------
// the loop, the readout
// ---------------------------------------------------------------------------

const fmt = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? (n / 1e3).toFixed(1) + "k" : Math.round(n).toLocaleString("en-US"));
let last = performance.now();
let frames = 0;
let fpsTime = last;
let fps = 0;
let renderMs = 0;
/** A message shown in the readout (a setting that could not be honoured). */
let note = "";

function stats() {
  const info = renderer.info.render as { triangles: number };
  let visible = 0;
  let lodMs = 0;
  for (const b of batches) {
    visible += b.stats.visible;
    lodMs += b.stats.lastUpdateMs;
  }
  return {
    fps,
    renderMs,
    draws: handle.drawCalls(),
    triangles: info.triangles,
    instances: instanceCount,
    visible,
    lodMs,
    batches: batches.length,
    terrainTriangles: terrainMesh.stats.triangles,
    loadMs,
  };
}

function frame(): void {
  const now = performance.now();
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  controls.update(dt);
  camera.updateMatrixWorld();

  // the shadow follows what is in front of the camera, snapped to its texels so it does not shimmer
  const focus = camera.getWorldDirection(new Vector3()).multiplyScalar(SHADOW_RANGE * 0.6).add(camera.position);
  const texel = (2 * SHADOW_RANGE) / sun.shadow.mapSize.x;
  focus.set(Math.round(focus.x / texel) * texel, heightAt(focus.x, focus.z), Math.round(focus.z / texel) * texel);
  sun.target.position.copy(focus);
  sun.position.copy(focus).addScaledVector(SUN_DIRECTION, 800);

  const t = performance.now();
  renderer.render(scene, camera);
  renderMs = renderMs * 0.9 + (performance.now() - t) * 0.1;

  frames++;
  if (now - fpsTime > 500) {
    fps = (frames * 1000) / (now - fpsTime);
    frames = 0;
    fpsTime = now;
    const s = stats();
    hud.innerHTML =
      `<b>${settings.draw === "nanite" ? "three-nanite" : settings.draw === "impostors" ? "impostors" : "plain InstancedMesh"}</b> · ${handle.backend}\n` +
      (note ? `<b>${note}</b>\n` : "") +
      `${fps.toFixed(0)} fps · render() ${s.renderMs.toFixed(2)} ms\n` +
      `draw calls ${s.draws} · triangles ${fmt(s.triangles)}\n` +
      `${fmt(s.instances)} placed` +
      (settings.draw === "nanite" ? ` · ${fmt(s.visible)} in view · ${s.batches} batches\nlevel choice ${s.lodMs.toFixed(2)} ms · terrain ${fmt(s.terrainTriangles)} tris` : "") +
      (settings.draw === "impostors" && impostors ? `\natlases ${Math.round(impostors.memoryMB)} MB · baked in ${(impostors.bakeMs / 1000).toFixed(1)} s` : "");
  }
}
renderer.setAnimationLoop(frame);

// for scripts and benchmarks
(window as unknown as { valley: unknown }).valley = {
  settings,
  stats,
  camera,
  controls,
  batches: () => batches,
  set(partial: Partial<Settings>) {
    Object.assign(settings, partial);
    if ("density" in partial) populate();
    else applySettings();
    if (partial.camera) {
      if (partial.camera === "tour") controls.startTour();
      else controls.mode = partial.camera;
    }
  },
  ready: true,
};
