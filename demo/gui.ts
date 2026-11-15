import GUI from "lil-gui";
import type { Controller } from "lil-gui";
import type { DebugColorMode } from "../src/index.js";
import { THRESHOLD_MAX, THRESHOLD_MIN, GRID_MAX } from "./params.js";
import type { ModelName, Params, RendererKind } from "./params.js";

/** Settings that are not part of the URL. */
export interface ToggleState {
  wireframe: boolean;
  shadows: boolean;
  freezeLod: boolean;
  compare: boolean;
}

export interface GuiHandlers {
  renderer(kind: RendererKind): void;
  model(model: ModelName): void;
  grid(n: number): void;
  threshold(px: number): void;
  debug(mode: DebugColorMode): void;
  toggles(): void;
  freeze(on: boolean): void;
  download(): void;
}

export interface DemoGui {
  /** Adds (or renames) the entry for a dropped file and selects it. */
  showDroppedFile(name: string): void;
}

const LOG_MIN = Math.log2(THRESHOLD_MIN);
const LOG_MAX = Math.log2(THRESHOLD_MAX);

export function createGui(params: Params, toggles: ToggleState, on: GuiHandlers): DemoGui {
  const gui = new GUI({ title: "three-nanite" });

  gui.add(params, "renderer", ["webgl", "webgpu"]).onChange(on.renderer);
  const modelOptions: Record<string, ModelName> = { bunny: "bunny", knot: "knot" };
  let model: Controller = gui.add(params, "model", modelOptions).onChange(on.model);
  gui.add(params, "grid", 1, GRID_MAX, 1).name("grid N (N x N)").onFinishChange(on.grid);

  // the threshold slider is logarithmic: it edits log2(px) and shows the pixel value next to it
  const slider = { log2: Math.log2(params.threshold) };
  const readout = { px: "" };
  const showPx = () => (readout.px = params.threshold.toFixed(2));
  showPx();
  gui.add(slider, "log2", LOG_MIN, LOG_MAX, 0.01).name("errorThreshold (log2 px)").onChange((v: number) => {
    params.threshold = 2 ** v;
    showPx();
    on.threshold(params.threshold);
  });
  gui.add(readout, "px").name("  = pixels").disable().listen();

  gui.add(params, "debug", ["none", "cluster", "lod"]).name("debugColors").onChange(on.debug);
  gui.add(toggles, "wireframe").onChange(on.toggles);
  gui.add(toggles, "shadows").onChange(on.toggles);
  gui.add(toggles, "freezeLod").name("freeze LOD").onChange(on.freeze);
  gui.add(toggles, "compare").name("compare: full detail").onChange(on.toggles);
  gui.add({ download: on.download }, "download").name("Download .meshlets");

  return {
    showDroppedFile(name) {
      modelOptions[`file: ${name}`] = "file";
      for (const key of Object.keys(modelOptions)) if (key.startsWith("file: ") && key !== `file: ${name}`) delete modelOptions[key];
      model = model.options(modelOptions).onChange(on.model);
      params.model = "file";
      model.updateDisplay();
    },
  };
}
