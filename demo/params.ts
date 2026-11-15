import type { DebugColorMode } from "../src/index.js";

export type RendererKind = "webgl" | "webgpu";
/** `file` is a model dropped onto the page; it cannot be linked. */
export type ModelName = "bunny" | "knot" | "file";

/** The state mirrored in the URL query. */
export interface Params {
  renderer: RendererKind;
  model: ModelName;
  grid: number;
  threshold: number;
  debug: DebugColorMode;
}

export const GRID_MAX = 30;
export const THRESHOLD_MIN = 0.25;
export const THRESHOLD_MAX = 64;

const DEFAULTS: Params = { renderer: "webgl", model: "knot", grid: 8, threshold: 1, debug: "none" };

function pick<T extends string>(value: string | null, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback;
}

function number(value: string | null, min: number, max: number, fallback: number): number {
  const n = Number(value);
  return value !== null && Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

export function readParams(search = location.search): Params {
  const q = new URLSearchParams(search);
  return {
    renderer: pick(q.get("renderer"), ["webgl", "webgpu"], DEFAULTS.renderer),
    model: pick(q.get("model"), ["bunny", "knot"], DEFAULTS.model),
    grid: Math.round(number(q.get("grid"), 1, GRID_MAX, DEFAULTS.grid)),
    threshold: number(q.get("threshold"), THRESHOLD_MIN, THRESHOLD_MAX, DEFAULTS.threshold),
    debug: pick(q.get("debug"), ["none", "cluster", "lod"], DEFAULTS.debug),
  };
}

/** Mirrors the state into the address bar without adding history entries. */
export function writeParams(p: Params): void {
  const q = new URLSearchParams(location.search);
  q.set("renderer", p.renderer);
  q.set("model", p.model === "file" ? DEFAULTS.model : p.model);
  q.set("grid", String(p.grid));
  q.set("threshold", String(Math.round(p.threshold * 100) / 100));
  q.set("debug", p.debug);
  history.replaceState(null, "", `?${q}`);
}
