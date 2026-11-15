import type { MeshletMesh, MeshletMeshStats } from "../src/index.js";

/** Test hook for automated checks. */
export interface DemoApi {
  /** True after the first frame that drew the built mesh. */
  ready: boolean;
  error: string | null;
  renderer: "webgl" | "webgpu";
  /** e.g. `WebGLRenderer`, `WebGPURenderer (WebGPU backend)`, `WebGPURenderer (WebGL fallback)`. */
  backend: string;
  frames: number;
  mesh: MeshletMesh | null;
  stats(): MeshletMeshStats & { fullTriangles: number; fps: number };
}

declare global {
  interface Window {
    __demo: DemoApi;
  }
}
