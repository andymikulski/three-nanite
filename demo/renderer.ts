import { WebGLRenderer } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { RendererKind } from "./params.js";

export interface RendererHandle {
  renderer: WebGLRenderer | WebGPURenderer;
  /** Human readable back-end, e.g. `WebGPURenderer (WebGPU backend)`. */
  backend: string;
  /** Draw calls of the last `render()`. */
  drawCalls(): number;
}

/** Creates the renderer of the requested kind; `three/webgpu` is only loaded when needed. */
export async function createRenderer(kind: RendererKind): Promise<RendererHandle> {
  let handle: RendererHandle;
  if (kind === "webgpu") {
    const { WebGPURenderer } = await import("three/webgpu");
    const renderer = new WebGPURenderer({ antialias: true });
    await renderer.init();
    // `isWebGPUBackend` is not declared on three's `Backend` base type
    const native = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend === true;
    handle = {
      renderer,
      backend: `WebGPURenderer (${native ? "WebGPU backend" : "WebGL fallback"})`,
      drawCalls: () => renderer.info.render.drawCalls,
    };
  } else {
    const renderer = new WebGLRenderer({ antialias: true });
    handle = { renderer, backend: "WebGLRenderer", drawCalls: () => renderer.info.render.calls };
  }
  const { renderer } = handle;
  renderer.shadowMap.enabled = true;
  return handle;
}
