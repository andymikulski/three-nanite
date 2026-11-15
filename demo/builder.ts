import type { BufferGeometry } from "three";
import type { MeshletData } from "../src/index.js";
import type { BuildMessage, BuildRequest } from "./build.worker.js";

export interface BuildResult {
  data: MeshletData;
  /** Wall-clock milliseconds, including the structured clone of the input and output. */
  ms: number;
}

/**
 * Builds the cluster hierarchy in a module worker so the page stays responsive and the
 * progress overlay can repaint while the build runs.
 */
export function buildInWorker(
  geometry: BufferGeometry,
  onProgress: (level: number, clusterCount: number) => void,
): Promise<BuildResult> {
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./build.worker.ts", import.meta.url), { type: "module" });
    const finish = (settle: () => void) => {
      worker.terminate();
      settle();
    };
    worker.onmessage = (event: MessageEvent<BuildMessage>) => {
      const message = event.data;
      if (message.type === "progress") onProgress(message.level, message.clusterCount);
      else if (message.type === "done") finish(() => resolve({ data: message.data, ms: performance.now() - t0 }));
      else finish(() => reject(new Error(message.message)));
    };
    worker.onerror = (event) => finish(() => reject(new Error(event.message || "build worker failed")));

    const request: BuildRequest = {
      position: geometry.getAttribute("position").array as Float32Array,
      normal: geometry.getAttribute("normal").array as Float32Array,
      index: geometry.getIndex()!.array as Uint16Array | Uint32Array,
    };
    worker.postMessage(request); // copied, the main thread keeps using the geometry
  });
}
