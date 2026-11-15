/// <reference lib="webworker" />
import { BufferAttribute, BufferGeometry } from "three";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import type { MeshletData } from "../src/index.js";

/** Request: a welded, indexed geometry as plain arrays. */
export interface BuildRequest {
  position: Float32Array;
  normal: Float32Array;
  index: Uint16Array | Uint32Array;
}

export type BuildMessage =
  | { type: "progress"; level: number; clusterCount: number }
  | { type: "done"; data: MeshletData }
  | { type: "error"; message: string };

function post(message: BuildMessage): void {
  postMessage(message);
}

self.onmessage = async (event: MessageEvent<BuildRequest>) => {
  try {
    const { position, normal, index } = event.data;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(position, 3));
    geometry.setAttribute("normal", new BufferAttribute(normal, 3));
    geometry.setIndex(new BufferAttribute(index, 1));
    const data = await buildMeshletsFromGeometry(geometry, {
      onProgress: ({ level, clusterCount }) => post({ type: "progress", level, clusterCount }),
    });
    post({ type: "done", data });
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
