/// <reference lib="webworker" />
import { BufferAttribute, BufferGeometry } from "three";
import { buildMeshletsFromGeometry } from "../../src/build/index.js";
import type { BuildFromGeometryOptions } from "../../src/build/index.js";
import type { MeshletData } from "../../src/index.js";

export interface BakeRequest {
  id: number;
  attributes: Record<string, { array: Float32Array; itemSize: number }>;
  index: Uint16Array | Uint32Array;
  options: BuildFromGeometryOptions;
}

export type BakeReply = { id: number; data: MeshletData } | { id: number; error: string };

self.onmessage = async (event: MessageEvent<BakeRequest>) => {
  const { id, attributes, index, options } = event.data;
  try {
    const geometry = new BufferGeometry();
    for (const [name, a] of Object.entries(attributes)) geometry.setAttribute(name, new BufferAttribute(a.array, a.itemSize));
    geometry.setIndex(new BufferAttribute(index, 1));
    const data = await buildMeshletsFromGeometry(geometry, options);
    postMessage({ id, data } satisfies BakeReply);
  } catch (error) {
    postMessage({ id, error: error instanceof Error ? error.message : String(error) } satisfies BakeReply);
  }
};
