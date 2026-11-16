/**
 * `three-nanite/build` — the offline half of the library.
 *
 * It pulls in meshoptimizer's WebAssembly modules, so it lives in its own
 * entry point: an app that only loads pre-baked `.meshlets` files never pays
 * for it. Runs in browsers, workers and Node.
 */
import type { BufferGeometry } from "three";
import { geometryToMeshletInput } from "../three/geometry.js";
import type { MeshletData } from "../types.js";
import { buildMeshlets, type BuildMeshletsOptions } from "./buildMeshlets.js";

export { buildMeshlets, type BuildMeshletsOptions } from "./buildMeshlets.js";
export { partitionClusters } from "./partition.js";
export { unweldMeshlets, weldVertices, type WeldResult } from "./weld.js";

/** Options for {@link buildMeshletsFromGeometry}. */
export interface BuildFromGeometryOptions extends BuildMeshletsOptions {
  /** Non-position attributes to keep. Default: all of them. */
  attributes?: string[];
}

/**
 * Builds the cluster LOD hierarchy for a three.js geometry.
 *
 * For good results the geometry should be indexed and welded (run
 * `BufferGeometryUtils.mergeVertices` first if it is not): simplification
 * works on shared vertices.
 */
export function buildMeshletsFromGeometry(
  geometry: BufferGeometry,
  options: BuildFromGeometryOptions = {},
): Promise<MeshletData> {
  const { attributes, ...buildOptions } = options;
  return buildMeshlets(geometryToMeshletInput(geometry, attributes ? { attributes } : undefined), buildOptions);
}
