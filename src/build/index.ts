/**
 * `three-nanite/build` — the offline half of the library.
 *
 * It pulls in meshoptimizer's WebAssembly modules, so it lives in its own
 * entry point: an app that only loads pre-baked `.meshlets` files never pays
 * for it. Runs in browsers, workers and Node.
 */
import type { BufferGeometry } from "three";
import { colorMapFromMaterial } from "../three/colorMap.js";
import { geometryToMeshletInput } from "../three/geometry.js";
import type { MeshletData } from "../types.js";
import { buildMeshlets, type BuildMeshletsOptions } from "./buildMeshlets.js";

export { buildMeshlets, type BuildMeshletsOptions } from "./buildMeshlets.js";
export { type ColorMap } from "./appearance.js";
export { colorMapFromMaterial } from "../three/colorMap.js";
export { partitionClusters } from "./partition.js";
export { unweldMeshlets, weldVertices, type WeldResult } from "./weld.js";

/** Options for {@link buildMeshletsFromGeometry}. */
export interface BuildFromGeometryOptions extends BuildMeshletsOptions {
  /** Non-position attributes to keep. Default: all of them. */
  attributes?: string[];
  /**
   * The material the geometry is drawn with. Its `map` (with its `alphaTest`) becomes `colorMap`,
   * or, for a material with `vertexColors` and no map, `vertexColors` is set: the levels are then
   * also judged by what they change to the eye (see `BuildMeshletsOptions.colorMap`). Give it for
   * any model that gets its colours from a palette or an atlas. Ignored where `colorMap` or
   * `vertexColors` is given, and where the texture's pixels cannot be read (see
   * `colorMapFromMaterial`).
   */
  material?: { map?: unknown; alphaTest?: number; vertexColors?: boolean } | null;
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
  const { attributes, material, ...buildOptions } = options;
  if (material && buildOptions.colorMap === undefined && buildOptions.vertexColors === undefined) {
    const colorMap = colorMapFromMaterial(material as Parameters<typeof colorMapFromMaterial>[0]);
    if (colorMap) buildOptions.colorMap = colorMap;
    else if (material.vertexColors === true && !material.map) buildOptions.vertexColors = true;
  }
  return buildMeshlets(geometryToMeshletInput(geometry, attributes ? { attributes } : undefined), buildOptions);
}
