/**
 * `three-nanite/core` -- the part of the library that needs neither three.js
 * nor meshoptimizer: the data types, LOD selection, validation and the file
 * format. Use it to load pre-baked `.meshlets` files and drive your own
 * renderer.
 */
export type { MeshletAttribute, MeshletData, MeshletInput, TypedArray } from "../types.js";

export { ClusterSelector, countTriangles, getLodErrors, lodScale } from "./select.js";
export { computeDrawRanges, rangesOfCut, type DrawRanges } from "./ranges.js";
export {
  deserializeMeshlets,
  MESHLET_FILE_EXTENSION,
  serializeMeshlets,
  type SerializeOptions,
} from "./serialize.js";
export { validateMeshletData } from "./validate.js";
