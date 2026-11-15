/**
 * `three-nanite` — the runtime half of the library. It has no dependency on
 * meshoptimizer; building hierarchies lives in `three-nanite/build`.
 */
export type { MeshletAttribute, MeshletData, MeshletInput, TypedArray } from "./types.js";

// rendering
export {
  MeshletMesh,
  type DebugColorMode,
  type MeshletMeshOptions,
  type MeshletMeshStats,
} from "./three/MeshletMesh.js";

// static LODs and geometry conversion
export { createLodChain, createLodGeometry, geometryToMeshletInput, getLodErrors } from "./three/geometry.js";

// files
export { deserializeMeshlets, MESHLET_FILE_EXTENSION, serializeMeshlets } from "./core/serialize.js";

// low level: LOD selection without three.js objects
export { ClusterSelector, countTriangles, lodScale } from "./core/select.js";
export { validateMeshletData } from "./core/validate.js";
