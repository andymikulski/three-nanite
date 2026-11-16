/**
 * Shared data contract for three-nanite.
 *
 * Everything here is plain data (typed arrays + numbers), so a `MeshletData`
 * can be built in a worker or a Node bake script, transferred, serialized to
 * a file, and handed to the runtime without touching three.js.
 *
 * The model follows meshoptimizer's `clusterlod.h` (itself modelled on
 * Nanite):
 *
 *  - The mesh is split into CLUSTERS of at most `maxTriangles` triangles.
 *  - Clusters are partitioned into GROUPS. Each group is merged, simplified
 *    with its outer boundary locked, and the result is split into new,
 *    coarser clusters. Those new clusters record the group they came from in
 *    `clusterRefined`. Repeating this yields a DAG.
 *  - Simplification never creates vertices (border dilation, a build option,
 *    appends some to the input's), so EVERY cluster of EVERY level indexes the
 *    one shared vertex buffer.
 *  - Each group stores the sphere + error of its simplified form. Errors and
 *    spheres are monotonic up the DAG, which is what makes any cut crack-free.
 *
 * A cluster `c` is drawn for a given view iff
 *
 *     over(clusterGroup[c])  &&  (clusterRefined[c] < 0 || !over(clusterRefined[c]))
 *
 * where `over(g)` means "group g's simplified form has a projected error above
 * the threshold". Terminal groups have `groupError = Infinity`, so `over` is
 * always true for them.
 */

export type TypedArray =
  | Float32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array;

/** One vertex attribute, laid out exactly like a three.js BufferAttribute. */
export interface MeshletAttribute {
  /** three.js attribute name: "position", "normal", "uv", "uv1", "color", "tangent", ... */
  name: string;
  /** Length is `vertexCount * itemSize`. */
  array: TypedArray;
  itemSize: number;
  normalized: boolean;
}

/** The complete cluster LOD hierarchy for one mesh. */
export interface MeshletData {
  /** Format version of this structure. */
  version: 1;

  // ---- vertices (shared by every cluster of every LOD level) -------------
  vertexCount: number;
  /**
   * Vertex attributes. `attributes[0]` is ALWAYS `position`
   * (Float32Array, itemSize 3, not normalized), except for hierarchy-only data
   * (for example `deserializeMeshlets` of a file written with
   * `attributes: false`), where the array is empty. Such data can drive
   * `ClusterSelector` but not `MeshletMesh`.
   */
  attributes: MeshletAttribute[];

  // ---- triangles ---------------------------------------------------------
  /**
   * Concatenated triangle lists of all clusters. Values are indices into the
   * shared vertex buffer. Cluster `c` owns
   * `indices[clusterIndexStart[c] .. clusterIndexStart[c] + clusterIndexCount[c])`.
   * Clusters are stored in order, without gaps: `clusterIndexStart[c+1] ===
   * clusterIndexStart[c] + clusterIndexCount[c]`.
   */
  indices: Uint32Array;

  // ---- clusters (structure of arrays, length clusterCount) ---------------
  clusterCount: number;
  clusterIndexStart: Uint32Array;
  clusterIndexCount: Uint32Array;
  /**
   * Tight bounding sphere of the cluster's own triangles, `[x, y, z, r]` per
   * cluster. For CULLING ONLY (frustum, raycast). Never use for LOD.
   */
  clusterBounds: Float32Array;
  /** Group this cluster belongs to (the group that gets simplified into coarser clusters). */
  clusterGroup: Uint32Array;
  /**
   * Group this cluster was produced FROM by simplification (the finer
   * group), or -1 for clusters of the original mesh (LOD 0).
   */
  clusterRefined: Int32Array;

  // ---- groups (structure of arrays, length groupCount) -------------------
  groupCount: number;
  /**
   * Clusters are sorted by group, so group `g` owns clusters
   * `[groupClusterStart[g] .. groupClusterStart[g] + groupClusterCount[g])`.
   */
  groupClusterStart: Uint32Array;
  groupClusterCount: Uint32Array;
  /**
   * LOD sphere of the group's simplified form, `[x, y, z, r]` per group, in
   * mesh space. Monotonic: it encloses the LOD spheres of every group below
   * it in the DAG.
   */
  groupBounds: Float32Array;
  /**
   * Accumulated simplification error of the group's simplified form, in mesh
   * space units. Monotonic (>= the error of every group below it).
   * `Infinity` for terminal groups (could not be simplified further).
   */
  groupError: Float32Array;
  /** DAG level the group was generated at; 0 = groups of original clusters. */
  groupDepth: Uint16Array;

  // ---- derived / convenience --------------------------------------------
  /** Ids of all terminal groups (`groupError === Infinity`). Traversal starts here. */
  rootGroups: Uint32Array;
  /** `max(groupDepth) + 1`. */
  levelCount: number;
  /** Bounding sphere of the whole mesh `[x, y, z, r]`. */
  boundingSphere: Float32Array;
  /** Axis-aligned bounding box of the whole mesh `[minX, minY, minZ, maxX, maxY, maxZ]`. */
  boundingBox: Float32Array;
}

/** Raw mesh handed to the builder. */
export interface MeshletInput {
  /** `attributes[0]` must be `position` (Float32Array, itemSize 3). */
  attributes: MeshletAttribute[];
  /** Triangle list. If omitted the mesh is treated as non-indexed (0, 1, 2, ...). */
  indices?: Uint32Array | Uint16Array | null;
}
