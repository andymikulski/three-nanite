/**
 * Offline cluster LOD builder. A TypeScript port of `clodBuild` from
 * meshoptimizer's clusterlod.h, adapted to the JS/wasm API of meshoptimizer:
 *
 *  - every group is compacted into LOCAL arrays before it is handed to wasm
 *    (the wrappers copy whole vertex arrays on every call, so passing the
 *    global arrays per group would be O(groups * vertexCount));
 *  - `meshopt_partitionClusters` is ported in `partition.ts`;
 *  - the output is the flat structure-of-arrays `MeshletData`.
 */
import { MeshoptClusterizer } from "meshoptimizer/clusterizer";
import type { MeshletBuffers } from "meshoptimizer/clusterizer";
import { MeshoptSimplifier } from "meshoptimizer/simplifier";
import type { SimplifierFlags } from "meshoptimizer/simplifier";
import type { MeshletAttribute, MeshletData, MeshletInput, TypedArray } from "../types.js";
import { partitionClusters } from "./partition.js";

export interface BuildMeshletsOptions {
  /** Max triangles per cluster, 4..256. Default 128. */
  maxTriangles?: number;
  /** Max vertices per cluster. Default `min(maxTriangles, 256)`. */
  maxVertices?: number;
  /** Min triangles per cluster (buildMeshletsFlex). Default `floor(maxTriangles / 3)`. */
  minTriangles?: number;
  /** Target number of clusters per group. Default 16. */
  partitionSize?: number;
  /** buildMeshletsFlex split factor. Default 2. */
  splitFactor?: number;
  /** Fraction of triangles kept by each simplification step. Default 0.5. */
  simplifyRatio?: number;
  /** A group is stuck (terminal) if simplified > merged * threshold. Default 0.85. */
  simplifyThreshold?: number;
  /** Weight of the previous (finer) error in the error accumulation. Default 1. */
  errorMergePrevious?: number;
  /** Additive weight of the current simplification error. Default 0. */
  errorMergeAdditive?: number;
  /** Use sloppy simplification when regular simplification gets stuck. Default true. */
  fallbackSloppy?: boolean;
  /** Error multiplier for sloppy results. Default 2. */
  sloppyErrorFactor?: number;
  /**
   * Caps the error a simplification step records at this many times the edge length of the triangles
   * it started from (per triangle the longer of its shortest edge and half its longest; the largest in
   * the group counts). A level is then drawn at the latest when the triangles under it are about
   * `errorThreshold / errorEdgeLimit` pixels long, however large its measured error (meshoptimizer's
   * `simplify_error_edge_limit`). It is meant for dense connected surfaces whose attribute error is
   * out of proportion to their triangles. 0 (the default) records the measured error.
   *
   * Not for foliage made of loose cards: simplifying those removes cards without making the rest any
   * longer, so every level gets the same cap and the whole mesh drops to its coarsest level at once.
   */
  errorEdgeLimit?: number;
  /**
   * The hierarchy normally ends when one cluster is left, which leaves up to `maxTriangles` triangles
   * drawn however far away the mesh is. With this the last cluster keeps being simplified, each step a
   * level of its own, until it has at most this many triangles or does not simplify any more. Default 8;
   * `Infinity` ends at one cluster, as meshoptimizer's clusterlod does.
   */
  coarsestTriangles?: number;
  /** Per-attribute-name weight for attribute-aware simplification (applied to every component). Default `{ normal: 0.5 }`. */
  attributeWeights?: Record<string, number>;
  /** Attribute names whose seams get the meshopt "Protect" vertex bit. Default `["uv", "uv1", "uv2", "uv3"]`. */
  protectAttributes?: string[];
  /**
   * Per vertex, non-zero for a vertex that must be kept where it is at every level (a position is
   * locked as a whole: locking one of its vertices locks all of them). Use it for the vertices a mesh
   * shares with a neighbouring mesh that is simplified on its own, such as the edges of terrain tiles,
   * so no crack opens between the two. Locked vertices limit how coarse the mesh can become.
   */
  lockedVertices?: ArrayLike<number>;
  /** Called once per DAG level. */
  onProgress?: (info: { level: number; clusterCount: number }) => void;
}

const FLT_MAX = 3.4028234663852886e38;
const LOCK = 1;
const PROTECT = 2;
const SCRATCH = 128; // bit 7: "seen by a previous group" (lockBoundary)
const MAX_ATTRIBUTE_FLOATS = 32;

interface Cluster {
  /** Triangle list, global vertex ids. */
  indices: Uint32Array;
  /** Group that produced this cluster, or -1 for LOD 0. */
  refined: number;
  /** LOD sphere (inherited group bounds) and its error. Used for monotonic bounds only. */
  lod: [number, number, number, number];
  error: number;
  /** Tight sphere of the cluster's own triangles. Culling only. */
  tight: [number, number, number, number];
}

interface SimplifyAttributes {
  /** Interleaved floats per vertex, or null if there are no weighted attributes. */
  data: Float32Array | null;
  stride: number;
  weights: number[];
}

export async function buildMeshlets(input: MeshletInput, options: BuildMeshletsOptions = {}): Promise<MeshletData> {
  await Promise.all([MeshoptClusterizer.ready, MeshoptSimplifier.ready]);

  // ---- options --------------------------------------------------------------
  const maxTriangles = options.maxTriangles ?? 128;
  if (!Number.isInteger(maxTriangles) || maxTriangles < 4 || maxTriangles > 256) {
    throw new Error(`buildMeshlets: maxTriangles must be an integer in [4, 256], got ${maxTriangles}`);
  }
  const maxVertices = options.maxVertices ?? Math.min(maxTriangles, 256);
  if (!Number.isInteger(maxVertices) || maxVertices < 3 || maxVertices > 256) {
    throw new Error(`buildMeshlets: maxVertices must be an integer in [3, 256], got ${maxVertices}`);
  }
  const minTriangles = options.minTriangles ?? Math.max(1, Math.floor(maxTriangles / 3));
  if (!Number.isInteger(minTriangles) || minTriangles < 1 || minTriangles > maxTriangles) {
    throw new Error(`buildMeshlets: minTriangles must be an integer in [1, maxTriangles], got ${minTriangles}`);
  }
  const partitionSize = options.partitionSize ?? 16;
  if (!(partitionSize >= 1)) throw new Error("buildMeshlets: partitionSize must be >= 1");
  const splitFactor = options.splitFactor ?? 2;
  const simplifyRatio = options.simplifyRatio ?? 0.5;
  const simplifyThreshold = options.simplifyThreshold ?? 0.85;
  const errorMergePrevious = options.errorMergePrevious ?? 1;
  const errorMergeAdditive = options.errorMergeAdditive ?? 0;
  const fallbackSloppy = options.fallbackSloppy ?? true;
  const sloppyErrorFactor = options.sloppyErrorFactor ?? 2;
  const errorEdgeLimit = options.errorEdgeLimit ?? 0;
  if (!(errorEdgeLimit >= 0)) throw new Error("buildMeshlets: errorEdgeLimit must be >= 0");
  const coarsestTriangles = options.coarsestTriangles ?? 8;
  if (!(coarsestTriangles >= 1)) throw new Error("buildMeshlets: coarsestTriangles must be >= 1");
  const onProgress = options.onProgress;

  // ---- input ----------------------------------------------------------------
  const attributes = input.attributes;
  if (!attributes || attributes.length === 0) throw new Error("buildMeshlets: input has no attributes");
  const posAttr = attributes[0];
  if (posAttr.name !== "position" || !(posAttr.array instanceof Float32Array) || posAttr.itemSize !== 3) {
    throw new Error("buildMeshlets: attributes[0] must be 'position' (Float32Array, itemSize 3)");
  }
  const positions = posAttr.array;
  if (positions.length === 0 || positions.length % 3 !== 0) throw new Error("buildMeshlets: empty or malformed position array");
  const vertexCount = positions.length / 3;

  for (const a of attributes) {
    if (a.array.length !== vertexCount * a.itemSize) {
      throw new Error(`buildMeshlets: attribute '${a.name}' length does not match vertex count`);
    }
  }

  const rawIndices = input.indices ?? null;
  let indices: Uint32Array;
  if (rawIndices) {
    if (rawIndices.length === 0) throw new Error("buildMeshlets: empty mesh (no indices)");
    if (rawIndices.length % 3 !== 0) throw new Error("buildMeshlets: index count must be divisible by 3");
    indices = new Uint32Array(rawIndices); // always a copy: filtered in place below
  } else {
    if (vertexCount % 3 !== 0) throw new Error("buildMeshlets: non-indexed mesh needs a vertex count divisible by 3");
    indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
  }

  // drop degenerate triangles; also validates ids
  {
    let w = 0;
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i];
      const b = indices[i + 1];
      const c = indices[i + 2];
      if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
        throw new Error("buildMeshlets: index out of range");
      }
      if (a === b || b === c || a === c) continue;
      if (w !== i) {
        indices[w] = a;
        indices[w + 1] = b;
        indices[w + 2] = c;
      }
      w += 3;
    }
    if (w !== indices.length) indices = indices.slice(0, w);
  }
  if (indices.length === 0) throw new Error("buildMeshlets: mesh has no non-degenerate triangles");

  // ---- global per-vertex data (built once) ------------------------------------
  // position-only remap: vertices with the same position map to the same canonical id
  const remap = MeshoptSimplifier.generatePositionRemap(positions, 3);
  const locks = new Uint8Array(vertexCount);
  setProtectBits(locks, remap, attributes, options.protectAttributes ?? ["uv", "uv1", "uv2", "uv3"]);
  // user locks, per position (canonical vertex)
  let pinned: Uint8Array | null = null;
  if (options.lockedVertices) {
    if (options.lockedVertices.length !== vertexCount) {
      throw new Error("buildMeshlets: lockedVertices length does not match vertex count");
    }
    pinned = new Uint8Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) if (options.lockedVertices[i]) pinned[remap[i]] = LOCK;
  }
  const simpAttrs = buildSimplifyAttributes(attributes, vertexCount, options.attributeWeights ?? { normal: 0.5 });

  // ---- LOD 0 ----------------------------------------------------------------
  const flags: SimplifierFlags[] = ["ErrorAbsolute", "ErrorClamped", "Permissive"];

  let clusters: Cluster[];
  {
    const buffers = MeshoptClusterizer.buildMeshletsFlex(indices, positions, 3, maxVertices, minTriangles, maxTriangles, 0, splitFactor);
    clusters = makeClusters(buffers, null, MeshoptClusterizer.computeMeshletBounds(buffers, positions, 3), -1);
    for (const c of clusters) {
      c.lod = [c.tight[0], c.tight[1], c.tight[2], c.tight[3]];
      c.error = 0;
    }
  }

  // ---- output accumulators ----------------------------------------------------
  const outChunks: Uint32Array[] = [];
  let outIndexOffset = 0;
  const cIndexStart: number[] = [];
  const cIndexCount: number[] = [];
  const cBounds: number[] = [];
  const cGroup: number[] = [];
  const cRefined: number[] = [];
  const gClusterStart: number[] = [];
  const gClusterCount: number[] = [];
  const gBounds: number[] = [];
  const gError: number[] = [];
  const gDepth: number[] = [];

  /** Emits a group with its member clusters; returns the group id (used as `refined` of new clusters). */
  const outputGroup = (members: Cluster[], bounds: Bound, depth: number): number => {
    const gid = gClusterStart.length;
    gClusterStart.push(cIndexStart.length);
    gClusterCount.push(members.length);
    gBounds.push(bounds.x, bounds.y, bounds.z, bounds.r);
    gError.push(bounds.error);
    gDepth.push(depth);

    for (const c of members) {
      cIndexStart.push(outIndexOffset);
      cIndexCount.push(c.indices.length);
      outChunks.push(c.indices);
      outIndexOffset += c.indices.length;
      cBounds.push(c.tight[0], c.tight[1], c.tight[2], c.tight[3]);
      cGroup.push(gid);
      cRefined.push(c.refined);
    }
    return gid;
  };

  // ---- scratch state for per-group compaction ----------------------------------
  const globalToLocal = new Int32Array(vertexCount).fill(-1);
  const scratch = new GroupScratch(simpAttrs.stride);

  let depth = 0;

  // merge and simplify clusters until we can't merge anymore; a single cluster left is a group of
  // its own (nothing to lock but the caller's pins) while it is above `coarsestTriangles`
  while (clusters.length > 1 || (clusters.length === 1 && clusters[0].indices.length / 3 > coarsestTriangles)) {
    onProgress?.({ level: depth, clusterCount: clusters.length });

    const groupOffsets = partitionLevel(clusters, partitionSize, positions, remap, vertexCount);
    lockBoundary(locks, clusters, groupOffsets, remap, pinned);

    const pending: Cluster[] = [];

    for (let gi = 0; gi + 1 < groupOffsets.length; gi++) {
      const members = clusters.slice(groupOffsets[gi], groupOffsets[gi + 1]);

      let mergedLength = 0;
      for (const m of members) mergedLength += m.indices.length;
      const merged = new Uint32Array(mergedLength);
      let o = 0;
      for (const m of members) {
        merged.set(m.indices, o);
        o += m.indices.length;
      }

      const targetSize = Math.floor((merged.length / 3) * simplifyRatio) * 3;

      // bounds and error monotonicity: use merged member bounds, not the precise bounds of the geometry
      const bounds = boundsMerge(members);

      // compact the group into local arrays
      const localCount = scratch.compact(merged, positions, locks, simpAttrs, globalToLocal);
      const localPositions = scratch.positions.subarray(0, localCount * 3);

      const sres = simplifyLocal(
        scratch,
        localCount,
        localPositions,
        simpAttrs,
        flags,
        targetSize,
        fallbackSloppy,
        sloppyErrorFactor,
        errorEdgeLimit,
      );

      if (sres.indices.length > merged.length * simplifyThreshold) {
        // stuck: terminal group, abandon the merge
        bounds.error = Infinity;
        outputGroup(members, bounds, depth);
        scratch.release(globalToLocal, localCount);
        continue;
      }

      const err = sres.error;
      bounds.error = Math.fround(Math.max(bounds.error * errorMergePrevious, err) + err * errorMergeAdditive);

      const refined = outputGroup(members, bounds, depth);

      // clusterize the simplified group (local ids; map back to global)
      const buffers = MeshoptClusterizer.buildMeshletsFlex(
        sres.indices,
        localPositions,
        3,
        maxVertices,
        minTriangles,
        maxTriangles,
        0,
        splitFactor,
      );
      const created = makeClusters(buffers, scratch.localToGlobal, MeshoptClusterizer.computeMeshletBounds(buffers, localPositions, 3), refined);
      for (const c of created) {
        // inherit the group-merged bounds so the next level merges them conservatively
        c.lod = [bounds.x, bounds.y, bounds.z, bounds.r];
        c.error = bounds.error;
        pending.push(c);
      }

      scratch.release(globalToLocal, localCount);
    }

    clusters = pending;
    depth++;
  }

  if (clusters.length > 0) {
    onProgress?.({ level: depth, clusterCount: clusters.length });
    const b = boundsMerge(clusters);
    b.error = Infinity;
    outputGroup(clusters, b, depth);
  }

  // ---- assemble MeshletData --------------------------------------------------
  const clusterCount = cIndexStart.length;
  const groupCount = gClusterStart.length;

  const outIndices = new Uint32Array(outIndexOffset);
  {
    let w = 0;
    for (const chunk of outChunks) {
      outIndices.set(chunk, w);
      w += chunk.length;
    }
  }

  const rootList: number[] = [];
  let maxDepth = 0;
  for (let g = 0; g < groupCount; g++) {
    if (gError[g] === Infinity) rootList.push(g);
    if (gDepth[g] > maxDepth) maxDepth = gDepth[g];
  }

  const sphere = MeshoptClusterizer.computeSphereBounds(positions, 3);
  const box = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k];
      if (v < box[k]) box[k] = v;
      if (v > box[k + 3]) box[k + 3] = v;
    }
  }

  return {
    version: 1,
    vertexCount,
    attributes: attributes.slice(),
    indices: outIndices,
    clusterCount,
    clusterIndexStart: Uint32Array.from(cIndexStart),
    clusterIndexCount: Uint32Array.from(cIndexCount),
    clusterBounds: Float32Array.from(cBounds),
    clusterGroup: Uint32Array.from(cGroup),
    clusterRefined: Int32Array.from(cRefined),
    groupCount,
    groupClusterStart: Uint32Array.from(gClusterStart),
    groupClusterCount: Uint32Array.from(gClusterCount),
    groupBounds: Float32Array.from(gBounds),
    groupError: Float32Array.from(gError),
    groupDepth: Uint16Array.from(gDepth),
    rootGroups: Uint32Array.from(rootList),
    levelCount: maxDepth + 1,
    boundingSphere: new Float32Array([sphere.centerX, sphere.centerY, sphere.centerZ, sphere.radius]),
    boundingBox: box,
  };
}

// ---------------------------------------------------------------------------------------------
// clusters
// ---------------------------------------------------------------------------------------------

interface Bound {
  x: number;
  y: number;
  z: number;
  r: number;
  error: number;
}

type BoundsLike = { centerX: number; centerY: number; centerZ: number; radius: number };

/** Converts meshlet buffers into clusters with global triangle lists. `localToGlobal` null means ids are already global. */
function makeClusters(buffers: MeshletBuffers, localToGlobal: Uint32Array | null, bounds: BoundsLike[], refined: number): Cluster[] {
  const { meshlets, vertices, triangles, meshletCount } = buffers;
  const out: Cluster[] = new Array(meshletCount);

  // each meshlet owns its slice of one shared buffer
  let total = 0;
  for (let i = 0; i < meshletCount; i++) total += meshlets[i * 4 + 3] * 3;
  const all = new Uint32Array(total);

  let w = 0;
  for (let i = 0; i < meshletCount; i++) {
    const vo = meshlets[i * 4];
    const to = meshlets[i * 4 + 1];
    const n = meshlets[i * 4 + 3] * 3;
    const begin = w;
    if (localToGlobal) {
      for (let j = 0; j < n; j++) all[w++] = localToGlobal[vertices[vo + triangles[to + j]]];
    } else {
      for (let j = 0; j < n; j++) all[w++] = vertices[vo + triangles[to + j]];
    }
    const b = bounds[i];
    const c: [number, number, number, number] = [b.centerX, b.centerY, b.centerZ, b.radius];
    out[i] = { indices: all.subarray(begin, w), refined, lod: c, error: 0, tight: c };
  }
  return out;
}

/** Merged LOD sphere of the given clusters; error is the max of member errors. */
function boundsMerge(members: Cluster[]): Bound {
  const n = members.length;
  const centers = new Float32Array(n * 3);
  const radii = new Float32Array(n);
  let error = 0;
  for (let i = 0; i < n; i++) {
    const m = members[i];
    centers[i * 3] = m.lod[0];
    centers[i * 3 + 1] = m.lod[1];
    centers[i * 3 + 2] = m.lod[2];
    radii[i] = m.lod[3];
    if (m.error > error) error = m.error;
  }
  const s = MeshoptClusterizer.computeSphereBounds(centers, 3, radii, 1);
  return { x: s.centerX, y: s.centerY, z: s.centerZ, r: s.radius, error };
}

// ---------------------------------------------------------------------------------------------
// partitioning and locking
// ---------------------------------------------------------------------------------------------

/**
 * Reorders `clusters` so every partition (group) is contiguous; returns group offsets
 * (`groups + 1` entries).
 */
function partitionLevel(
  clusters: Cluster[],
  partitionSize: number,
  positions: Float32Array,
  remap: Uint32Array,
  vertexCount: number,
): number[] {
  if (clusters.length <= partitionSize) return [0, clusters.length];

  const counts = new Uint32Array(clusters.length);
  let total = 0;
  for (let i = 0; i < clusters.length; i++) {
    counts[i] = clusters[i].indices.length;
    total += counts[i];
  }
  const remapped = new Uint32Array(total);
  let w = 0;
  for (const c of clusters) {
    const idx = c.indices;
    for (let j = 0; j < idx.length; j++) remapped[w++] = remap[idx[j]];
  }

  const { partition, count } = partitionClusters(remapped, counts, positions, vertexCount, partitionSize);

  // stable counting sort by partition id
  const offsets = new Array<number>(count + 1).fill(0);
  for (let i = 0; i < clusters.length; i++) offsets[partition[i] + 1]++;
  for (let i = 1; i <= count; i++) offsets[i] += offsets[i - 1];

  const cursor = offsets.slice(0, count);
  const sorted: Cluster[] = new Array(clusters.length);
  for (let i = 0; i < clusters.length; i++) sorted[cursor[partition[i]]++] = clusters[i];
  for (let i = 0; i < sorted.length; i++) clusters[i] = sorted[i];

  return offsets;
}

/**
 * Locks every vertex position that is shared with a different group, so group
 * boundaries are untouched by simplification. Bit 7 is scratch. The Protect
 * bit (seams) set up earlier is preserved. `pinned` holds the caller's own locks
 * (per canonical vertex), which apply at every level.
 */
function lockBoundary(
  locks: Uint8Array,
  clusters: Cluster[],
  groupOffsets: number[],
  remap: Uint32Array,
  pinned: Uint8Array | null,
): void {
  for (let i = 0; i < locks.length; i++) locks[i] &= ~(LOCK | SCRATCH);

  for (let g = 0; g + 1 < groupOffsets.length; g++) {
    const begin = groupOffsets[g];
    const end = groupOffsets[g + 1];

    // lock vertices already seen by a prior group
    for (let j = begin; j < end; j++) {
      const idx = clusters[j].indices;
      for (let k = 0; k < idx.length; k++) {
        const r = remap[idx[k]];
        locks[r] |= locks[r] >> 7;
      }
    }

    // mark as seen
    for (let j = begin; j < end; j++) {
      const idx = clusters[j].indices;
      for (let k = 0; k < idx.length; k++) locks[remap[idx[k]]] |= SCRATCH;
    }
  }

  // lock all vertices of a position consistently; keep the protect bit
  for (let i = 0; i < locks.length; i++) {
    locks[i] = (locks[remap[i]] & LOCK) | (locks[i] & PROTECT);
  }
  if (pinned) for (let i = 0; i < locks.length; i++) locks[i] |= pinned[remap[i]];
}

// ---------------------------------------------------------------------------------------------
// attributes
// ---------------------------------------------------------------------------------------------

function setProtectBits(locks: Uint8Array, remap: Uint32Array, attributes: MeshletAttribute[], names: string[]): void {
  const protectedAttrs = attributes.filter((a, i) => i > 0 && names.includes(a.name));
  if (protectedAttrs.length === 0) return;

  for (let i = 0; i < locks.length; i++) {
    const r = remap[i];
    if (r === i) continue;
    for (const a of protectedAttrs) {
      const arr = a.array;
      const s = a.itemSize;
      let differs = false;
      for (let k = 0; k < s; k++) {
        if (arr[i * s + k] !== arr[r * s + k]) {
          differs = true;
          break;
        }
      }
      if (differs) {
        locks[i] |= PROTECT;
        break;
      }
    }
  }
}

function normalizationDivisor(array: TypedArray, normalized: boolean): number {
  if (!normalized) return 1;
  if (array instanceof Uint8Array) return 255;
  if (array instanceof Int8Array) return 127;
  if (array instanceof Uint16Array) return 65535;
  if (array instanceof Int16Array) return 32767;
  if (array instanceof Uint32Array) return 4294967295;
  if (array instanceof Int32Array) return 2147483647;
  return 1;
}

function buildSimplifyAttributes(attributes: MeshletAttribute[], vertexCount: number, weights: Record<string, number>): SimplifyAttributes {
  const used: MeshletAttribute[] = [];
  const w: number[] = [];
  let stride = 0;

  for (let i = 1; i < attributes.length; i++) {
    const a = attributes[i];
    const weight = weights[a.name];
    if (!(weight > 0)) continue;
    used.push(a);
    for (let k = 0; k < a.itemSize; k++) w.push(weight);
    stride += a.itemSize;
  }

  if (stride > MAX_ATTRIBUTE_FLOATS) {
    throw new Error(`buildMeshlets: weighted attributes use ${stride} floats per vertex, at most ${MAX_ATTRIBUTE_FLOATS} are supported`);
  }
  if (stride === 0) return { data: null, stride: 0, weights: [] };

  const data = new Float32Array(vertexCount * stride);
  let base = 0;
  for (const a of used) {
    const arr = a.array;
    const s = a.itemSize;
    const div = normalizationDivisor(arr, a.normalized);
    const signedNormalized = a.normalized && (arr instanceof Int8Array || arr instanceof Int16Array || arr instanceof Int32Array);

    for (let v = 0; v < vertexCount; v++) {
      for (let k = 0; k < s; k++) {
        let x = arr[v * s + k] / div;
        if (signedNormalized && x < -1) x = -1;
        data[v * stride + base + k] = x;
      }
    }
    base += s;
  }

  return { data, stride, weights: w };
}

// ---------------------------------------------------------------------------------------------
// per-group local compaction and simplification
// ---------------------------------------------------------------------------------------------

/** Growable scratch buffers for one group's local data. */
class GroupScratch {
  positions = new Float32Array(0);
  attributes = new Float32Array(0);
  locks = new Uint8Array(0);
  localToGlobal = new Uint32Array(0);
  localIndices = new Uint32Array(0);
  private localCountIndices = 0;
  private readonly attrStride: number;

  constructor(attrStride: number) {
    this.attrStride = attrStride;
  }

  /** Compacts the vertices used by `merged` (global ids) into local arrays; returns the local vertex count. */
  compact(merged: Uint32Array, positions: Float32Array, locks: Uint8Array, attrs: SimplifyAttributes, globalToLocal: Int32Array): number {
    if (this.localToGlobal.length < merged.length) {
      this.localToGlobal = new Uint32Array(merged.length);
      this.positions = new Float32Array(merged.length * 3);
      this.locks = new Uint8Array(merged.length);
      this.attributes = new Float32Array(merged.length * this.attrStride);
    }
    if (this.localIndices.length < merged.length) this.localIndices = new Uint32Array(merged.length);

    const l2g = this.localToGlobal;
    const lpos = this.positions;
    const llock = this.locks;
    const lattr = this.attributes;
    const stride = this.attrStride;
    const gattr = attrs.data;
    const localIndices = this.localIndices;

    let count = 0;
    for (let i = 0; i < merged.length; i++) {
      const g = merged[i];
      let l = globalToLocal[g];
      if (l < 0) {
        l = count++;
        globalToLocal[g] = l;
        l2g[l] = g;
        lpos[l * 3] = positions[g * 3];
        lpos[l * 3 + 1] = positions[g * 3 + 1];
        lpos[l * 3 + 2] = positions[g * 3 + 2];
        llock[l] = locks[g];
        if (gattr) for (let k = 0; k < stride; k++) lattr[l * stride + k] = gattr[g * stride + k];
      }
      localIndices[i] = l;
    }
    this.localCountIndices = merged.length;
    return count;
  }

  /** Local index list of the last compacted group. */
  get indices(): Uint32Array {
    return this.localIndices.subarray(0, this.localCountIndices);
  }

  /** Resets the global->local map entries touched by the last group. */
  release(globalToLocal: Int32Array, localCount: number): void {
    for (let l = 0; l < localCount; l++) globalToLocal[this.localToGlobal[l]] = -1;
  }
}

interface SimplifyResult {
  /** LOCAL vertex ids. */
  indices: Uint32Array;
  /** Absolute error in mesh units. */
  error: number;
}

function simplifyLocal(
  scratch: GroupScratch,
  localCount: number,
  localPositions: Float32Array,
  attrs: SimplifyAttributes,
  flags: SimplifierFlags[],
  targetSize: number,
  fallbackSloppy: boolean,
  sloppyErrorFactor: number,
  errorEdgeLimit: number,
): SimplifyResult {
  const localIndices = scratch.indices;
  const localLocks = scratch.locks.subarray(0, localCount);
  const localAttrs = scratch.attributes.subarray(0, localCount * attrs.stride);

  let [result, error] = MeshoptSimplifier.simplifyWithAttributes(
    localIndices,
    localPositions,
    3,
    localAttrs,
    attrs.stride,
    attrs.weights,
    localLocks,
    targetSize,
    FLT_MAX,
    flags,
  );

  // sloppy returns a relative error; scale to absolute
  if (result.length > targetSize && fallbackSloppy) {
    const [sloppy, relError] = MeshoptSimplifier.simplifySloppy(localIndices, localPositions, 3, localLocks, targetSize, FLT_MAX);
    result = sloppy;
    error = relError * MeshoptSimplifier.getScale(localPositions, 3) * sloppyErrorFactor;
  }

  // never emit an empty group: it would break DAG invariants
  if (result.length === 0) result = localIndices.slice(0, 3);

  // optionally limit the error by the edge length of the triangles simplified (clusterlod.h
  // simplify_error_edge_limit): what a step gets wrong is no bigger than the triangles it replaced
  if (errorEdgeLimit > 0) {
    let maxEdgeSq = 0;
    for (let i = 0; i < localIndices.length; i += 3) {
      const a = localIndices[i] * 3;
      const b = localIndices[i + 1] * 3;
      const c = localIndices[i + 2] * 3;
      const ab = distSq(localPositions, a, b);
      const ac = distSq(localPositions, a, c);
      const bc = distSq(localPositions, b, c);
      const emax = Math.max(ab, ac, bc);
      const emin = Math.min(ab, ac, bc);
      // the shortest edge, so thin triangles count as thin, with a stopgap for long slivers
      maxEdgeSq = Math.max(maxEdgeSq, emin, emax / 4);
    }
    error = Math.min(error, Math.sqrt(maxEdgeSq) * errorEdgeLimit);
  }

  return { indices: result, error };
}

function distSq(p: Float32Array, a: number, b: number): number {
  const x = p[a] - p[b];
  const y = p[a + 1] - p[b + 1];
  const z = p[a + 2] - p[b + 2];
  return x * x + y * y + z * z;
}
