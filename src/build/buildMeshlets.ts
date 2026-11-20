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
import { AppearanceCheck, type ColorMap } from "./appearance.js";
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
  /**
   * The base colour texture of the mesh. With it every simplification step is also judged by what
   * it changes TO THE EYE: the group is drawn before and after, small and from all sides, and a
   * region that has changed colour counts as an error of its own size, so the step waits until
   * that region is about a pixel across.
   *
   * The simplifier's own error is how far the surface moves, and that alone lets through steps
   * that move it very little and repaint a lot: the raised rim of a flat roof closing over its
   * floor, a panel sinking into the wall behind it, a face taking its neighbour's texture across a
   * seam. Models that get their colours from a palette or an atlas are full of these; give them
   * this. It costs build time (about twice, for a mesh whose groups are not one colour) and
   * nothing at run time. `alphaTest` in it makes the holes of a cut-out count as holes.
   */
  colorMap?: ColorMap;
  /**
   * As `colorMap`, for a mesh that is coloured by its `color` attribute (a material with
   * `vertexColors`): steps are judged by those colours, and their seams are protected like those
   * of a uv set. Default false. Ignored when `colorMap` is given.
   */
  vertexColors?: boolean;
  /**
   * When a group gets nowhere with the protected seams intact, simplify it again letting collapses
   * cross them. Each protected attribute then counts towards the error with a weight of at least 1,
   * so seams are crossed where it costs least. This is what lets meshes whose every edge is a seam
   * simplify at all: low-poly models textured from an atlas, anything with a lightmap layout in a
   * second uv set. Default true.
   */
  fallbackUnprotected?: boolean;
  /**
   * Use sloppy simplification when a group gets nowhere any other way (after `fallbackUnprotected`).
   * It ignores topology and seams altogether and is the last resort. Default true.
   */
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
  /**
   * Makes up for the area a simplification step removes at open borders by moving the borders that
   * are left outwards, so that every level covers what the level before covered (after meshoptimizer's
   * `simplify_dilate_borders`). Loose pieces with open borders, such as leaf cards, can only be
   * simplified by removing them: without this a canopy thins out with every level and is all but gone
   * once a card is a pixel large; with it the cards that are left grow by what was removed, and since
   * larger cards carry a larger error, the levels also spread out over distance.
   *
   * Whether that is what the full-detail mesh looks like from afar depends on the material. It is for
   * opaque pieces and for cut-outs that keep their coverage when minified (alpha to coverage, alpha
   * hashing, coverage-preserving mipmaps). A plain `alphaTest` cut-out loses its coverage in the
   * mipmaps: the full-detail canopy itself thins out with distance, and a hierarchy built WITHOUT
   * dilation matches it, with far fewer triangles. Hence the default, false.
   *
   * A moved vertex is a NEW vertex (the finer levels keep the original), so the result can have more
   * vertices than the input and then no longer references the input arrays: a mesh made only of cards
   * about doubles, closed surfaces add none. No vertex leaves the bounds of the input.
   */
  dilateBorders?: boolean;
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
  /** Interleaved floats per vertex, or null if there are no weighted or protected attributes. */
  data: Float32Array | null;
  stride: number;
  /** Weight of every float: the caller's, 0 for attributes that are only protected. */
  weights: number[];
  /**
   * Weights for simplifying across protected seams (protected attributes weigh at least 1), or null
   * when there is nothing protected.
   */
  seamWeights: number[] | null;
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
  const fallbackUnprotected = options.fallbackUnprotected ?? true;
  const sloppyErrorFactor = options.sloppyErrorFactor ?? 2;
  const errorEdgeLimit = options.errorEdgeLimit ?? 0;
  if (!(errorEdgeLimit >= 0)) throw new Error("buildMeshlets: errorEdgeLimit must be >= 0");
  const coarsestTriangles = options.coarsestTriangles ?? 8;
  if (!(coarsestTriangles >= 1)) throw new Error("buildMeshlets: coarsestTriangles must be >= 1");
  const dilateBorders = options.dilateBorders ?? false;
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
  /** Vertices of the input; border dilation may add more (see `vertices`). */
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
  const useVertexColors = options.vertexColors === true && !options.colorMap;
  const protectAttributes = options.protectAttributes ?? (useVertexColors ? ["uv", "uv1", "uv2", "uv3", "color"] : ["uv", "uv1", "uv2", "uv3"]);
  const appearance = options.colorMap || useVertexColors ? AppearanceCheck.create(attributes, options.colorMap) : null;
  const seam = seamTest(attributes, protectAttributes);
  if (seam) for (let i = 0; i < vertexCount; i++) if (remap[i] !== i && seam(i, remap[i])) locks[i] |= PROTECT;
  // user locks, per position (canonical vertex)
  let pinned: Uint8Array | null = null;
  if (options.lockedVertices) {
    if (options.lockedVertices.length !== vertexCount) {
      throw new Error("buildMeshlets: lockedVertices length does not match vertex count");
    }
    pinned = new Uint8Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) if (options.lockedVertices[i]) pinned[remap[i]] = LOCK;
  }
  const simpAttrs = buildSimplifyAttributes(
    attributes,
    vertexCount,
    options.attributeWeights ?? { normal: 0.5 },
    fallbackUnprotected ? protectAttributes : [],
  );
  const vertices = new VertexStore(positions, remap, locks);

  // Bounds of the mesh: those of the input. Border dilation keeps the vertices it adds inside them,
  // so a coarse level never reaches further than the mesh it stands for, and culling and LOD
  // distances stay as tight as the input allows.
  const meshSphere = MeshoptClusterizer.computeSphereBounds(positions, 3);
  const boundingSphere = new Float32Array([meshSphere.centerX, meshSphere.centerY, meshSphere.centerZ, meshSphere.radius]);
  const boundingBox = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i + k];
      if (v < boundingBox[k]) boundingBox[k] = v;
      if (v > boundingBox[k + 3]) boundingBox[k + 3] = v;
    }
  }
  const dilator = dilateBorders ? new BorderDilator(boundingBox, boundingSphere) : null;

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
  const scratch = new GroupScratch(simpAttrs.stride);

  let depth = 0;

  // merge and simplify clusters until we can't merge anymore; a single cluster left is a group of
  // its own (nothing to lock but the caller's pins) while it is above `coarsestTriangles`
  while (clusters.length > 1 || (clusters.length === 1 && clusters[0].indices.length / 3 > coarsestTriangles)) {
    onProgress?.({ level: depth, clusterCount: clusters.length });

    const groupOffsets = partitionLevel(clusters, partitionSize, vertices.positions, vertices.remap, vertices.count);
    lockBoundary(vertices.locks, vertices.count, clusters, groupOffsets, vertices.remap, pinned);

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
      const localCount = scratch.compact(merged, vertices, simpAttrs);
      const localPositions = scratch.positions.subarray(0, localCount * 3);

      const sres = simplifyLocal(
        scratch,
        localCount,
        localPositions,
        simpAttrs,
        flags,
        targetSize,
        Math.floor(merged.length * simplifyThreshold),
        fallbackSloppy,
        sloppyErrorFactor,
        errorEdgeLimit,
        appearance,
        vertices.origin,
      );

      if (sres.indices.length > merged.length * simplifyThreshold) {
        // stuck: terminal group, abandon the merge
        bounds.error = Infinity;
        outputGroup(members, bounds, depth);
        continue;
      }

      const err = sres.error;
      bounds.error = Math.fround(Math.max(bounds.error * errorMergePrevious, err) + err * errorMergeAdditive);

      const refined = outputGroup(members, bounds, depth);

      // the group is out with its own clusters; what replaces it may now get its borders moved
      // (new vertices, which the local arrays then stand for)
      if (dilator) dilator.dilate(scratch.indices, sres.indices, scratch, localCount, vertices, bounds, seam);

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

  // the vertices: the input's own arrays, unless border dilation added some
  const finalCount = vertices.count;
  const finalPositions = vertices.positions.subarray(0, finalCount * 3);
  let outAttributes = attributes.slice();
  if (finalCount > vertexCount) {
    const origin = vertices.origin;
    outAttributes = attributes.map((a, index) => {
      const ctor = a.array.constructor as new (length: number) => TypedArray;
      const array = new ctor(finalCount * a.itemSize);
      array.set(a.array);
      if (index === 0) {
        array.set(finalPositions.subarray(vertexCount * 3), vertexCount * 3);
      } else {
        const s = a.itemSize;
        const from = a.array;
        for (let v = vertexCount; v < finalCount; v++) {
          const o = origin[v] * s;
          for (let k = 0; k < s; k++) array[v * s + k] = from[o + k];
        }
      }
      return { name: a.name, array, itemSize: a.itemSize, normalized: a.normalized };
    });
  }

  return {
    version: 1,
    vertexCount: finalCount,
    attributes: outAttributes,
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
    boundingSphere,
    boundingBox,
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
  vertexCount: number,
  clusters: Cluster[],
  groupOffsets: number[],
  remap: Uint32Array,
  pinned: Uint8Array | null,
): void {
  for (let i = 0; i < vertexCount; i++) locks[i] &= ~(LOCK | SCRATCH);

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
  for (let i = 0; i < vertexCount; i++) {
    locks[i] = (locks[remap[i]] & LOCK) | (locks[i] & PROTECT);
  }
  // (vertices added by border dilation are past the end of `pinned`: they were free to move)
  if (pinned) for (let i = 0; i < pinned.length; i++) locks[i] |= pinned[remap[i]];
}

// ---------------------------------------------------------------------------------------------
// attributes
// ---------------------------------------------------------------------------------------------

/**
 * Returns a test for "vertices `i` and `r` (INPUT vertex ids at the same position) differ in one of
 * the named attributes", which is what gets a vertex the meshopt "Protect" bit; null when none of
 * the names is present.
 */
function seamTest(attributes: MeshletAttribute[], names: string[]): ((i: number, r: number) => boolean) | null {
  const protectedAttrs = attributes.filter((a, i) => i > 0 && names.includes(a.name));
  if (protectedAttrs.length === 0) return null;

  return (i, r) => {
    for (const a of protectedAttrs) {
      const arr = a.array;
      const s = a.itemSize;
      for (let k = 0; k < s; k++) if (arr[i * s + k] !== arr[r * s + k]) return true;
    }
    return false;
  };
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

function buildSimplifyAttributes(
  attributes: MeshletAttribute[],
  vertexCount: number,
  weights: Record<string, number>,
  protect: string[],
): SimplifyAttributes {
  const used: MeshletAttribute[] = [];
  const w: number[] = [];
  const seamW: number[] = [];
  let stride = 0;
  let anyProtected = false;

  for (let i = 1; i < attributes.length; i++) {
    const a = attributes[i];
    const weight = weights[a.name] > 0 ? weights[a.name] : 0;
    const isProtected = protect.includes(a.name);
    if (weight === 0 && !isProtected) continue;
    // an attribute that is only protected would add floats without ever weighing in: leave it out when there is no room
    if (weight === 0 && stride + a.itemSize > MAX_ATTRIBUTE_FLOATS) continue;
    used.push(a);
    if (isProtected) anyProtected = true;
    for (let k = 0; k < a.itemSize; k++) {
      w.push(weight);
      seamW.push(isProtected ? Math.max(weight, 1) : weight);
    }
    stride += a.itemSize;
  }

  if (stride > MAX_ATTRIBUTE_FLOATS) {
    throw new Error(`buildMeshlets: weighted attributes use ${stride} floats per vertex, at most ${MAX_ATTRIBUTE_FLOATS} are supported`);
  }
  if (stride === 0) return { data: null, stride: 0, weights: [], seamWeights: null };

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

  return { data, stride, weights: w, seamWeights: anyProtected ? seamW : null };
}

// ---------------------------------------------------------------------------------------------
// vertices and border dilation
// ---------------------------------------------------------------------------------------------

/** The vertices of the hierarchy: the input's, followed by the ones border dilation adds. */
class VertexStore {
  /** Number of vertices. */
  count: number;
  /** xyz per vertex. The input's own array until a vertex is added, then a copy. */
  positions: Float32Array;
  /** Canonical vertex of each vertex (the first one at its position). */
  remap: Uint32Array;
  /** meshopt lock bits of each vertex. */
  locks: Uint8Array;
  /** Input vertex each vertex takes its attributes from. */
  origin: Uint32Array;
  /** A per-vertex map for whoever needs one for a moment; -1 everywhere between uses. */
  scratch: Int32Array;

  constructor(positions: Float32Array, remap: Uint32Array, locks: Uint8Array) {
    const n = positions.length / 3;
    this.count = n;
    this.positions = positions;
    this.remap = remap;
    this.locks = locks;
    this.origin = new Uint32Array(n);
    for (let i = 0; i < n; i++) this.origin[i] = i;
    this.scratch = new Int32Array(n).fill(-1);
  }

  /**
   * Adds a copy of vertex `source` at a new position and returns its id. `canonical` is the vertex
   * already added at that position, or -1 when this is the first one there.
   */
  add(source: number, x: number, y: number, z: number, canonical: number, lock: number): number {
    const id = this.count;
    if (id === this.remap.length) this.grow();
    this.count = id + 1;
    this.positions[id * 3] = x;
    this.positions[id * 3 + 1] = y;
    this.positions[id * 3 + 2] = z;
    this.remap[id] = canonical < 0 ? id : canonical;
    this.locks[id] = lock;
    this.origin[id] = this.origin[source];
    return id;
  }

  private grow(): void {
    const n = this.count;
    const capacity = Math.max(n + 1024, Math.ceil(n * 1.5));
    const positions = new Float32Array(capacity * 3);
    positions.set(this.positions.subarray(0, n * 3));
    this.positions = positions;
    const remap = new Uint32Array(capacity);
    remap.set(this.remap.subarray(0, n));
    this.remap = remap;
    const locks = new Uint8Array(capacity);
    locks.set(this.locks.subarray(0, n));
    this.locks = locks;
    const origin = new Uint32Array(capacity);
    origin.set(this.origin.subarray(0, n));
    this.origin = origin;
    this.scratch = new Int32Array(capacity).fill(-1);
  }
}

/**
 * Port of `dilateBorders` from meshoptimizer's clusterlod.h, for a hierarchy whose levels share one
 * vertex buffer.
 *
 * It compares the area of the triangles on OPEN BORDERS (an edge with no neighbour in the group,
 * other than a locked one on the group's own boundary) before and after a simplification step. If
 * the step lost some of that area (but not most of it), every border left is moved outwards, along
 * the plane of its triangles, far enough to win the area back: by `lost area / border length`.
 *
 * Unlike the reference, a border stops at the bounds of the input mesh. Left alone, the last few
 * cards of a canopy would each grow to the area of hundreds and reach far outside the tree, and the
 * bounds of the whole mesh, which every level is culled and measured by, would grow with them.
 *
 * The reference moves vertices in place and asks the caller to copy the vertices of every cluster.
 * Here the finer levels keep drawing the original vertices, so a vertex that moves is added as a new
 * vertex and the simplified triangles are pointed at it.
 */
class BorderDilator {
  /** Directed edges `a * positions + b` of the triangles last passed to `fillEdges`; -1 = empty. */
  private table = new Float64Array(0);
  private mask = 0;
  private positionCount = 0;
  /** Per local vertex: the id of its position within the group. */
  private pid = new Int32Array(0);
  /** Per position id. */
  private locked = new Uint8Array(0);
  /** Per position id: summed edge normals (xyz, each weighted by its edge length) and edge lengths. */
  private offsets = new Float64Array(0);
  /** Per position id: where it is. */
  private base = new Float32Array(0);
  /** Per position id: 0 = stays, 1 = moves to `target`. */
  private moved = new Uint8Array(0);
  private target = new Float32Array(0);
  /** Per position id: the first vertex added at `target`, or -1. */
  private canonical = new Int32Array(0);
  /** Per local vertex: already replaced by a new vertex. */
  private replaced = new Uint8Array(0);

  /**
   * @param box    Box no vertex is moved out of, `[minX, minY, minZ, maxX, maxY, maxZ]`.
   * @param sphere Sphere no vertex is moved out of, `[x, y, z, r]`.
   */
  constructor(
    private readonly box: Float32Array,
    private readonly sphere: Float32Array,
  ) {}

  /**
   * @param merged     The group before simplification (local vertex ids).
   * @param simplified The group after it (local vertex ids).
   * @param scratch    The group's local arrays. Positions and `localToGlobal` of moved vertices are
   *                   updated in place, so the clusters made from `simplified` use the new vertices.
   * @param bounds     LOD sphere the new clusters inherit; grown to hold the moved vertices.
   * @param seam       See `seamTest`.
   * @returns Number of vertices added.
   */
  dilate(
    merged: Uint32Array,
    simplified: Uint32Array,
    scratch: GroupScratch,
    localCount: number,
    vertices: VertexStore,
    bounds: Bound,
    seam: ((i: number, r: number) => boolean) | null,
  ): number {
    const positions = scratch.positions;
    const l2g = scratch.localToGlobal;

    // ---- positions of the group, numbered densely (vertices at one position share an id)
    if (this.pid.length < localCount) {
      this.pid = new Int32Array(localCount);
      this.locked = new Uint8Array(localCount);
      this.offsets = new Float64Array(localCount * 4);
      this.base = new Float32Array(localCount * 3);
      this.moved = new Uint8Array(localCount);
      this.target = new Float32Array(localCount * 3);
      this.canonical = new Int32Array(localCount);
      this.replaced = new Uint8Array(localCount);
    }
    const { pid, locked, base } = this;
    const remap = vertices.remap;
    const ids = vertices.scratch;
    let count = 0;
    for (let l = 0; l < localCount; l++) {
      const r = remap[l2g[l]];
      let p = ids[r];
      if (p < 0) {
        p = count++;
        ids[r] = p;
        // every vertex of a position carries the same lock bit
        locked[p] = scratch.locks[l] & LOCK;
        base[p * 3] = positions[l * 3];
        base[p * 3 + 1] = positions[l * 3 + 1];
        base[p * 3 + 2] = positions[l * 3 + 2];
      }
      pid[l] = p;
    }
    for (let l = 0; l < localCount; l++) ids[remap[l2g[l]]] = -1;
    this.positionCount = count;

    // ---- did the step lose border area?
    this.fillEdges(merged);
    const oldArea = this.borderArea(merged, positions);
    if (oldArea === 0) return 0;

    this.fillEdges(simplified);
    const newArea = this.borderArea(simplified, positions);
    // moderate shrinkage only: growing what is left of a step that removed nearly everything is not safe
    if (newArea >= oldArea || newArea < oldArea / 4) return 0;

    // What to win back: the border area that went, but never more than the group lost in all. The
    // two differ when triangles stop counting as border (pieces that fold onto each other, say),
    // and taking the border figure alone, as the reference does, then makes the group grow.
    const simplifiedArea = triangleArea(simplified, positions);
    const lost = Math.min(oldArea - newArea, triangleArea(merged, positions) - simplifiedArea);
    if (!(lost > 0)) return 0;

    // ---- edge normals of the borders that are left (the table holds the edges of `simplified`)
    const { offsets, moved, target, canonical, replaced } = this;
    offsets.fill(0, 0, count * 4);
    moved.fill(0, 0, count);
    canonical.fill(-1, 0, count);
    replaced.fill(0, 0, localCount);

    let perimeter = 0;
    for (let i = 0; i < simplified.length; i += 3) {
      for (let e = 0; e < 3; e++) {
        const la = simplified[i + e];
        const lb = simplified[i + (e === 2 ? 0 : e + 1)];
        const a = pid[la];
        const b = pid[lb];
        if ((locked[a] & locked[b]) !== 0 || this.hasEdge(b, a)) continue;

        const lc = simplified[i + ((e + 2) % 3)];
        const ax = positions[la * 3];
        const ay = positions[la * 3 + 1];
        const az = positions[la * 3 + 2];
        const ex = positions[lb * 3] - ax;
        const ey = positions[lb * 3 + 1] - ay;
        const ez = positions[lb * 3 + 2] - az;
        const el = Math.sqrt(ex * ex + ey * ey + ez * ez);

        // normal of the edge in the plane of its triangle, pointing away from the third vertex
        const cx = ax - positions[lc * 3];
        const cy = ay - positions[lc * 3 + 1];
        const cz = az - positions[lc * 3 + 2];
        const cp = (cx * ex + cy * ey + cz * ez) / (el > 0 ? el * el : 1);
        const nx = cx - ex * cp;
        const ny = cy - ey * cp;
        const nz = cz - ez * cp;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        // unit normal weighted by the edge length
        const ns = nl > 0 ? el / nl : 0;

        if (locked[a] === 0) {
          offsets[a * 4] += nx * ns;
          offsets[a * 4 + 1] += ny * ns;
          offsets[a * 4 + 2] += nz * ns;
          offsets[a * 4 + 3] += el;
        }
        if (locked[b] === 0) {
          offsets[b * 4] += nx * ns;
          offsets[b * 4 + 1] += ny * ns;
          offsets[b * 4 + 2] += nz * ns;
          offsets[b * 4 + 3] += el;
        }
        perimeter += el;
      }
    }
    if (!(perimeter > 0)) return 0;

    // ---- the direction each border position moves in, per unit of distance: the average normal,
    // divided by its length twice (once to normalise it, once more to lengthen the step into a miter
    // where two borders meet at a corner)
    let corners = 0;
    let movable = 0;
    for (let p = 0; p < count; p++) {
      const weight = offsets[p * 4 + 3];
      if (!(weight > 0)) {
        offsets[p * 4 + 3] = 0;
        continue;
      }
      const nx = offsets[p * 4] / weight;
      const ny = offsets[p * 4 + 1] / weight;
      const nz = offsets[p * 4 + 2] / weight;
      const nn = Math.max(0.15, nx * nx + ny * ny + nz * nz);
      offsets[p * 4] = nx / nn;
      offsets[p * 4 + 1] = ny / nn;
      offsets[p * 4 + 2] = nz / nn;
      offsets[p * 4 + 3] = 1;
      moved[p] = 1;
      movable++;
      // the average of two unit normals has the length cos(turn / 2)
      corners += Math.sqrt((1 - Math.min(1, nn)) / nn);
    }
    if (movable === 0) return 0;

    // ---- how far: until the area lost is won back
    // Moving the borders by d gains about `perimeter * d` along the edges plus `d * d * tan(turn / 2)`
    // at every corner. (The reference leaves the corners out; cards are mostly corner, and with it
    // they grow by a sixth too much with every level.) That is the first guess; pieces of irregular
    // shape need the gain measured, which a few secant steps do.
    const guess =
      corners > 1e-12 * perimeter
        ? (Math.sqrt(perimeter * perimeter + 4 * corners * lost) - perimeter) / (2 * corners)
        : lost / perimeter;
    let distance = guess;
    let previousDistance = 0;
    let previousGain = 0;
    for (let step = 0; step < 4; step++) {
      this.place(count, distance);
      // over all triangles: a border vertex that moves also stretches the triangles behind it
      const gain = this.movedArea(simplified) - simplifiedArea;
      if (Math.abs(gain - lost) <= 0.01 * lost) break;
      // no more to gain (the bounds of the mesh are in the way), or no progress
      if (!(gain > previousGain) || step === 3) break;
      const next = distance + ((lost - gain) * (distance - previousDistance)) / (gain - previousGain);
      previousDistance = distance;
      previousGain = gain;
      // stay near the first guess: a border that cannot gain the area is not pushed ever further
      distance = Math.min(Math.max(next, guess * 0.25), guess * 2);
      if (distance === previousDistance) break;
    }
    this.place(count, distance);

    // ---- the group's LOD sphere holds what moved
    for (let p = 0; p < count; p++) {
      if (moved[p] === 0) continue;
      const dx = target[p * 3] - bounds.x;
      const dy = target[p * 3 + 1] - bounds.y;
      const dz = target[p * 3 + 2] - bounds.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > bounds.r) bounds.r = Math.fround(d * (1 + 1e-6));
    }

    // ---- every vertex at a moved position becomes a vertex of its own
    let added = 0;
    for (let i = 0; i < simplified.length; i++) {
      const l = simplified[i];
      const p = pid[l];
      if (moved[p] === 0 || replaced[l] !== 0) continue;

      const first = canonical[p];
      const source = l2g[l];
      // a vertex that differs from the first one at its position in a protected attribute is a seam
      const lock = first >= 0 && seam !== null && seam(vertices.origin[source], vertices.origin[first]) ? PROTECT : 0;
      const id = vertices.add(source, target[p * 3], target[p * 3 + 1], target[p * 3 + 2], first, lock);
      if (first < 0) canonical[p] = id;
      l2g[l] = id;
      positions[l * 3] = target[p * 3];
      positions[l * 3 + 1] = target[p * 3 + 1];
      positions[l * 3 + 2] = target[p * 3 + 2];
      replaced[l] = 1;
      added++;
    }
    return added;
  }

  /** Writes to `target` where every moving position is when the borders move by `distance`. */
  private place(count: number, distance: number): void {
    const { base, offsets, moved, target, box, sphere } = this;
    for (let p = 0; p < count; p++) {
      if (moved[p] === 0) continue;
      let x = base[p * 3] + offsets[p * 4] * distance;
      let y = base[p * 3 + 1] + offsets[p * 4 + 1] * distance;
      let z = base[p * 3 + 2] + offsets[p * 4 + 2] * distance;

      // not past the bounds of the input: its box, then its sphere
      x = x < box[0] ? box[0] : x > box[3] ? box[3] : x;
      y = y < box[1] ? box[1] : y > box[4] ? box[4] : y;
      z = z < box[2] ? box[2] : z > box[5] ? box[5] : z;
      const sx = x - sphere[0];
      const sy = y - sphere[1];
      const sz = z - sphere[2];
      const sd = Math.sqrt(sx * sx + sy * sy + sz * sz);
      if (sd > sphere[3]) {
        // a hair inside, so that float32 rounding cannot put it back out
        const f = (sphere[3] / sd) * (1 - 1e-6);
        x = sphere[0] + sx * f;
        y = sphere[1] + sy * f;
        z = sphere[2] + sz * f;
      }
      target[p * 3] = x;
      target[p * 3 + 1] = y;
      target[p * 3 + 2] = z;
    }
  }

  private fillEdges(indices: Uint32Array): void {
    let size = 16;
    while (size < indices.length * 2) size *= 2;
    if (this.table.length < size) this.table = new Float64Array(size);
    const table = this.table;
    const mask = (this.mask = size - 1);
    table.fill(-1, 0, size);

    const pid = this.pid;
    const n = this.positionCount;
    for (let i = 0; i < indices.length; i += 3) {
      for (let e = 0; e < 3; e++) {
        const a = pid[indices[i + e]];
        const b = pid[indices[i + (e === 2 ? 0 : e + 1)]];
        const key = a * n + b;
        let h = (Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca6b)) & mask;
        while (table[h] !== -1 && table[h] !== key) h = (h + 1) & mask;
        table[h] = key;
      }
    }
  }

  private hasEdge(a: number, b: number): boolean {
    const table = this.table;
    const mask = this.mask;
    const key = a * this.positionCount + b;
    let h = (Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca6b)) & mask;
    while (table[h] !== -1) {
      if (table[h] === key) return true;
      h = (h + 1) & mask;
    }
    return false;
  }

  /** Area of the triangles `indices` with every moving position at its `target`. */
  private movedArea(indices: Uint32Array): number {
    const { pid, base, moved, target } = this;
    let area = 0;
    for (let i = 0; i < indices.length; i += 3) {
      const a = pid[indices[i]];
      const b = pid[indices[i + 1]];
      const c = pid[indices[i + 2]];
      const pa = moved[a] !== 0 ? target : base;
      const pb = moved[b] !== 0 ? target : base;
      const pc = moved[c] !== 0 ? target : base;
      const ax = pa[a * 3];
      const ay = pa[a * 3 + 1];
      const az = pa[a * 3 + 2];
      const ux = pb[b * 3] - ax;
      const uy = pb[b * 3 + 1] - ay;
      const uz = pb[b * 3 + 2] - az;
      const vx = pc[c * 3] - ax;
      const vy = pc[c * 3 + 1] - ay;
      const vz = pc[c * 3 + 2] - az;
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      area += Math.sqrt(nx * nx + ny * ny + nz * nz) * 0.5;
    }
    return area;
  }

  /** Area of the triangles with at least one open, unlocked edge. The table must hold the edges of `indices`. */
  private borderArea(indices: Uint32Array, positions: Float32Array): number {
    const { pid, locked } = this;
    let area = 0;
    for (let i = 0; i < indices.length; i += 3) {
      const la = indices[i];
      const lb = indices[i + 1];
      const lc = indices[i + 2];
      const a = pid[la];
      const b = pid[lb];
      const c = pid[lc];
      const border =
        ((locked[a] & locked[b]) === 0 && !this.hasEdge(b, a)) ||
        ((locked[b] & locked[c]) === 0 && !this.hasEdge(c, b)) ||
        ((locked[c] & locked[a]) === 0 && !this.hasEdge(a, c));
      if (!border) continue;

      const ax = positions[la * 3];
      const ay = positions[la * 3 + 1];
      const az = positions[la * 3 + 2];
      const ux = positions[lb * 3] - ax;
      const uy = positions[lb * 3 + 1] - ay;
      const uz = positions[lb * 3 + 2] - az;
      const vx = positions[lc * 3] - ax;
      const vy = positions[lc * 3 + 1] - ay;
      const vz = positions[lc * 3 + 2] - az;
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      area += Math.sqrt(nx * nx + ny * ny + nz * nz) * 0.5;
    }
    return area;
  }
}

// ---------------------------------------------------------------------------------------------
// per-group local compaction and simplification
// ---------------------------------------------------------------------------------------------

/** Growable scratch buffers for one group's local data. */
class GroupScratch {
  positions = new Float32Array(0);
  attributes = new Float32Array(0);
  locks = new Uint8Array(0);
  private otherLocks = new Uint8Array(0);
  localToGlobal = new Uint32Array(0);
  localIndices = new Uint32Array(0);
  private localCountIndices = 0;
  private readonly attrStride: number;

  constructor(attrStride: number) {
    this.attrStride = attrStride;
  }

  /** Compacts the vertices used by `merged` (global ids) into local arrays; returns the local vertex count. */
  compact(merged: Uint32Array, vertices: VertexStore, attrs: SimplifyAttributes): number {
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
    const { positions, locks, origin, scratch: globalToLocal } = vertices;

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
        if (gattr) {
          // attributes are those of the input vertex a vertex was made from
          const o = origin[g] * stride;
          for (let k = 0; k < stride; k++) lattr[l * stride + k] = gattr[o + k];
        }
      }
      localIndices[i] = l;
    }
    this.localCountIndices = merged.length;
    // the map is only needed while compacting: leave it clean for the next user
    for (let l = 0; l < count; l++) globalToLocal[l2g[l]] = -1;
    return count;
  }

  /** Local index list of the last compacted group. */
  get indices(): Uint32Array {
    return this.localIndices.subarray(0, this.localCountIndices);
  }

  /** A second lock array of `count` entries, for a simplification with other locks than `locks`. */
  unprotected(count: number): Uint8Array {
    if (this.otherLocks.length < count) this.otherLocks = new Uint8Array(Math.max(count, this.locks.length));
    return this.otherLocks.subarray(0, count);
  }
}

/** Total area of a triangle list. */
function triangleArea(indices: Uint32Array, positions: Float32Array): number {
  let area = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    area += Math.sqrt(nx * nx + ny * ny + nz * nz) * 0.5;
  }
  return area;
}

interface SimplifyResult {
  /** LOCAL vertex ids. */
  indices: Uint32Array;
  /** Absolute error in mesh units. */
  error: number;
}

/** A group is as good as simplified to its target when it is this close to it. */
const NEAR_TARGET = 1.1;
/** A step that goes further is worth an error this many times that of the most careful one. */
const WORTH = 2;

/** Of two ways of simplifying a group, the one to keep (`stuckSize`: above it a result got nowhere). */
function better(a: SimplifyResult, b: SimplifyResult, stuckSize: number): SimplifyResult {
  const small = a.indices.length <= b.indices.length ? a : b;
  const large = small === a ? b : a;
  // the one that goes further, unless only the other gets anywhere or it costs out of proportion
  if (large.indices.length > stuckSize) return small;
  return small.error <= large.error * WORTH ? small : large;
}

/**
 * Simplifies a group towards `targetSize` indices, trying ever less careful ways until one gets
 * there, and keeping the one that goes furthest for an error in proportion (`better`):
 *
 *  1. every seam of the mesh is kept (hard edges, uv seams, whatever the index buffer separates);
 *  2. only the seams of the protected attributes are kept;
 *  3. no seam is kept (`fallbackUnprotected`);
 *  4. topology is not kept either (`fallbackSloppy`), when nothing else got anywhere.
 *
 * The error of each is the simplifier's or, when there is something to go by (`appearance`), the
 * size of what it changes to the eye, whichever is larger.
 */
function simplifyLocal(
  scratch: GroupScratch,
  localCount: number,
  localPositions: Float32Array,
  attrs: SimplifyAttributes,
  flags: SimplifierFlags[],
  targetSize: number,
  stuckSize: number,
  fallbackSloppy: boolean,
  sloppyErrorFactor: number,
  errorEdgeLimit: number,
  appearance: AppearanceCheck | null,
  origin: Uint32Array,
): SimplifyResult {
  const localIndices = scratch.indices;
  const localLocks = scratch.locks.subarray(0, localCount);
  const localAttrs = scratch.attributes.subarray(0, localCount * attrs.stride);

  const localToGlobal = scratch.localToGlobal;
  const seen = appearance !== null && appearance.prepare(localIndices, localPositions, localCount, (l) => origin[localToGlobal[l]]);
  const judged = (indices: Uint32Array, error: number): SimplifyResult => ({
    indices,
    error: seen ? Math.max(error, appearance!.error(indices, error)) : error,
  });

  const strict = flags.filter((f) => f !== "Permissive");
  let [found, foundError] = MeshoptSimplifier.simplifyWithAttributes(
    localIndices,
    localPositions,
    3,
    localAttrs,
    attrs.stride,
    attrs.weights,
    localLocks,
    targetSize,
    FLT_MAX,
    strict,
  );
  let best = judged(found, foundError);

  if (best.indices.length > targetSize * NEAR_TARGET) {
    [found, foundError] = MeshoptSimplifier.simplifyWithAttributes(
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
    best = better(best, judged(found, foundError), stuckSize);

    // across the protected seams too, at the price of the attribute error that costs
    if (best.indices.length > targetSize * NEAR_TARGET && attrs.seamWeights !== null) {
      let anyProtected = false;
      const open = scratch.unprotected(localCount);
      for (let l = 0; l < localCount; l++) {
        if ((localLocks[l] & PROTECT) !== 0) anyProtected = true;
        open[l] = localLocks[l] & ~PROTECT;
      }
      if (anyProtected) {
        [found, foundError] = MeshoptSimplifier.simplifyWithAttributes(
          localIndices,
          localPositions,
          3,
          localAttrs,
          attrs.stride,
          attrs.seamWeights,
          open,
          targetSize,
          FLT_MAX,
          flags,
        );
        best = better(best, judged(found, foundError), stuckSize);
      }
    }

    // sloppy returns a relative error; scale to absolute
    if (best.indices.length > stuckSize && fallbackSloppy) {
      const [sloppy, relError] = MeshoptSimplifier.simplifySloppy(localIndices, localPositions, 3, localLocks, targetSize, FLT_MAX);
      best = better(best, judged(sloppy, relError * MeshoptSimplifier.getScale(localPositions, 3) * sloppyErrorFactor), stuckSize);
    }
  }
  let result = best.indices;
  let error = best.error;

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
