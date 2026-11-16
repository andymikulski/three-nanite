import type { MeshletData } from "../types.js";
import { ClusterSelector } from "./select.js";

/**
 * The WHOLE-MESH levels of detail of a hierarchy.
 *
 * A cut of the DAG normally mixes detail levels within one object, which is what makes it worth
 * having for an object the camera is close to. Seen from a few radii away the same object needs
 * one detail level throughout, and then its cut is one of a small, fixed family: the
 * view-independent cuts `selectByError(e)`, which only change when `e` crosses a group's error.
 * Those are the levels listed here, finest (the original mesh, error 0) first.
 *
 * Because the family is small and shared by every instance, instances that are on the same level
 * can be drawn together with ordinary instancing, and finding the level of an instance is a
 * binary search instead of a walk of the DAG. `MeshletMesh` does both.
 *
 * Each level is stored as INDEX RUNS: maximal contiguous ranges of `data.indices`. Coarse levels
 * are usually a single run.
 */
export interface LodLevels {
  /** Number of levels. */
  count: number;
  /**
   * Error of each level, ascending, in mesh space units; `errors[0]` is 0. Level `j` is the cut
   * `selectByError(errors[j])` and may be drawn wherever an error of `errors[j]` is acceptable.
   */
  errors: Float64Array;
  /** Triangles and clusters of each level. */
  triangles: Uint32Array;
  clusters: Uint32Array;
  /** Deepest DAG level among the clusters of each level (for colouring). */
  depth: Uint16Array;
  /** Runs of level `j`: `runStart/runCount[runOffset[j] .. runOffset[j + 1])`, in units of indices. */
  runOffset: Uint32Array;
  runStart: Uint32Array;
  runCount: Uint32Array;
  /**
   * True when every distinct group error is a level, so every view-independent cut is listed.
   * False when the list was thinned to `maxLevels` (an error between two kept levels then uses the
   * finer one).
   */
  exact: boolean;
}

/** Options of {@link computeLodLevels}. */
export interface LodLevelOptions {
  /** Upper bound on the number of levels. Default 48. */
  maxLevels?: number;
}

/**
 * Lists the whole-mesh levels of a hierarchy.
 *
 * Small meshes get one level per distinct group error. Large ones, which have hundreds of
 * distinct errors, are thinned so that consecutive levels are at least a constant factor apart in
 * error (the factor grows until `maxLevels` fit; it starts at 2^(1/4), about a sixth of a
 * halving of triangles per level).
 */
export function computeLodLevels(data: MeshletData, options: LodLevelOptions = {}): LodLevels {
  const maxLevels = Math.max(3, Math.floor(options.maxLevels ?? 48));

  const finite: number[] = [];
  for (let g = 0; g < data.groupCount; g++) {
    const e = data.groupError[g];
    if (e > 0 && Number.isFinite(e)) finite.push(e);
  }
  finite.sort((a, b) => a - b);
  const distinct: number[] = [0];
  for (const e of finite) if (e !== distinct[distinct.length - 1]) distinct.push(e);

  let chosen = distinct;
  let exact = true;
  if (distinct.length > maxLevels) {
    exact = false;
    const last = distinct[distinct.length - 1];
    for (let step = 2 ** 0.25; ; step *= 2 ** 0.125) {
      chosen = [0];
      let previous = 0;
      for (let i = 1; i < distinct.length - 1; i++) {
        const e = distinct[i];
        // the first positive error is always a level; after it, levels are `step` apart
        if (previous === 0 || e >= previous * step) {
          chosen.push(e);
          previous = e;
        }
      }
      // the coarsest cut is always available
      chosen.push(last);
      if (chosen.length <= maxLevels) break;
    }
  }

  const count = chosen.length;
  const errors = Float64Array.from(chosen);
  const triangles = new Uint32Array(count);
  const clusters = new Uint32Array(count);
  const depth = new Uint16Array(count);
  const runOffset = new Uint32Array(count + 1);
  const starts: number[] = [];
  const counts: number[] = [];

  const selector = new ClusterSelector(data);
  const cut = new Uint32Array(data.clusterCount);
  const { clusterIndexStart, clusterIndexCount, clusterGroup, groupDepth } = data;

  for (let j = 0; j < count; j++) {
    const n = selector.selectByError(cut, errors[j]);
    // clusters are stored in order and without gaps: consecutive ids are contiguous indices
    const ids = cut.subarray(0, n).sort();
    let indexTotal = 0;
    let deepest = 0;
    let runStart = -1;
    let runEnd = -1;
    for (let i = 0; i < n; i++) {
      const c = ids[i];
      const start = clusterIndexStart[c];
      const size = clusterIndexCount[c];
      indexTotal += size;
      const d = groupDepth[clusterGroup[c]];
      if (d > deepest) deepest = d;
      if (start === runEnd) {
        runEnd += size;
      } else {
        if (runEnd > runStart) {
          starts.push(runStart);
          counts.push(runEnd - runStart);
        }
        runStart = start;
        runEnd = start + size;
      }
    }
    if (runEnd > runStart) {
      starts.push(runStart);
      counts.push(runEnd - runStart);
    }
    triangles[j] = indexTotal / 3;
    clusters[j] = n;
    depth[j] = deepest;
    runOffset[j + 1] = starts.length;
  }

  return {
    count,
    errors,
    triangles,
    clusters,
    depth,
    runOffset,
    runStart: Uint32Array.from(starts),
    runCount: Uint32Array.from(counts),
    exact,
  };
}

/** The coarsest level whose error is at most `error` (mesh space units, `>= 0`). */
export function levelForError(levels: LodLevels, error: number): number {
  const errors = levels.errors;
  let lo = 0;
  let hi = levels.count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (errors[mid] <= error) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * The level a view-dependent cut is equal to, or -1 when it mixes levels.
 *
 * @param within `ClusterSelector.lastWithin` of the `select` that produced the cut.
 * @param over   `ClusterSelector.lastOver` of the same call.
 */
export function levelOfCut(levels: LodLevels, within: number, over: number): number {
  if (!(within < over)) return -1;
  // any level with an error in [within, over) is this very cut
  const j = levelForError(levels, within);
  if (levels.errors[j] === within) return j;
  return j + 1 < levels.count && levels.errors[j + 1] < over ? j + 1 : -1;
}

/**
 * Writes the triangle list of level `level` to `out` (which must hold `levels.triangles[level] * 3`
 * entries) and returns the number of indices written.
 */
export function copyLevelIndices(
  data: MeshletData,
  levels: LodLevels,
  level: number,
  out: Uint32Array | Uint16Array,
): number {
  const source = data.indices;
  let w = 0;
  for (let r = levels.runOffset[level]; r < levels.runOffset[level + 1]; r++) {
    const start = levels.runStart[r];
    const size = levels.runCount[r];
    for (let i = 0; i < size; i++) out[w++] = source[start + i];
  }
  return w;
}
