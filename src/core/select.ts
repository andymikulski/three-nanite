import type { MeshletData } from "../types.js";

/**
 * Converts a screen-space error budget into the scale factor `k` used by
 * {@link ClusterSelector.select}.
 *
 * A group's simplified form is "over budget" (must be refined) when
 *
 *     perspective:   groupError > k * max(distance - radius, distance to the mesh sphere, znear)
 *     orthographic:  groupError > k
 *
 * @param errorThresholdPx  Allowed geometric error, in pixels.
 * @param viewportHeightPx  Height of the render target, in pixels.
 * @param projectionScaleY  `camera.projectionMatrix.elements[5]`
 *                          (cot(fovY / 2) for perspective, 2 / (top - bottom) * zoom for orthographic).
 * @param worldScale        Uniform world scale of the mesh instance. Only matters for
 *                          orthographic cameras; perspective error is scale invariant
 *                          because the camera position is expressed in mesh space.
 */
export function lodScale(
  errorThresholdPx: number,
  viewportHeightPx: number,
  projectionScaleY: number,
  worldScale = 1,
  orthographic = false,
): number {
  const k = errorThresholdPx / (0.5 * viewportHeightPx * projectionScaleY);
  return orthographic ? k / worldScale : k;
}

/**
 * Selects the crack-free set of clusters (a "cut" of the DAG) for a view.
 *
 * The traversal starts at the terminal groups and only descends into groups
 * whose simplified form is over budget, so the cost is proportional to the
 * size of the result, not to the size of the mesh.
 *
 * One selector can be reused for any number of `select` calls (it keeps
 * scratch memory sized to the mesh). It is NOT re-entrant.
 */
export class ClusterSelector {
  readonly data: MeshletData;

  /**
   * After a `select`: the smallest error among the groups found over budget (`Infinity` if none)
   * and the largest among those found within budget (0 if none). When `lastWithin < lastOver` the
   * cut is exactly the view-independent cut of any error in `[lastWithin, lastOver)`: the whole
   * mesh at one level of detail (see `LodLevels`).
   */
  lastOver = Infinity;
  lastWithin = 0;

  /** Per group: `gen` => visited & not over, `gen + 1` => visited & over. */
  private readonly state: Uint32Array;
  private readonly stack: Uint32Array;
  private gen = 0;

  constructor(data: MeshletData) {
    this.data = data;
    this.state = new Uint32Array(data.groupCount);
    this.stack = new Uint32Array(data.groupCount);
  }

  /**
   * View-dependent selection.
   *
   * The distance to a group is never taken smaller than the distance to the bounding sphere of
   * the mesh. Group spheres only grow up the DAG (each encloses the spheres below it) and end up
   * well outside the mesh, which would otherwise refine a mesh seen from outside as if the camera
   * were much nearer. All geometry is inside the mesh sphere, so the bound stays valid, and
   * clamping every group by the same value keeps the decisions monotonic (crack-free).
   *
   * @param out   Receives cluster ids. Must hold at least `data.clusterCount` entries.
   * @param px,py,pz  Camera position in MESH space (ignored when `orthographic`).
   * @param k     See {@link lodScale}.
   * @param znear Camera near distance in mesh space units (`camera.near / worldScale`).
   * @returns Number of cluster ids written to `out`.
   */
  select(
    out: Uint32Array,
    px: number,
    py: number,
    pz: number,
    k: number,
    znear: number,
    orthographic = false,
  ): number {
    const d = this.data;
    const state = this.state;
    const stack = this.stack;
    const groupError = d.groupError;
    const groupBounds = d.groupBounds;
    const groupClusterStart = d.groupClusterStart;
    const groupClusterCount = d.groupClusterCount;
    const clusterRefined = d.clusterRefined;
    const roots = d.rootGroups;

    // two values per call: gen = "not over", gen + 1 = "over"
    if (this.gen >= 0xfffffff0) {
      state.fill(0);
      this.gen = 0;
    }
    const gen = (this.gen += 2);

    let sp = 0;
    for (let i = 0; i < roots.length; i++) {
      const g = roots[i];
      state[g] = gen + 1;
      stack[sp++] = g;
    }

    // no group is nearer than the mesh itself
    let floor = znear;
    if (!orthographic) {
      const s = d.boundingSphere;
      const dx = s[0] - px;
      const dy = s[1] - py;
      const dz = s[2] - pz;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - s[3];
      if (dist > floor) floor = dist;
    }

    let minOver = Infinity;
    let maxWithin = 0;
    let n = 0;
    while (sp > 0) {
      const g = stack[--sp];
      const start = groupClusterStart[g];
      const end = start + groupClusterCount[g];

      for (let c = start; c < end; c++) {
        const r = clusterRefined[c];
        if (r < 0) {
          out[n++] = c;
          continue;
        }

        let s = state[r];
        if (s !== gen && s !== gen + 1) {
          const err = groupError[r];
          let over: boolean;
          if (orthographic) {
            over = err > k;
          } else {
            const o = r * 4;
            const dx = groupBounds[o] - px;
            const dy = groupBounds[o + 1] - py;
            const dz = groupBounds[o + 2] - pz;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) - groupBounds[o + 3];
            over = err > k * (dist > floor ? dist : floor);
          }
          if (over) {
            s = gen + 1;
            stack[sp++] = r;
            if (err < minOver) minOver = err;
          } else {
            s = gen;
            if (err > maxWithin) maxWithin = err;
          }
          state[r] = s;
        }

        // the finer group is within budget => this (coarser) cluster is the one to draw
        if (s === gen) out[n++] = c;
      }
    }

    this.lastOver = minOver;
    this.lastWithin = maxWithin;
    return n;
  }

  /**
   * View-independent selection: the coarsest cut whose accumulated error is
   * at most `maxError` (mesh space units). `0` returns the original mesh,
   * `Infinity` the coarsest available.
   */
  selectByError(out: Uint32Array, maxError: number): number {
    return this.select(out, 0, 0, 0, maxError, 0, true);
  }
}

/** Total number of triangles in a list of clusters. */
export function countTriangles(data: MeshletData, clusters: ArrayLike<number>, count = clusters.length): number {
  let indices = 0;
  for (let i = 0; i < count; i++) indices += data.clusterIndexCount[clusters[i]];
  return indices / 3;
}

/**
 * Suggested error thresholds for {@link createLodGeometry}, one per DAG
 * level, coarsest last: for each depth the largest finite `groupError` among
 * its groups. Depths without a finite error are skipped and the result is
 * strictly increasing. Passing `0` yields the original mesh and `Infinity`
 * the coarsest cut.
 */
export function getLodErrors(data: MeshletData): number[] {
  const maxPerDepth = new Float64Array(data.levelCount).fill(-1);
  for (let g = 0; g < data.groupCount; g++) {
    const e = data.groupError[g];
    const d = data.groupDepth[g];
    if (Number.isFinite(e) && d < data.levelCount && e > maxPerDepth[d]) maxPerDepth[d] = e;
  }

  const errors: number[] = [];
  for (let d = 0; d < data.levelCount; d++) {
    const e = maxPerDepth[d];
    if (e < 0) continue;
    if (errors.length === 0 || e > errors[errors.length - 1]) errors.push(e);
  }
  return errors;
}
