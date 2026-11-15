import type { MeshletData } from "../types.js";

/**
 * Converts a screen-space error budget into the scale factor `k` used by
 * {@link ClusterSelector.select}.
 *
 * A group's simplified form is "over budget" (must be refined) when
 *
 *     perspective:   groupError > k * max(distance - radius, znear)
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
            over = err > k * (dist > znear ? dist : znear);
          }
          s = over ? gen + 1 : gen;
          state[r] = s;
          if (over) stack[sp++] = r;
        }

        // the finer group is within budget => this (coarser) cluster is the one to draw
        if (s === gen) out[n++] = c;
      }
    }

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
