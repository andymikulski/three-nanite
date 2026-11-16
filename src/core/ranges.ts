import type { MeshletData } from "../types.js";

/**
 * The units a cut is DRAWN in.
 *
 * A cluster is drawn iff its own group is over budget and the group it was simplified from is not
 * (see `types.ts`), so two clusters with the same `clusterGroup` and the same `clusterRefined` are
 * always drawn together or not at all. The builder stores such clusters next to each other, and
 * cluster index ranges are contiguous, so every run of them is ONE contiguous index range: a draw
 * range. A cut is therefore a set of draw ranges, typically 3 to 16 times fewer than its clusters
 * (the whole finest level of a group is a single range).
 *
 * LOD selection is unchanged and still works in clusters ({@link ClusterSelector}); draw ranges
 * only decide how many draws the result costs. Use {@link rangesOfCut} to convert.
 */
export interface DrawRanges {
  /** Number of draw ranges. */
  count: number;
  /** Draw range of each cluster (length `clusterCount`). */
  ofCluster: Uint32Array;
  /** First cluster of each range; its clusters are `firstCluster[r] .. firstCluster[r] + clusterCount[r])`. */
  firstCluster: Uint32Array;
  clusterCount: Uint32Array;
  /** The range in `data.indices`. */
  indexStart: Uint32Array;
  indexCount: Uint32Array;
  /** Sphere around the member clusters' tight spheres, `[x, y, z, r]` per range. For culling only. */
  bounds: Float32Array;
}

/**
 * Computes the draw ranges of a hierarchy. With `merge = false` every cluster is its own range
 * (one draw per cluster, as needed to colour or cull clusters individually).
 */
export function computeDrawRanges(data: MeshletData, merge = true): DrawRanges {
  const n = data.clusterCount;
  const { clusterGroup, clusterRefined, clusterIndexStart, clusterIndexCount, clusterBounds } = data;
  const ofCluster = new Uint32Array(n);

  let count = 0;
  for (let c = 0; c < n; c++) {
    const joins =
      merge &&
      c > 0 &&
      clusterGroup[c] === clusterGroup[c - 1] &&
      clusterRefined[c] === clusterRefined[c - 1] &&
      clusterIndexStart[c] === clusterIndexStart[c - 1] + clusterIndexCount[c - 1];
    if (!joins) count++;
    ofCluster[c] = count - 1;
  }

  const firstCluster = new Uint32Array(count);
  const clusterCount = new Uint32Array(count);
  const indexStart = new Uint32Array(count);
  const indexCount = new Uint32Array(count);
  const bounds = new Float32Array(count * 4);

  for (let c = 0; c < n; c++) {
    const r = ofCluster[c];
    const o = c * 4;
    if (clusterCount[r] === 0) {
      firstCluster[r] = c;
      indexStart[r] = clusterIndexStart[c];
      bounds[r * 4] = clusterBounds[o];
      bounds[r * 4 + 1] = clusterBounds[o + 1];
      bounds[r * 4 + 2] = clusterBounds[o + 2];
      bounds[r * 4 + 3] = clusterBounds[o + 3];
    } else {
      enclose(bounds, r * 4, clusterBounds[o], clusterBounds[o + 1], clusterBounds[o + 2], clusterBounds[o + 3]);
    }
    clusterCount[r]++;
    indexCount[r] += clusterIndexCount[c];
  }

  return { count, ofCluster, firstCluster, clusterCount, indexStart, indexCount, bounds };
}

/** Grows the sphere at `out[o..o+4)` to enclose the sphere `(x, y, z, r)`. */
function enclose(out: Float32Array, o: number, x: number, y: number, z: number, r: number): void {
  const dx = x - out[o];
  const dy = y - out[o + 1];
  const dz = z - out[o + 2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const r0 = out[o + 3];
  if (d + r <= r0) return;
  if (d + r0 <= r) {
    out[o] = x;
    out[o + 1] = y;
    out[o + 2] = z;
    out[o + 3] = r;
    return;
  }
  const k = (d + r - r0) / (2 * d);
  out[o] += dx * k;
  out[o + 1] += dy * k;
  out[o + 2] += dz * k;
  // a hair of slack for the float32 rounding of the centre
  out[o + 3] = ((d + r + r0) / 2) * (1 + 1e-6);
}

/**
 * Converts a cut from cluster ids to draw range ids, in place.
 *
 * @param cut   Cluster ids as written by `ClusterSelector.select`; receives the range ids.
 * @param count Number of cluster ids in `cut`.
 * @returns Number of range ids written.
 */
export function rangesOfCut(ranges: DrawRanges, cut: Uint32Array, count: number): number {
  const { ofCluster, firstCluster } = ranges;
  let n = 0;
  for (let i = 0; i < count; i++) {
    const c = cut[i];
    const r = ofCluster[c];
    // all clusters of a range are in the cut together; its first cluster stands for it
    if (firstCluster[r] === c) cut[n++] = r;
  }
  return n;
}
