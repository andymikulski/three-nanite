/**
 * Port of meshoptimizer's `meshopt_partitionClusters` (src/partition.cpp).
 *
 * Groups clusters into partitions of roughly `targetPartitionSize` clusters,
 * greedily merging clusters that share the most vertices (and, when positions
 * are given, that are spatially close). The JS/wasm build of meshoptimizer
 * does not expose this function.
 *
 * Based on: Takio Kurita. An efficient agglomerative clustering algorithm
 * using a heap. 1991.
 */

/** To avoid excessive recursion on malformed inputs, switch to bisection after this depth. */
const MERGE_DEPTH_CUTOFF = 40;

export interface PartitionResult {
  /** Partition id per cluster, in `[0, count)`. */
  partition: Uint32Array;
  count: number;
}

/**
 * @param clusterIndices      Position-remapped vertex ids, concatenated per cluster.
 * @param clusterIndexCounts  Number of entries in `clusterIndices` per cluster.
 * @param vertexPositions     Tightly packed xyz floats, or null for connectivity-only partitioning.
 * @param vertexCount         Number of vertices (ids in `clusterIndices` are `< vertexCount`).
 * @param targetPartitionSize Target number of clusters per partition (a partition may reach target + target / 3).
 */
export function partitionClusters(
  clusterIndices: Uint32Array,
  clusterIndexCounts: Uint32Array,
  vertexPositions: Float32Array | null,
  vertexCount: number,
  targetPartitionSize: number,
): PartitionResult {
  if (!(targetPartitionSize > 0)) throw new Error("partitionClusters: targetPartitionSize must be positive");

  const clusterCount = clusterIndexCounts.length;
  const totalIndexCount = clusterIndices.length;
  const maxPartitionSize = targetPartitionSize + Math.floor(targetPartitionSize / 3);

  const partition = new Uint32Array(clusterCount);
  if (clusterCount === 0) return { partition, count: 0 };

  // ---- filter duplicate vertices within each cluster ----------------------
  const used = new Uint8Array(vertexCount);
  const filtered = new Uint32Array(totalIndexCount);
  const offsets = new Uint32Array(clusterCount + 1);

  let write = 0;
  let start = 0;
  for (let i = 0; i < clusterCount; i++) {
    offsets[i] = write;
    const n = clusterIndexCounts[i];
    for (let j = 0; j < n; j++) {
      const v = clusterIndices[start + j];
      if (v >= vertexCount) throw new Error("partitionClusters: vertex id out of range");
      filtered[write] = v;
      write += 1 - used[v];
      used[v] = 1;
    }
    for (let j = offsets[i]; j < write; j++) used[filtered[j]] = 0;
    start += n;
  }
  if (start !== totalIndexCount) throw new Error("partitionClusters: index counts do not sum to clusterIndices.length");
  offsets[clusterCount] = write;

  // ---- cluster adjacency with shared vertex counts ------------------------
  const refStart = new Uint32Array(vertexCount + 1);
  for (let i = 0; i < clusterCount; i++) {
    for (let j = offsets[i]; j < offsets[i + 1]; j++) refStart[filtered[j]]++;
  }

  // worst-case adjacency size: every other cluster can be adjacent at most once
  let totalAdjacency = 0;
  for (let i = 0; i < clusterCount; i++) {
    let count = 0;
    for (let j = offsets[i]; j < offsets[i + 1]; j++) count += refStart[filtered[j]] - 1;
    totalAdjacency += count < clusterCount - 1 ? count : clusterCount - 1;
  }

  const adjOffsets = new Uint32Array(clusterCount + 1);
  const adjClusters = new Uint32Array(totalAdjacency);
  const adjShared = new Uint32Array(totalAdjacency);

  let totalRefs = 0;
  for (let i = 0; i < vertexCount; i++) {
    const count = refStart[i];
    refStart[i] = totalRefs;
    totalRefs += count;
  }

  const refData = new Uint32Array(totalRefs);
  for (let i = 0; i < clusterCount; i++) {
    for (let j = offsets[i]; j < offsets[i + 1]; j++) refData[refStart[filtered[j]]++] = i;
  }

  // refStart now holds the end of each vertex's data; shift to get the start
  refStart.copyWithin(1, 0, vertexCount);
  refStart[0] = 0;

  const slots = new Uint32Array(clusterCount).fill(0xffffffff);

  for (let i = 0; i < clusterCount; i++) {
    const base = adjOffsets[i];
    let count = 0;

    for (let j = offsets[i]; j < offsets[i + 1]; j++) {
      const v = filtered[j];
      const refEnd = refStart[v + 1];
      for (let k = refStart[v]; k < refEnd; k++) {
        const c = refData[k];
        if (c === i) continue;

        // slots is never cleared across clusters; adjClusters[slot] confirms ownership
        const slot = slots[c];
        if (slot < count && adjClusters[base + slot] === c) {
          adjShared[base + slot]++;
        } else {
          slots[c] = count;
          adjClusters[base + count] = c;
          adjShared[base + count] = 1;
          count++;
        }
      }
    }

    adjOffsets[i + 1] = adjOffsets[i] + count;
  }

  // ---- group state (structure of arrays) ----------------------------------
  const gGroup = new Int32Array(clusterCount); // current group id of each cluster, or -1 while being processed
  const gNext = new Int32Array(clusterCount); // linked list of clusters in a group
  const gSize = new Uint32Array(clusterCount); // 0 unless root
  const gVertices = new Uint32Array(clusterCount);
  const gCenter = new Float32Array(clusterCount * 3);
  const gRadius = new Float32Array(clusterCount);
  const sharedAcc = new Uint32Array(clusterCount);

  // binary min-heap keyed by `order`
  const heapId = new Uint32Array(clusterCount);
  const heapOrder = new Int32Array(clusterCount);

  const heapPush = (size: number, id: number, order: number): void => {
    let i = size;
    heapId[i] = id;
    heapOrder[i] = order;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!(heapOrder[i] < heapOrder[p])) break;
      const ti = heapId[i];
      const to = heapOrder[i];
      heapId[i] = heapId[p];
      heapOrder[i] = heapOrder[p];
      heapId[p] = ti;
      heapOrder[p] = to;
      i = p;
    }
  };

  let popId = 0;
  const heapPop = (size: number): void => {
    popId = heapId[0];
    size--;
    heapId[0] = heapId[size];
    heapOrder[0] = heapOrder[size];

    let i = 0;
    while (i * 2 + 1 < size) {
      let j = i * 2 + 1;
      if (j + 1 < size && heapOrder[j + 1] < heapOrder[j]) j++;
      if (heapOrder[j] >= heapOrder[i]) break;
      const ti = heapId[i];
      const to = heapOrder[i];
      heapId[i] = heapId[j];
      heapOrder[i] = heapOrder[j];
      heapId[j] = ti;
      heapOrder[j] = to;
      i = j;
    }
  };

  const mergeBounds = (target: number, source: number): void => {
    const r1 = gRadius[target];
    const r2 = gRadius[source];
    const dx = gCenter[source * 3] - gCenter[target * 3];
    const dy = gCenter[source * 3 + 1] - gCenter[target * 3 + 1];
    const dz = gCenter[source * 3 + 2] - gCenter[target * 3 + 2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (d + r1 < r2) {
      gCenter[target * 3] = gCenter[source * 3];
      gCenter[target * 3 + 1] = gCenter[source * 3 + 1];
      gCenter[target * 3 + 2] = gCenter[source * 3 + 2];
      gRadius[target] = r2;
      return;
    }

    if (d + r2 > r1) {
      const k = d > 0 ? (d + r2 - r1) / (2 * d) : 0;
      gCenter[target * 3] += dx * k;
      gCenter[target * 3 + 1] += dy * k;
      gCenter[target * 3 + 2] += dz * k;
      gRadius[target] = (d + r2 + r1) / 2;
    }
  };

  const boundsScore = (target: number, source: number): number => {
    const r1 = gRadius[target];
    const r2 = gRadius[source];
    const dx = gCenter[source * 3] - gCenter[target * 3];
    const dy = gCenter[source * 3 + 1] - gCenter[target * 3 + 1];
    const dz = gCenter[source * 3 + 2] - gCenter[target * 3 + 2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const mr = d + r1 < r2 ? r2 : d + r2 < r1 ? r1 : (d + r2 + r1) / 2;
    return mr > 0 ? r1 / mr : 0;
  };

  const useBounds = vertexPositions !== null;

  let bestShared = 0;
  const pickGroupToMerge = (id: number): number => {
    const groupRsqrt = 1 / Math.sqrt(gVertices[id]);

    let bestGroup = -1;
    let bestScore = 0;
    bestShared = 0;

    // accumulate shared counts per adjacent group over all clusters of this group
    for (let ci = id; ci >= 0; ci = gNext[ci]) {
      for (let adj = adjOffsets[ci]; adj !== adjOffsets[ci + 1]; adj++) {
        const other = gGroup[adjClusters[adj]];
        if (other >= 0) sharedAcc[other] += adjShared[adj];
      }
    }

    // score each adjacent group; resets sharedAcc in the process
    for (let ci = id; ci >= 0; ci = gNext[ci]) {
      for (let adj = adjOffsets[ci]; adj !== adjOffsets[ci + 1]; adj++) {
        const other = gGroup[adjClusters[adj]];
        if (other < 0 || sharedAcc[other] === 0) continue;

        // reset before any early out so the next call sees zeros
        const shared = sharedAcc[other];
        sharedAcc[other] = 0;

        if (gSize[id] + gSize[other] > maxPartitionSize) continue;

        const otherRsqrt = 1 / Math.sqrt(gVertices[other]);

        // normalize by the expected boundary of each group (also keeps scoring symmetric)
        let score = shared * (groupRsqrt + otherRsqrt);

        // favor merging nearby groups
        if (useBounds) score *= 1 + 0.4 * boundsScore(id, other);

        if (score > bestScore) {
          bestGroup = other;
          bestScore = score;
          bestShared = shared;
        }
      }
    }

    return bestGroup;
  };

  const tailOf = (id: number): number => {
    let tail = id;
    while (gNext[tail] >= 0) tail = gNext[tail];
    return tail;
  };

  // ---- spatial merge of leftovers (uses bounds only) ----------------------
  const mergeLeaf = (order: Uint32Array, base: number, count: number): void => {
    for (let i = 0; i < count; i++) {
      const id = order[base + i];
      if (gSize[id] === 0 || gSize[id] >= targetPartitionSize) continue;

      let bestScore = -1;
      let bestGroup = -1;

      for (let j = 0; j < count; j++) {
        const other = order[base + j];
        if (id === other || gSize[other] === 0) continue;
        if (gSize[id] + gSize[other] > maxPartitionSize) continue;

        const score = boundsScore(id, other);
        if (score > bestScore) {
          bestScore = score;
          bestGroup = other;
        }
      }

      // merge `id` INTO bestGroup so more groups may join the same target
      if (bestGroup !== -1) {
        gNext[tailOf(bestGroup)] = id;

        // vertex counts are not updated; unused by the spatial merge
        gSize[bestGroup] += gSize[id];
        gSize[id] = 0;

        mergeBounds(bestGroup, id);
        gRadius[id] = 0;
      }
    }
  };

  const mergePartition = (order: Uint32Array, base: number, count: number, axis: number, pivot: number): number => {
    let m = 0;

    // invariant: [0, m) are < pivot, [m, i) are >= pivot
    for (let i = 0; i < count; i++) {
      const v = gCenter[order[base + i] * 3 + axis];

      const t = order[base + m];
      order[base + m] = order[base + i];
      order[base + i] = t;

      if (v < pivot) m++;
    }

    return m;
  };

  const leafSize = 8;
  const mergeSpatial = (order: Uint32Array, base: number, count: number, depth: number): void => {
    let total = 0;
    for (let i = 0; i < count; i++) total += gSize[order[base + i]];

    if (total <= maxPartitionSize || count <= leafSize) {
      mergeLeaf(order, base, count);
      return;
    }

    const mean = [0, 0, 0];
    const vars = [0, 0, 0];
    let runc = 1;
    let runs = 1;

    // Welford's algorithm over group centers
    for (let i = 0; i < count; i++, runc += 1, runs = 1 / runc) {
      const g = order[base + i] * 3;
      for (let k = 0; k < 3; k++) {
        const p = gCenter[g + k];
        const delta = p - mean[k];
        mean[k] += delta * runs;
        vars[k] += delta * (p - mean[k]);
      }
    }

    // split along the axis of largest variance
    const axis = vars[0] >= vars[1] && vars[0] >= vars[2] ? 0 : vars[1] >= vars[2] ? 1 : 2;

    let middle = mergePartition(order, base, count, axis, mean[axis]);

    // enforce balance; this also bounds recursion depth on pathological inputs
    if (middle <= leafSize >> 1 || count - middle <= leafSize >> 1 || depth >= MERGE_DEPTH_CUTOFF) middle = count >> 1;

    mergeSpatial(order, base, middle, depth + 1);
    mergeSpatial(order, base + middle, count - middle, depth + 1);
  };

  // ---- create singleton groups --------------------------------------------
  let pending = 0;

  for (let i = 0; i < clusterCount; i++) {
    gGroup[i] = i;
    gNext[i] = -1;
    gSize[i] = 1;
    gVertices[i] = offsets[i + 1] - offsets[i];
    if (gVertices[i] === 0) throw new Error("partitionClusters: empty cluster");

    if (vertexPositions) computeClusterBounds(filtered, offsets[i], offsets[i + 1], vertexPositions, gCenter, gRadius, i);

    heapPush(pending++, i, gVertices[i]);
  }

  // ---- iteratively merge the smallest group with its best neighbour -------
  while (pending > 0) {
    heapPop(pending--);
    const top = popId;

    // merged into another group earlier
    if (gSize[top] === 0) continue;

    // detach clusters so they cannot be merged into themselves; reattached on reinsertion
    for (let i = top; i >= 0; i = gNext[i]) gGroup[i] = -1;

    // large enough, emit as is
    if (gSize[top] >= targetPartitionSize) continue;

    const bestGroup = pickGroupToMerge(top);

    // cannot grow any more, emit as is
    if (bestGroup === -1) continue;

    gNext[tailOf(top)] = bestGroup;

    gSize[top] += gSize[bestGroup];
    gVertices[top] += gVertices[bestGroup];
    // O(1) approximation of the merged vertex count
    gVertices[top] = gVertices[top] > bestShared ? gVertices[top] - bestShared : 1;

    gSize[bestGroup] = 0;
    gVertices[bestGroup] = 0;

    if (vertexPositions) {
      mergeBounds(top, bestGroup);
      gRadius[bestGroup] = 0;
    }

    for (let i = top; i >= 0; i = gNext[i]) gGroup[i] = top;

    heapPush(pending++, top, gVertices[top]);
  }

  // ---- final pass: merge small groups by spatial locality alone -----------
  if (vertexPositions) {
    const mergeOrder = new Uint32Array(clusterCount);
    let n = 0;
    for (let i = 0; i < clusterCount; i++) if (gSize[i] !== 0) mergeOrder[n++] = i;

    mergeSpatial(mergeOrder, 0, n, 0);
  }

  // ---- output ---------------------------------------------------------------
  let nextGroup = 0;

  for (let i = 0; i < clusterCount; i++) {
    if (gSize[i] === 0) continue;

    for (let j = i; j >= 0; j = gNext[j]) partition[j] = nextGroup;
    nextGroup++;
  }

  return { partition, count: nextGroup };
}

/** Bounding sphere of a cluster: mean of vertex positions as center, farthest vertex as radius. */
function computeClusterBounds(
  indices: Uint32Array,
  begin: number,
  end: number,
  positions: Float32Array,
  outCenter: Float32Array,
  outRadius: Float32Array,
  slot: number,
): void {
  let cx = 0;
  let cy = 0;
  let cz = 0;

  for (let j = begin; j < end; j++) {
    const p = indices[j] * 3;
    cx += positions[p];
    cy += positions[p + 1];
    cz += positions[p + 2];
  }

  const n = end - begin;
  if (n > 0) {
    cx /= n;
    cy /= n;
    cz /= n;
  }

  let radiusSq = 0;
  for (let j = begin; j < end; j++) {
    const p = indices[j] * 3;
    const dx = positions[p] - cx;
    const dy = positions[p + 1] - cy;
    const dz = positions[p + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > radiusSq) radiusSq = d2;
  }

  outCenter[slot * 3] = cx;
  outCenter[slot * 3 + 1] = cy;
  outCenter[slot * 3 + 2] = cz;
  outRadius[slot] = Math.sqrt(radiusSq);
}
