import { describe, expect, it } from "vitest";
import { partitionClusters } from "../src/build/partition.js";

interface ClusterGrid {
  indices: Uint32Array;
  counts: Uint32Array;
  positions: Float32Array;
  vertexCount: number;
  /** Clusters per side. */
  side: number;
}

/**
 * `side x side` clusters, each a `cell x cell` block of quads, over one shared
 * vertex grid. Neighbouring clusters share the vertices of their common border.
 */
function makeClusterGrid(side: number, cell = 4): ClusterGrid {
  const n = side * cell;
  const w = n + 1;
  const positions = new Float32Array(w * w * 3);
  for (let j = 0; j < w; j++) {
    for (let i = 0; i < w; i++) {
      positions[(j * w + i) * 3] = i;
      positions[(j * w + i) * 3 + 1] = j;
    }
  }

  const indices: number[] = [];
  const counts: number[] = [];
  for (let cj = 0; cj < side; cj++) {
    for (let ci = 0; ci < side; ci++) {
      let count = 0;
      for (let j = 0; j < cell; j++) {
        for (let i = 0; i < cell; i++) {
          const a = (cj * cell + j) * w + ci * cell + i;
          const b = a + 1;
          const c = a + w;
          const d = c + 1;
          indices.push(a, b, d, a, d, c);
          count += 6;
        }
      }
      counts.push(count);
    }
  }
  return { indices: Uint32Array.from(indices), counts: Uint32Array.from(counts), positions, vertexCount: w * w, side };
}

function partitionSizes(partition: Uint32Array, count: number): number[] {
  const sizes = new Array<number>(count).fill(0);
  for (const p of partition) sizes[p]++;
  return sizes;
}

/** Fraction of 4-neighbour cluster pairs of the grid that share a partition. */
function sameFraction(partition: Uint32Array, side: number): number {
  let same = 0;
  let total = 0;
  for (let j = 0; j < side; j++) {
    for (let i = 0; i < side; i++) {
      const c = j * side + i;
      if (i + 1 < side) {
        total++;
        if (partition[c] === partition[c + 1]) same++;
      }
      if (j + 1 < side) {
        total++;
        if (partition[c] === partition[c + side]) same++;
      }
    }
  }
  return same / total;
}

describe("partitionClusters", () => {
  for (const withPositions of [true, false]) {
    describe(withPositions ? "with positions" : "connectivity only", () => {
      it("assigns every cluster a valid id and respects the size bound", () => {
        for (const target of [4, 8, 16, 24]) {
          const g = makeClusterGrid(12);
          const { partition, count } = partitionClusters(g.indices, g.counts, withPositions ? g.positions : null, g.vertexCount, target);
          expect(partition.length).toBe(144);
          for (const p of partition) expect(p).toBeLessThan(count);

          const sizes = partitionSizes(partition, count);
          const max = target + Math.floor(target / 3);
          for (const s of sizes) {
            expect(s).toBeGreaterThan(0); // ids are dense
            expect(s).toBeLessThanOrEqual(max);
          }
          expect(sizes.reduce((a, b) => a + b, 0)).toBe(144);
          // partitions should mostly reach the target, not stay singletons
          expect(count).toBeLessThanOrEqual(Math.ceil((144 / target) * 2));
        }
      });

      it("groups adjacent clusters together (spatial coherence)", () => {
        const g = makeClusterGrid(16);
        const { partition, count } = partitionClusters(g.indices, g.counts, withPositions ? g.positions : null, g.vertexCount, 16);
        expect(count).toBeGreaterThanOrEqual(256 / 21);
        // perfect 4x4 blocks give 0.8; random assignment into ~16 groups gives ~0.06
        expect(sameFraction(partition, g.side)).toBeGreaterThan(0.5);

        // a partition's cluster footprint must be compact: bounding box area close to its size
        const minI = new Array<number>(count).fill(Infinity);
        const maxI = new Array<number>(count).fill(-Infinity);
        const minJ = new Array<number>(count).fill(Infinity);
        const maxJ = new Array<number>(count).fill(-Infinity);
        const sizes = partitionSizes(partition, count);
        for (let c = 0; c < 256; c++) {
          const p = partition[c];
          const i = c % 16;
          const j = Math.floor(c / 16);
          minI[p] = Math.min(minI[p], i);
          maxI[p] = Math.max(maxI[p], i);
          minJ[p] = Math.min(minJ[p], j);
          maxJ[p] = Math.max(maxJ[p], j);
        }
        let area = 0;
        for (let p = 0; p < count; p++) area += (maxI[p] - minI[p] + 1) * (maxJ[p] - minJ[p] + 1);
        expect(area / 256).toBeLessThan(2.5);
        expect(sizes.length).toBe(count);
      });

      it("is deterministic", () => {
        const g = makeClusterGrid(10);
        const a = partitionClusters(g.indices, g.counts, withPositions ? g.positions : null, g.vertexCount, 8);
        const b = partitionClusters(g.indices, g.counts, withPositions ? g.positions : null, g.vertexCount, 8);
        expect(a.count).toBe(b.count);
        expect(a.partition).toEqual(b.partition);
      });
    });
  }

  it("keeps disconnected clusters apart without positions and merges them spatially with positions", () => {
    // 6 disjoint single-triangle clusters
    const indices = new Uint32Array(6 * 3);
    const positions = new Float32Array(18 * 3);
    for (let c = 0; c < 6; c++) {
      for (let k = 0; k < 3; k++) {
        indices[c * 3 + k] = c * 3 + k;
        positions[(c * 3 + k) * 3] = c * 10 + (k === 1 ? 1 : 0);
        positions[(c * 3 + k) * 3 + 1] = k === 2 ? 1 : 0;
      }
    }
    const counts = new Uint32Array(6).fill(3);

    const none = partitionClusters(indices, counts, null, 18, 4);
    expect(none.count).toBe(6);

    const spatial = partitionClusters(indices, counts, positions, 18, 4);
    expect(spatial.count).toBeLessThan(6);
    const sizes = partitionSizes(spatial.partition, spatial.count);
    for (const s of sizes) expect(s).toBeLessThanOrEqual(5); // 4 + floor(4 / 3)
  });

  it("handles duplicate vertices within a cluster and trivial inputs", () => {
    const empty = partitionClusters(new Uint32Array(0), new Uint32Array(0), null, 0, 4);
    expect(empty.count).toBe(0);

    // two clusters sharing vertex 1; cluster 0 lists it twice
    const r = partitionClusters(new Uint32Array([0, 1, 1, 1, 2, 3]), new Uint32Array([3, 3]), null, 4, 4);
    expect(r.count).toBe(1);
    expect(Array.from(r.partition)).toEqual([0, 0]);

    expect(() => partitionClusters(new Uint32Array([0, 9]), new Uint32Array([2]), null, 4, 4)).toThrow(/out of range/);
    expect(() => partitionClusters(new Uint32Array([0, 1]), new Uint32Array([1]), null, 4, 4)).toThrow(/sum/);
  });

  it("partitions a random cluster soup without violating invariants", () => {
    let seed = 12345;
    const rand = (): number => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const vertexCount = 400;
    const positions = new Float32Array(vertexCount * 3);
    for (let i = 0; i < positions.length; i++) positions[i] = rand() * 100;

    const clusterCount = 90;
    const indices: number[] = [];
    const counts: number[] = [];
    for (let c = 0; c < clusterCount; c++) {
      const n = 3 + Math.floor(rand() * 30);
      const base = Math.floor(rand() * (vertexCount - 40));
      for (let j = 0; j < n; j++) indices.push(base + Math.floor(rand() * 40));
      counts.push(n);
    }

    for (const usePositions of [false, true]) {
      const { partition, count } = partitionClusters(
        Uint32Array.from(indices),
        Uint32Array.from(counts),
        usePositions ? positions : null,
        vertexCount,
        16,
      );
      const sizes = partitionSizes(partition, count);
      for (const s of sizes) {
        expect(s).toBeGreaterThan(0);
        expect(s).toBeLessThanOrEqual(21);
      }
      expect(sizes.reduce((a, b) => a + b, 0)).toBe(clusterCount);
    }
  });
});
