import { beforeAll, describe, expect, it } from "vitest";
import { buildMeshlets } from "../src/build/buildMeshlets.js";
import { ClusterSelector, getLodErrors } from "../src/core/select.js";
import { validateMeshletData } from "../src/core/validate.js";
import type { MeshletData, MeshletInput } from "../src/types.js";

const N = 64; // quads per side

/** Deterministic height at integer grid coordinates (world space, so neighbouring tiles agree exactly). */
function height(gx: number, gy: number): number {
  let h = Math.imul(gx, 374761393) ^ Math.imul(gy, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return ((h >>> 0) / 4294967296) * 0.004;
}

interface Grid {
  input: MeshletInput;
  /** Vertices on any of the four border edges (every copy of a duplicated vertex). */
  border: number[];
  /** Vertices to lock: for duplicated vertices only the first copy. */
  lockOne: Uint8Array;
  /** Vertices on the border at grid x = N (the edge shared with a tile to the right). */
  right: Uint8Array;
  /** Vertices on the border at grid x = 0 (the edge shared with a tile to the left). */
  left: Uint8Array;
}

/**
 * N x N quad plane at grid offset `ox`, with a little height noise. With `dupLeft` the vertices of the
 * left border exist twice (same position, different normal), used by alternating quad rows.
 */
function makeGrid(ox = 0, dupLeft = false): Grid {
  const stride = N + 1;
  const base = stride * stride;
  const total = dupLeft ? base + stride : base;
  const position = new Float32Array(total * 3);
  const normal = new Float32Array(total * 3);
  const put = (v: number, i: number, j: number, tilt: number): void => {
    position.set([(ox + i) / N, j / N, height(ox + i, j)], v * 3);
    normal.set([tilt, 0, Math.sqrt(1 - tilt * tilt)], v * 3);
  };
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) put(j * stride + i, i, j, 0);
  if (dupLeft) for (let j = 0; j <= N; j++) put(base + j, 0, j, 0.3);

  const border: number[] = [];
  const lockOne = new Uint8Array(total);
  const left = new Uint8Array(total);
  const right = new Uint8Array(total);
  for (let j = 0; j <= N; j++) {
    for (let i = 0; i <= N; i++) {
      if (i !== 0 && i !== N && j !== 0 && j !== N) continue;
      const v = j * stride + i;
      border.push(v);
      lockOne[v] = 1;
      if (i === 0) left[v] = 1;
      if (i === N) right[v] = 1;
    }
  }
  if (dupLeft) {
    for (let j = 0; j <= N; j++) {
      border.push(base + j);
      left[base + j] = 1;
    }
  }

  const indices = new Uint32Array(N * N * 6);
  let w = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const useDup = dupLeft && i === 0 && j % 2 === 1; // odd rows use the duplicated left vertices
      const a = useDup ? base + j : j * stride + i;
      const d = useDup ? base + j + 1 : (j + 1) * stride + i;
      const b = j * stride + i + 1;
      const c = (j + 1) * stride + i + 1;
      indices.set([a, b, c, a, c, d], w);
      w += 6;
    }
  }

  return {
    input: {
      attributes: [
        { name: "position", array: position, itemSize: 3, normalized: false },
        { name: "normal", array: normal, itemSize: 3, normalized: false },
      ],
      indices,
    },
    border,
    lockOne,
    left,
    right,
  };
}

const posKey = (p: Float32Array, v: number): string => `${p[v * 3]},${p[v * 3 + 1]},${p[v * 3 + 2]}`;

/** The triangles (as vertex ids) of the cut chosen by `selectByError`. */
function cut(data: MeshletData, selector: ClusterSelector, maxError: number): number[] {
  const out = new Uint32Array(data.clusterCount);
  const n = selector.selectByError(out, maxError);
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = data.clusterIndexStart[out[i]];
    for (let k = 0; k < data.clusterIndexCount[out[i]]; k++) ids.push(data.indices[s + k]);
  }
  return ids;
}

/** Positions (as keys) of the vertices that a triangle of the cut uses. */
function usedPositions(data: MeshletData, ids: number[]): Set<string> {
  const p = data.attributes[0].array as Float32Array;
  return new Set(ids.map((v) => posKey(p, v)));
}

/**
 * The open edges of the cut (used by exactly one triangle) whose endpoints are both flagged in `mask`,
 * keyed by endpoint positions. On a tile this is the seam its neighbour has to match. Edges used by two
 * triangles are not part of it: simplification may leave a zero-area sliver of three border vertices,
 * whose long edge lies on the border but is closed by the triangle on its other side.
 */
function seamEdges(data: MeshletData, ids: number[], mask: Uint8Array): Set<string> {
  const p = data.attributes[0].array as Float32Array;
  const uses = new Map<string, number>();
  for (let t = 0; t < ids.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = ids[t + e];
      const b = ids[t + ((e + 1) % 3)];
      if (!mask[a] || !mask[b]) continue;
      const ka = posKey(p, a);
      const kb = posKey(p, b);
      if (ka === kb) continue;
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  }
  return new Set([...uses].filter(([, n]) => n === 1).map(([key]) => key));
}

function missingBorder(grid: Grid, data: MeshletData, maxError = Infinity): string[] {
  const used = usedPositions(data, cut(data, new ClusterSelector(data), maxError));
  const p = grid.input.attributes[0].array as Float32Array;
  return grid.border.map((v) => posKey(p, v)).filter((k) => !used.has(k));
}

function triangleCount(data: MeshletData, maxError: number): number {
  return cut(data, new ClusterSelector(data), maxError).length / 3;
}

describe("buildMeshlets: lockedVertices", () => {
  let plain: Grid;
  let plainFree: MeshletData;
  let plainLocked: MeshletData;
  let dup: Grid;
  let dupFree: MeshletData;
  let dupLocked: MeshletData;
  let tileA: Grid;
  let tileB: Grid;
  let dataA: MeshletData;
  let dataB: MeshletData;

  beforeAll(async () => {
    plain = makeGrid();
    plainFree = await buildMeshlets(plain.input);
    plainLocked = await buildMeshlets(plain.input, { lockedVertices: plain.lockOne });

    dup = makeGrid(0, true);
    dupFree = await buildMeshlets(dup.input);
    // only the first copy of each duplicated vertex is flagged
    dupLocked = await buildMeshlets(dup.input, { lockedVertices: dup.lockOne });

    tileA = makeGrid(0);
    tileB = makeGrid(N);
    dataA = await buildMeshlets(tileA.input, { lockedVertices: tileA.right });
    dataB = await buildMeshlets(tileB.input, { lockedVertices: tileB.left });
  }, 120_000);

  it("keeps every locked border vertex at the coarsest cut, and the option is what does it", () => {
    expect(missingBorder(plain, plainFree).length).toBeGreaterThan(0);
    expect(missingBorder(plain, plainLocked)).toEqual([]);
  });

  it("locking one vertex of a position locks its duplicates", () => {
    // the grid really has duplicated border vertices with distinct normals
    const normal = dup.input.attributes[1].array as Float32Array;
    const position = dup.input.attributes[0].array as Float32Array;
    const copies = dup.border.filter((v) => dup.left[v]);
    expect(copies.length).toBe(2 * (N + 1));
    const first = copies[0];
    const twin = copies.find((v) => v !== first && posKey(position, v) === posKey(position, first));
    expect(twin).toBeDefined();
    expect(normal[twin! * 3]).not.toBe(normal[first * 3]);
    // exactly one of every pair is flagged
    expect(dup.lockOne[first] + dup.lockOne[twin!]).toBe(1);

    expect(missingBorder(dup, dupFree).length).toBeGreaterThan(0);
    expect(missingBorder(dup, dupLocked)).toEqual([]);
  });

  it("never opens a crack between two tiles that share an edge", () => {
    const pa = dataA.attributes[0].array as Float32Array;
    const pb = dataB.attributes[0].array as Float32Array;
    // the two tiles agree on the shared border positions
    const sharedA = new Set(tileA.border.filter((v) => tileA.right[v]).map((v) => posKey(pa, v)));
    const sharedB = new Set(tileB.border.filter((v) => tileB.left[v]).map((v) => posKey(pb, v)));
    expect(sharedA.size).toBe(N + 1);
    expect(sharedB).toEqual(sharedA);

    const errors = getLodErrors(dataA);
    expect(errors.length).toBeGreaterThan(1);
    const thresholds = [0, errors[Math.floor(errors.length / 2)], errors[errors.length - 1], Infinity];

    const selA = new ClusterSelector(dataA);
    const selB = new ClusterSelector(dataB);
    const reference = seamEdges(dataA, cut(dataA, selA, 0), tileA.right);
    expect(reference.size).toBe(N);

    for (const ea of thresholds) {
      for (const eb of thresholds) {
        const edgesA = seamEdges(dataA, cut(dataA, selA, ea), tileA.right);
        const edgesB = seamEdges(dataB, cut(dataB, selB, eb), tileB.left);
        expect(edgesA, `A at ${ea}`).toEqual(reference);
        expect(edgesB, `B at ${eb}`).toEqual(reference);
      }
    }
  });

  it("throws when lockedVertices has the wrong length", async () => {
    const n = plain.input.attributes[0].array.length / 3;
    await expect(buildMeshlets(plain.input, { lockedVertices: new Uint8Array(n - 1) })).rejects.toThrow(/lockedVertices/);
    await expect(buildMeshlets(plain.input, { lockedVertices: new Uint8Array(n + 1) })).rejects.toThrow(/lockedVertices/);
  });

  it("still validates and still reduces", () => {
    for (const data of [plainLocked, dupLocked, dataA, dataB]) {
      expect(validateMeshletData(data)).toEqual([]);
      expect(triangleCount(data, Infinity)).toBeLessThan(triangleCount(data, 0));
    }
    expect(triangleCount(plainLocked, 0)).toBe(2 * N * N);
  });
});
