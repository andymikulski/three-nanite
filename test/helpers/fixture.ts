import type { MeshletData } from "../../src/types.js";

/**
 * A tiny hand-built, valid MeshletData that does not need the builder:
 *
 *  - an N x N quad grid in the XY plane spanning [-1, 1], facing +Z
 *    (N = 8 => 81 vertices, 128 triangles), with position / normal / uv
 *  - LOD 0: four clusters, one per quadrant (32 triangles each), all in group 0
 *  - LOD 1: one cluster of 32 triangles using every second vertex, produced
 *    from group 0 (`clusterRefined = 0`), sitting in terminal group 1
 *
 * Cluster ids: 0..3 = LOD 0 quadrants, 4 = coarse.
 * Group 0: error `FIXTURE_ERROR`, sphere centred on the origin with radius sqrt(2).
 * Group 1: terminal (error Infinity).
 */
export const FIXTURE_ERROR = 0.05;
export const FIXTURE_GRID = 8;

export function makeFixture(): MeshletData {
  const N = FIXTURE_GRID;
  const side = N + 1;
  const vertexCount = side * side;

  const position = new Float32Array(vertexCount * 3);
  const normal = new Float32Array(vertexCount * 3);
  const uv = new Float32Array(vertexCount * 2);
  for (let j = 0; j < side; j++) {
    for (let i = 0; i < side; i++) {
      const v = j * side + i;
      position[v * 3] = (i / N) * 2 - 1;
      position[v * 3 + 1] = (j / N) * 2 - 1;
      position[v * 3 + 2] = 0;
      normal[v * 3 + 2] = 1;
      uv[v * 2] = i / N;
      uv[v * 2 + 1] = j / N;
    }
  }

  const quad = (out: number[], i0: number, j0: number, step: number) => {
    const a = j0 * side + i0;
    const b = j0 * side + i0 + step;
    const c = (j0 + step) * side + i0;
    const d = (j0 + step) * side + i0 + step;
    out.push(a, b, d, a, d, c);
  };

  const clusters: number[][] = [];
  const half = N / 2;
  for (let qj = 0; qj < 2; qj++) {
    for (let qi = 0; qi < 2; qi++) {
      const tris: number[] = [];
      for (let j = 0; j < half; j++) for (let i = 0; i < half; i++) quad(tris, qi * half + i, qj * half + j, 1);
      clusters.push(tris);
    }
  }
  const coarse: number[] = [];
  for (let j = 0; j < N; j += 2) for (let i = 0; i < N; i += 2) quad(coarse, i, j, 2);
  clusters.push(coarse);

  const clusterCount = clusters.length;
  const indices = new Uint32Array(clusters.reduce((n, c) => n + c.length, 0));
  const clusterIndexStart = new Uint32Array(clusterCount);
  const clusterIndexCount = new Uint32Array(clusterCount);
  const clusterBounds = new Float32Array(clusterCount * 4);
  let offset = 0;
  clusters.forEach((tris, c) => {
    indices.set(tris, offset);
    clusterIndexStart[c] = offset;
    clusterIndexCount[c] = tris.length;
    offset += tris.length;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const v of tris) {
      minX = Math.min(minX, position[v * 3]);
      maxX = Math.max(maxX, position[v * 3]);
      minY = Math.min(minY, position[v * 3 + 1]);
      maxY = Math.max(maxY, position[v * 3 + 1]);
    }
    clusterBounds[c * 4] = (minX + maxX) / 2;
    clusterBounds[c * 4 + 1] = (minY + maxY) / 2;
    clusterBounds[c * 4 + 2] = 0;
    clusterBounds[c * 4 + 3] = Math.hypot(maxX - minX, maxY - minY) / 2;
  });

  const r = Math.SQRT2;
  return {
    version: 1,
    vertexCount,
    attributes: [
      { name: "position", array: position, itemSize: 3, normalized: false },
      { name: "normal", array: normal, itemSize: 3, normalized: false },
      { name: "uv", array: uv, itemSize: 2, normalized: false },
    ],
    indices,
    clusterCount,
    clusterIndexStart,
    clusterIndexCount,
    clusterBounds,
    clusterGroup: new Uint32Array([0, 0, 0, 0, 1]),
    clusterRefined: new Int32Array([-1, -1, -1, -1, 0]),
    groupCount: 2,
    groupClusterStart: new Uint32Array([0, 4]),
    groupClusterCount: new Uint32Array([4, 1]),
    groupBounds: new Float32Array([0, 0, 0, r, 0, 0, 0, r]),
    groupError: new Float32Array([FIXTURE_ERROR, Infinity]),
    groupDepth: new Uint16Array([0, 1]),
    rootGroups: new Uint32Array([1]),
    levelCount: 2,
    boundingSphere: new Float32Array([0, 0, 0, r]),
    boundingBox: new Float32Array([-1, -1, 0, 1, 1, 0]),
  };
}
