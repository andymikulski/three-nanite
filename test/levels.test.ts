import { beforeAll, describe, expect, it } from "vitest";
import { TorusKnotGeometry } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { computeLodLevels, copyLevelIndices, levelForError, levelOfCut } from "../src/core/levels.js";
import { ClusterSelector, countTriangles, lodScale } from "../src/core/select.js";
import type { MeshletData } from "../src/types.js";
import { FIXTURE_ERROR, makeFixture } from "./helpers/fixture.js";

const sortedIds = (out: Uint32Array, n: number) => Array.from(out.subarray(0, n)).sort((a, b) => a - b);

/** Small deterministic generator in [0, 1). */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

describe("LodLevels on the fixture", () => {
  const data = makeFixture();
  const levels = computeLodLevels(data);

  it("lists the original mesh and the coarse cluster as two levels", () => {
    expect(levels.count).toBe(2);
    expect(levels.errors[0]).toBe(0);
    expect(levels.errors[1]).toBe(data.groupError[0]);
    expect(Array.from(levels.triangles)).toEqual([128, 32]);
    expect(Array.from(levels.clusters)).toEqual([4, 1]);
    expect(Array.from(levels.depth)).toEqual([0, 1]);
    expect(levels.exact).toBe(true);

    expect(levels.runOffset[1] - levels.runOffset[0]).toBe(1);
    expect(levels.runStart[levels.runOffset[0]]).toBe(0);
    expect(levels.runCount[levels.runOffset[0]]).toBe(384);
    expect(levels.runOffset[2] - levels.runOffset[1]).toBe(1);
    expect(levels.runStart[levels.runOffset[1]]).toBe(384);
    expect(levels.runCount[levels.runOffset[1]]).toBe(96);
  });

  it("levelForError returns the coarsest level whose error is within the budget", () => {
    expect(levelForError(levels, 0)).toBe(0);
    expect(levelForError(levels, FIXTURE_ERROR * 0.99)).toBe(0);
    expect(levelForError(levels, data.groupError[0])).toBe(1);
    expect(levelForError(levels, Infinity)).toBe(1);
  });

  it("levelOfCut recognises a whole-mesh cut and nothing else", () => {
    const selector = new ClusterSelector(data);
    const out = new Uint32Array(data.clusterCount);
    const k = lodScale(1, 1080, 1.7);

    selector.select(out, 0, 0, 2, k, 0.1);
    expect(levelOfCut(levels, selector.lastWithin, selector.lastOver)).toBe(0);

    selector.select(out, 0, 0, 1000, k, 0.1);
    expect(levelOfCut(levels, selector.lastWithin, selector.lastOver)).toBe(1);
  });
});

describe("LodLevels on a built hierarchy", () => {
  let data: MeshletData;
  let selector: ClusterSelector;
  let out: Uint32Array;

  beforeAll(async () => {
    data = await buildMeshletsFromGeometry(mergeVertices(new TorusKnotGeometry(1, 0.35, 320, 48)));
    selector = new ClusterSelector(data);
    out = new Uint32Array(data.clusterCount);
  });

  it("has ascending errors from 0 and shrinking triangle counts that match the cuts", () => {
    const levels = computeLodLevels(data);
    expect(levels.count).toBeGreaterThan(2);
    expect(levels.errors[0]).toBe(0);
    for (let j = 1; j < levels.count; j++) {
      expect(levels.errors[j]).toBeGreaterThan(levels.errors[j - 1]);
      expect(levels.triangles[j]).toBeLessThanOrEqual(levels.triangles[j - 1]);
    }
    const finest = selector.selectByError(out, 0);
    expect(levels.triangles[0]).toBe(countTriangles(data, out, finest));
    const coarsest = selector.selectByError(out, Infinity);
    expect(levels.triangles[levels.count - 1]).toBe(countTriangles(data, out, coarsest));
  });

  it("stores each level as the triangles of its view-independent cut, in ascending, non-adjacent runs", () => {
    const levels = computeLodLevels(data);
    for (let j = 0; j < levels.count; j++) {
      const n = selector.selectByError(out, levels.errors[j]);
      const ids = sortedIds(out, n);
      const expected: number[] = [];
      for (const c of ids) {
        const start = data.clusterIndexStart[c];
        for (let i = 0; i < data.clusterIndexCount[c]; i++) expected.push(data.indices[start + i]);
      }

      const actual = new Uint32Array(levels.triangles[j] * 3);
      expect(copyLevelIndices(data, levels, j, actual)).toBe(actual.length);
      expect(actual.length).toBe(expected.length);
      expect(Array.from(actual)).toEqual(expected);

      let total = 0;
      let previousEnd = -1;
      for (let r = levels.runOffset[j]; r < levels.runOffset[j + 1]; r++) {
        total += levels.runCount[r];
        expect(levels.runStart[r]).toBeGreaterThan(previousEnd);
        previousEnd = levels.runStart[r] + levels.runCount[r];
      }
      expect(total).toBe(levels.triangles[j] * 3);
    }
  });

  it("thins to maxLevels, keeping the finest and the coarsest cut; a large maxLevels keeps every error", () => {
    const thinned = computeLodLevels(data, { maxLevels: 8 });
    expect(thinned.count).toBeLessThanOrEqual(8);
    expect(thinned.exact).toBe(false);
    expect(thinned.errors[0]).toBe(0);
    let largest = 0;
    for (let g = 0; g < data.groupCount; g++) {
      const e = data.groupError[g];
      if (Number.isFinite(e) && e > largest) largest = e;
    }
    expect(thinned.errors[thinned.count - 1]).toBe(largest);

    const full = computeLodLevels(data, { maxLevels: 100000 });
    const distinct = new Set<number>();
    for (let g = 0; g < data.groupCount; g++) {
      const e = data.groupError[g];
      if (Number.isFinite(e) && e > 0) distinct.add(e);
    }
    expect(full.exact).toBe(true);
    expect(full.count).toBe(1 + distinct.size);
  });

  it("levelOfCut only names a level when the view-dependent cut is exactly that level", () => {
    const levels = computeLodLevels(data, { maxLevels: 100000 });
    const k = lodScale(1, 1080, 1 / Math.tan((50 * Math.PI) / 360));
    const znear = 0.1;
    const s = data.boundingSphere;
    const radius = s[3];
    const rand = lcg(12345);

    let levelHits = 0;
    let mixed = 0;
    let nearMixed = 0;
    const positions = 40;
    for (let p = 0; p < positions; p++) {
      const t = p / (positions - 1);
      const distance = radius * 1.1 * Math.pow(300 / 1.1, t);
      // a random direction on the sphere
      const z = rand() * 2 - 1;
      const phi = rand() * Math.PI * 2;
      const rxy = Math.sqrt(1 - z * z);
      const px = s[0] + distance * rxy * Math.cos(phi);
      const py = s[1] + distance * rxy * Math.sin(phi);
      const pz = s[2] + distance * z;

      const n = selector.select(out, px, py, pz, k, znear);
      const level = levelOfCut(levels, selector.lastWithin, selector.lastOver);
      if (level >= 0) {
        levelHits++;
        const selected = sortedIds(out, n);
        const m = selector.selectByError(out, levels.errors[level]);
        expect(selected).toEqual(sortedIds(out, m));
      } else {
        mixed++;
        if (distance < radius * 3) nearMixed++;
      }
    }
    expect(levelHits).toBeGreaterThan(0);
    expect(mixed).toBeGreaterThan(0);
    expect(nearMixed).toBeGreaterThan(0);
  });

  it("never selects more than the whole-mesh cut that is safe at the nearest point of the mesh", () => {
    const k = lodScale(1, 1080, 1 / Math.tan((50 * Math.PI) / 360));
    const znear = 0.1;
    const s = data.boundingSphere;
    const radius = s[3];
    const rand = lcg(777);

    for (const surfaceDistance of [0.05, 0.3, 1, 2, 5, 20, 100]) {
      for (let t = 0; t < 3; t++) {
        const z = rand() * 2 - 1;
        const phi = rand() * Math.PI * 2;
        const rxy = Math.sqrt(1 - z * z);
        const distance = radius + surfaceDistance;
        const px = s[0] + distance * rxy * Math.cos(phi);
        const py = s[1] + distance * rxy * Math.sin(phi);
        const pz = s[2] + distance * z;

        const d = Math.sqrt((s[0] - px) ** 2 + (s[1] - py) ** 2 + (s[2] - pz) ** 2) - radius;
        expect(d).toBeGreaterThan(0);

        const n = selector.select(out, px, py, pz, k, znear);
        const viewTriangles = countTriangles(data, out, n);
        const m = selector.selectByError(out, k * Math.max(d, znear));
        const levelTriangles = countTriangles(data, out, m);
        expect(viewTriangles).toBeLessThanOrEqual(levelTriangles);
      }
    }
  });
});
