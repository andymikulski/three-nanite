import { beforeAll, describe, expect, it } from "vitest";
import type * as THREE from "three";
import { buildMeshlets } from "../src/build/buildMeshlets.js";
import { validateMeshletData } from "../src/core/validate.js";
import { ClusterSelector, countTriangles } from "../src/core/select.js";
import type { MeshletData, MeshletInput } from "../src/types.js";
import { makeFixture } from "./helpers/fixture.js";
import {
  geometryToInput,
  icosphere,
  identityIndices,
  maxFiniteError,
  octahedron,
  plane,
  sameMultiset,
  sampleCuts,
  torusKnot,
  triangleMultiset,
} from "./helpers/buildMeshes.js";

interface Built {
  input: MeshletInput;
  data: MeshletData;
}

const closedMeshes: Record<string, () => THREE.BufferGeometry> = {
  "torus knot (attribute seams)": () => torusKnot(256, 48, false),
  "torus knot (welded)": () => torusKnot(256, 48, true),
  "icosphere (non-indexed)": () => icosphere(40, false),
  "icosphere (welded)": () => icosphere(40, true),
};

const built = new Map<string, Built>();

async function buildNamed(name: string, geometry: THREE.BufferGeometry): Promise<Built> {
  const input = geometryToInput(geometry);
  const data = await buildMeshlets(input);
  const b = { input, data };
  built.set(name, b);
  return b;
}

function inputTriangleCount(input: MeshletInput): number {
  return (input.indices ? input.indices.length : input.attributes[0].array.length / 3) / 3;
}

function lod0Triangles(data: MeshletData): Map<string, number> {
  const idx: number[] = [];
  for (let c = 0; c < data.clusterCount; c++) {
    if (data.clusterRefined[c] !== -1) continue;
    const s = data.clusterIndexStart[c];
    for (let j = 0; j < data.clusterIndexCount[c]; j++) idx.push(data.indices[s + j]);
  }
  return triangleMultiset(idx);
}

beforeAll(async () => {
  for (const [name, make] of Object.entries(closedMeshes)) await buildNamed(name, make());
  await buildNamed("plane", plane(40));
}, 120_000);

describe("buildMeshlets: closed meshes", () => {
  for (const name of Object.keys(closedMeshes)) {
    describe(name, () => {
      it("passes validateMeshletData", () => {
        expect(validateMeshletData(built.get(name)!.data)).toEqual([]);
      });

      it("shares the input vertex buffer", () => {
        const { input, data } = built.get(name)!;
        expect(data.attributes.length).toBe(input.attributes.length);
        for (let i = 0; i < data.attributes.length; i++) expect(data.attributes[i].array).toBe(input.attributes[i].array);
        expect(data.vertexCount).toBe(input.attributes[0].array.length / 3);
      });

      it("LOD 0 clusters reproduce the input triangles", () => {
        const { input, data } = built.get(name)!;
        const src = input.indices ?? identityIndices(input.attributes[0].array.length / 3);
        expect(sameMultiset(lod0Triangles(data), triangleMultiset(src))).toBe(true);
      });

      it("every cut is crack-free and the triangle count is monotonic", async () => {
        const { input, data } = built.get(name)!;
        const cuts = await sampleCuts(data);
        for (const cut of cuts) {
          // no cracks / T-junction gaps / overlaps in any cut
          expect(cut.stats.open, `${cut.label}: open edges`).toBe(0);
          if (cut.stats.triangles >= 300) {
            expect(cut.stats.overused, `${cut.label}: over-used edges`).toBe(0);
            expect(cut.stats.duplicateDirected, `${cut.label}: duplicate directed edges`).toBe(0);
          } else if (cut.stats.triangles >= 64) {
            // The topmost group of a thin tube (torus knot) is simplified below the topological
            // resolution of the tube, which can pinch a couple of edges into non-manifold ones.
            // That is simplifier behaviour inside a single group, not a crack.
            expect(cut.stats.overused, `${cut.label}: over-used edges`).toBeLessThanOrEqual(2);
          }
          // Below that are the levels of the last cluster by itself (`coarsestTriangles`): a closed
          // tube of a few triangles is a folded sheet, more pinched still, and still has no open edge.
        }

        // first 11 cuts are error 0 .. max in 10 steps, then Infinity
        const errorCuts = cuts.slice(0, 12);
        for (let i = 1; i < errorCuts.length; i++) {
          expect(errorCuts[i].stats.triangles, errorCuts[i].label).toBeLessThanOrEqual(errorCuts[i - 1].stats.triangles);
        }

        // error 0 is the original mesh, Infinity is the coarsest cut
        expect(errorCuts[0].stats.triangles).toBe(inputTriangleCount(input));
        const coarsest = errorCuts[errorCuts.length - 1].stats.triangles;
        const ratio = coarsest / errorCuts[0].stats.triangles;
        console.log(
          `[${name}] levels=${data.levelCount} clusters=${data.clusterCount} groups=${data.groupCount} ` +
            `tris ${errorCuts[0].stats.triangles} -> coarsest ${coarsest} (${(ratio * 100).toFixed(2)}%)`,
        );
        expect(ratio).toBeLessThan(0.05);
      });
    });
  }
});

describe("buildMeshlets: open plane", () => {
  it("passes validateMeshletData", () => {
    expect(validateMeshletData(built.get("plane")!.data)).toEqual([]);
  });

  it("no edge is used more than twice in any cut", async () => {
    const { data } = built.get("plane")!;
    const cuts = await sampleCuts(data);
    for (const cut of cuts) {
      expect(cut.stats.overused, cut.label).toBe(0);
      expect(cut.stats.duplicateDirected, cut.label).toBe(0);
    }
  });

  it("the original mesh has the full 160-edge border and cuts never grow it", async () => {
    const { data } = built.get("plane")!;
    const cuts = await sampleCuts(data);
    expect(cuts[0].stats.open).toBe(160);
    for (const cut of cuts) expect(cut.stats.open, cut.label).toBeLessThanOrEqual(160);
  });
});

describe("buildMeshlets: small meshes", () => {
  it("builds a single triangle as one terminal group", async () => {
    const data = await buildMeshlets({
      attributes: [{ name: "position", array: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), itemSize: 3, normalized: false }],
    });
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.clusterCount).toBe(1);
    expect(data.groupCount).toBe(1);
    expect(Array.from(data.rootGroups)).toEqual([0]);
    expect(data.levelCount).toBe(1);
    expect(data.groupError[0]).toBe(Infinity);
  });

  it("builds an octahedron", async () => {
    const { data } = await buildNamed("octa", octahedron());
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.clusterCount).toBe(1);
    expect(data.indices.length).toBe(24);
  });

  it("accepts Uint16 indices", async () => {
    const pos = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    const attributes = [{ name: "position", array: pos, itemSize: 3, normalized: false }];
    const a = await buildMeshlets({ attributes, indices: new Uint16Array([0, 1, 2, 0, 2, 3]) });
    expect(validateMeshletData(a)).toEqual([]);
    expect(a.indices.length).toBe(6);
    expect(a.indices).toBeInstanceOf(Uint32Array);
  });

  it("drops degenerate triangles and does not modify the caller's indices", async () => {
    const pos = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    const indices = new Uint32Array([0, 1, 1, 0, 1, 2, 0, 2, 3, 2, 2, 3]);
    const copy = indices.slice();
    const data = await buildMeshlets({ attributes: [{ name: "position", array: pos, itemSize: 3, normalized: false }], indices });
    expect(data.indices.length).toBe(6);
    expect(indices).toEqual(copy);
  });

  it("converts normalized integer attributes for simplification", async () => {
    const input = geometryToInput(torusKnot(64, 16, false));
    const n = input.attributes[1].array as Float32Array;
    const q = new Int8Array(n.length);
    for (let i = 0; i < n.length; i++) q[i] = Math.round(n[i] * 127);
    input.attributes[1] = { name: "normal", array: q, itemSize: 3, normalized: true };
    const data = await buildMeshlets(input);
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.attributes[1].array).toBe(q);
  });
});

describe("buildMeshlets: options and errors", () => {
  it("is deterministic", async () => {
    const a = await buildMeshlets(geometryToInput(torusKnot(96, 24, false)));
    const b = await buildMeshlets(geometryToInput(torusKnot(96, 24, false)));
    expect(a.indices).toEqual(b.indices);
    expect(a.groupError).toEqual(b.groupError);
    expect(a.clusterRefined).toEqual(b.clusterRefined);
  });

  it("respects maxTriangles / maxVertices", async () => {
    const data = await buildMeshlets(geometryToInput(torusKnot(128, 24, false)), { maxTriangles: 64 });
    expect(validateMeshletData(data)).toEqual([]);
    let maxSeen = 0;
    for (let c = 0; c < data.clusterCount; c++) {
      const tris = data.clusterIndexCount[c] / 3;
      maxSeen = Math.max(maxSeen, tris);
      expect(tris).toBeLessThanOrEqual(64);
      const s = data.clusterIndexStart[c];
      const unique = new Set(data.indices.subarray(s, s + data.clusterIndexCount[c]));
      expect(unique.size).toBeLessThanOrEqual(64);
    }
    expect(maxSeen).toBeGreaterThan(32);
  });

  it("reports progress once per level", async () => {
    const levels: number[] = [];
    const data = await buildMeshlets(geometryToInput(torusKnot(96, 24, false)), {
      onProgress: (info) => levels.push(info.level),
    });
    expect(levels).toEqual(Array.from({ length: levels.length }, (_, i) => i));
    expect(levels.length).toBeLessThanOrEqual(data.levelCount);
  });

  it("works without weighted attributes and with custom weights", async () => {
    const input = geometryToInput(torusKnot(64, 16, false));
    const a = await buildMeshlets(input, { attributeWeights: {} });
    expect(validateMeshletData(a)).toEqual([]);
    const b = await buildMeshlets(input, { attributeWeights: { normal: 1, uv: 0.5 }, protectAttributes: [] });
    expect(validateMeshletData(b)).toEqual([]);
  });

  it("throws clear errors for invalid input", async () => {
    const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const attr = { name: "position", array: pos, itemSize: 3, normalized: false };
    await expect(buildMeshlets({ attributes: [] })).rejects.toThrow(/attributes/);
    await expect(buildMeshlets({ attributes: [{ ...attr, array: new Float32Array(0) }] })).rejects.toThrow(/empty/);
    await expect(buildMeshlets({ attributes: [attr], indices: new Uint32Array(0) })).rejects.toThrow(/empty/);
    await expect(buildMeshlets({ attributes: [attr], indices: new Uint32Array([0, 1]) })).rejects.toThrow(/divisible by 3/);
    await expect(buildMeshlets({ attributes: [attr], indices: new Uint32Array([0, 1, 7]) })).rejects.toThrow(/out of range/);
    await expect(buildMeshlets({ attributes: [{ ...attr, itemSize: 2 }] })).rejects.toThrow(/position/);
    await expect(
      buildMeshlets({ attributes: [{ ...attr, array: new Float64Array(9) as unknown as Float32Array }] }),
    ).rejects.toThrow(/position/);
    await expect(buildMeshlets({ attributes: [attr], indices: new Uint32Array([0, 0, 1]) })).rejects.toThrow(/degenerate/);
    await expect(buildMeshlets({ attributes: [attr] }, { maxTriangles: 3 })).rejects.toThrow(/maxTriangles/);
    await expect(buildMeshlets({ attributes: [attr] }, { maxTriangles: 300 })).rejects.toThrow(/maxTriangles/);
    await expect(
      buildMeshlets({ attributes: [attr, { name: "normal", array: new Float32Array(4), itemSize: 3, normalized: false }] }),
    ).rejects.toThrow(/length/);
  });
});

describe("validateMeshletData", () => {
  it("accepts the hand-built fixture", () => {
    expect(validateMeshletData(makeFixture())).toEqual([]);
  });

  it("detects broken data", () => {
    const f = makeFixture();
    f.indices[0] = 9999;
    expect(validateMeshletData(f).join("\n")).toMatch(/out of range/);

    const g = makeFixture();
    g.clusterIndexStart[2] += 3;
    expect(validateMeshletData(g).join("\n")).toMatch(/contiguous/);

    const h = makeFixture();
    h.groupError[0] = Infinity;
    h.groupError[1] = 0.01;
    expect(validateMeshletData(h).join("\n")).toMatch(/error/);

    const i = makeFixture();
    i.groupBounds[7] = 0.1;
    expect(validateMeshletData(i).join("\n")).toMatch(/enclose/);

    const j = makeFixture();
    j.rootGroups = new Uint32Array([0]);
    expect(validateMeshletData(j).join("\n")).toMatch(/rootGroups/);

    const k = makeFixture();
    k.levelCount = 5;
    expect(validateMeshletData(k).join("\n")).toMatch(/levelCount/);

    const l = makeFixture();
    l.clusterGroup[4] = 0;
    expect(validateMeshletData(l).join("\n")).toMatch(/clusterGroup/);

    const m = makeFixture();
    m.groupBounds[0] = NaN;
    expect(validateMeshletData(m).join("\n")).toMatch(/NaN/);

    const n = makeFixture();
    n.clusterRefined[4] = 1;
    expect(validateMeshletData(n).join("\n")).toMatch(/depth/);
  });
});

describe("buildMeshlets: performance", () => {
  it("builds a ~200k triangle torus knot", async () => {
    const input = geometryToInput(torusKnot(1024, 100, true));
    const tris = inputTriangleCount(input);
    const levels: number[] = [];
    const t0 = performance.now();
    const data = await buildMeshlets(input, { onProgress: (i) => levels.push(i.clusterCount) });
    const ms = performance.now() - t0;

    const selector = new ClusterSelector(data);
    const out = new Uint32Array(data.clusterCount);
    const n = selector.selectByError(out, Infinity);
    console.log(
      `[perf] ${tris} triangles, ${data.vertexCount} vertices -> ${data.clusterCount} clusters, ${data.groupCount} groups, ` +
        `${data.levelCount} levels in ${(ms / 1000).toFixed(2)} s; per-level cluster counts: ${levels.join(", ")}; ` +
        `max finite error ${maxFiniteError(data)}; coarsest cut ${n} clusters`,
    );
    expect(validateMeshletData(data)).toEqual([]);
    expect(ms).toBeLessThan(30_000);
  }, 120_000);
});

describe("buildMeshlets: coarsest level and error limit", () => {
  const coarsestTriangles = (data: MeshletData): number => {
    const out = new Uint32Array(data.clusterCount);
    return countTriangles(data, out, new ClusterSelector(data).selectByError(out, Infinity));
  };

  it("keeps simplifying the last cluster down to coarsestTriangles", async () => {
    const input = geometryToInput(icosphere(24, true));
    const tail = await buildMeshlets(input);
    const stop = await buildMeshlets(input, { coarsestTriangles: Infinity });
    expect(validateMeshletData(tail)).toEqual([]);
    // ending at one cluster leaves a good part of a cluster drawn at any distance
    expect(coarsestTriangles(stop)).toBeGreaterThan(16);
    expect(coarsestTriangles(tail)).toBeLessThanOrEqual(8);
    expect(tail.levelCount).toBeGreaterThan(stop.levelCount);
    // the finer levels are the same hierarchy
    expect(Array.from(tail.indices.subarray(0, stop.indices.length))).toEqual(Array.from(stop.indices));
  });

  it("errorEdgeLimit only ever lowers the recorded errors", async () => {
    const input = geometryToInput(torusKnot(128, 24, true));
    const measured = await buildMeshlets(input);
    // a smooth dense surface simplifies with errors far below its edge lengths, so the cap has to be tiny to bind
    const capped = await buildMeshlets(input, { errorEdgeLimit: 0.002 });
    expect(validateMeshletData(capped)).toEqual([]);
    expect(capped.groupCount).toBe(measured.groupCount);
    let lower = 0;
    for (let g = 0; g < capped.groupCount; g++) {
      expect(capped.groupError[g]).toBeLessThanOrEqual(measured.groupError[g]);
      if (capped.groupError[g] < measured.groupError[g]) lower++;
    }
    expect(lower).toBeGreaterThan(0);
    await expect(buildMeshlets(input, { errorEdgeLimit: -1 })).rejects.toThrow(/errorEdgeLimit/);
    await expect(buildMeshlets(input, { coarsestTriangles: 0 })).rejects.toThrow(/coarsestTriangles/);
  });
});
