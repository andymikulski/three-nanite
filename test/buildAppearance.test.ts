import { beforeAll, describe, expect, it } from "vitest";
import { AppearanceCheck, type ColorMap } from "../src/build/appearance.js";
import { buildMeshlets } from "../src/build/buildMeshlets.js";
import { computeLodLevels, copyLevelIndices } from "../src/core/levels.js";
import type { MeshletData, MeshletInput } from "../src/types.js";
import { makeColorMap, roofBuilding } from "./helpers/roofBuilding.js";

interface Row {
  level: number;
  triangles: number;
  error: number;
  seen: number;
}

interface Measured {
  data: MeshletData;
  levels: ReturnType<typeof computeLodLevels>;
  rows: Row[];
}

/** The appearance error of every level of `data` against level 0. */
function measure(data: MeshletData, input: MeshletInput, colorMap: ColorMap | undefined): Measured {
  const levels = computeLodLevels(data, { maxLevels: 100000 });
  const indicesOf = (j: number): Uint32Array => {
    const out = new Uint32Array(levels.triangles[j] * 3);
    copyLevelIndices(data, levels, j, out);
    return out;
  };
  const check = AppearanceCheck.create(input.attributes, colorMap)!;
  expect(check).not.toBeNull();
  const positions = data.attributes[0].array as Float32Array;
  const base = indicesOf(0);
  check.prepare(base, positions, data.vertexCount, (l) => l);
  const rows: Row[] = [];
  for (let j = 1; j < levels.count; j++) {
    rows.push({ level: j, triangles: levels.triangles[j], error: levels.errors[j], seen: check.error(indicesOf(j), levels.errors[j]) });
  }
  return { data, levels, rows };
}

const slack = (error: number): number => error * 1.5 + 0.3;
const table = (name: string, m: Measured): string =>
  `${name}: level 0 has ${m.levels.triangles[0]} triangles\n` +
  m.rows.map((r) => `  level ${r.level}: triangles ${r.triangles}, error ${r.error.toFixed(4)}, seen ${r.seen.toFixed(4)}`).join("\n");

describe("buildMeshlets: colorMap and vertexColors", () => {
  const roof = roofBuilding();
  const colorMap = makeColorMap();
  let withMap: Measured;
  let without: Measured;

  beforeAll(async () => {
    const a = await buildMeshlets(roof.input, { colorMap });
    const b = await buildMeshlets(roof.input);
    withMap = measure(a, roof.input, colorMap);
    without = measure(b, roof.input, colorMap);
    console.log(table("with colorMap", withMap) + "\n" + table("without colorMap", without));
  }, 60_000);

  it("G: with a colour map no level changes more to the eye than its error says", () => {
    expect(withMap.rows.length).toBeGreaterThan(0);
    for (const r of withMap.rows) {
      expect(r.seen, `level ${r.level}: ${r.triangles} triangles, error ${r.error}`).toBeLessThanOrEqual(slack(r.error));
    }
  });

  // G, second part, left out on purpose: on this fixture the build WITHOUT a colour map is not worse.
  // It comes out identical to the one with it (same levels, errors 0.0029 / 0.0034 / 0.0076 / 2.236 /
  // 7.457 for 188 / 93 / 45 / 21 / 5 triangles, seen 0 for all but the last, 4.79 against 7.457),
  // because the unprotected-by-colour build already keeps the uv seams between floor and rim, and the
  // simplifier's own error is large wherever the roof closes. Also true at grid 12 and 24.
  it("the build without a colour map is the same here (the option changes nothing the fixture can show)", () => {
    expect(withMap.rows).toEqual(without.rows);
  });

  it("with the seams unprotected, the colour map makes the step that closes the roof cost more", async () => {
    const off = await buildMeshlets(roof.input, { protectAttributes: [] });
    const on = await buildMeshlets(roof.input, { protectAttributes: [], colorMap });
    const a = measure(off, roof.input, colorMap);
    const b = measure(on, roof.input, colorMap);
    console.log(table("protectAttributes [] without colorMap", a) + "\n" + table("protectAttributes [] with colorMap", b));
    for (const r of b.rows) expect(r.seen, `level ${r.level}`).toBeLessThanOrEqual(slack(r.error));
    // for each triangle count both reach, the colour map build never charges less
    for (const rb of b.rows) {
      const ra = a.rows.find((r) => r.triangles === rb.triangles);
      if (ra) expect(rb.error).toBeGreaterThanOrEqual(ra.error - 1e-6);
    }
    expect(b.rows.some((rb) => a.rows.some((ra) => ra.triangles === rb.triangles && rb.error > ra.error + 0.1))).toBe(true);
  });

  it("H: both builds end at few triangles", () => {
    for (const m of [withMap, without]) {
      expect(m.levels.triangles[m.levels.count - 1]).toBeLessThanOrEqual(64);
    }
  });

  it("H: the colour map build is never both coarser and cheaper than the other", () => {
    const a = withMap.levels;
    const b = without.levels;
    const n = Math.min(a.count, b.count);
    for (let j = 0; j < n; j++) {
      const notCoarser = a.errors[j] >= b.errors[j] - 1e-6;
      const moreTriangles = a.triangles[j] > b.triangles[j];
      expect(notCoarser || moreTriangles, `level ${j}: with (error ${a.errors[j]}, ${a.triangles[j]} tri), without (error ${b.errors[j]}, ${b.triangles[j]} tri)`).toBe(true);
    }
  });

  it("H: the largest error of the colour map build is not smaller", () => {
    const largest = (m: Measured): number => m.levels.errors[m.levels.count - 1];
    expect(largest(withMap)).toBeGreaterThanOrEqual(largest(without) * 0.99);
  });

  it("I: vertexColors judges the steps by the color attribute", async () => {
    const colored = roofBuilding({ vertexColors: true });
    const data = await buildMeshlets(colored.input, { vertexColors: true });
    const m = measure(data, colored.input, undefined);
    console.log(table("vertexColors", m));
    expect(m.rows.length).toBeGreaterThan(0);
    for (const r of m.rows) {
      expect(r.seen, `level ${r.level}: ${r.triangles} triangles, error ${r.error}`).toBeLessThanOrEqual(slack(r.error));
    }
  });

  it("J: a colour map for a mesh without uv changes nothing", async () => {
    const plain = roofBuilding({ noUv: true });
    expect(plain.input.attributes.map((a) => a.name)).toEqual(["position", "normal"]);
    const on = await buildMeshlets(plain.input, { colorMap });
    const off = await buildMeshlets(plain.input);
    expect(Array.from(on.indices)).toEqual(Array.from(off.indices));
    expect(Array.from(on.groupError)).toEqual(Array.from(off.groupError));
  });
});
