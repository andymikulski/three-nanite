import { describe, expect, it } from "vitest";
import { AppearanceCheck, type ColorMap } from "../src/build/appearance.js";
import type { MeshletAttribute } from "../src/types.js";
import { makeColorMap, roofBuilding } from "./helpers/roofBuilding.js";

const identity = (l: number): number => l;

function without(indices: Uint32Array, start: number, end: number): number[] {
  return [...indices.subarray(0, start), ...indices.subarray(end)];
}

describe("AppearanceCheck.create", () => {
  const roof = roofBuilding();
  const attributes = roof.input.attributes;

  it("returns null with no colour map and no color attribute", () => {
    expect(AppearanceCheck.create(attributes, undefined)).toBeNull();
  });

  it("returns null when the colour map names a uv attribute that is not there", () => {
    expect(AppearanceCheck.create(attributes, { ...makeColorMap(), attribute: "uv7" })).toBeNull();
    const noUv = attributes.filter((a) => a.name !== "uv");
    expect(AppearanceCheck.create(noUv, makeColorMap())).toBeNull();
  });

  it("returns an instance with a colour map and a uv attribute", () => {
    expect(AppearanceCheck.create(attributes, makeColorMap())).toBeInstanceOf(AppearanceCheck);
  });

  it("returns an instance with no map but a 3-component color attribute", () => {
    const colored = roofBuilding({ vertexColors: true }).input.attributes;
    expect(AppearanceCheck.create(colored, undefined)).toBeInstanceOf(AppearanceCheck);
    // a 2-component color is not enough
    const thin: MeshletAttribute[] = [colored[0], { name: "color", array: new Float32Array(roof.vertexCount * 2), itemSize: 2, normalized: false }];
    expect(AppearanceCheck.create(thin, undefined)).toBeNull();
  });
});

describe("AppearanceCheck", () => {
  it("B: the same triangles before and after is an error of exactly 0", () => {
    const roof = roofBuilding();
    const check = AppearanceCheck.create(roof.input.attributes, makeColorMap())!;
    expect(check.prepare(roof.indices, roof.positions, roof.vertexCount, identity)).toBe(true);
    expect(check.error(roof.indices, 0)).toBe(0);
  });

  it("C: one colour throughout is nothing to compare", () => {
    const roof = roofBuilding({ floorUsesWallUv: true });
    const check = AppearanceCheck.create(roof.input.attributes, makeColorMap())!;
    expect(check.prepare(roof.indices, roof.positions, roof.vertexCount, identity)).toBe(false);
    // closing the roof over its floor changes nothing then
    const lid = [...without(roof.indices, roof.floorStart, roof.floorEnd), ...lidTriangles(roof.rimInnerCorners)];
    expect(check.error(Uint32Array.from(lid), 0.4)).toBe(0);
  });

  it("D: a lid over the recessed floor is seen as a change about the size of the floor", () => {
    const roof = roofBuilding();
    const check = AppearanceCheck.create(roof.input.attributes, makeColorMap())!;
    expect(check.prepare(roof.indices, roof.positions, roof.vertexCount, identity)).toBe(true);
    const after = Uint32Array.from([...without(roof.indices, roof.floorStart, roof.floorEnd), ...lidTriangles(roof.rimInnerCorners)]);
    const seen = check.error(after, 0.4);
    expect(seen).toBeGreaterThanOrEqual(2);
    expect(seen).toBeLessThanOrEqual(6);
  });

  it("E: a shift of the floor that the geometric error explains is not counted", () => {
    const lift = 0.05;
    const roof = roofBuilding({ floorCopyLift: lift });
    const check = AppearanceCheck.create(roof.input.attributes, makeColorMap())!;
    expect(check.prepare(roof.indices, roof.positions, roof.vertexCount, identity)).toBe(true);
    const moved = Array.from(roof.indices);
    for (let i = roof.floorStart; i < roof.floorEnd; i++) moved[i] = roof.indices[i] - roof.floorFirstVertex + roof.floorCopyFirstVertex;
    expect(check.error(Uint32Array.from(moved), lift)).toBe(0);
  });

  describe("F: holes with alphaTest", () => {
    // a 4 x 4 quad at y = 0, uv 0..1: the colours of the map are only met between its corners, and a
    // coordinate of exactly 1 is the last texel, not the first of the next repeat
    const positions = new Float32Array([0, 0, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4]);
    const e = 1;
    const uv = new Float32Array([0, 0, e, 0, e, e, 0, e]);
    const attributes: MeshletAttribute[] = [
      { name: "position", array: positions, itemSize: 3, normalized: false },
      { name: "uv", array: uv, itemSize: 2, normalized: false },
    ];
    const quad = Uint32Array.from([0, 1, 2, 0, 2, 3]);
    // 2 x 2 pixels, two rows: left column / right column
    const pixels = (left: number[], right: number[]): Uint8Array => new Uint8Array([...left, ...right, ...left, ...right]);

    const removed = (map: ColorMap): { prepared: boolean; error: number } => {
      const check = AppearanceCheck.create(attributes, map)!;
      const prepared = check.prepare(quad, positions, 4, identity);
      return { prepared, error: check.error(new Uint32Array(0), 0) };
    };

    it("removing a half-hole quad costs less than removing an opaque one", () => {
      const holes = removed({ data: pixels([200, 180, 150, 255], [200, 180, 150, 0]), width: 2, height: 2, alphaTest: 0.5 });
      const opaque = removed({ data: pixels([200, 180, 150, 255], [40, 44, 48, 255]), width: 2, height: 2 });
      expect(holes.prepared).toBe(true);
      expect(opaque.prepared).toBe(true);
      expect(holes.error).toBeGreaterThan(0);
      expect(holes.error).toBeLessThan(opaque.error);
    });
  });
});

function lidTriangles(c: [number, number, number, number]): number[] {
  return [c[0], c[1], c[2], c[0], c[2], c[3]];
}
