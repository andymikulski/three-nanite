import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Float32BufferAttribute,
  InterleavedBuffer,
  InterleavedBufferAttribute,
  SphereGeometry,
} from "three";
import { describe, expect, it } from "vitest";
import {
  createLodChain,
  createLodGeometry,
  geometryToMeshletInput,
  getLodErrors,
} from "../src/three/geometry.js";
import { FIXTURE_ERROR, makeFixture } from "./helpers/fixture.js";

function expectPositionFirst(input: ReturnType<typeof geometryToMeshletInput>): void {
  const p = input.attributes[0];
  expect(p.name).toBe("position");
  expect(p.array).toBeInstanceOf(Float32Array);
  expect(p.itemSize).toBe(3);
  expect(p.normalized).toBe(false);
}

/** Triangle corner positions (flattened) in index order. */
function trianglePositions(g: BufferGeometry): number[] {
  const pos = g.getAttribute("position");
  const idx = g.getIndex()!;
  const out: number[] = [];
  for (let i = 0; i < idx.count; i++) {
    const v = idx.getX(i);
    out.push(pos.getX(v), pos.getY(v), pos.getZ(v));
  }
  return out;
}

describe("geometryToMeshletInput", () => {
  it("converts an indexed BoxGeometry", () => {
    const box = new BoxGeometry();
    const input = geometryToMeshletInput(box);
    expectPositionFirst(input);
    expect(input.attributes.map((a) => a.name)).toEqual(["position", "normal", "uv"]);
    expect(input.attributes[0].array).toBe(box.getAttribute("position").array);
    expect(input.attributes[1].array).toBe(box.getAttribute("normal").array);
    expect(input.attributes[2].itemSize).toBe(2);
    expect(input.indices).toBe(box.getIndex()!.array);
  });

  it("converts an indexed SphereGeometry with a Uint16 index", () => {
    const sphere = new SphereGeometry(1, 16, 12);
    const input = geometryToMeshletInput(sphere);
    expectPositionFirst(input);
    expect(input.indices).toBeInstanceOf(Uint16Array);
    expect(input.indices!.length).toBe(sphere.getIndex()!.count);
  });

  it("returns null indices for non-indexed geometry", () => {
    const input = geometryToMeshletInput(new BoxGeometry().toNonIndexed());
    expectPositionFirst(input);
    expect(input.indices).toBeNull();
    expect(input.attributes.length).toBe(3);
  });

  it("converts non-Uint16/32 indices to Uint32Array", () => {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
    g.setIndex(new BufferAttribute(new Uint8Array([0, 1, 2]), 1));
    const input = geometryToMeshletInput(g);
    expect(input.indices).toBeInstanceOf(Uint32Array);
    expect(Array.from(input.indices!)).toEqual([0, 1, 2]);
  });

  it("de-interleaves interleaved attributes", () => {
    // per vertex: pos(3) uv(2) pad(1)
    const stride = 6;
    const raw = new Float32Array([
      0, 0, 0, 0.0, 0.0, 99,
      1, 0, 0, 1.0, 0.0, 99,
      0, 1, 0, 0.0, 1.0, 99,
      1, 1, 0, 1.0, 1.0, 99,
    ]);
    const buffer = new InterleavedBuffer(raw, stride);
    const g = new BufferGeometry();
    g.setAttribute("uv", new InterleavedBufferAttribute(buffer, 2, 3));
    g.setAttribute("position", new InterleavedBufferAttribute(buffer, 3, 0));
    g.setIndex([0, 1, 2, 2, 1, 3]);

    const input = geometryToMeshletInput(g);
    expectPositionFirst(input);
    expect(input.attributes.map((a) => a.name)).toEqual(["position", "uv"]);
    expect(Array.from(input.attributes[0].array)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    expect(input.attributes[0].array.length).toBe(12);
    expect(Array.from(input.attributes[1].array)).toEqual([0, 0, 1, 0, 0, 1, 1, 1]);
    expect(input.attributes[1].itemSize).toBe(2);
    expect(input.attributes[1].array).not.toBe(raw);
  });

  it("keeps the element type of interleaved non-float attributes and normalized flags", () => {
    const raw = new Uint8Array([10, 20, 30, 255, 1, 2, 3, 128, 4, 5, 6, 0]);
    const buffer = new InterleavedBuffer(raw, 4);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
    g.setAttribute("color", new InterleavedBufferAttribute(buffer, 3, 1, true));
    const input = geometryToMeshletInput(g);
    const color = input.attributes[1];
    expect(color.array).toBeInstanceOf(Uint8Array);
    expect(color.normalized).toBe(true);
    expect(color.itemSize).toBe(3);
    expect(Array.from(color.array)).toEqual([20, 30, 255, 2, 3, 128, 5, 6, 0]);
  });

  it("keeps a normalized Uint8 color attribute as-is", () => {
    const g = new BoxGeometry();
    const colors = new Uint8Array(24 * 4).fill(200);
    g.setAttribute("color", new BufferAttribute(colors, 4, true));
    const input = geometryToMeshletInput(g);
    const color = input.attributes.find((a) => a.name === "color")!;
    expect(color.array).toBe(colors);
    expect(color.normalized).toBe(true);
    expect(color.itemSize).toBe(4);
    // stable order: normal, uv, color
    expect(input.attributes.map((a) => a.name)).toEqual(["position", "normal", "uv", "color"]);
  });

  it("orders attributes: known names first, then alphabetically", () => {
    const g = new BoxGeometry();
    const n = g.getAttribute("position").count;
    for (const name of ["zeta", "alpha", "uv1", "tangent", "color"]) {
      g.setAttribute(name, new Float32BufferAttribute(new Float32Array(n * 2), 2));
    }
    expect(geometryToMeshletInput(g).attributes.map((a) => a.name)).toEqual([
      "position", "normal", "tangent", "uv", "uv1", "color", "alpha", "zeta",
    ]);
  });

  it("converts normalized integer positions to Float32", () => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(new Int16Array([32767, 0, -32767, 0, 32767, 0, -32768, 0, 0]), 3, true));
    const input = geometryToMeshletInput(g);
    expectPositionFirst(input);
    expect(input.attributes[0].array[0]).toBeCloseTo(1, 5);
    expect(input.attributes[0].array[2]).toBeCloseTo(-1, 5);
    expect(input.attributes[0].array[6]).toBe(-1);
  });

  it("converts non-normalized integer positions to Float32", () => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(new Int16Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3));
    const input = geometryToMeshletInput(g);
    expectPositionFirst(input);
    expect(Array.from(input.attributes[0].array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("honours the attributes filter", () => {
    const box = new BoxGeometry();
    expect(geometryToMeshletInput(box, { attributes: ["uv"] }).attributes.map((a) => a.name)).toEqual(["position", "uv"]);
    expect(geometryToMeshletInput(box, { attributes: [] }).attributes.map((a) => a.name)).toEqual(["position"]);
    expect(geometryToMeshletInput(box, { attributes: ["normal", "uv", "missing"] }).attributes.length).toBe(3);
  });

  it("throws without a position attribute", () => {
    const g = new BufferGeometry();
    g.setAttribute("normal", new Float32BufferAttribute([0, 0, 1], 3));
    expect(() => geometryToMeshletInput(g)).toThrow(/no position attribute/);
  });
});

describe("createLodGeometry", () => {
  const data = makeFixture();

  it("error 0 selects the fine clusters", () => {
    const g = createLodGeometry(data, 0);
    expect(g.getIndex()!.count / 3).toBe(128);
    const expected = Array.from(data.indices.subarray(0, 4 * 96));
    expect(Array.from(g.getIndex()!.array)).toEqual(expected);
  });

  it("error at or above the group error selects the coarse cluster", () => {
    for (const err of [FIXTURE_ERROR, 1, Infinity]) {
      const g = createLodGeometry(data, err);
      expect(g.getIndex()!.count / 3).toBe(32);
      expect(Array.from(g.getIndex()!.array)).toEqual(Array.from(data.indices.subarray(4 * 96, 4 * 96 + 96)));
    }
    // float32 error of group 0 is slightly above the double 0.05, so a strictly smaller value refines
    expect(createLodGeometry(data, 0.04).getIndex()!.count / 3).toBe(128);
  });

  it("uses Uint16Array indices for small meshes", () => {
    expect(createLodGeometry(data, 0).getIndex()!.array).toBeInstanceOf(Uint16Array);
    expect(createLodGeometry(data, 0, { compact: true }).getIndex()!.array).toBeInstanceOf(Uint16Array);
  });

  it("uses Uint32Array indices above 65536 vertices", () => {
    const big = makeFixture();
    const n = 70000;
    big.vertexCount = n;
    big.attributes = [{ name: "position", array: new Float32Array(n * 3), itemSize: 3, normalized: false }];
    expect(createLodGeometry(big, 0).getIndex()!.array).toBeInstanceOf(Uint32Array);
    expect(createLodGeometry(big, 0, { compact: true }).getIndex()!.array).toBeInstanceOf(Uint16Array);
  });

  it("non-compact geometry shares the attribute arrays", () => {
    const a = createLodGeometry(data, 0);
    const b = createLodGeometry(data, Infinity);
    data.attributes.forEach((attr) => {
      expect(a.getAttribute(attr.name).array).toBe(attr.array);
      expect(b.getAttribute(attr.name).array).toBe(attr.array);
      expect(a.getAttribute(attr.name).itemSize).toBe(attr.itemSize);
    });
    expect(a.getAttribute("position").count).toBe(81);
  });

  it("compact geometry keeps only referenced vertices", () => {
    const full = createLodGeometry(data, Infinity);
    const compact = createLodGeometry(data, Infinity, { compact: true });
    // coarse level uses every second vertex of a 9x9 grid: 5x5
    expect(compact.getAttribute("position").count).toBe(25);
    expect(compact.getAttribute("normal").count).toBe(25);
    expect(compact.getAttribute("uv").count).toBe(25);
    expect(compact.getAttribute("uv").array.length).toBe(50);
    expect(compact.getAttribute("position").array).toBeInstanceOf(Float32Array);
    expect(trianglePositions(compact)).toEqual(trianglePositions(full));

    // first-use order: indices of the first triangle are 0, 1, 2
    expect(Array.from(compact.getIndex()!.array.subarray(0, 3))).toEqual([0, 1, 2]);

    const fineFull = createLodGeometry(data, 0);
    const fineCompact = createLodGeometry(data, 0, { compact: true });
    expect(fineCompact.getAttribute("position").count).toBe(81);
    expect(trianglePositions(fineCompact)).toEqual(trianglePositions(fineFull));
  });

  it("compact geometry does not modify the source data", () => {
    const fresh = makeFixture();
    createLodGeometry(fresh, Infinity, { compact: true });
    expect(Array.from(fresh.indices)).toEqual(Array.from(makeFixture().indices));
  });

  it("sets bounding volumes from the data", () => {
    const g = createLodGeometry(data, 0);
    expect(g.boundingSphere!.radius).toBeCloseTo(Math.SQRT2, 5);
    expect(g.boundingSphere!.center.toArray()).toEqual([0, 0, 0]);
    expect(g.boundingBox!.min.toArray()).toEqual([-1, -1, 0]);
    expect(g.boundingBox!.max.toArray()).toEqual([1, 1, 0]);
  });
});

describe("getLodErrors", () => {
  it("returns the finite per-depth maxima", () => {
    const errors = getLodErrors(makeFixture());
    expect(errors.length).toBe(1);
    expect(errors[0]).toBeCloseTo(FIXTURE_ERROR, 6);
  });

  it("is strictly increasing and skips non-increasing / empty depths", () => {
    const d = makeFixture();
    d.groupCount = 5;
    d.levelCount = 4;
    d.groupError = new Float32Array([0.5, 0.2, 0.2, Infinity, 0.9]);
    d.groupDepth = new Uint16Array([0, 0, 1, 3, 3]);
    // depth0 max 0.5, depth1 max 0.2 (dropped), depth2 empty, depth3 max finite 0.9
    const errors = getLodErrors(d);
    expect(errors.length).toBe(2);
    expect(errors[0]).toBeCloseTo(0.5, 6);
    expect(errors[1]).toBeCloseTo(0.9, 6);
  });
});

describe("createLodChain", () => {
  it("builds original + one geometry per error", () => {
    const chain = createLodChain(makeFixture());
    expect(chain.length).toBe(2);
    expect(chain.map((g) => g.getIndex()!.count / 3)).toEqual([128, 32]);
    expect(chain.map((g) => g.userData.triangleCount)).toEqual([128, 32]);
    expect(chain[0].userData.lodError).toBe(0);
    expect(chain[1].userData.lodError).toBeCloseTo(FIXTURE_ERROR, 6);
  });

  it("supports compact and maxLevels", () => {
    const compact = createLodChain(makeFixture(), { compact: true });
    expect(compact[1].getAttribute("position").count).toBe(25);
    expect(createLodChain(makeFixture(), { maxLevels: 1 }).length).toBe(1);
    expect(createLodChain(makeFixture(), { maxLevels: 5 }).length).toBe(2);
  });
});
