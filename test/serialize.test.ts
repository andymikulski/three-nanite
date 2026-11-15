import { describe, expect, it } from "vitest";
import { MESHLET_FILE_EXTENSION, deserializeMeshlets, serializeMeshlets } from "../src/core/serialize.js";
import type { MeshletData, TypedArray } from "../src/types.js";
import { makeFixture } from "./helpers/fixture.js";

const ARRAY_FIELDS = [
  "indices",
  "clusterIndexStart",
  "clusterIndexCount",
  "clusterBounds",
  "clusterGroup",
  "clusterRefined",
  "groupClusterStart",
  "groupClusterCount",
  "groupBounds",
  "groupError",
  "groupDepth",
  "rootGroups",
  "boundingSphere",
  "boundingBox",
] as const;

function expectSameArray(actual: TypedArray, expected: TypedArray): void {
  expect(actual.constructor).toBe(expected.constructor);
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(Object.is(actual[i], expected[i])).toBe(true);
}

function expectSameData(actual: MeshletData, expected: MeshletData): void {
  expect(actual.version).toBe(1);
  for (const k of ["vertexCount", "clusterCount", "groupCount", "levelCount"] as const) {
    expect(actual[k]).toBe(expected[k]);
  }
  expect(actual.attributes.length).toBe(expected.attributes.length);
  expected.attributes.forEach((e, i) => {
    const a = actual.attributes[i];
    expect(a.name).toBe(e.name);
    expect(a.itemSize).toBe(e.itemSize);
    expect(a.normalized).toBe(e.normalized);
    expectSameArray(a.array, e.array);
  });
  for (const f of ARRAY_FIELDS) expectSameArray(actual[f], expected[f]);
}

function fixtureWithExtras(): MeshletData {
  const data = makeFixture();
  const n = data.vertexCount;
  const color = new Uint8Array(n * 4);
  for (let i = 0; i < color.length; i++) color[i] = (i * 7) & 255;
  const ids = new Uint16Array(n * 2);
  for (let i = 0; i < ids.length; i++) ids[i] = i * 3;
  data.attributes.push(
    { name: "color", array: color, itemSize: 4, normalized: true },
    { name: "skinIndex", array: ids, itemSize: 2, normalized: false },
  );
  return data;
}

describe("serializeMeshlets / deserializeMeshlets", () => {
  it("exports the file extension", () => {
    expect(MESHLET_FILE_EXTENSION).toBe(".meshlets");
  });

  it("round-trips the fixture", () => {
    const data = makeFixture();
    const out = deserializeMeshlets(serializeMeshlets(data));
    expectSameData(out, data);
    expect(out.groupError[1]).toBe(Infinity);
    expect(out.attributes[0].name).toBe("position");
  });

  it("round-trips extra attributes with order, itemSize and normalized", () => {
    const data = fixtureWithExtras();
    const out = deserializeMeshlets(serializeMeshlets(data));
    expectSameData(out, data);
    expect(out.attributes.map((a) => a.name)).toEqual(["position", "normal", "uv", "color", "skinIndex"]);
    expect(out.attributes[3].array).toBeInstanceOf(Uint8Array);
    expect(out.attributes[3].normalized).toBe(true);
    expect(out.attributes[4].array).toBeInstanceOf(Uint16Array);
    expect(out.attributes[4].normalized).toBe(false);
  });

  it("writes the documented preamble", () => {
    const buf = serializeMeshlets(makeFixture());
    const bytes = new Uint8Array(buf);
    expect(String.fromCharCode(...bytes.subarray(0, 4))).toBe("MSHL");
    const dv = new DataView(buf);
    expect(dv.getUint32(4, true)).toBe(1);
    expect(dv.getUint32(8, true) % 8).toBe(0);
    expect(dv.getUint32(12, true)).toBe(0);
    expect(buf.byteLength % 8).toBe(0);
  });

  it("starts every section on an 8-byte boundary", () => {
    const buf = serializeMeshlets(fixtureWithExtras());
    const dv = new DataView(buf);
    const headerLen = dv.getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 16, headerLen))) as {
      sections: { name: string; offset: number; length: number; attribute?: boolean }[];
    };
    expect(header.sections.length).toBe(5 + 14);
    expect(header.sections[0]).toMatchObject({ name: "position", attribute: true, itemSize: 3, normalized: false });
    for (const s of header.sections) {
      expect(s.offset % 8).toBe(0);
      expect(s.offset).toBeGreaterThanOrEqual(16 + headerLen);
    }
  });

  it("is zero-copy on an ArrayBuffer", () => {
    const data = makeFixture();
    const buf = serializeMeshlets(data);
    const out = deserializeMeshlets(buf);
    expect(out.indices.buffer).toBe(buf);
    expect(out.groupError.buffer).toBe(buf);
    expect(out.attributes[0].array.buffer).toBe(buf);
    expect(out.attributes[2].array.buffer).toBe(buf);
  });

  it("is zero-copy on an aligned Uint8Array window", () => {
    const data = makeFixture();
    const file = new Uint8Array(serializeMeshlets(data));
    const host = new Uint8Array(file.length + 32);
    host.set(file, 16);
    const out = deserializeMeshlets(host.subarray(16, 16 + file.length));
    expect(out.indices.buffer).toBe(host.buffer);
    expect(out.indices.byteOffset % 4).toBe(0);
    expectSameData(out, data);
  });

  it("copies misaligned sections when the byteOffset is odd", () => {
    const data = fixtureWithExtras();
    const file = new Uint8Array(serializeMeshlets(data));
    const host = new Uint8Array(file.length + 8);
    host.set(file, 3);
    const out = deserializeMeshlets(host.subarray(3, 3 + file.length));
    expectSameData(out, data);
    expect(out.indices.buffer).not.toBe(host.buffer);
    // u8 sections need no alignment and may still be views
    expect(out.attributes[3].array.buffer).toBe(host.buffer);
  });

  it("accepts a Uint8Array with a non-zero even byteOffset", () => {
    const data = makeFixture();
    const file = new Uint8Array(serializeMeshlets(data));
    const host = new Uint8Array(file.length + 8);
    host.set(file, 2);
    expectSameData(deserializeMeshlets(host.subarray(2, 2 + file.length)), data);
  });

  it("rejects bad magic", () => {
    const bytes = new Uint8Array(serializeMeshlets(makeFixture()));
    bytes[0] = 0x58;
    expect(() => deserializeMeshlets(bytes)).toThrow("Not a three-nanite meshlet file");
    expect(() => deserializeMeshlets(new ArrayBuffer(2))).toThrow("Not a three-nanite meshlet file");
  });

  it("rejects an unsupported version", () => {
    const buf = serializeMeshlets(makeFixture());
    new DataView(buf).setUint32(4, 7, true);
    expect(() => deserializeMeshlets(buf)).toThrow("Unsupported meshlet file version 7");
  });

  it("rejects truncated buffers", () => {
    const buf = serializeMeshlets(makeFixture());
    expect(() => deserializeMeshlets(buf.slice(0, 10))).toThrow(/^Corrupt meshlet file: /);
    expect(() => deserializeMeshlets(buf.slice(0, 40))).toThrow(/^Corrupt meshlet file: /);
    expect(() => deserializeMeshlets(buf.slice(0, buf.byteLength - 8))).toThrow(/^Corrupt meshlet file: /);
    expect(() => deserializeMeshlets(buf.slice(0, buf.byteLength >> 1))).toThrow(/^Corrupt meshlet file: /);
  });

  it("rejects a garbled header", () => {
    const bytes = new Uint8Array(serializeMeshlets(makeFixture()));
    bytes[16] = 0x7e;
    expect(() => deserializeMeshlets(bytes)).toThrow(/^Corrupt meshlet file: /);
  });
});
