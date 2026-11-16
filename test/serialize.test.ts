import { describe, expect, it } from "vitest";
import { MeshBasicMaterial } from "three";
import { ClusterSelector } from "../src/core/select.js";
import { MESHLET_FILE_EXTENSION, deserializeMeshlets, serializeMeshlets } from "../src/core/serialize.js";
import { validateMeshletData } from "../src/core/validate.js";
import { createLodGeometry } from "../src/three/geometry.js";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
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

function readHeader(buf: ArrayBuffer): { indexBytes?: number; sections: { name: string; type: string }[] } {
  const headerLen = new DataView(buf).getUint32(8, true);
  return JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 16, headerLen)));
}

function selectedClusters(data: MeshletData, maxError: number): number[] {
  const out = new Uint32Array(data.clusterCount);
  const n = new ClusterSelector(data).selectByError(out, maxError);
  return Array.from(out.subarray(0, n)).sort((a, b) => a - b);
}

describe("hierarchy-only files", () => {
  it("round-trips with { attributes: false }", () => {
    const data = makeFixture();
    const buf = serializeMeshlets(data, { attributes: false });
    expect(readHeader(buf).sections.some((s) => "attribute" in s)).toBe(false);
    expect(buf.byteLength).toBeLessThan(serializeMeshlets(data).byteLength);

    const out = deserializeMeshlets(buf);
    expect(out.attributes).toEqual([]);
    expect(out.vertexCount).toBe(data.vertexCount);
    for (const f of ARRAY_FIELDS) expectSameArray(out[f], data[f]);
    expect(validateMeshletData(out)).toEqual([]);
  });

  it("still validates indices against vertexCount without attributes", () => {
    const out = deserializeMeshlets(serializeMeshlets(makeFixture(), { attributes: false }));
    out.vertexCount = 10;
    expect(validateMeshletData(out).join("; ")).toMatch(/out of range/);
  });

  it("selects the same clusters as the original", () => {
    const data = makeFixture();
    const out = deserializeMeshlets(serializeMeshlets(data, { attributes: false }));
    for (const error of [0, 0.01, 0.05, 1, Infinity]) {
      expect(selectedClusters(out, error)).toEqual(selectedClusters(data, error));
    }
    expect(selectedClusters(out, 0)).toEqual([0, 1, 2, 3]);
    expect(selectedClusters(out, Infinity)).toEqual([4]);
  });

  it("combines with compactIndices", () => {
    const data = makeFixture();
    const out = deserializeMeshlets(serializeMeshlets(data, { attributes: false, compactIndices: true }));
    expect(out.attributes).toEqual([]);
    expectSameArray(out.indices, data.indices);
  });

  it("cannot be rendered: MeshletMesh and createLodGeometry throw", () => {
    const hierarchyOnly = deserializeMeshlets(serializeMeshlets(makeFixture(), { attributes: false }));
    const message = "three-nanite: this MeshletData holds no vertex attributes (hierarchy only)";
    expect(() => new MeshletMesh(hierarchyOnly, new MeshBasicMaterial())).toThrow(message);
    expect(() => createLodGeometry(hierarchyOnly, 0)).toThrow(message);
  });

  it("rejects an attribute-less file whose first attribute is not a position", () => {
    const data = makeFixture();
    data.attributes.reverse();
    expect(() => deserializeMeshlets(serializeMeshlets(data))).toThrow(/first attribute/);
  });
});

describe("compact indices", () => {
  it("stores 16-bit indices and widens them on load", () => {
    const data = makeFixture();
    const plain = serializeMeshlets(data);
    const compact = serializeMeshlets(data, { compactIndices: true });

    expect(readHeader(compact).indexBytes).toBe(2);
    expect(readHeader(compact).sections.find((s) => s.name === "indices")?.type).toBe("u16");
    expect(readHeader(plain).indexBytes).toBeUndefined();
    expect(compact.byteLength).toBeLessThan(plain.byteLength);
    expect(compact.byteLength % 8).toBe(0);

    const out = deserializeMeshlets(compact);
    expect(out.indices).toBeInstanceOf(Uint32Array);
    expectSameData(out, data);
    expect(validateMeshletData(out)).toEqual([]);
  });

  it("widens misaligned 16-bit sections too", () => {
    const data = makeFixture();
    const file = new Uint8Array(serializeMeshlets(data, { compactIndices: true }));
    const host = new Uint8Array(file.length + 8);
    host.set(file, 3);
    expectSameData(deserializeMeshlets(host.subarray(3, 3 + file.length)), data);
  });

  it("stays 32-bit when vertexCount > 65536", () => {
    const data = makeFixture();
    data.vertexCount = 70000; // not a valid mesh, but serialization does not validate
    const buf = serializeMeshlets(data, { compactIndices: true, attributes: false });
    const header = readHeader(buf);
    expect(header.indexBytes).toBeUndefined();
    expect(header.sections.find((s) => s.name === "indices")?.type).toBe("u32");
    expectSameArray(deserializeMeshlets(buf).indices, data.indices);
  });

  it("uses 16-bit up to exactly 65536 vertices", () => {
    const data = makeFixture();
    data.vertexCount = 65536;
    expect(readHeader(serializeMeshlets(data, { compactIndices: true, attributes: false })).indexBytes).toBe(2);
  });

  it("rejects a header whose indexBytes disagrees with the section type", () => {
    const buf = serializeMeshlets(makeFixture(), { compactIndices: true });
    const bytes = new Uint8Array(buf);
    const text = new TextDecoder().decode(bytes.subarray(16, 16 + new DataView(buf).getUint32(8, true)));
    expect(text).toContain('"indexBytes":2');
    const patched = new Uint8Array(buf.slice(0));
    patched.set(new TextEncoder().encode(text.replace('"indexBytes":2', '"indexBytes":4')), 16);
    expect(() => deserializeMeshlets(patched)).toThrow(/^Corrupt meshlet file: /);
  });
});

