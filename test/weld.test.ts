import { describe, expect, it } from "vitest";
import { TorusKnotGeometry } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshlets, unweldMeshlets, weldVertices } from "../src/build/index.js";
import { geometryToMeshletInput } from "../src/three/geometry.js";
import { validateMeshletData } from "../src/core/validate.js";
import type { MeshletAttribute, MeshletInput } from "../src/types.js";

function knotSoup(): { soup: MeshletInput; merged: number } {
  const geometry = new TorusKnotGeometry(1, 0.35, 100, 16);
  const merged = mergeVertices(geometry).getAttribute("position").count;
  return { soup: geometryToMeshletInput(geometry.toNonIndexed()), merged };
}

/** Compares vertex `a` of `x` with vertex `b` of `y`, attribute by attribute. */
function sameVertex(x: MeshletInput, a: number, y: MeshletInput, b: number): boolean {
  return x.attributes.every((attr, i) => {
    const other = y.attributes[i];
    for (let k = 0; k < attr.itemSize; k++) {
      if (!Object.is(attr.array[a * attr.itemSize + k], other.array[b * attr.itemSize + k])) return false;
    }
    return true;
  });
}

function attr(name: string, array: MeshletAttribute["array"], itemSize: number, normalized = false): MeshletAttribute {
  return { name, array, itemSize, normalized };
}

describe("weldVertices", () => {
  it("welds a non-indexed torus knot back to the vertex count of mergeVertices", () => {
    const { soup, merged } = knotSoup();
    expect(soup.indices).toBeNull();
    expect(soup.attributes.map((a) => a.name)).toEqual(["position", "normal", "uv"]);

    const { input, source } = weldVertices(soup);
    const welded = input.attributes[0].array.length / 3;
    expect(welded).toBe(merged);
    expect(welded).toBeLessThan(soup.attributes[0].array.length / 3);
    expect(source.length).toBe(welded);
    expect(input.indices).toBeInstanceOf(Uint32Array);
    expect(input.indices!.length).toBe(soup.attributes[0].array.length / 3);
    for (let i = 0; i < input.indices!.length; i++) expect(input.indices![i]).toBeLessThan(welded);
  });

  it("points source at a soup vertex with the same attributes, and keeps triangles intact", () => {
    const { soup } = knotSoup();
    const { input, source } = weldVertices(soup);
    const welded = source.length;

    // first-occurrence order: ids appear in increasing order of first use
    let next = 0;
    for (const id of input.indices!) {
      if (id === next) next++;
      else expect(id).toBeLessThan(next);
    }
    expect(next).toBe(welded);

    for (let i = 0; i < welded; i++) expect(sameVertex(input, i, soup, source[i])).toBe(true);
    // source is the FIRST soup vertex of its kind
    for (let i = 1; i < welded; i++) expect(source[i]).toBeGreaterThan(source[i - 1]);

    // every welded triangle corner equals the soup corner it came from
    for (let k = 0; k < input.indices!.length; k++) {
      expect(sameVertex(input, input.indices![k], soup, k)).toBe(true);
    }
  });

  it("handles indexed input and drops unreferenced vertices", () => {
    // vertices 0 and 3 are identical, vertex 4 is unused
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 5, 5, 1, 1, 0]);
    const out = weldVertices({
      attributes: [attr("position", position, 3)],
      indices: new Uint16Array([0, 1, 2, 3, 2, 5]),
    });
    expect(Array.from(out.input.indices!)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(out.source)).toEqual([0, 1, 2, 5]);
    expect(Array.from(out.input.attributes[0].array)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
  });

  it("does not weld vertices that differ in any attribute", () => {
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const uv = new Float32Array([0, 0, 1, 0, 0, 1, 0.5, 0.5, 1, 0, 0, 1]);
    const out = weldVertices({ attributes: [attr("position", position, 3), attr("uv", uv, 2)] });
    expect(out.source.length).toBe(4);
    expect(out.input.indices!.length).toBe(6);
  });

  it("compares bits: 0 and -0 stay distinct", () => {
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, -0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const out = weldVertices({ attributes: [attr("position", position, 3)] });
    expect(out.source.length).toBe(4);
  });

  it("drops a triangle that becomes degenerate", () => {
    // second triangle: corners 3 and 4 are the same point, so it collapses
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3]);
    const out = weldVertices({ attributes: [attr("position", position, 3)] });
    expect(out.input.indices!.length).toBe(3);
    expect(out.source.length).toBe(3);
    expect(Array.from(out.source)).toEqual([0, 1, 2]);
  });

  it("preserves attribute classes, itemSize, name and normalized", () => {
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const color = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    const id = new Uint16Array([7, 8, 9, 7, 8, 9]);
    const out = weldVertices({
      attributes: [attr("position", position, 3), attr("color", color, 4, true), attr("id", id, 1)],
    });
    expect(out.source.length).toBe(3);
    const [p, c, i] = out.input.attributes;
    expect(p.array).toBeInstanceOf(Float32Array);
    expect(c.array).toBeInstanceOf(Uint8Array);
    expect(c.array).not.toBeInstanceOf(Float32Array);
    expect(i.array).toBeInstanceOf(Uint16Array);
    expect([c.name, c.itemSize, c.normalized]).toEqual(["color", 4, true]);
    expect([i.name, i.itemSize, i.normalized]).toEqual(["id", 1, false]);
    expect(c.array.length).toBe(12);
    expect(Array.from(c.array)).toEqual([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
  });

  it("honors the byteOffset of attribute views", () => {
    const backing = new Float32Array(30);
    backing.set([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], 5);
    const position = backing.subarray(5, 23);
    const out = weldVertices({ attributes: [attr("position", position, 3)] });
    expect(out.source.length).toBe(3);
  });

  it("rejects malformed input", () => {
    const position = new Float32Array(9);
    expect(() => weldVertices({ attributes: [attr("position", position, 3)], indices: new Uint32Array([0, 1, 3]) })).toThrow(
      "index out of range",
    );
    expect(() => weldVertices({ attributes: [attr("position", new Float32Array(12), 3)] })).toThrow("divisible by 3");
    expect(() => weldVertices({ attributes: [attr("position", position, 3), attr("uv", new Float32Array(4), 2)] })).toThrow(
      "does not match",
    );
  });
});

describe("weld -> buildMeshlets -> unweldMeshlets", () => {
  it("expresses the hierarchy in the soup's own vertex numbering", async () => {
    const { soup } = knotSoup();
    const soupVertices = soup.attributes[0].array.length / 3;
    const { input, source } = weldVertices(soup);
    const welded = await buildMeshlets(input);
    const weldedIndices = welded.indices;
    const weldedPositions = welded.attributes[0].array;

    const data = unweldMeshlets(welded, source, soup);
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.vertexCount).toBe(soupVertices);
    expect(data.attributes[0].name).toBe("position");
    expect(data.attributes[0].array).toBe(soup.attributes[0].array);
    expect(data.indices).not.toBe(welded.indices);
    expect(data.indices).toBeInstanceOf(Uint32Array);
    expect(data.indices.length).toBe(weldedIndices.length);
    for (let i = 0; i < data.indices.length; i++) expect(data.indices[i]).toBeLessThan(soupVertices);

    // everything else is shared, not copied
    expect(data.groupError).toBe(welded.groupError);
    expect(data.clusterIndexStart).toBe(welded.clusterIndexStart);
    expect(data.rootGroups).toBe(welded.rootGroups);

    // the soup positions referenced equal the welded positions referenced before unwelding
    let seed = 12345;
    const random = (n: number): number => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed % n;
    };
    const soupPositions = soup.attributes[0].array;
    for (let t = 0; t < 40; t++) {
      const c = random(data.clusterCount);
      const tri = random(data.clusterIndexCount[c] / 3);
      const at = data.clusterIndexStart[c] + tri * 3;
      for (let k = 0; k < 3; k++) {
        const soupVertex = data.indices[at + k];
        const weldedVertex = weldedIndices[at + k];
        for (let axis = 0; axis < 3; axis++) {
          expect(soupPositions[soupVertex * 3 + axis]).toBe(weldedPositions[weldedVertex * 3 + axis]);
        }
      }
    }
  });

  it("rejects a source that does not match the hierarchy", async () => {
    const { soup } = knotSoup();
    const { input, source } = weldVertices(soup);
    const welded = await buildMeshlets(input);
    expect(() => unweldMeshlets(welded, source.subarray(1), soup)).toThrow("source has");
  });
});
