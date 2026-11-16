import type { MeshletAttribute, MeshletData, MeshletInput, TypedArray } from "../types.js";

/** Result of {@link weldVertices}. */
export interface WeldResult {
  /** The welded mesh: unique vertices in first-occurrence order, and an index over them. */
  input: MeshletInput;
  /** For each welded vertex, the index of the first vertex of the original input it was made from. */
  source: Uint32Array;
}

/** Allocates a zeroed array of the same element type as `like`. */
function allocLike(like: TypedArray, length: number): TypedArray {
  const Ctor = like.constructor as new (length: number) => TypedArray;
  return new Ctor(length);
}

/**
 * Welds vertices whose attributes are all bitwise identical. Works on indexed and unindexed input
 * (unindexed: vertices 0,1,2 are the first triangle, and so on). Triangles that become degenerate
 * (two corners welded to the same vertex) are dropped.
 *
 * Attributes are compared by their raw bytes, so `0` and `-0` are different and no float is converted.
 * Vertices are emitted in the order of their first use by a kept triangle; vertices that no kept
 * triangle uses are dropped. The output attributes keep the typed-array class, `itemSize`, `name` and
 * `normalized` of the input.
 *
 * @throws Error if an attribute length does not match the vertex count, the vertex count of an
 *         unindexed input is not divisible by 3, or an index is out of range.
 */
export function weldVertices(input: MeshletInput): WeldResult {
  const attributes = input.attributes;
  if (!attributes || attributes.length === 0) throw new Error("weldVertices: input has no attributes");
  const first = attributes[0];
  if (first.name !== "position" || !(first.array instanceof Float32Array) || first.itemSize !== 3) {
    throw new Error("weldVertices: attributes[0] must be 'position' (Float32Array, itemSize 3)");
  }
  const vertexCount = first.array.length / 3;
  if (!Number.isInteger(vertexCount)) throw new Error("weldVertices: malformed position array");
  for (const a of attributes) {
    if (a.array.length !== vertexCount * a.itemSize) {
      throw new Error(`weldVertices: attribute '${a.name}' length does not match vertex count`);
    }
  }

  const rawIndices = input.indices ?? null;
  const cornerCount = rawIndices ? rawIndices.length : vertexCount;
  if (cornerCount % 3 !== 0) throw new Error("weldVertices: index count must be divisible by 3");

  // ---- raw byte views --------------------------------------------------------
  const attributeCount = attributes.length;
  const bytes: Uint8Array[] = new Array(attributeCount);
  const strides = new Int32Array(attributeCount); // bytes per vertex of each attribute
  for (let a = 0; a < attributeCount; a++) {
    const array = attributes[a].array;
    bytes[a] = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    strides[a] = attributes[a].itemSize * array.BYTES_PER_ELEMENT;
  }

  // ---- pass 1: canonical original vertex for every vertex a triangle refers to ---------------
  // canon[v] = first vertex (in index order) whose attributes equal those of v; -1 = not seen yet
  const canon = new Int32Array(vertexCount).fill(-1);
  let tableSize = 2;
  while (tableSize < vertexCount * 2) tableSize *= 2;
  const mask = tableSize - 1;
  const table = new Int32Array(tableSize).fill(-1); // open addressing, linear probing; holds vertex ids

  const hashOf = (v: number): number => {
    let h = 0x811c9dc5;
    for (let a = 0; a < attributeCount; a++) {
      const b = bytes[a];
      const stride = strides[a];
      const end = (v + 1) * stride;
      for (let i = v * stride; i < end; i++) h = Math.imul(h ^ b[i], 0x01000193);
    }
    // avalanche, so that the low bits used for the table slot depend on every byte
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    h ^= h >>> 13;
    return h;
  };

  const equal = (u: number, v: number): boolean => {
    for (let a = 0; a < attributeCount; a++) {
      const b = bytes[a];
      const stride = strides[a];
      let i = u * stride;
      let j = v * stride;
      for (const end = i + stride; i < end; i++, j++) if (b[i] !== b[j]) return false;
    }
    return true;
  };

  const canonical = (v: number): number => {
    const known = canon[v];
    if (known >= 0) return known;
    let slot = hashOf(v) & mask;
    for (;;) {
      const other = table[slot];
      if (other < 0) {
        table[slot] = v;
        canon[v] = v;
        return v;
      }
      if (equal(other, v)) {
        canon[v] = other;
        return other;
      }
      slot = (slot + 1) & mask;
    }
  };

  const cornerVertex = (k: number): number => {
    const v = rawIndices ? rawIndices[k] : k;
    if (v >= vertexCount) throw new Error("weldVertices: index out of range");
    return v;
  };

  // ---- pass 2: emit kept triangles, numbering welded vertices on first use ---------------------
  const welded = new Int32Array(vertexCount).fill(-1); // canonical original vertex -> welded id
  const outIndices = new Uint32Array(cornerCount);
  const sourceIds = new Uint32Array(vertexCount);
  let outCorners = 0;
  let outVertices = 0;

  for (let k = 0; k < cornerCount; k += 3) {
    const a = canonical(cornerVertex(k));
    const b = canonical(cornerVertex(k + 1));
    const c = canonical(cornerVertex(k + 2));
    if (a === b || b === c || a === c) continue;

    for (let corner = 0; corner < 3; corner++) {
      const v = corner === 0 ? a : corner === 1 ? b : c;
      let id = welded[v];
      if (id < 0) {
        id = outVertices++;
        welded[v] = id;
        sourceIds[id] = v;
      }
      outIndices[outCorners++] = id;
    }
  }

  const source = sourceIds.slice(0, outVertices);
  const outAttributes: MeshletAttribute[] = attributes.map((attr) => {
    const itemSize = attr.itemSize;
    const from = attr.array;
    const to = allocLike(from, outVertices * itemSize);
    for (let i = 0; i < outVertices; i++) {
      const s = source[i] * itemSize;
      const d = i * itemSize;
      for (let k = 0; k < itemSize; k++) to[d + k] = from[s + k];
    }
    return { name: attr.name, array: to, itemSize, normalized: attr.normalized };
  });

  return {
    input: { attributes: outAttributes, indices: outIndices.slice(0, outCorners) },
    source,
  };
}

/**
 * Rewrites a hierarchy built on a welded mesh so that it indexes the original vertices again:
 * every index i becomes source[i], `attributes` become the original ones and `vertexCount` the original
 * count. Everything else (clusters, groups, bounds, errors) is shared with `data`, not copied.
 *
 * @param data     Hierarchy built from `weldVertices(original).input`.
 * @param source   `WeldResult.source` of that weld.
 * @param original The input that was passed to {@link weldVertices}.
 */
export function unweldMeshlets(data: MeshletData, source: Uint32Array, original: MeshletInput): MeshletData {
  const position = original.attributes[0];
  if (!position || position.name !== "position" || position.itemSize !== 3) {
    throw new Error("unweldMeshlets: original.attributes[0] must be 'position' with itemSize 3");
  }
  if (source.length !== data.vertexCount) {
    throw new Error(
      `unweldMeshlets: source has ${source.length} entries, the hierarchy has ${data.vertexCount} vertices` +
        (data.vertexCount > source.length
          ? " (it was built with dilateBorders, which adds vertices: a hierarchy that is mapped back must be built without it)"
          : ""),
    );
  }

  const indices = new Uint32Array(data.indices.length);
  for (let i = 0; i < indices.length; i++) indices[i] = source[data.indices[i]];

  return {
    ...data,
    vertexCount: position.array.length / 3,
    attributes: original.attributes.slice(),
    indices,
  };
}
