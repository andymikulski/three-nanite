import { Box3, BufferAttribute, BufferGeometry, Sphere, Vector3 } from "three";
import type { InterleavedBufferAttribute } from "three";
import { ClusterSelector, getLodErrors } from "../core/select.js";
import type { MeshletAttribute, MeshletData, MeshletInput, TypedArray } from "../types.js";

export { getLodErrors };

/** Array types three.js may hand us; wider than {@link TypedArray}. */
type ThreeArray = TypedArray | Uint8ClampedArray | Float64Array;

/** Order in which well-known attributes are emitted; everything else follows alphabetically. */
const ATTRIBUTE_ORDER = ["normal", "tangent", "uv", "uv1", "uv2", "uv3", "color"];

/** Allocates a zeroed array of the same element type as `like`. */
function allocLike(like: TypedArray, length: number): TypedArray {
  if (like instanceof Float32Array) return new Float32Array(length);
  if (like instanceof Uint32Array) return new Uint32Array(length);
  if (like instanceof Uint16Array) return new Uint16Array(length);
  if (like instanceof Uint8Array) return new Uint8Array(length);
  if (like instanceof Int32Array) return new Int32Array(length);
  if (like instanceof Int16Array) return new Int16Array(length);
  return new Int8Array(length);
}

/** Maps an array onto one of the element types a MeshletAttribute supports (copying only if it must). */
function toSupported(array: ThreeArray): TypedArray {
  if (array instanceof Uint8ClampedArray) return new Uint8Array(array.buffer, array.byteOffset, array.length);
  if (array instanceof Float64Array) return Float32Array.from(array);
  return array;
}

/** Integer range used to expand a normalized component to [0, 1] / [-1, 1] (1 for float arrays). */
function normalizationDivisor(array: ThreeArray): number {
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray) return 255;
  if (array instanceof Uint16Array) return 65535;
  if (array instanceof Uint32Array) return 4294967295;
  if (array instanceof Int8Array) return 127;
  if (array instanceof Int16Array) return 32767;
  if (array instanceof Int32Array) return 2147483647;
  return 1;
}

/**
 * Converts a three.js BufferGeometry into builder input.
 *
 * - `attributes[0]` is always `position`, as a tightly packed `Float32Array`
 *   with `itemSize` 3 (other element types are converted, expanding normalized
 *   integers to floats).
 * - The other attributes follow in a stable order: `normal`, `tangent`, `uv`,
 *   `uv1`, `uv2`, `uv3`, `color`, then any others alphabetically. They keep
 *   their element type, `itemSize` and `normalized` flag. Interleaved
 *   attributes are de-interleaved into tight arrays; plain tight attributes
 *   reuse their array without copying.
 * - `indices` is the geometry's index (a `Uint16Array` or `Uint32Array`, other
 *   types are converted to `Uint32Array`), or `null` for non-indexed geometry.
 *
 * Morph attributes, geometry groups and `drawRange` are ignored.
 *
 * @param options.attributes Names of the non-position attributes to keep (default: all).
 * @throws Error if the geometry has no usable `position` attribute.
 */
export function geometryToMeshletInput(
  geometry: BufferGeometry,
  options: { attributes?: string[] } = {},
): MeshletInput {
  // three's attribute typing also admits GLBufferAttribute, which has no CPU data
  const source = (name: string): BufferAttribute | InterleavedBufferAttribute => {
    const attr = geometry.getAttribute(name) as BufferAttribute | InterleavedBufferAttribute;
    if (!(attr as { array?: unknown }).array) {
      throw new Error(`three-nanite: attribute "${name}" has no CPU-side array`);
    }
    return attr;
  };

  const positionSource = geometry.getAttribute("position") as BufferAttribute | InterleavedBufferAttribute | undefined;
  if (!positionSource) throw new Error("three-nanite: geometry has no position attribute");
  if (positionSource.itemSize < 2) throw new Error("three-nanite: position attribute needs at least 2 components");
  const vertexCount = positionSource.count;

  const attributes: MeshletAttribute[] = [
    { name: "position", array: extractPosition(source("position")), itemSize: 3, normalized: false },
  ];

  const wanted = options.attributes ? new Set(options.attributes) : null;
  const names = Object.keys(geometry.attributes).filter(
    (n) => n !== "position" && (!wanted || wanted.has(n)),
  );
  const rank = (n: string): number => {
    const i = ATTRIBUTE_ORDER.indexOf(n);
    return i < 0 ? ATTRIBUTE_ORDER.length : i;
  };
  names.sort((a, b) => rank(a) - rank(b) || (a < b ? -1 : a > b ? 1 : 0));

  for (const name of names) {
    const attr = source(name);
    if (attr.count !== vertexCount) {
      throw new Error(`three-nanite: attribute "${name}" has ${attr.count} items, position has ${vertexCount}`);
    }
    attributes.push({ name, array: extractAttribute(attr), itemSize: attr.itemSize, normalized: attr.normalized });
  }

  let indices: Uint16Array | Uint32Array | null = null;
  const index = geometry.getIndex();
  if (index) {
    const array = index.array;
    indices =
      array instanceof Uint16Array || array instanceof Uint32Array ? array : Uint32Array.from(array);
  }

  return { attributes, indices };
}

function isInterleaved(attr: BufferAttribute | InterleavedBufferAttribute): attr is InterleavedBufferAttribute {
  return (attr as InterleavedBufferAttribute).isInterleavedBufferAttribute === true;
}

/** Tight copy of any attribute as Float32 xyz. */
function extractPosition(attr: BufferAttribute | InterleavedBufferAttribute): Float32Array {
  const interleaved = isInterleaved(attr);
  if (!interleaved && attr.array instanceof Float32Array) {
    const need = attr.count * attr.itemSize;
    if (attr.itemSize === 3) return attr.array.length === need ? attr.array : attr.array.subarray(0, need);
  }

  const src = attr.array;
  const stride = interleaved ? attr.data.stride : attr.itemSize;
  const offset = interleaved ? attr.offset : 0;
  const comps = Math.min(attr.itemSize, 3);
  const scale = attr.normalized ? 1 / normalizationDivisor(src) : 1;
  const signed = src instanceof Int8Array || src instanceof Int16Array || src instanceof Int32Array;

  const out = new Float32Array(attr.count * 3);
  for (let i = 0; i < attr.count; i++) {
    const s = i * stride + offset;
    for (let k = 0; k < comps; k++) {
      const v = src[s + k] * scale;
      out[i * 3 + k] = signed && v < -1 && attr.normalized ? -1 : v;
    }
  }
  return out;
}

/** Tight array of the attribute's own element type; reuses the source array when it is already tight. */
function extractAttribute(attr: BufferAttribute | InterleavedBufferAttribute): TypedArray {
  const src = toSupported(attr.array);
  const need = attr.count * attr.itemSize;
  if (!isInterleaved(attr)) return src.length === need ? src : src.subarray(0, need);

  const stride = attr.data.stride;
  const offset = attr.offset;
  const itemSize = attr.itemSize;
  const out = allocLike(src, need);
  for (let i = 0; i < attr.count; i++) {
    const s = i * stride + offset;
    const d = i * itemSize;
    for (let k = 0; k < itemSize; k++) out[d + k] = src[s + k];
  }
  return out;
}

/** Copies the listed vertices of `src` (`order[i]` = old index of new vertex i). */
function gather(src: MeshletAttribute, order: Uint32Array, count: number): TypedArray {
  const itemSize = src.itemSize;
  const from = src.array;
  const out = allocLike(from, count * itemSize);
  for (let i = 0; i < count; i++) {
    const s = order[i] * itemSize;
    const d = i * itemSize;
    for (let k = 0; k < itemSize; k++) out[d + k] = from[s + k];
  }
  return out;
}

/**
 * Builds static geometry for one view-independent LOD: the crack-free cut of
 * the cluster DAG whose accumulated error is at most `maxError` (mesh space
 * units; compared at float32 precision like the stored errors). `0` gives the original mesh, `Infinity` the coarsest available.
 *
 * By default the geometry's attributes are `BufferAttribute`s over the very
 * same arrays as `data.attributes` (no copy), so several LODs share vertex
 * memory on the CPU. With `compact: true`, unreferenced vertices are dropped
 * and attribute arrays and indices are rebuilt (first-use order).
 */
export function createLodGeometry(
  data: MeshletData,
  maxError: number,
  options: { compact?: boolean } = {},
): BufferGeometry {
  assertHasAttributes(data);
  const clusters = new Uint32Array(data.clusterCount);
  const n = new ClusterSelector(data).selectByError(clusters, Math.fround(maxError)); // errors are stored as float32
  return buildGeometry(data, clusters.subarray(0, n), options.compact === true);
}

function assertHasAttributes(data: MeshletData): void {
  if (data.attributes.length === 0) {
    throw new Error("three-nanite: this MeshletData holds no vertex attributes (hierarchy only)");
  }
}

function buildGeometry(data: MeshletData, selected: Uint32Array, compact: boolean): BufferGeometry {
  // ascending cluster ids: deterministic output and best index locality
  selected.sort();

  let indexCount = 0;
  for (let i = 0; i < selected.length; i++) indexCount += data.clusterIndexCount[selected[i]];

  const source = data.indices;
  const indices = new Uint32Array(indexCount);
  let w = 0;
  for (let i = 0; i < selected.length; i++) {
    const c = selected[i];
    const start = data.clusterIndexStart[c];
    indices.set(source.subarray(start, start + data.clusterIndexCount[c]), w);
    w += data.clusterIndexCount[c];
  }

  const geometry = new BufferGeometry();
  let vertexCount = data.vertexCount;

  if (compact) {
    const remap = new Uint32Array(data.vertexCount).fill(0xffffffff);
    const order = new Uint32Array(data.vertexCount);
    let count = 0;
    for (let i = 0; i < indices.length; i++) {
      const v = indices[i];
      let m = remap[v];
      if (m === 0xffffffff) {
        m = count;
        remap[v] = count;
        order[count++] = v;
      }
      indices[i] = m;
    }
    vertexCount = count;
    for (const a of data.attributes) {
      geometry.setAttribute(a.name, new BufferAttribute(gather(a, order, count), a.itemSize, a.normalized));
    }
  } else {
    for (const a of data.attributes) {
      geometry.setAttribute(a.name, new BufferAttribute(a.array, a.itemSize, a.normalized));
    }
  }

  const index16 = vertexCount <= 65536 ? Uint16Array.from(indices) : null;
  geometry.setIndex(new BufferAttribute(index16 ?? indices, 1));

  const s = data.boundingSphere;
  geometry.boundingSphere = new Sphere(new Vector3(s[0], s[1], s[2]), s[3]);
  const b = data.boundingBox;
  geometry.boundingBox = new Box3(new Vector3(b[0], b[1], b[2]), new Vector3(b[3], b[4], b[5]));
  return geometry;
}

/**
 * Convenience LOD chain: the original mesh followed by one geometry per
 * {@link getLodErrors} entry (finest to coarsest). Each geometry carries
 * `userData.lodError` and `userData.triangleCount`. Levels whose triangle
 * count is not smaller than the previous kept level are skipped.
 *
 * @param options.compact   Passed to {@link createLodGeometry}.
 * @param options.maxLevels Upper bound on the chain length; the original and
 *                          the coarsest level are always kept, the rest are
 *                          evenly spaced among the candidates.
 */
export function createLodChain(
  data: MeshletData,
  options: { compact?: boolean; maxLevels?: number } = {},
): BufferGeometry[] {
  const selector = new ClusterSelector(data);
  const scratch = new Uint32Array(data.clusterCount);

  const candidates: { error: number; triangles: number }[] = [];
  for (const error of [0, ...getLodErrors(data)]) {
    const n = selector.selectByError(scratch, error);
    let indexTotal = 0;
    for (let i = 0; i < n; i++) indexTotal += data.clusterIndexCount[scratch[i]];
    const triangles = indexTotal / 3;
    if (candidates.length > 0 && triangles >= candidates[candidates.length - 1].triangles) continue;
    candidates.push({ error, triangles });
  }

  let kept = candidates;
  const max = options.maxLevels;
  if (max !== undefined && max < candidates.length) {
    if (max <= 1) {
      kept = candidates.slice(0, 1);
    } else {
      kept = [];
      for (let i = 0; i < max; i++) kept.push(candidates[Math.round((i * (candidates.length - 1)) / (max - 1))]);
    }
  }

  return kept.map(({ error, triangles }) => {
    const geometry = createLodGeometry(data, error, { compact: options.compact });
    geometry.userData.lodError = error;
    geometry.userData.triangleCount = triangles;
    return geometry;
  });
}
