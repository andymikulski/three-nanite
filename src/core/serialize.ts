import type { MeshletAttribute, MeshletData, TypedArray } from "../types.js";

/** Conventional file extension for serialized meshlet data. */
export const MESHLET_FILE_EXTENSION = ".meshlets";

/*
 * Container layout (little-endian):
 *
 *   0   4  magic "MSHL"
 *   4   4  uint32 format version (1)
 *   8   4  uint32 JSON header byte length (multiple of 8)
 *  12   4  uint32 reserved (0)
 *  16   N  JSON header, UTF-8, space padded
 *  ...     payload: every section on an 8-byte boundary (offsets are from the start of the file)
 */

const MAGIC = [0x4d, 0x53, 0x48, 0x4c]; // "MSHL"
const FORMAT_VERSION = 1;
const PREAMBLE_BYTES = 16;

type SectionType = "f32" | "u8" | "u16" | "u32" | "i8" | "i16" | "i32";

interface SectionEntry {
  name: string;
  type: SectionType;
  offset: number;
  length: number;
  attribute?: true;
  itemSize?: number;
  normalized?: boolean;
}

interface Header {
  version: number;
  vertexCount: number;
  clusterCount: number;
  groupCount: number;
  levelCount: number;
  sections: SectionEntry[];
}

const BYTES: Record<SectionType, number> = { f32: 4, u8: 1, u16: 2, u32: 4, i8: 1, i16: 2, i32: 4 };

/** The non-attribute sections and the element type each one must have. */
const FIXED_SECTIONS = {
  indices: "u32",
  clusterIndexStart: "u32",
  clusterIndexCount: "u32",
  clusterBounds: "f32",
  clusterGroup: "u32",
  clusterRefined: "i32",
  groupClusterStart: "u32",
  groupClusterCount: "u32",
  groupBounds: "f32",
  groupError: "f32",
  groupDepth: "u16",
  rootGroups: "u32",
  boundingSphere: "f32",
  boundingBox: "f32",
} as const satisfies Record<string, SectionType>;

type FixedName = keyof typeof FIXED_SECTIONS;
const FIXED_NAMES = Object.keys(FIXED_SECTIONS) as FixedName[];

function assertLittleEndian(): void {
  if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) {
    throw new Error("three-nanite: big-endian platforms are not supported");
  }
}

function typeOf(array: TypedArray): SectionType {
  if (array instanceof Float32Array) return "f32";
  if (array instanceof Uint32Array) return "u32";
  if (array instanceof Uint16Array) return "u16";
  if (array instanceof Uint8Array) return "u8";
  if (array instanceof Int32Array) return "i32";
  if (array instanceof Int16Array) return "i16";
  if (array instanceof Int8Array) return "i8";
  throw new Error("Unsupported typed array type in meshlet data");
}

function makeView(type: SectionType, buffer: ArrayBufferLike, byteOffset: number, length: number): TypedArray {
  switch (type) {
    case "f32": return new Float32Array(buffer, byteOffset, length);
    case "u8": return new Uint8Array(buffer, byteOffset, length);
    case "u16": return new Uint16Array(buffer, byteOffset, length);
    case "u32": return new Uint32Array(buffer, byteOffset, length);
    case "i8": return new Int8Array(buffer, byteOffset, length);
    case "i16": return new Int16Array(buffer, byteOffset, length);
    case "i32": return new Int32Array(buffer, byteOffset, length);
  }
}

const align8 = (n: number): number => (n + 7) & ~7;

/**
 * Serializes meshlet data into a single binary container (see
 * {@link MESHLET_FILE_EXTENSION}). The result can be written to disk, fetched,
 * and handed to {@link deserializeMeshlets}, which maps it without copying.
 */
export function serializeMeshlets(data: MeshletData): ArrayBuffer {
  assertLittleEndian();

  const sources: { entry: SectionEntry; array: TypedArray }[] = [];
  for (const a of data.attributes) {
    sources.push({
      entry: {
        name: a.name,
        type: typeOf(a.array),
        offset: 0,
        length: a.array.length,
        attribute: true,
        itemSize: a.itemSize,
        normalized: a.normalized,
      },
      array: a.array,
    });
  }
  for (const name of FIXED_NAMES) {
    const array = data[name] as TypedArray;
    sources.push({ entry: { name, type: typeOf(array), offset: 0, length: array.length }, array });
  }

  const encoder = new TextEncoder();
  const buildHeader = (payloadStart: number): { json: Uint8Array; total: number } => {
    let offset = payloadStart;
    for (const { entry } of sources) {
      entry.offset = offset;
      offset = align8(offset + entry.length * BYTES[entry.type]);
    }
    const header: Header = {
      version: FORMAT_VERSION,
      vertexCount: data.vertexCount,
      clusterCount: data.clusterCount,
      groupCount: data.groupCount,
      levelCount: data.levelCount,
      sections: sources.map((s) => s.entry),
    };
    return { json: encoder.encode(JSON.stringify(header)), total: offset };
  };

  // Offsets are absolute, so the header size depends on the offsets' digit counts.
  // Growing the payload start only grows the JSON, so this converges in a few rounds.
  let payloadStart = PREAMBLE_BYTES;
  let built = buildHeader(payloadStart);
  for (;;) {
    const needed = PREAMBLE_BYTES + align8(built.json.length);
    if (needed <= payloadStart) break;
    payloadStart = needed;
    built = buildHeader(payloadStart);
  }
  const headerBytes = payloadStart - PREAMBLE_BYTES;

  const out = new Uint8Array(built.total);
  const view = new DataView(out.buffer);
  out.set(MAGIC, 0);
  view.setUint32(4, FORMAT_VERSION, true);
  view.setUint32(8, headerBytes, true);
  view.setUint32(12, 0, true);
  out.fill(0x20, PREAMBLE_BYTES, payloadStart);
  out.set(built.json, PREAMBLE_BYTES);

  for (const { entry, array } of sources) {
    out.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), entry.offset);
  }
  return out.buffer;
}

/**
 * Reads a container written by {@link serializeMeshlets}.
 *
 * Typed arrays are views directly on `buffer` (zero copy) whenever their
 * address is aligned for the element size; a misaligned section (for example
 * a `Uint8Array` with an odd `byteOffset`) is copied individually. Do not
 * mutate or detach the input buffer while the result is in use.
 *
 * @throws Error if the data is not a meshlet file, has an unsupported version, or is corrupt.
 */
export function deserializeMeshlets(buffer: ArrayBuffer | Uint8Array): MeshletData {
  assertLittleEndian();

  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const base = bytes.buffer;
  const baseOffset = bytes.byteOffset;
  const size = bytes.byteLength;

  if (size < 4 || MAGIC.some((m, i) => bytes[i] !== m)) {
    throw new Error("Not a three-nanite meshlet file");
  }
  if (size < PREAMBLE_BYTES) throw new Error("Corrupt meshlet file: truncated preamble");

  const dv = new DataView(base, baseOffset, size);
  const version = dv.getUint32(4, true);
  if (version !== FORMAT_VERSION) throw new Error(`Unsupported meshlet file version ${version}`);

  const headerBytes = dv.getUint32(8, true);
  if (PREAMBLE_BYTES + headerBytes > size) throw new Error("Corrupt meshlet file: truncated header");

  let header: Header;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(PREAMBLE_BYTES, PREAMBLE_BYTES + headerBytes))) as Header;
  } catch {
    throw new Error("Corrupt meshlet file: invalid header");
  }
  if (
    !header ||
    !Array.isArray(header.sections) ||
    !isCount(header.vertexCount) ||
    !isCount(header.clusterCount) ||
    !isCount(header.groupCount) ||
    !isCount(header.levelCount)
  ) {
    throw new Error("Corrupt meshlet file: invalid header");
  }

  const payloadStart = PREAMBLE_BYTES + headerBytes;
  const attributes: MeshletAttribute[] = [];
  const fixed = new Map<string, TypedArray>();

  for (const s of header.sections) {
    if (
      !s ||
      typeof s.name !== "string" ||
      !Object.hasOwn(BYTES, s.type) ||
      !isCount(s.offset) ||
      !isCount(s.length)
    ) {
      throw new Error("Corrupt meshlet file: invalid section entry");
    }
    const elementBytes = BYTES[s.type];
    if (s.offset < payloadStart || s.offset + s.length * elementBytes > size) {
      throw new Error(`Corrupt meshlet file: section "${s.name}" lies outside the file`);
    }

    const byteOffset = baseOffset + s.offset;
    const array =
      byteOffset % elementBytes === 0
        ? makeView(s.type, base, byteOffset, s.length)
        : copySection(s, bytes.subarray(s.offset, s.offset + s.length * elementBytes));

    if (s.attribute) {
      if (!isCount(s.itemSize) || s.itemSize < 1 || s.length !== header.vertexCount * s.itemSize) {
        throw new Error(`Corrupt meshlet file: attribute "${s.name}" has an invalid size`);
      }
      attributes.push({ name: s.name, array, itemSize: s.itemSize, normalized: s.normalized === true });
    } else {
      fixed.set(s.name, array);
      const expected = (FIXED_SECTIONS as Record<string, SectionType>)[s.name];
      if (expected !== undefined && expected !== s.type) {
        throw new Error(`Corrupt meshlet file: section "${s.name}" has type ${s.type}, expected ${expected}`);
      }
    }
  }

  if (attributes.length === 0 || attributes[0].name !== "position" || attributes[0].itemSize !== 3) {
    throw new Error("Corrupt meshlet file: the first attribute must be a 3-component position");
  }

  const get = <T extends TypedArray>(name: FixedName, length?: number): T => {
    const array = fixed.get(name);
    if (!array) throw new Error(`Corrupt meshlet file: missing section "${name}"`);
    if (length !== undefined && array.length !== length) {
      throw new Error(`Corrupt meshlet file: section "${name}" has length ${array.length}, expected ${length}`);
    }
    return array as T;
  };

  const { clusterCount, groupCount } = header;
  return {
    version: 1,
    vertexCount: header.vertexCount,
    attributes,
    indices: get<Uint32Array>("indices"),
    clusterCount,
    clusterIndexStart: get<Uint32Array>("clusterIndexStart", clusterCount),
    clusterIndexCount: get<Uint32Array>("clusterIndexCount", clusterCount),
    clusterBounds: get<Float32Array>("clusterBounds", clusterCount * 4),
    clusterGroup: get<Uint32Array>("clusterGroup", clusterCount),
    clusterRefined: get<Int32Array>("clusterRefined", clusterCount),
    groupCount,
    groupClusterStart: get<Uint32Array>("groupClusterStart", groupCount),
    groupClusterCount: get<Uint32Array>("groupClusterCount", groupCount),
    groupBounds: get<Float32Array>("groupBounds", groupCount * 4),
    groupError: get<Float32Array>("groupError", groupCount),
    groupDepth: get<Uint16Array>("groupDepth", groupCount),
    rootGroups: get<Uint32Array>("rootGroups"),
    levelCount: header.levelCount,
    boundingSphere: get<Float32Array>("boundingSphere", 4),
    boundingBox: get<Float32Array>("boundingBox", 6),
  };
}

function isCount(n: unknown): n is number {
  return typeof n === "number" && Number.isSafeInteger(n) && n >= 0;
}

/** Copies one section into a fresh, aligned buffer. */
function copySection(s: SectionEntry, source: Uint8Array): TypedArray {
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  return makeView(s.type, copy.buffer, 0, s.length);
}
