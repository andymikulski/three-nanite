import type { ColorMap } from "../../src/build/appearance.js";
import type { MeshletInput } from "../../src/types.js";

/** A "building with a recessed roof", unwelded per face, y up. */
export const HALF = 5;
export const HEIGHT = 12;
export const RIM_HALF = 4.4;
export const RECESS = 0.4;
export const FLOOR_Y = HEIGHT - RECESS;
const DEFAULT_GRID = 6;
export const GRID = DEFAULT_GRID;

export const WALL_RGB = [200, 180, 150] as const;
export const FLOOR_RGB = [40, 44, 48] as const;
export const WALL_UV = [0.25, 0.5] as const;
export const FLOOR_UV = [0.75, 0.5] as const;

export function makeColorMap(): ColorMap {
  return { data: new Uint8Array([...WALL_RGB, 255, ...FLOOR_RGB, 255]), width: 2, height: 1 };
}

export interface RoofBuilding {
  input: MeshletInput;
  vertexCount: number;
  positions: Float32Array;
  /** The input's triangle list. */
  indices: Uint32Array;
  /** Range of `indices` (in index entries) that is the roof floor. */
  floorStart: number;
  floorEnd: number;
  /** Four vertices at the corners of the inner square at y = HEIGHT, carrying the wall uv. */
  rimInnerCorners: [number, number, number, number];
  /** Vertex range [floorFirstVertex, floorFirstVertex + floorVertexCount) of the floor. */
  floorFirstVertex: number;
  floorVertexCount: number;
}

type V3 = [number, number, number];

export interface RoofOptions {
  /** Give the floor the wall's uv too (one colour throughout). */
  floorUsesWallUv?: boolean;
  /** Add a `color` attribute (3 floats) and leave out `uv`. */
  vertexColors?: boolean;
  /** Leave out `uv`. */
  noUv?: boolean;
  /** Extra copy of the floor's vertices this much higher (appended after all others). */
  floorCopyLift?: number;
  /** Quads per side of a grid (default GRID). */
  grid?: number;
}

export function roofBuilding(options: RoofOptions = {}): RoofBuilding & { floorCopyFirstVertex: number } {
  const position: number[] = [];
  const normal: number[] = [];
  const uv: number[] = [];
  const color: number[] = [];
  const indices: number[] = [];

  const colorOf = (u: readonly number[]): readonly number[] => (u === FLOOR_UV ? FLOOR_RGB : WALL_RGB);

  function addVertex(p: V3, n: V3, u: readonly number[]): number {
    const id = position.length / 3;
    position.push(...p);
    normal.push(...n);
    uv.push(...u);
    color.push(...colorOf(u).map((c) => c / 255));
    return id;
  }

  /** A quad p0..p3 (a loop), wound so that its geometric normal faces `n`. */
  function quad(p: [V3, V3, V3, V3], n: V3, u: readonly number[]): number[] {
    const e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    const e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    const cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const flip = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2] < 0;
    const loop = flip ? [p[0], p[3], p[2], p[1]] : p;
    const ids = loop.map((q) => addVertex(q, n, u));
    indices.push(ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]);
    return ids;
  }

  /** A grid of GRID x GRID quads over the parallelogram o + s * du + t * dv. */
  function grid(o: V3, du: V3, dv: V3, n: V3, u: readonly number[]): void {
    const GRID = options.grid ?? DEFAULT_GRID;
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const at = (s: number, t: number): V3 => [
          o[0] + du[0] * (s / GRID) + dv[0] * (t / GRID),
          o[1] + du[1] * (s / GRID) + dv[1] * (t / GRID),
          o[2] + du[2] * (s / GRID) + dv[2] * (t / GRID),
        ];
        quad([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)], n, u);
      }
    }
  }

  const H = HALF;
  // outer walls
  grid([-H, 0, H], [2 * H, 0, 0], [0, HEIGHT, 0], [0, 0, 1], WALL_UV); // +z
  grid([H, 0, -H], [-2 * H, 0, 0], [0, HEIGHT, 0], [0, 0, -1], WALL_UV); // -z
  grid([H, 0, H], [0, 0, -2 * H], [0, HEIGHT, 0], [1, 0, 0], WALL_UV); // +x
  grid([-H, 0, -H], [0, 0, 2 * H], [0, HEIGHT, 0], [-1, 0, 0], WALL_UV); // -x

  // rim top: four trapezoids between the outer and the inner square
  const outer: V3[] = [
    [-H, HEIGHT, -H],
    [H, HEIGHT, -H],
    [H, HEIGHT, H],
    [-H, HEIGHT, H],
  ];
  const r = RIM_HALF;
  const inner: V3[] = [
    [-r, HEIGHT, -r],
    [r, HEIGHT, -r],
    [r, HEIGHT, r],
    [-r, HEIGHT, r],
  ];
  const rimInner: number[] = [];
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const ids = quad([outer[i], outer[j], inner[j], inner[i]], [0, 1, 0], WALL_UV);
    // which of the quad's vertices is inner[i]: the one at that position
    const at = (q: V3): number => ids.find((id) => position[id * 3] === q[0] && position[id * 3 + 1] === q[1] && position[id * 3 + 2] === q[2])!;
    rimInner.push(at(inner[i]));
  }

  // inner rim walls, facing inwards
  const lo = FLOOR_Y;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const a = inner[i];
    const b = inner[j];
    const dx = b[0] - a[0];
    const dz = b[2] - a[2];
    // inward: towards the centre
    const n: V3 = Math.abs(dx) > 0 ? [0, 0, a[2] > 0 ? -1 : 1] : [a[0] > 0 ? -1 : 1, 0, 0];
    void dz;
    quad([a, b, [b[0], lo, b[2]], [a[0], lo, a[2]]], n, WALL_UV);
  }

  // the floor
  const floorStart = indices.length;
  const floorFirstVertex = position.length / 3;
  grid([-r, FLOOR_Y, -r], [2 * r, 0, 0], [0, 0, 2 * r], [0, 1, 0], options.floorUsesWallUv ? WALL_UV : FLOOR_UV);
  const floorEnd = indices.length;
  const floorVertexCount = position.length / 3 - floorFirstVertex;

  // optional second copy of the floor's vertices, lifted
  const floorCopyFirstVertex = position.length / 3;
  if (options.floorCopyLift !== undefined) {
    for (let v = floorFirstVertex; v < floorFirstVertex + floorVertexCount; v++) {
      addVertex(
        [position[v * 3], position[v * 3 + 1] + options.floorCopyLift, position[v * 3 + 2]],
        [0, 1, 0],
        options.floorUsesWallUv ? WALL_UV : FLOOR_UV,
      );
    }
  }

  const positions = Float32Array.from(position);
  const attributes: MeshletInput["attributes"] = [
    { name: "position", array: positions, itemSize: 3, normalized: false },
    { name: "normal", array: Float32Array.from(normal), itemSize: 3, normalized: false },
  ];
  if (options.vertexColors) {
    attributes.push({ name: "color", array: Float32Array.from(color), itemSize: 3, normalized: false });
  } else if (!options.noUv) {
    attributes.push({ name: "uv", array: Float32Array.from(uv), itemSize: 2, normalized: false });
  }
  const index = Uint32Array.from(indices);
  return {
    input: { attributes, indices: index },
    vertexCount: positions.length / 3,
    positions,
    indices: index,
    floorStart,
    floorEnd,
    rimInnerCorners: rimInner as [number, number, number, number],
    floorFirstVertex,
    floorVertexCount,
    floorCopyFirstVertex,
  };
}
