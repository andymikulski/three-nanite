import { beforeAll, describe, expect, it } from "vitest";
import { MeshBasicMaterial, PerspectiveCamera, TorusKnotGeometry } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshlets } from "../src/build/buildMeshlets.js";
import { unweldMeshlets, weldVertices } from "../src/build/weld.js";
import { computeLodLevels, copyLevelIndices } from "../src/core/levels.js";
import { ClusterSelector, countTriangles, getLodErrors } from "../src/core/select.js";
import { validateMeshletData } from "../src/core/validate.js";
import { geometryToMeshletInput } from "../src/three/geometry.js";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
import type { MeshletData, MeshletInput } from "../src/types.js";

// ---------------------------------------------------------------------------------------------
// meshes
// ---------------------------------------------------------------------------------------------

/** Small deterministic generator, uniform in [0, 1). */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const CARD_UV = [0, 0, 1, 0, 1, 1, 0, 1];

/** `n` separate square cards (side 0.05, own 4 vertices each) in the cube [-1, 1]^3. */
function cards(n: number, seed = 7): MeshletInput {
  const random = lcg(seed);
  const position = new Float32Array(n * 12);
  const normal = new Float32Array(n * 12);
  const uv = new Float32Array(n * 8);
  const indices = new Uint32Array(n * 6);
  const half = 0.025;
  for (let c = 0; c < n; c++) {
    const cx = random() * 2 - 1;
    const cy = random() * 2 - 1;
    const cz = random() * 2 - 1;
    // random orientation: a unit normal (rejection sampled in the ball), a tangent basis, a spin
    let nx = 0;
    let ny = 0;
    let nz = 1;
    for (;;) {
      const x = random() * 2 - 1;
      const y = random() * 2 - 1;
      const z = random() * 2 - 1;
      const l = Math.hypot(x, y, z);
      if (l > 0.1 && l <= 1) {
        nx = x / l;
        ny = y / l;
        nz = z / l;
        break;
      }
    }
    // any vector not parallel to the normal
    const ax = Math.abs(nx) < 0.9 ? 1 : 0;
    const ay = ax === 1 ? 0 : 1;
    let tx = ay * nz;
    let ty = -ax * nz;
    let tz = ax * ny - ay * nx;
    const tl = Math.hypot(tx, ty, tz);
    tx /= tl;
    ty /= tl;
    tz /= tl;
    const bx = ny * tz - nz * ty;
    const by = nz * tx - nx * tz;
    const bz = nx * ty - ny * tx;
    const spin = random() * Math.PI * 2;
    const cs = Math.cos(spin);
    const sn = Math.sin(spin);
    const ux = tx * cs + bx * sn;
    const uy = ty * cs + by * sn;
    const uz = tz * cs + bz * sn;
    const vx = -tx * sn + bx * cs;
    const vy = -ty * sn + by * cs;
    const vz = -tz * sn + bz * cs;
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ];
    for (let k = 0; k < 4; k++) {
      const [a, b] = corners[k];
      const v = c * 4 + k;
      position[v * 3] = cx + (ux * a + vx * b) * half;
      position[v * 3 + 1] = cy + (uy * a + vy * b) * half;
      position[v * 3 + 2] = cz + (uz * a + vz * b) * half;
      normal[v * 3] = nx;
      normal[v * 3 + 1] = ny;
      normal[v * 3 + 2] = nz;
      uv[v * 2] = CARD_UV[k * 2];
      uv[v * 2 + 1] = CARD_UV[k * 2 + 1];
    }
    indices.set([c * 4, c * 4 + 1, c * 4 + 2, c * 4, c * 4 + 2, c * 4 + 3], c * 6);
  }
  return {
    attributes: [
      { name: "position", array: position, itemSize: 3, normalized: false },
      { name: "normal", array: normal, itemSize: 3, normalized: false },
      { name: "uv", array: uv, itemSize: 2, normalized: false },
    ],
    indices,
  };
}

/** Flat n x n grid of quads in the XY plane; every quad has its own 4 vertices and its own uv square. */
function seamGrid(n: number): MeshletInput {
  const position = new Float32Array(n * n * 12);
  const normal = new Float32Array(n * n * 12);
  const uv = new Float32Array(n * n * 8);
  const indices = new Uint32Array(n * n * 6);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const q = j * n + i;
      for (let k = 0; k < 4; k++) {
        const v = q * 4 + k;
        const du = CARD_UV[k * 2];
        const dv = CARD_UV[k * 2 + 1];
        position.set([(i + du) / n, (j + dv) / n, 0], v * 3);
        normal.set([0, 0, 1], v * 3);
        uv.set([du, dv], v * 2);
      }
      indices.set([q * 4, q * 4 + 1, q * 4 + 2, q * 4, q * 4 + 2, q * 4 + 3], q * 6);
    }
  }
  return {
    attributes: [
      { name: "position", array: position, itemSize: 3, normalized: false },
      { name: "normal", array: normal, itemSize: 3, normalized: false },
      { name: "uv", array: uv, itemSize: 2, normalized: false },
    ],
    indices,
  };
}

function torusKnotInput(): MeshletInput {
  return geometryToMeshletInput(mergeVertices(new TorusKnotGeometry(1, 0.35, 160, 32)));
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

function vertexCountOf(input: MeshletInput): number {
  return input.attributes[0].array.length / 3;
}

function triangleCountOf(input: MeshletInput): number {
  return input.indices ? input.indices.length / 3 : vertexCountOf(input) / 3;
}

/** Index list of a whole-mesh level. */
function levelIndices(data: MeshletData, levels: ReturnType<typeof computeLodLevels>, level: number): Uint32Array {
  const out = new Uint32Array(levels.triangles[level] * 3);
  copyLevelIndices(data, levels, level, out);
  return out;
}

function totalArea(positions: ArrayLike<number>, indices: ArrayLike<number>): number {
  let area = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    area += Math.sqrt(nx * nx + ny * ny + nz * nz) * 0.5;
  }
  return area;
}

/** Triangle count of the cut `selectByError(maxError)`. */
function cutTriangles(data: MeshletData, maxError: number): number {
  const out = new Uint32Array(data.clusterCount);
  const n = new ClusterSelector(data).selectByError(out, maxError);
  return countTriangles(data, out, n);
}

function tri(a: number, b: number, c: number): string {
  return [a, b, c].sort((x, y) => x - y).join(",");
}

function copyArrays(input: MeshletInput): Float32Array[] {
  return input.attributes.map((a) => Float32Array.from(a.array as Float32Array));
}

// ---------------------------------------------------------------------------------------------
// dilateBorders
// ---------------------------------------------------------------------------------------------

describe("buildMeshlets: dilateBorders", () => {
  let input: MeshletInput;
  let inputCopies: Float32Array[];
  let inputVertices: number;
  let plain: MeshletData;
  let dilated: MeshletData;
  let dilatedAgain: MeshletData;
  let plainLevels: ReturnType<typeof computeLodLevels>;
  let dilatedLevels: ReturnType<typeof computeLodLevels>;

  beforeAll(async () => {
    input = cards(3000);
    inputCopies = copyArrays(input);
    inputVertices = vertexCountOf(input);
    plain = await buildMeshlets(input);
    dilated = await buildMeshlets(input, { dilateBorders: true });
    dilatedAgain = await buildMeshlets(input, { dilateBorders: true });
    plainLevels = computeLodLevels(plain, { maxLevels: 100000 });
    dilatedLevels = computeLodLevels(dilated, { maxLevels: 100000 });
  }, 60_000);

  it("is off by default: the input's vertices are used as they are", () => {
    expect(plain.vertexCount).toBe(inputVertices);
    expect(plain.attributes[0].array).toBe(input.attributes[0].array);
  });

  it("adds vertices after the input's, keeps the input's values, and copies the other attributes", () => {
    expect(validateMeshletData(dilated)).toEqual([]);
    expect(dilated.vertexCount).toBeGreaterThan(inputVertices);

    for (let a = 0; a < input.attributes.length; a++) {
      const source = input.attributes[a];
      const out = dilated.attributes[a];
      expect(out.name).toBe(source.name);
      expect(out.itemSize).toBe(source.itemSize);
      expect(out.array.length).toBe(dilated.vertexCount * source.itemSize);
      const head = out.array.subarray(0, inputVertices * source.itemSize);
      expect(Array.from(head)).toEqual(Array.from(inputCopies[a]));
      // the input itself was not touched
      expect(Array.from(source.array)).toEqual(Array.from(inputCopies[a]));
    }

    const uv = dilated.attributes[2].array;
    const normal = dilated.attributes[1].array;
    for (let v = inputVertices; v < dilated.vertexCount; v++) {
      const u = uv[v * 2];
      const w = uv[v * 2 + 1];
      const isCorner = [0, 1, 2, 3].some((k) => u === CARD_UV[k * 2] && w === CARD_UV[k * 2 + 1]);
      expect(isCorner, `uv of vertex ${v}`).toBe(true);
      const length = Math.hypot(normal[v * 3], normal[v * 3 + 1], normal[v * 3 + 2]);
      expect(Math.abs(length - 1)).toBeLessThan(1e-6);
    }
  });

  it("moves no vertex out of the input's bounding box or sphere, and keeps the input's bounds", () => {
    const p = input.attributes[0].array as Float32Array;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        if (p[i + k] < min[k]) min[k] = p[i + k];
        if (p[i + k] > max[k]) max[k] = p[i + k];
      }
    }
    const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const tolerance = 1e-6 * size;
    const out = dilated.attributes[0].array as Float32Array;
    const [cx, cy, cz, r] = dilated.boundingSphere;
    for (let v = 0; v < dilated.vertexCount; v++) {
      for (let k = 0; k < 3; k++) {
        const x = out[v * 3 + k];
        expect(x).toBeGreaterThanOrEqual(min[k] - tolerance);
        expect(x).toBeLessThanOrEqual(max[k] + tolerance);
      }
      const d = Math.hypot(out[v * 3] - cx, out[v * 3 + 1] - cy, out[v * 3 + 2] - cz);
      expect(d).toBeLessThanOrEqual(r * (1 + 1e-5));
    }

    for (let i = 0; i < 6; i++) expect(dilated.boundingBox[i]).toBeCloseTo(plain.boundingBox[i], 5);
    for (let i = 0; i < 4; i++) expect(dilated.boundingSphere[i]).toBeCloseTo(plain.boundingSphere[i], 5);
  });

  it("keeps the area of every level that still has a tenth of the triangles; without it the area drains away", () => {
    const area = (data: MeshletData, levels: typeof dilatedLevels, level: number): number =>
      totalArea(data.attributes[0].array, levelIndices(data, levels, level));

    const base = area(dilated, dilatedLevels, 0);
    const baseTriangles = dilatedLevels.triangles[0];
    expect(dilatedLevels.count).toBeGreaterThan(3);
    let checked = 0;
    for (let j = 1; j < dilatedLevels.count; j++) {
      if (dilatedLevels.triangles[j] < baseTriangles * 0.1) continue;
      const ratio = area(dilated, dilatedLevels, j) / base;
      expect(ratio, `dilated level ${j} (${dilatedLevels.triangles[j]} triangles)`).toBeGreaterThanOrEqual(0.85);
      expect(ratio, `dilated level ${j} (${dilatedLevels.triangles[j]} triangles)`).toBeLessThanOrEqual(1.15);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);

    const plainBase = area(plain, plainLevels, 0);
    const target = plainLevels.triangles[0] * 0.25;
    let closest = 0;
    for (let j = 1; j < plainLevels.count; j++) {
      if (Math.abs(plainLevels.triangles[j] - target) < Math.abs(plainLevels.triangles[closest] - target)) closest = j;
    }
    expect(closest).toBeGreaterThan(0);
    expect(area(plain, plainLevels, closest) / plainBase).toBeLessThan(0.6);
  });

  it("spreads the errors of the levels out over a wider range", () => {
    const spread = (data: MeshletData): number => {
      const e = getLodErrors(data);
      return e[e.length - 1] / e[0];
    };
    const withDilation = spread(dilated);
    const without = spread(plain);
    expect(withDilation, `dilated ${withDilation}, plain ${without}`).toBeGreaterThanOrEqual(2 * without);
  });

  it("is deterministic", () => {
    expect(dilatedAgain.vertexCount).toBe(dilated.vertexCount);
    expect(Array.from(dilatedAgain.indices)).toEqual(Array.from(dilated.indices));
    expect(Array.from(dilatedAgain.attributes[0].array)).toEqual(Array.from(dilated.attributes[0].array));
  });

  it("changes nothing for a closed surface", async () => {
    const closed = torusKnotInput();
    const off = await buildMeshlets(closed, { dilateBorders: false });
    const on = await buildMeshlets(closed, { dilateBorders: true });
    expect(on.vertexCount).toBe(vertexCountOf(closed));
    for (let i = 0; i < closed.attributes.length; i++) expect(on.attributes[i].array).toBe(closed.attributes[i].array);
    expect(Array.from(on.indices)).toEqual(Array.from(off.indices));
    expect(Array.from(on.groupError)).toEqual(Array.from(off.groupError));
    expect(Array.from(on.groupBounds)).toEqual(Array.from(off.groupBounds));
  });

  it("neither moves nor removes locked vertices", async () => {
    const small = cards(600);
    const locked = new Uint8Array(vertexCountOf(small));
    locked.fill(1, 0, 40 * 4);
    const data = await buildMeshlets(small, { dilateBorders: true, lockedVertices: locked });
    expect(validateMeshletData(data)).toEqual([]);

    const levels = computeLodLevels(data, { maxLevels: 100000 });
    expect(levels.count).toBeGreaterThan(2);
    for (let j = 0; j < levels.count; j++) {
      const present = new Set<string>();
      const indices = levelIndices(data, levels, j);
      for (let i = 0; i < indices.length; i += 3) present.add(tri(indices[i], indices[i + 1], indices[i + 2]));
      for (let c = 0; c < 40; c++) {
        const a = c * 4;
        expect(present.has(tri(a, a + 1, a + 2)), `level ${j}, card ${c}, first triangle`).toBe(true);
        expect(present.has(tri(a, a + 2, a + 3)), `level ${j}, card ${c}, second triangle`).toBe(true);
      }
    }
    // and the locked vertices were not given new positions
    const p = data.attributes[0].array;
    const original = small.attributes[0].array;
    for (let v = 0; v < 160 * 3; v++) expect(p[v]).toBe(original[v]);
  });

  it("makes unweldMeshlets refuse the hierarchy", async () => {
    const source = cards(300);
    const n = source.indices!.length;
    // triangle soup: three vertices of their own per triangle
    const soupAttributes = source.attributes.map((a) => {
      const array = new Float32Array(n * a.itemSize);
      for (let i = 0; i < n; i++) {
        const from = source.indices![i] * a.itemSize;
        for (let k = 0; k < a.itemSize; k++) array[i * a.itemSize + k] = (a.array as Float32Array)[from + k];
      }
      return { ...a, array };
    });
    const soup: MeshletInput = { attributes: soupAttributes, indices: null };

    const welded = weldVertices(soup);
    const data = await buildMeshlets(welded.input, { dilateBorders: true });
    expect(data.vertexCount).toBeGreaterThan(vertexCountOf(welded.input));
    expect(() => unweldMeshlets(data, welded.source, soup)).toThrow(/dilateBorders/);
  });

  it("can be drawn by a MeshletMesh", () => {
    const material = new MeshBasicMaterial();
    let mesh!: MeshletMesh;
    expect(() => {
      mesh = new MeshletMesh(dilated, material);
    }).not.toThrow();
    expect(mesh.batched.geometry.getAttribute("position").count).toBe(dilated.vertexCount);

    const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 100000);
    camera.position.set(0, 0, 5000);
    camera.updateMatrixWorld(true);
    mesh.update(camera, 1080);
    expect(mesh.stats.triangles).toBeGreaterThan(0);
    expect(mesh.stats.triangles).toBeLessThan(triangleCountOf(input) * 0.1);
  });
});

// ---------------------------------------------------------------------------------------------
// fallbackUnprotected
// ---------------------------------------------------------------------------------------------

describe("buildMeshlets: fallbackUnprotected", () => {
  const N = 48;
  let grid: MeshletInput;
  let gridDefault: MeshletData;
  let gridStrict: MeshletData;
  let gridNoSloppy: MeshletData;

  beforeAll(async () => {
    grid = seamGrid(N);
    gridDefault = await buildMeshlets(grid);
    gridStrict = await buildMeshlets(grid, { fallbackUnprotected: false });
    gridNoSloppy = await buildMeshlets(grid, { fallbackUnprotected: false, fallbackSloppy: false });
  }, 60_000);

  it("without it and without the sloppy fallback, a mesh of nothing but seams cannot be simplified", () => {
    expect(validateMeshletData(gridNoSloppy)).toEqual([]);
    expect(cutTriangles(gridNoSloppy, Infinity)).toBe(triangleCountOf(grid));
  });

  it("by default lets such a mesh simplify across its seams", () => {
    expect(validateMeshletData(gridDefault)).toEqual([]);
    expect(gridDefault.levelCount).toBeGreaterThanOrEqual(3);
    expect(cutTriangles(gridDefault, Infinity)).toBeLessThan(triangleCountOf(grid) * 0.05);
  });

  it("changes nothing when nothing is protected", async () => {
    const on = await buildMeshlets(grid, { protectAttributes: [], fallbackUnprotected: true });
    const off = await buildMeshlets(grid, { protectAttributes: [], fallbackUnprotected: false });
    expect(Array.from(on.indices)).toEqual(Array.from(off.indices));
    expect(Array.from(on.groupError)).toEqual(Array.from(off.groupError));
    expect(Array.from(on.clusterRefined)).toEqual(Array.from(off.clusterRefined));
  });

  it("changes nothing when there are no seams", async () => {
    const knot = torusKnotInput();
    const noUv: MeshletInput = { attributes: knot.attributes.filter((a) => a.name !== "uv"), indices: knot.indices };
    expect(noUv.attributes.map((a) => a.name)).toEqual(["position", "normal"]);
    const on = await buildMeshlets(noUv, { fallbackUnprotected: true });
    const off = await buildMeshlets(noUv, { fallbackUnprotected: false });
    expect(Array.from(on.indices)).toEqual(Array.from(off.indices));
    expect(Array.from(on.groupError)).toEqual(Array.from(off.groupError));
  });

  it("charges a collapse across a seam for the uv error it causes", () => {
    // The grid is flat, so no collapse moves the surface: sloppy simplification, which only measures
    // positions, records an error of 0 and would swap the levels at any distance. Crossing a seam is
    // counted with the uv change (protected attributes weigh at least 1), so its levels wait until
    // that change is small on screen.
    const crossing = getLodErrors(gridDefault)[0];
    const sloppy = getLodErrors(gridStrict)[0];
    expect(sloppy).toBe(0);
    expect(crossing).toBeGreaterThan(0);
  });
});
