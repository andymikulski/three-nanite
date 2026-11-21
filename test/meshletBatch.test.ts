import { beforeAll, describe, expect, it, vi } from "vitest";
import {
  BufferGeometry,
  Color,
  Frustum,
  Matrix4,
  MeshBasicMaterial,
  MeshDepthMaterial,
  MeshStandardMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  Quaternion,
  Raycaster,
  REVISION,
  Scene,
  SphereGeometry,
  Texture,
  TorusKnotGeometry,
  Vector3,
} from "three";
import type { Camera } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { levelForError, lodScale, MeshletBatch } from "../src/index.js";
import type { MeshletBatchOptions } from "../src/index.js";
import type { MeshletData } from "../src/types.js";
import { LOD_PALETTE } from "../src/three/palette.js";

const H = 1080;
const BUILD_TIMEOUT = 120000;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

let knot: MeshletData;
let ball: MeshletData;

beforeAll(async () => {
  const options = { attributes: ["normal", "uv"] };
  knot = await buildMeshletsFromGeometry(mergeVertices(new TorusKnotGeometry(1, 0.35, 320, 48)), options);
  ball = await buildMeshletsFromGeometry(mergeVertices(new SphereGeometry(1, 24, 16)), options);
}, BUILD_TIMEOUT);

function makeBatch(options?: MeshletBatchOptions, material?: ConstructorParameters<typeof MeshletBatch>[1]): MeshletBatch {
  return new MeshletBatch([knot, ball], material, options);
}

function makeCamera(x: number, y: number, z: number, far = 1e7): PerspectiveCamera {
  const camera = new PerspectiveCamera(50, 16 / 9, 0.1, far);
  camera.position.set(x, y, z);
  camera.updateMatrixWorld(true);
  return camera;
}

const translation = (x: number, y = 0, z = 0) => new Matrix4().makeTranslation(x, y, z);

/** A small seeded generator (mulberry32): numbers in [0, 1). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function direction(rand: () => number): Vector3 {
  const v = new Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5);
  return v.lengthSq() > 0 ? v.normalize() : v.set(1, 0, 0);
}

/** An instance of `mesh` whose bounding sphere is centred on `point` (unit scale, no rotation). */
function matrixAt(data: MeshletData, point: Vector3): Matrix4 {
  const c = data.boundingSphere;
  return translation(point.x - c[0], point.y - c[1], point.z - c[2]);
}

function update(batch: MeshletBatch, camera: Camera): boolean {
  camera.updateMatrixWorld(true);
  batch.updateMatrixWorld(true);
  return batch.update(camera, H);
}

// ---------------------------------------------------------------------------
// private parts
// ---------------------------------------------------------------------------

interface Raw {
  drawCount: number;
  drawCounts: Int32Array;
  drawOffsets: Int32Array;
  drawSeen: Int32Array;
  drawShadow: Int32Array;
  drawAll: Int32Array;
}
const raw = (batch: MeshletBatch) => batch as unknown as Raw;

interface BatchedRaw {
  _indirectTexture: { image: { data: Uint32Array } };
  _mergedDrawCount: number;
  _mergedDrawStarts: Int32Array;
  _mergedDrawCounts: Int32Array;
  _mergedDrawInstanceCounts: Int32Array;
  _multiDrawBytesPerElement: number;
}
const batchedRaw = (batch: MeshletBatch) => batch.batched as unknown as BatchedRaw;

// ---------------------------------------------------------------------------
// fake renderers and hooks
// ---------------------------------------------------------------------------

function makeWebGL() {
  const gl = { TRIANGLES: 4, UNSIGNED_SHORT: 5123, UNSIGNED_INT: 5125, drawElementsInstanced: vi.fn() };
  const extension = { multiDrawElementsInstancedWEBGL: vi.fn() };
  const setValue = vi.fn();
  const state = { multiDraw: true };
  const renderer = {
    getRenderTarget: () => null,
    getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
    getContext: () => gl,
    extensions: { get: (name: string) => (name === "WEBGL_multi_draw" && state.multiDraw ? extension : null) },
    properties: { get: () => ({ currentProgram: { getUniforms: () => ({ setValue }) } }) },
    info: { render: { calls: 0, triangles: 0 } },
  };
  return { renderer, gl, extension, setValue, state };
}

function makeWebGPU() {
  return {
    isWebGPURenderer: true,
    getRenderTarget: () => null,
    getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
  };
}

type Renderer = ReturnType<typeof makeWebGL>["renderer"] | ReturnType<typeof makeWebGPU>;

function before(batch: MeshletBatch, renderer: Renderer, camera: Camera, scene: Scene = new Scene()): void {
  const b = batch.batched;
  (b.onBeforeRender as unknown as (...a: unknown[]) => void)(renderer, scene, camera, b.geometry, b.material, null);
}
function after(batch: MeshletBatch, renderer: Renderer, camera: Camera, scene: Scene = new Scene()): void {
  const b = batch.batched;
  (b.onAfterRender as unknown as (...a: unknown[]) => void)(renderer, scene, camera, b.geometry, b.material, null);
}
function beforeShadow(batch: MeshletBatch, renderer: Renderer, camera: Camera, shadowCamera: Camera): void {
  const b = batch.batched;
  (b.onBeforeShadow as unknown as (...a: unknown[]) => void)(renderer, b, camera, shadowCamera, b.geometry, new MeshDepthMaterial(), null);
}
function afterShadow(batch: MeshletBatch, renderer: Renderer, camera: Camera, shadowCamera: Camera): void {
  const b = batch.batched;
  (b.onAfterShadow as unknown as (...a: unknown[]) => void)(renderer, b, camera, shadowCamera, b.geometry, new MeshDepthMaterial(), null);
}

// ---------------------------------------------------------------------------
// reading the draw lists as the renderers would
// ---------------------------------------------------------------------------

interface SubDraw {
  d: number;
  ids: number[];
  indexCount: number;
  offset: number;
}

/** WebGLRenderer: sub-draw `d` draws `counts[d]` instances `T[T[d] + k]`. */
function readGL(batch: MeshletBatch, which: "seen" | "shadow" | "all"): SubDraw[] {
  const r = raw(batch);
  const T = batchedRaw(batch)._indirectTexture.image.data;
  const counts = which === "seen" ? r.drawSeen : which === "shadow" ? r.drawShadow : r.drawAll;
  const out: SubDraw[] = [];
  for (let d = 0; d < r.drawCount; d++) {
    const ids: number[] = [];
    for (let k = 0; k < counts[d]; k++) ids.push(T[T[d] + k]);
    out.push({ d, ids, indexCount: r.drawCounts[d], offset: r.drawOffsets[d] });
  }
  return out;
}

interface Entry {
  e: number;
  ids: number[];
  indexStart: number;
  indexCount: number;
}

/** WebGPURenderer: entries draw consecutive runs of the id list. */
function readGPU(batch: MeshletBatch): Entry[] {
  const b = batchedRaw(batch);
  const T = b._indirectTexture.image.data;
  const out: Entry[] = [];
  let first = 0;
  for (let e = 0; e < b._mergedDrawCount; e++) {
    const n = b._mergedDrawInstanceCounts[e];
    const ids: number[] = [];
    for (let k = 0; k < n; k++) ids.push(T[first + k]);
    out.push({ e, ids, indexStart: b._mergedDrawStarts[e] / b._multiDrawBytesPerElement, indexCount: b._mergedDrawCounts[e] });
    first += n;
  }
  return out;
}

/** Where, in the index buffer, the indices of (mesh, level) begin. */
function levelStart(batch: MeshletBatch, mesh: number, level: number): number {
  let at = 0;
  for (let k = 0; k < batch.levels.length; k++) {
    for (let j = 0; j < batch.levels[k].count; j++) {
      if (k === mesh && j === level) return at;
      at += batch.levels[k].triangles[j] * 3;
    }
  }
  throw new Error("no such level");
}

/** 40 instances around the origin, a dozen of them in front of a camera looking down -z, the rest in any direction. */
function scatter(batch: MeshletBatch, n = 40, seed = 0xfeed): Matrix4[] {
  const rand = seeded(seed);
  const out: Matrix4[] = [];
  for (let i = 0; i < n; i++) {
    const mesh = i % 2;
    const data = mesh === 0 ? knot : ball;
    const point =
      i < 14
        ? new Vector3((rand() - 0.5) * 10, (rand() - 0.5) * 6, -(5 + rand() * 200))
        : direction(rand).multiplyScalar(5 + rand() * 200);
    const s = i % 7 === 3 ? 2 : 1;
    const m = matrixAt(data, point);
    if (s !== 1) {
      m.premultiply(new Matrix4().makeScale(s, s, s));
    }
    out.push(m);
    batch.addInstance(mesh, m);
  }
  return out;
}

/** The instances whose bounding sphere intersects the view of `camera`. */
function expectedInView(batch: MeshletBatch, camera: Camera): number[] {
  const frustum = new Frustum().setFromProjectionMatrix(new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
  const out: number[] = [];
  const m = new Matrix4();
  for (let i = 0; i < batch.count; i++) {
    batch.getMatrixAt(i, m);
    const data = batch.meshes[batch.getMeshAt(i)];
    const c = new Vector3(data.boundingSphere[0], data.boundingSphere[1], data.boundingSphere[2]).applyMatrix4(m).applyMatrix4(batch.matrixWorld);
    const r = data.boundingSphere[3] * m.getMaxScaleOnAxis() * batch.matrixWorld.getMaxScaleOnAxis();
    if (frustum.intersectsSphere({ center: c, radius: r } as never)) out.push(i);
  }
  return out;
}

const sorted = (a: number[]) => [...a].sort((p, q) => p - q);

/** Group checks shared by the WebGL and WebGPU readers: one mesh and level per run, the index range of that level. */
function expectGroupsRight(batch: MeshletBatch, runs: Array<{ ids: number[]; indexCount: number; indexStart: number }>, label: string): void {
  for (const run of runs) {
    if (run.ids.length === 0) continue;
    const mesh = batch.getMeshAt(run.ids[0]);
    const level = batch.getLevelAt(run.ids[0]);
    for (const id of run.ids) {
      expect(batch.getMeshAt(id), `${label}: mesh of ${id}`).toBe(mesh);
      expect(batch.getLevelAt(id), `${label}: level of ${id}`).toBe(level);
    }
    expect(run.indexCount, `${label}: index count`).toBe(batch.levels[mesh].triangles[level] * 3);
    expect(run.indexStart, `${label}: index start`).toBe(levelStart(batch, mesh, level));
  }
}

/** F's first check: reading the lists as the main pass would for the shadow pass (all of them). */
function expectAllListed(batch: MeshletBatch): void {
  const r = raw(batch);
  const bpe = (batch.batched.geometry.index!.array as Uint16Array | Uint32Array).BYTES_PER_ELEMENT;
  const runs = readGL(batch, "all");
  const seenIds: number[] = [];
  for (const run of runs) {
    seenIds.push(...run.ids);
    expectGroupsRight(batch, [{ ids: run.ids, indexCount: run.indexCount, indexStart: run.offset / bpe }], `sub-draw ${run.d}`);
    expect(run.ids.length).toBeGreaterThan(0);
  }
  expect(sorted(seenIds)).toEqual(Array.from({ length: batch.count }, (_, i) => i));
  expect(r.drawCount).toBe(runs.length);
}

// ===========================================================================
// A. construction
// ===========================================================================

describe("MeshletBatch", () => {
  it("A. construction: validation, statistics, shared geometry and index type", async () => {
    expect(() => new MeshletBatch([])).toThrow();

    const onlyNormal = await buildMeshletsFromGeometry(mergeVertices(new SphereGeometry(1, 24, 16)), { attributes: ["normal"] });
    expect(() => new MeshletBatch([knot, onlyNormal])).toThrow();
    // (a shallow copy with an attribute dropped)
    const dropped: MeshletData = { ...ball, attributes: ball.attributes.slice(0, 2) };
    expect(() => new MeshletBatch([knot, dropped])).toThrow();

    const batch = makeBatch();
    expect(batch.stats.meshes).toBe(2);
    expect(batch.meshes.length).toBe(2);
    expect(batch.levels.length).toBe(2);
    expect(batch.batched.isBatchedMesh).toBe(true);
    expect(batch.batched.parent).toBe(batch);

    const geometry = batch.batched.geometry;
    expect(geometry.getAttribute("position").count).toBe(knot.vertexCount + ball.vertexCount);
    expect(knot.vertexCount + ball.vertexCount).toBeLessThanOrEqual(65535);
    let total = 0;
    for (const levels of batch.levels) for (let j = 0; j < levels.count; j++) total += levels.triangles[j] * 3;
    expect(geometry.index!.count).toBe(total);
    expect(geometry.index!.array).toBeInstanceOf(Uint16Array);

    // more than 65535 vertices together: 32 bit indices
    const copies = Math.floor(65535 / knot.vertexCount) + 1;
    const many = new MeshletBatch(Array.from({ length: copies }, () => knot));
    const manyGeometry = many.batched.geometry;
    expect(manyGeometry.getAttribute("position").count).toBe(copies * knot.vertexCount);
    expect(copies * knot.vertexCount).toBeGreaterThan(65535);
    let manyTotal = 0;
    for (const levels of many.levels) for (let j = 0; j < levels.count; j++) manyTotal += levels.triangles[j] * 3;
    expect(manyGeometry.index!.count).toBe(manyTotal);
    expect(manyGeometry.index!.array).toBeInstanceOf(Uint32Array);
    // and the batch of the copies draws with them
    many.addInstance(copies - 1);
    many.update(makeCamera(0, 0, 5), H);
    const gl = makeWebGL();
    before(many, gl.renderer, makeCamera(0, 0, 5));
    expect(raw(many).drawCount).toBe(1);
    many.dispose();
    batch.dispose();
  }, BUILD_TIMEOUT);

  // =========================================================================
  // B. instances
  // =========================================================================

  it("B. addInstance, getMatrixAt, getMeshAt, range errors", () => {
    const batch = makeBatch();
    expect(batch.addInstance(0)).toBe(0);
    expect(batch.addInstance(1, translation(1, 2, 3))).toBe(1);
    expect(batch.addInstance(0, translation(-4, 5, 6))).toBe(2);
    expect(() => batch.addInstance(2)).toThrow(RangeError);
    expect(() => batch.addInstance(-1)).toThrow(RangeError);
    expect(batch.count).toBe(3);
    expect(batch.stats.instances).toBe(3);

    const m = new Matrix4();
    batch.getMatrixAt(1, m);
    expect(m.elements).toEqual(translation(1, 2, 3).elements);
    batch.getMatrixAt(0, m);
    expect(m.elements).toEqual(new Matrix4().elements);

    const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.7);
    const full = new Matrix4().compose(new Vector3(9, 8, 7), q, new Vector3(2, 2, 2));
    batch.setMatrixAt(2, full);
    batch.getMatrixAt(2, m);
    for (let e = 0; e < 16; e++) expect(m.elements[e]).toBeCloseTo(full.elements[e], 12);

    expect([0, 1, 2].map((i) => batch.getMeshAt(i))).toEqual([0, 1, 0]);
    expect(() => batch.setMatrixAt(3, new Matrix4())).toThrow();
    expect(() => batch.setMatrixAt(-1, new Matrix4())).toThrow();
    batch.dispose();
  });

  // =========================================================================
  // C. levels
  // =========================================================================

  it("C. levels follow the distance, the same as levelForError", () => {
    const batch = makeBatch({ drawCost: 0 });
    const distances = [0.2, 3, 8, 20, 50, 120, 300, 800, 2000, 5000];
    const ids: number[][] = [[], []];
    for (const mesh of [0, 1]) {
      for (const dist of distances) ids[mesh].push(batch.addInstance(mesh, matrixAt(mesh === 0 ? knot : ball, new Vector3(0, 0, -dist))));
    }
    const camera = makeCamera(0, 0, 0);
    update(batch, camera);

    const k = lodScale(1, H, camera.projectionMatrix.elements[5], 1);
    const m = new Matrix4();
    for (const mesh of [0, 1]) {
      const data = mesh === 0 ? knot : ball;
      const seen = new Set<number>();
      let previous = -1;
      for (const i of ids[mesh]) {
        const level = batch.getLevelAt(i);
        seen.add(level);
        expect(level, `mesh ${mesh} instance ${i} does not decrease`).toBeGreaterThanOrEqual(previous);
        previous = level;
        expect(batch.getTriangleCountAt(i)).toBe(batch.levels[mesh].triangles[level]);

        batch.getMatrixAt(i, m);
        const centre = new Vector3(data.boundingSphere[0], data.boundingSphere[1], data.boundingSphere[2]).applyMatrix4(m);
        const d = Math.max(0, centre.distanceTo(camera.position) - data.boundingSphere[3]);
        const expected = levelForError(batch.levels[mesh], k * Math.max(d, camera.near));
        expect(level, `mesh ${mesh} instance ${i} (distance ${d})`).toBe(expected);
      }
      if (mesh === 0) expect(seen.size).toBeGreaterThanOrEqual(3);
    }
    // the camera is inside the bounding sphere of the first of each
    expect(batch.getLevelAt(ids[0][0])).toBe(0);
    expect(batch.getLevelAt(ids[1][0])).toBe(0);
    batch.dispose();
  });

  // =========================================================================
  // D. drawCost
  // =========================================================================

  describe("D. drawCost", () => {
    /** Instances of both meshes at distances over the whole range. */
    function fill(batch: MeshletBatch): number[] {
      const ids: number[] = [];
      for (let j = 0; j < 30; j++) {
        const dist = 3 * Math.pow(1.25, j);
        ids.push(batch.addInstance(0, matrixAt(knot, new Vector3(j * 0.1, 0, -dist))));
        ids.push(batch.addInstance(1, matrixAt(ball, new Vector3(-j * 0.1, 0, -dist))));
      }
      return ids;
    }
    const levelsOf = (batch: MeshletBatch, ids: number[]) => ids.map((i) => batch.getLevelAt(i));

    it("D. one level per mesh with a huge drawCost, never coarser than asked, taken at run time", () => {
      const camera = makeCamera(0, 0, 0);
      const exact = makeBatch({ drawCost: 0 });
      const ids = fill(exact);
      update(exact, camera);
      const asked = levelsOf(exact, ids);
      const smallIds = ids.filter((i) => exact.getMeshAt(i) === 1);
      const smallAsked = new Set(smallIds.map((i) => exact.getLevelAt(i)));
      expect(smallAsked.size).toBeGreaterThanOrEqual(2);

      const merged = makeBatch({ drawCost: 1e9 });
      fill(merged);
      update(merged, camera);
      for (const mesh of [0, 1]) {
        const mine = ids.filter((i) => merged.getMeshAt(i) === mesh);
        const drawn = new Set(mine.map((i) => merged.getLevelAt(i)));
        expect(drawn.size, `mesh ${mesh} on one level`).toBe(1);
        expect([...drawn][0]).toBe(Math.min(...mine.map((i) => exact.getLevelAt(i))));
      }

      const middle = makeBatch();
      fill(middle);
      update(middle, camera);
      for (const b of [merged, middle]) {
        for (const i of ids) expect(b.getLevelAt(i), `instance ${i}`).toBeLessThanOrEqual(asked[i]);
      }

      // run time: nothing else changes
      const batch = makeBatch({ drawCost: 0 });
      fill(batch);
      update(batch, camera);
      expect(levelsOf(batch, ids)).toEqual(asked);
      batch.drawCost = 1e9;
      update(batch, camera);
      expect(levelsOf(batch, ids)).toEqual(levelsOf(merged, ids));
      expect(levelsOf(batch, ids)).not.toEqual(asked);
      batch.drawCost = 0;
      update(batch, camera);
      expect(levelsOf(batch, ids)).toEqual(asked);
      for (const b of [exact, merged, middle, batch]) b.dispose();
    });
  });

  // =========================================================================
  // E. incremental equals from scratch
  // =========================================================================

  describe("E. incremental equals from scratch", () => {
    const INSTANCES = 80;

    function makePlacements(rand: () => number): Array<{ mesh: number; matrix: Matrix4 }> {
      const out: Array<{ mesh: number; matrix: Matrix4 }> = [];
      for (let i = 0; i < INSTANCES; i++) {
        const position = new Vector3((rand() - 0.5) * 400, (rand() - 0.5) * 400, (rand() - 0.5) * 400);
        const s = i % 12 === 5 ? 0.3 + rand() * 3 : 1;
        const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), rand() * 6);
        out.push({ mesh: rand() < 0.5 ? 0 : 1, matrix: new Matrix4().compose(position, q, new Vector3(s, s, s)) });
      }
      return out;
    }

    function stepKind(rand: () => number): "tiny" | "small" | "jump" | "visit" {
      const r = rand();
      return r < 0.35 ? "tiny" : r < 0.7 ? "small" : r < 0.88 ? "jump" : "visit";
    }

    function makePair(options: MeshletBatchOptions, placements: Array<{ mesh: number; matrix: Matrix4 }>): [MeshletBatch, MeshletBatch] {
      const pair: [MeshletBatch, MeshletBatch] = [makeBatch(options), makeBatch(options)];
      for (const b of pair) placements.forEach((p) => b.addInstance(p.mesh, p.matrix));
      return pair;
    }

    function evaluateFromScratch(fresh: MeshletBatch, camera: Camera): void {
      const threshold = fresh.errorThreshold;
      fresh.errorThreshold = threshold + 1;
      update(fresh, camera);
      fresh.errorThreshold = threshold;
      update(fresh, camera);
    }

    function expectSame(walker: MeshletBatch, fresh: MeshletBatch, step: number, where: string): void {
      for (let i = 0; i < INSTANCES; i++) {
        const message = `step ${step}, instance ${i} (mesh ${walker.getMeshAt(i)}), ${where}`;
        expect(walker.getLevelAt(i), `level, ${message}`).toBe(fresh.getLevelAt(i));
        expect(walker.getTriangleCountAt(i), `triangles, ${message}`).toBe(fresh.getTriangleCountAt(i));
      }
    }

    const costs: Array<[string, MeshletBatchOptions]> = [
      ["drawCost 0", { drawCost: 0 }],
      ["the default drawCost", {}],
    ];

    for (const [name, options] of costs) {
      it(`E. walking the camera (${name})`, () => {
        const rand = seeded(0xc0ffee);
        const placements = makePlacements(rand);
        const [walker, fresh] = makePair(options, placements);
        const camera = makeCamera(0, 0, 300);
        const position = camera.position;
        const levels = new Set<number>();
        for (let step = 0; step < 300; step++) {
          const kind = stepKind(rand);
          if (kind === "tiny") position.addScaledVector(direction(rand), 1e-3);
          else if (kind === "small") position.addScaledVector(direction(rand), 0.5);
          else if (kind === "jump") position.addScaledVector(direction(rand), 50 + rand() * 250);
          else {
            const target = new Vector3().setFromMatrixPosition(placements[Math.floor(rand() * INSTANCES)].matrix);
            position.copy(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
          }
          update(walker, camera);
          evaluateFromScratch(fresh, camera);
          expectSame(walker, fresh, step, `camera at ${position.toArray().join(", ")}`);
          for (let i = 0; i < INSTANCES; i++) levels.add(walker.getLevelAt(i));
        }
        expect(levels.size).toBeGreaterThanOrEqual(4);
      });

      it(`E. moving the batch (${name})`, () => {
        const rand = seeded(0xbadf00d);
        const placements = makePlacements(rand);
        const [walker, fresh] = makePair(options, placements);
        const camera = makeCamera(20, 10, 100);
        const where = new Vector3();
        const levels = new Set<number>();
        for (let step = 0; step < 300; step++) {
          const kind = stepKind(rand);
          if (kind === "tiny") where.addScaledVector(direction(rand), 1e-3);
          else if (kind === "small") where.addScaledVector(direction(rand), 0.5);
          else if (kind === "jump") where.addScaledVector(direction(rand), 50 + rand() * 250);
          else {
            const target = new Vector3().setFromMatrixPosition(placements[Math.floor(rand() * INSTANCES)].matrix);
            where.copy(camera.position).sub(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
          }
          for (const b of [walker, fresh]) {
            b.position.copy(where);
            b.updateMatrixWorld(true);
          }
          update(walker, camera);
          evaluateFromScratch(fresh, camera);
          expectSame(walker, fresh, step, `batch at ${where.toArray().join(", ")}`);
          for (let i = 0; i < INSTANCES; i++) levels.add(walker.getLevelAt(i));
        }
        expect(levels.size).toBeGreaterThanOrEqual(4);
      });

      it(`E. moving the batch and setMatrixAt every 10 steps (${name})`, () => {
        const rand = seeded(0x5eed5);
        const placements = makePlacements(rand);
        const [walker, fresh] = makePair(options, placements);
        const camera = makeCamera(20, 10, 100);
        const where = new Vector3();
        for (let step = 0; step < 300; step++) {
          const kind = stepKind(rand);
          if (kind === "tiny") where.addScaledVector(direction(rand), 1e-3);
          else if (kind === "small") where.addScaledVector(direction(rand), 0.5);
          else if (kind === "jump") where.addScaledVector(direction(rand), 50 + rand() * 250);
          else {
            const target = new Vector3().setFromMatrixPosition(placements[Math.floor(rand() * INSTANCES)].matrix);
            where.copy(camera.position).sub(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
          }
          for (const b of [walker, fresh]) {
            b.position.copy(where);
            b.updateMatrixWorld(true);
          }
          if (step % 10 === 5) {
            const i = Math.floor(rand() * INSTANCES);
            const s = rand() < 0.3 ? 0.3 + rand() * 3 : 1;
            const target =
              rand() < 0.5
                ? camera.position.clone().sub(where).addScaledVector(direction(rand), rand() * 6)
                : new Vector3().setFromMatrixPosition(placements[i].matrix).addScaledVector(direction(rand), rand() * 200);
            const m = new Matrix4().compose(target, new Quaternion(), new Vector3(s, s, s));
            placements[i].matrix = m;
            walker.setMatrixAt(i, m);
            fresh.setMatrixAt(i, m);
          }
          update(walker, camera);
          evaluateFromScratch(fresh, camera);
          expectSame(walker, fresh, step, `batch at ${where.toArray().join(", ")}`);
        }
      });
    }
  });

  // =========================================================================
  // F. WebGL lists
  // =========================================================================

  describe("F. WebGL lists", () => {
    function setup(options?: MeshletBatchOptions) {
      const batch = makeBatch({ drawCost: 0, ...options });
      scatter(batch);
      const camera = makeCamera(0, 0, 0);
      update(batch, camera);
      const gl = makeWebGL();
      before(batch, gl.renderer, camera);
      return { batch, camera, gl };
    }

    it("F. drawAll lists every instance once, grouped by mesh and level", () => {
      const { batch } = setup();
      expectAllListed(batch);
      const levels = new Set<number>();
      for (let i = 0; i < batch.count; i++) levels.add(batch.getLevelAt(i));
      expect(levels.size).toBeGreaterThanOrEqual(2);
    });

    it("F. drawSeen lists exactly the instances in view", () => {
      const { batch, camera } = setup();
      const expected = expectedInView(batch, camera);
      expect(expected.length).toBeGreaterThan(0);
      expect(expected.length).toBeLessThan(batch.count);

      const bpe = (batch.batched.geometry.index!.array as Uint16Array).BYTES_PER_ELEMENT;
      const runs = readGL(batch, "seen");
      const ids: number[] = [];
      for (const run of runs) {
        ids.push(...run.ids);
        expectGroupsRight(batch, [{ ids: run.ids, indexCount: run.indexCount, indexStart: run.offset / bpe }], `sub-draw ${run.d}`);
      }
      expect(sorted(ids)).toEqual(expected);
      expect(batch.stats.visible).toBe(expected.length);
    });

    it("F. onAfterRender and onAfterShadow make the one multi-draw call", () => {
      const { batch, camera, gl } = setup();
      const r = raw(batch);
      const type = (batch.batched.geometry.index!.array as Uint16Array).BYTES_PER_ELEMENT === 2 ? 5123 : 5125;
      after(batch, gl.renderer, camera);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenCalledTimes(1);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenCalledWith(4, r.drawCounts, 0, type, r.drawOffsets, 0, r.drawSeen, 0, r.drawCount);
      expect(gl.renderer.info.render.calls).toBe(1);
      expect(gl.gl.drawElementsInstanced).not.toHaveBeenCalled();

      gl.extension.multiDrawElementsInstancedWEBGL.mockClear();
      const shadowCamera = new PerspectiveCamera(60, 1, 0.1, 500);
      beforeShadow(batch, gl.renderer, camera, shadowCamera);
      afterShadow(batch, gl.renderer, camera, shadowCamera);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenCalledTimes(1);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenCalledWith(4, r.drawCounts, 0, type, r.drawOffsets, 0, r.drawShadow, 0, r.drawCount);
      expect(gl.renderer.info.render.calls).toBe(2);
    });

    it("F. cullInstances false draws every instance in the main pass", () => {
      const { batch, camera, gl } = setup({ cullInstances: false });
      after(batch, gl.renderer, camera);
      const call = gl.extension.multiDrawElementsInstancedWEBGL.mock.calls[0];
      const counts = call[6] as Int32Array;
      const D = call[8] as number;
      let sum = 0;
      for (let d = 0; d < D; d++) sum += counts[d];
      expect(sum).toBe(batch.count);
      expect(batch.stats.visible).toBe(batch.count);
    });

    it("F. without WEBGL_multi_draw it draws each sub-draw with instances, after setting its number", () => {
      const { batch, camera, gl } = setup();
      gl.state.multiDraw = false;
      after(batch, gl.renderer, camera);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).not.toHaveBeenCalled();
      const r = raw(batch);
      const type = (batch.batched.geometry.index!.array as Uint16Array).BYTES_PER_ELEMENT === 2 ? 5123 : 5125;
      const withInstances: number[] = [];
      for (let d = 0; d < r.drawCount; d++) if (r.drawSeen[d] > 0) withInstances.push(d);
      expect(withInstances.length).toBeGreaterThan(0);
      expect(withInstances.length).toBeLessThan(r.drawCount); // some sub-draw has no instance in view
      expect(gl.gl.drawElementsInstanced).toHaveBeenCalledTimes(withInstances.length);
      expect(gl.setValue).toHaveBeenCalledTimes(withInstances.length);
      withInstances.forEach((d, n) => {
        expect(gl.setValue.mock.calls[n]).toEqual([gl.gl, "_gl_DrawID", d]);
        expect(gl.gl.drawElementsInstanced.mock.calls[n]).toEqual([4, r.drawCounts[d], type, r.drawOffsets[d], r.drawSeen[d]]);
        expect(gl.setValue.mock.invocationCallOrder[n]).toBeLessThan(gl.gl.drawElementsInstanced.mock.invocationCallOrder[n]);
      });
    });
  });

  // =========================================================================
  // G. WebGPU lists
  // =========================================================================

  it("G. WebGPU lists", () => {
    const batch = makeBatch({ drawCost: 0 });
    scatter(batch);
    const camera = makeCamera(0, 0, 0);
    update(batch, camera);
    const gpu = makeWebGPU();
    before(batch, gpu, camera);

    if (!(Number.parseInt(REVISION, 10) >= 187)) {
      expect(batchedRaw(batch)._mergedDrawCount).toBe(0);
      return;
    }

    const index = batch.batched.geometry.index!.array as Uint16Array;
    const b = batchedRaw(batch);
    expect(b._multiDrawBytesPerElement).toBe(index.BYTES_PER_ELEMENT);
    for (let e = 0; e < b._mergedDrawCount; e++) expect(b._mergedDrawStarts[e] % index.BYTES_PER_ELEMENT).toBe(0);

    const expected = expectedInView(batch, camera);
    expect(expected.length).toBeGreaterThan(0);
    expect(expected.length).toBeLessThan(batch.count);
    const entries = readGPU(batch);
    const drawn = entries.filter((e) => e.indexCount > 0);
    const ids: number[] = [];
    for (const entry of drawn) ids.push(...entry.ids);
    expect(sorted(ids)).toEqual(expected);
    expectGroupsRight(batch, drawn, "main pass");
    // (an entry with no count draws nothing, whatever its instances)
    for (const entry of entries) if (entry.indexCount === 0) expect(entry.ids.length).toBeGreaterThan(0);

    // the shadow pass
    const scene = new Scene();
    scene.overrideMaterial = { isShadowPassMaterial: true } as never;
    before(batch, gpu, camera, scene);
    const all = readGPU(batch);
    const allIds: number[] = [];
    for (const entry of all) {
      expect(entry.indexCount).toBeGreaterThan(0);
      allIds.push(...entry.ids);
    }
    expect(sorted(allIds)).toEqual(Array.from({ length: batch.count }, (_, i) => i));
    expectGroupsRight(batch, all, "shadow pass");
    expect(b._multiDrawBytesPerElement).toBe(index.BYTES_PER_ELEMENT);
    for (let e = 0; e < b._mergedDrawCount; e++) expect(b._mergedDrawStarts[e] % index.BYTES_PER_ELEMENT).toBe(0);

    // and the main pass again
    before(batch, gpu, camera);
    const again = readGPU(batch).filter((e) => e.indexCount > 0);
    expect(sorted(again.flatMap((e) => e.ids))).toEqual(expected);
  });

  // =========================================================================
  // H. visibility
  // =========================================================================

  it("H. setVisibleAt takes an instance out of every list and back", () => {
    const batch = makeBatch({ drawCost: 0 });
    scatter(batch);
    const camera = makeCamera(0, 0, 0);
    update(batch, camera);
    const gl = makeWebGL();
    before(batch, gl.renderer, camera);

    const inView = expectedInView(batch, camera);
    const i = inView[0];
    const listed = (which: "seen" | "all") => readGL(batch, which).flatMap((r) => r.ids);
    expect(listed("seen")).toContain(i);
    expect(listed("all")).toContain(i);
    expect(batch.getVisibleAt(i)).toBe(true);

    batch.setVisibleAt(i, false);
    expect(batch.getVisibleAt(i)).toBe(false);
    update(batch, camera);
    before(batch, gl.renderer, camera);
    expect(listed("seen")).not.toContain(i);
    expect(listed("all")).not.toContain(i);
    expect(sorted(listed("all"))).toEqual(Array.from({ length: batch.count }, (_, k) => k).filter((k) => k !== i));

    batch.setVisibleAt(i, true);
    expect(batch.getVisibleAt(i)).toBe(true);
    update(batch, camera);
    before(batch, gl.renderer, camera);
    expect(listed("seen")).toContain(i);
    expect(listed("all")).toContain(i);
    expect(sorted(listed("seen"))).toEqual(inView);
  });

  // =========================================================================
  // I. growth
  // =========================================================================

  it("I. growing the capacity", () => {
    const batch = makeBatch({ drawCost: 0, maxInstances: 2 });
    const gl = makeWebGL();
    const camera = makeCamera(0, 0, 0);
    const rand = seeded(0xabc);
    const matrices: Matrix4[] = [];
    const add = (n: number) => {
      for (let j = 0; j < n; j++) {
        const i = matrices.length;
        const m = new Matrix4().compose(
          new Vector3((rand() - 0.5) * 20, (rand() - 0.5) * 10, -(5 + rand() * 300)),
          new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), rand()),
          new Vector3(1 + (i % 3) * 0.5, 1 + (i % 3) * 0.5, 1 + (i % 3) * 0.5),
        );
        matrices.push(m);
        expect(batch.addInstance(i % 2, m)).toBe(i);
      }
    };
    add(2);
    update(batch, camera);
    before(batch, gl.renderer, camera);
    const old = batch.batched;

    expect(() => add(10)).not.toThrow();
    expect(batch.count).toBe(12);
    expect(batch.batched).not.toBe(old);
    expect(old.parent).toBeNull();
    expect(batch.children).not.toContain(old);
    expect(batch.children).toContain(batch.batched);

    const m = new Matrix4();
    for (let i = 0; i < 12; i++) {
      batch.batched.getMatrixAt(i, m);
      // (the draw object keeps them in a float32 texture)
      for (let e = 0; e < 16; e++) expect(m.elements[e]).toBeCloseTo(matrices[i].elements[e], 3);
      batch.getMatrixAt(i, m);
      for (let e = 0; e < 16; e++) expect(m.elements[e]).toBeCloseTo(matrices[i].elements[e], 12);
    }

    update(batch, camera);
    for (let i = 0; i < 12; i++) expect(batch.getLevelAt(i)).toBeGreaterThanOrEqual(0);
    before(batch, gl.renderer, camera);
    expectAllListed(batch);
  });

  // =========================================================================
  // J. materials
  // =========================================================================

  describe("J. materials", () => {
    const shader = () => ({ vertexShader: "a\n#include <batching_vertex>\nb" });

    it("J. a derived material and depth materials on WebGL, the given one on WebGPU", () => {
      const given = new MeshStandardMaterial();
      const spy = vi.fn();
      given.onBeforeCompile = spy;
      const batch = makeBatch({}, given);
      expect(batch.material).toBe(given);

      const check = () => {
        const derived = batch.batched.material as MeshStandardMaterial;
        expect(derived).not.toBe(given);
        expect(Object.getPrototypeOf(derived)).toBe(given);
        spy.mockClear();
        const s = shader();
        derived.onBeforeCompile(s as never, null as never);
        expect(spy).toHaveBeenCalled();
        expect(s.vertexShader).toContain("gl_InstanceID");
        expect(s.vertexShader).not.toContain("#include <batching_vertex>");
        expect(derived.customProgramCacheKey()).not.toBe(given.customProgramCacheKey());

        const depth = batch.batched.customDepthMaterial;
        expect(depth).toBeInstanceOf(MeshDepthMaterial);
        const d = shader();
        depth!.onBeforeCompile(d as never, null as never);
        expect(d.vertexShader).toContain("gl_InstanceID");
        expect(d.vertexShader).not.toContain("#include <batching_vertex>");
      };
      check(); // (no renderer met yet)

      const camera = makeCamera(0, 0, 0);
      batch.addInstance(0, translation(0, 0, -5));
      update(batch, camera);
      before(batch, makeWebGL().renderer, camera);
      check();

      const other = new MeshBasicMaterial();
      batch.material = other;
      expect(batch.material).toBe(other);
      expect(Object.getPrototypeOf(batch.batched.material)).toBe(other);
      expect(batch.batched.customDepthMaterial).toBeInstanceOf(MeshDepthMaterial);

      // WebGPU
      const gpuBatch = makeBatch({}, given);
      gpuBatch.addInstance(0, translation(0, 0, -5));
      update(gpuBatch, camera);
      before(gpuBatch, makeWebGPU(), camera);
      expect(gpuBatch.batched.material).toBe(given);
      expect(gpuBatch.batched.customDepthMaterial).toBeUndefined();
      gpuBatch.material = other;
      expect(gpuBatch.batched.material).toBe(other);
      expect(gpuBatch.batched.customDepthMaterial).toBeUndefined();
    });

    it("J. alphaTest and map reach the depth material", () => {
      const map = new Texture();
      const given = new MeshStandardMaterial({ alphaTest: 0.5, map });
      const batch = makeBatch({}, given);
      const depth = batch.batched.customDepthMaterial as MeshDepthMaterial;
      expect(depth.alphaTest).toBe(0.5);
      expect(depth.map).toBe(map);
      before(batch, makeWebGL().renderer, makeCamera(0, 0, 0));
      const again = batch.batched.customDepthMaterial as MeshDepthMaterial;
      expect(again.alphaTest).toBe(0.5);
      expect(again.map).toBe(map);
    });
  });

  // =========================================================================
  // K. forwarded properties
  // =========================================================================

  it("K. castShadow, receiveShadow, renderOrder are the draw object's, also after growth", () => {
    const batch = makeBatch({ maxInstances: 1 });
    batch.castShadow = true;
    batch.receiveShadow = true;
    batch.renderOrder = 7;
    expect(batch.batched.castShadow).toBe(true);
    expect(batch.batched.receiveShadow).toBe(true);
    expect(batch.batched.renderOrder).toBe(7);
    expect(batch.castShadow).toBe(true);
    expect(batch.renderOrder).toBe(7);

    const old = batch.batched;
    for (let i = 0; i < 5; i++) batch.addInstance(i % 2);
    expect(batch.batched).not.toBe(old);
    expect(batch.batched.castShadow).toBe(true);
    expect(batch.batched.receiveShadow).toBe(true);
    expect(batch.batched.renderOrder).toBe(7);
    expect(batch.castShadow).toBe(true);
    expect(batch.receiveShadow).toBe(true);
    expect(batch.renderOrder).toBe(7);

    batch.castShadow = false;
    batch.renderOrder = 2;
    expect(batch.batched.castShadow).toBe(false);
    expect(batch.batched.renderOrder).toBe(2);
  });

  // =========================================================================
  // L. raycasting
  // =========================================================================

  it("L. raycasting reports the batch, the instance and the mesh", () => {
    const knotAt = new Vector3(30, 0, -10);
    const ballAt = new Vector3(-30, 0, -10);

    const run = (offset: Vector3) => {
      const batch = makeBatch({ drawCost: 0 });
      batch.position.copy(offset);
      batch.updateMatrixWorld(true);
      const k = batch.addInstance(0, translation(knotAt.x, knotAt.y, knotAt.z));
      const s = batch.addInstance(1, translation(ballAt.x, ballAt.y, ballAt.z));

      // the knot: through a vertex of it, along its normal; the camera is inside its bounding sphere, so it is drawn whole
      const position = knot.attributes.find((a) => a.name === "position")!.array;
      const normal = knot.attributes.find((a) => a.name === "normal")!.array;
      const v = new Vector3(position[0], position[1], position[2]).add(knotAt).add(offset);
      const n = new Vector3(normal[0], normal[1], normal[2]).normalize();
      const kc = new Vector3(knot.boundingSphere[0], knot.boundingSphere[1], knot.boundingSphere[2]).add(knotAt).add(offset);
      update(batch, makeCamera(kc.x, kc.y, kc.z));
      expect(batch.getLevelAt(k)).toBe(0);
      const hits = new Raycaster(v.clone().addScaledVector(n, 2), n.clone().negate()).intersectObject(batch, true);
      expect(hits.length).toBeGreaterThan(0);
      for (const hit of hits) {
        expect(hit.object).toBe(batch);
        expect(hit.instanceId).toBe(k);
        expect(hit.batchId).toBe(0);
      }

      // the ball
      const bc = new Vector3(ball.boundingSphere[0], ball.boundingSphere[1], ball.boundingSphere[2]).add(ballAt).add(offset);
      update(batch, makeCamera(bc.x, bc.y, bc.z));
      expect(batch.getLevelAt(s)).toBe(0);
      const ballHits = new Raycaster(bc.clone().add(new Vector3(0, 0, 10)), new Vector3(0, 0, -1)).intersectObject(batch, true);
      expect(ballHits.length).toBeGreaterThan(0);
      for (const hit of ballHits) {
        expect(hit.object).toBe(batch);
        expect(hit.instanceId).toBe(s);
        expect(hit.batchId).toBe(1);
      }

      // nothing
      const none = new Raycaster(new Vector3(0, 500, 0).add(offset), new Vector3(0, 1, 0)).intersectObject(batch, true);
      expect(none.length).toBe(0);
      const miss = new Raycaster(new Vector3(0, 0, 0).add(offset), new Vector3(0, 0, -1)).intersectObject(batch, true);
      expect(miss.length).toBe(0);
      batch.dispose();
    };
    run(new Vector3());
    run(new Vector3(100, 20, -7));
  });

  // =========================================================================
  // M. orthographic
  // =========================================================================

  it("M. an orthographic camera gives every instance of a mesh the same level", () => {
    const batch = makeBatch({ drawCost: 0 });
    for (const dist of [1, 5, 20, 90, 400, 900]) {
      batch.addInstance(0, matrixAt(knot, new Vector3(0, 0, -dist)));
      batch.addInstance(1, matrixAt(ball, new Vector3(3, 0, -dist)));
    }
    const camera = new OrthographicCamera(-20, 20, 20, -20, 0.1, 2000);
    camera.position.set(0, 0, 0);
    camera.updateMatrixWorld(true);
    update(batch, camera);
    for (const mesh of [0, 1]) {
      const levels = new Set<number>();
      for (let i = 0; i < batch.count; i++) if (batch.getMeshAt(i) === mesh) levels.add(batch.getLevelAt(i));
      expect(levels.size, `mesh ${mesh}`).toBe(1);
    }
    // moving the camera along the view axis changes nothing
    const before0 = Array.from({ length: batch.count }, (_, i) => batch.getLevelAt(i));
    camera.position.set(0, 0, 300);
    update(batch, camera);
    expect(Array.from({ length: batch.count }, (_, i) => batch.getLevelAt(i))).toEqual(before0);
  });

  // =========================================================================
  // N. dispose
  // =========================================================================

  it("N. dispose does not throw", () => {
    const fresh = makeBatch();
    expect(() => fresh.dispose()).not.toThrow();

    const batch = makeBatch({ drawCost: 0 });
    batch.castShadow = true;
    scatter(batch);
    const camera = makeCamera(0, 0, 0);
    update(batch, camera);
    const gl = makeWebGL();
    before(batch, gl.renderer, camera);
    after(batch, gl.renderer, camera);
    const shadowCamera = new PerspectiveCamera(60, 1, 0.1, 500);
    beforeShadow(batch, gl.renderer, camera, shadowCamera);
    afterShadow(batch, gl.renderer, camera, shadowCamera);
    expect(() => batch.dispose()).not.toThrow();

    const gpuBatch = makeBatch();
    scatter(gpuBatch);
    update(gpuBatch, camera);
    before(gpuBatch, makeWebGPU(), camera);
    expect(() => gpuBatch.dispose()).not.toThrow();
    expect(new BufferGeometry()).toBeDefined();
  });

  it("O. debugColors 'lod' tints every instance by the depth of the level it is drawn at", () => {
    const colorsOf = (batch: MeshletBatch) => (batch.batched as unknown as { _colorsTexture: { image: { data: Float32Array } } | null })._colorsTexture;
    const expectTints = (batch: MeshletBatch, lod: boolean) => {
      const data = colorsOf(batch)!.image.data;
      for (let i = 0; i < batch.count; i++) {
        const levels = batch.levels[batch.getMeshAt(i)];
        const want = lod ? LOD_PALETTE[levels.depth[batch.getLevelAt(i)] % LOD_PALETTE.length] : new Color(1, 1, 1);
        expect([data[i * 4], data[i * 4 + 1], data[i * 4 + 2]]).toEqual([want.r, want.g, want.b].map(Math.fround));
      }
    };
    for (const renderer of [() => makeWebGL().renderer, makeWebGPU]) {
      const batch = makeBatch({ drawCost: 0, maxInstances: 8 });
      scatter(batch);
      const camera = makeCamera(0, 0, 0);
      update(batch, camera);
      before(batch, renderer(), camera);
      expect(colorsOf(batch)).toBeNull(); // (no colours until asked for)

      const version = (batch.batched.material as MeshStandardMaterial).version;
      batch.debugColors = "lod";
      before(batch, renderer(), camera);
      // (WebGPURenderer only notices the new colour texture through the material)
      expect((batch.batched.material as MeshStandardMaterial).version).toBeGreaterThan(version);
      const levels = new Set<number>();
      for (let i = 0; i < batch.count; i++) levels.add(batch.getLevelAt(i));
      expect(levels.size).toBeGreaterThanOrEqual(2);
      expectTints(batch, true);

      // moving away changes levels, and the tints follow
      const far = makeCamera(0, 0, 400);
      update(batch, far);
      before(batch, renderer(), far);
      expectTints(batch, true);

      // growing makes a new draw object, painted again
      const k = batch.count;
      for (let i = 0; i < k; i++) batch.addInstance(i % 2, batch.getMatrixAt(i, new Matrix4()));
      update(batch, far);
      before(batch, renderer(), far);
      expectTints(batch, true);

      batch.debugColors = "none";
      before(batch, renderer(), far);
      expectTints(batch, false);
    }
  });

  it("O. the derived material looks colours up per instance as it does matrices", () => {
    const batch = makeBatch({}, new MeshStandardMaterial());
    const s = { vertexShader: "#include <batching_vertex>\n#include <color_vertex>" };
    (batch.batched.material as MeshStandardMaterial).onBeforeCompile(s as never, null as never);
    expect(s.vertexShader).not.toContain("#include <color_vertex>");
    expect(s.vertexShader).toContain("getBatchingColor( getIndirectIndex( int( getIndirectIndex( gl_DrawID ) ) + gl_InstanceID ) )");
    expect(s.vertexShader).not.toContain("getBatchingColor( getIndirectIndex( gl_DrawID ) )");
  });

  describe("P. culling at scale, and shadow passes culled for their camera", () => {
    /** 600 instances spread over a wide area, so the culling goes by chunks. */
    function setup() {
      const batch = makeBatch({ drawCost: 0 });
      const rand = seeded(0xc0ffee);
      for (let i = 0; i < 600; i++) {
        const mesh = i % 2;
        batch.addInstance(mesh, matrixAt(mesh === 0 ? knot : ball, new Vector3((rand() - 0.5) * 600, (rand() - 0.5) * 20, (rand() - 0.5) * 600)));
      }
      const camera = makeCamera(0, 2, 0);
      camera.lookAt(30, 0, -100);
      camera.updateMatrixWorld(true);
      const shadow = new OrthographicCamera(-60, 60, 60, -60, 0.1, 2000);
      shadow.position.set(150, 300, -100);
      shadow.lookAt(60, 0, -60);
      shadow.updateMatrixWorld(true);
      return { batch, camera, shadow };
    }
    const union = (a: number[], b: number[]) => sorted([...new Set([...a, ...b])]);

    it("P. the main pass draws exactly the instances in view", () => {
      const { batch, camera } = setup();
      update(batch, camera);
      const gl = makeWebGL();
      before(batch, gl.renderer, camera);
      const expected = expectedInView(batch, camera);
      expect(expected.length).toBeGreaterThan(20);
      expect(expected.length).toBeLessThan(batch.count - 100);
      expect(sorted(readGL(batch, "seen").flatMap((r) => r.ids))).toEqual(expected);

      // a moved instance is culled where it now is
      const m = new Matrix4();
      const gone = expected[0];
      batch.setMatrixAt(gone, matrixAt(batch.meshes[batch.getMeshAt(gone)], new Vector3(0, 0, 400)));
      const outside = batch.count - 1 - [...Array(batch.count).keys()].reverse().findIndex((i) => !expected.includes(i));
      batch.getMatrixAt(outside, m);
      batch.setMatrixAt(outside, matrixAt(batch.meshes[batch.getMeshAt(outside)], new Vector3(3, 2, -10)));
      update(batch, camera);
      before(batch, gl.renderer, camera);
      const now = expectedInView(batch, camera);
      expect(now).toContain(outside);
      expect(now).not.toContain(gone);
      expect(sorted(readGL(batch, "seen").flatMap((r) => r.ids))).toEqual(now);
    });

    it("P. WebGL: a shadow pass draws the instances in view and those in the shadow camera's", () => {
      const { batch, camera, shadow } = setup();
      update(batch, camera);
      const gl = makeWebGL();
      beforeShadow(batch, gl.renderer, camera, shadow);
      afterShadow(batch, gl.renderer, camera, shadow);
      const r = raw(batch);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenLastCalledWith(4, r.drawCounts, 0, expect.any(Number), r.drawOffsets, 0, r.drawShadow, 0, r.drawCount);
      const main = expectedInView(batch, camera);
      const lit = expectedInView(batch, shadow);
      expect(lit.filter((i) => !main.includes(i)).length).toBeGreaterThan(5);
      const want = union(main, lit);
      expect(want.length).toBeLessThan(batch.count - 100);
      const bpe = (batch.batched.geometry.index!.array as Uint16Array | Uint32Array).BYTES_PER_ELEMENT;
      const runs = readGL(batch, "shadow");
      for (const run of runs) expectGroupsRight(batch, [{ ids: run.ids, indexCount: run.indexCount, indexStart: run.offset / bpe }], `shadow sub-draw ${run.d}`);
      expect(sorted(runs.flatMap((run) => run.ids))).toEqual(want);
      // the main pass that follows draws the view only
      before(batch, gl.renderer, camera);
      after(batch, gl.renderer, camera);
      expect(sorted(readGL(batch, "seen").flatMap((run) => run.ids))).toEqual(main);
      // a shadow camera it was not culled for gets everything
      const other = shadow.clone();
      other.position.x += 50;
      other.updateMatrixWorld(true);
      afterShadow(batch, gl.renderer, camera, other);
      expect(gl.extension.multiDrawElementsInstancedWEBGL).toHaveBeenLastCalledWith(4, r.drawCounts, 0, expect.any(Number), r.drawOffsets, 0, r.drawAll, 0, r.drawCount);
    });

    it("P. WebGPU: the same, through the merged draws", () => {
      const { batch, camera, shadow } = setup();
      update(batch, camera);
      const gpu = makeWebGPU();
      before(batch, gpu, camera);
      const main = expectedInView(batch, camera);
      const scene = new Scene();
      scene.overrideMaterial = { isShadowNodeMaterial: true } as never;
      beforeShadow(batch, gpu, camera, shadow);
      before(batch, gpu, shadow, scene);
      const drawn = readGPU(batch).filter((e) => e.indexCount > 0);
      expectGroupsRight(batch, drawn, "shadow pass");
      expect(sorted(drawn.flatMap((e) => e.ids))).toEqual(union(main, expectedInView(batch, shadow)));
      before(batch, gpu, camera);
      expect(sorted(readGPU(batch).filter((e) => e.indexCount > 0).flatMap((e) => e.ids))).toEqual(main);
    });
  });
});
