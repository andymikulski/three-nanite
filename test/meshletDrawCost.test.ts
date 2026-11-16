import { beforeAll, describe, expect, it } from "vitest";
import { Matrix4, MeshBasicMaterial, Object3D, PerspectiveCamera, Quaternion, Scene, TorusKnotGeometry, Vector3 } from "three";
import type { InstancedMesh } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
import type { MeshletMeshOptions } from "../src/three/MeshletMesh.js";
import type { MeshletData } from "../src/types.js";
import { makeFixture } from "./helpers/fixture.js";

const H = 1080;
const DEFAULT_DRAW_COST = 4096;

function makeCamera(x: number, y: number, z: number): PerspectiveCamera {
  const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 10000);
  camera.position.set(x, y, z);
  camera.updateMatrixWorld(true);
  return camera;
}

const translation = (x: number, y = 0, z = 0) => new Matrix4().makeTranslation(x, y, z);

function makeFixtureMesh(options?: MeshletMeshOptions): MeshletMesh {
  return new MeshletMesh(makeFixture(), new MeshBasicMaterial(), options);
}

/** The level mesh of fixture level `level` (level 0 draws indices from 0, level 1 from 384). */
function levelMeshOf(mesh: MeshletMesh, level: number): InstancedMesh {
  const found = mesh.levelMeshes.find((lm) => lm.geometry.drawRange.start === (level === 0 ? 0 : 384));
  if (found === undefined) throw new Error(`no mesh for level ${level}`);
  return found;
}

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

describe("MeshletMesh drawCost on the fixture", () => {
  /** Three instances: one right in front of the camera, two far away; with drawCost 0 they are on two levels. */
  function place(mesh: MeshletMesh): void {
    mesh.count = 3;
    mesh.setMatrixAt(0, translation(0, 0, 0));
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.setMatrixAt(2, translation(0, 1000, 0));
  }
  const near = () => makeCamera(0, 0, 5);

  it("merges levels: instances that sit on several levels share the finest of them", () => {
    const exact = makeFixtureMesh({ drawCost: 0 });
    place(exact);
    exact.update(near(), H);
    const levels = [0, 1, 2].map((i) => exact.getLevelAt(i));
    expect(new Set(levels).size).toBeGreaterThanOrEqual(2);
    expect(exact.stats.levelDraws).toBe(2);
    const finest = Math.min(...levels);

    const merged = makeFixtureMesh();
    expect(merged.drawCost).toBe(DEFAULT_DRAW_COST);
    place(merged);
    merged.update(near(), H);
    expect(merged.stats.levelDraws).toBe(1);
    expect(merged.stats.draws).toBe(1);
    for (let i = 0; i < 3; i++) expect(merged.getLevelAt(i)).toBe(finest);
    expect(merged.stats.triangles).toBe(3 * 128);
  });

  it("never draws an instance coarser than it asks for, on the fixture and on a spread of distances", () => {
    const exact = makeFixtureMesh({ drawCost: 0 });
    const merged = makeFixtureMesh();
    const n = 40;
    for (const mesh of [exact, merged]) {
      mesh.count = n;
      for (let i = 0; i < n; i++) mesh.setMatrixAt(i, translation(0, 0, -(2 + i * i * 1.5)));
    }
    const camera = makeCamera(0, 0, 0);
    exact.update(camera, H);
    merged.update(camera, H);
    const seen = new Set<number>();
    for (let i = 0; i < n; i++) {
      const asked = exact.getLevelAt(i);
      const drawn = merged.getLevelAt(i);
      seen.add(asked);
      if (asked === -1 || drawn === -1) continue;
      expect(drawn, `instance ${i}`).toBeLessThanOrEqual(asked);
    }
    expect(seen.size).toBeGreaterThanOrEqual(2); // the spread does reach both levels
  });

  it("takes a changed drawCost on the next update with an unchanged camera and instances", () => {
    const mesh = makeFixtureMesh();
    place(mesh);
    const camera = near();
    mesh.update(camera, H);
    expect([0, 1, 2].map((i) => mesh.getLevelAt(i))).toEqual([0, 0, 0]);
    expect(mesh.stats.levelDraws).toBe(1);

    mesh.drawCost = 0;
    mesh.update(camera, H);
    expect([0, 1, 2].map((i) => mesh.getLevelAt(i))).toEqual([0, 1, 1]);
    expect(mesh.stats.levelDraws).toBe(2);
    expect(mesh.stats.triangles).toBe(128 + 32 + 32);

    mesh.drawCost = DEFAULT_DRAW_COST;
    mesh.update(camera, H);
    expect([0, 1, 2].map((i) => mesh.getLevelAt(i))).toEqual([0, 0, 0]);
    expect(mesh.stats.levelDraws).toBe(1);
  });

  it("attaches level meshes while they have instances and detaches them when they empty", () => {
    const mesh = makeFixtureMesh({ drawCost: 0 });
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    const check = () => {
      expect(mesh.children.length).toBe(mesh.stats.levelDraws + (mesh.batched.parent === mesh ? 1 : 0));
      for (const lm of mesh.levelMeshes) expect(lm.parent).toBe(lm.count > 0 ? mesh : null);
    };

    // instance 0 on the fine level, instance 1 on the coarse one
    mesh.update(makeCamera(0, 0, 5), H);
    expect([mesh.getLevelAt(0), mesh.getLevelAt(1)]).toEqual([0, 1]);
    expect(levelMeshOf(mesh, 0).parent).toBe(mesh);
    expect(levelMeshOf(mesh, 1).parent).toBe(mesh);
    expect(mesh.stats.levelDraws).toBe(2);
    check();

    // far away: the fine level empties
    mesh.update(makeCamera(0, 0, 500), H);
    expect([mesh.getLevelAt(0), mesh.getLevelAt(1)]).toEqual([1, 1]);
    expect(levelMeshOf(mesh, 0).count).toBe(0);
    expect(levelMeshOf(mesh, 0).parent).toBeNull();
    expect(levelMeshOf(mesh, 1).parent).toBe(mesh);
    expect(mesh.stats.levelDraws).toBe(1);
    check();

    // and back
    mesh.update(makeCamera(0, 0, 5), H);
    expect(levelMeshOf(mesh, 0).parent).toBe(mesh);
    expect(levelMeshOf(mesh, 1).parent).toBe(mesh);
    check();

    // drawCost merges the two levels into one: the coarse one empties instead
    mesh.drawCost = DEFAULT_DRAW_COST;
    mesh.update(makeCamera(0, 0, 5), H);
    expect([mesh.getLevelAt(0), mesh.getLevelAt(1)]).toEqual([0, 0]);
    expect(levelMeshOf(mesh, 1).count).toBe(0);
    expect(levelMeshOf(mesh, 1).parent).toBeNull();
    expect(levelMeshOf(mesh, 0).parent).toBe(mesh);
    check();

    // no instance at all: nothing is in the scene
    mesh.count = 0;
    mesh.update(makeCamera(0, 0, 5), H);
    expect(mesh.children.length).toBe(0);
  });

  it("makes every draw mesh share the world matrix of the MeshletMesh, and still updates other children", () => {
    const mesh = makeFixtureMesh({ drawCost: 0 });
    const scene = new Scene();
    scene.add(mesh);
    mesh.position.set(5, 0, 0);
    const extra = new Object3D();
    extra.position.set(0, 2, 0);
    mesh.add(extra);
    scene.updateMatrixWorld();

    mesh.count = 2;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.update(makeCamera(5, 0, 5), H);
    expect(mesh.stats.levelDraws).toBe(2);
    scene.updateMatrixWorld();

    const drawMeshes = mesh.children.filter((c) => c !== extra);
    expect(drawMeshes.length).toBe(2);
    for (const child of drawMeshes) {
      expect(child.matrixWorld).toBe(mesh.matrixWorld);
      expect(child.matrixAutoUpdate).toBe(false);
      expect(child.matrixWorldAutoUpdate).toBe(false);
      expect(child.matrixWorld.elements[12]).toBe(5);
    }
    expect(new Vector3().setFromMatrixPosition(extra.matrixWorld).toArray()).toEqual([5, 2, 0]);

    // moving the object moves all of them, with no walk over the draw meshes
    mesh.position.set(-3, 1, 0);
    scene.updateMatrixWorld();
    for (const child of drawMeshes) expect(child.matrixWorld.elements[12]).toBe(-3);
    expect(new Vector3().setFromMatrixPosition(extra.matrixWorld).toArray()).toEqual([-3, 3, 0]);
  });

  describe("hiding levels that nothing of is in view during the renderer's walk", () => {
    /** One instance in front of the camera (looking down -z), one behind it, at the given depths. */
    function setup(frontZ: number, behindZ: number) {
      const mesh = makeFixtureMesh({ drawCost: 0 });
      mesh.count = 2;
      mesh.setMatrixAt(0, translation(0, 0, frontZ));
      mesh.setMatrixAt(1, translation(0, 0, behindZ));
      const camera = makeCamera(0, 0, 0);
      return { mesh, camera };
    }

    // [front z, behind z]: the far one is on the coarse level, the other on the fine one, either way round
    const arrangements: Array<[number, number]> = [
      [-3000, 40],
      [-40, 3000],
    ];

    for (const [frontZ, behindZ] of arrangements) {
      it(`hides the level of the instance behind the camera until the helper is reached (front ${frontZ}, behind ${behindZ})`, () => {
        const { mesh, camera } = setup(frontZ, behindZ);
        mesh.update(camera);
        const frontLevel = mesh.getLevelAt(0);
        const behindLevel = mesh.getLevelAt(1);
        expect(frontLevel).toBeGreaterThanOrEqual(0);
        expect(behindLevel).toBeGreaterThanOrEqual(0);
        expect(frontLevel).not.toBe(behindLevel);

        const front = levelMeshOf(mesh, frontLevel);
        const behind = levelMeshOf(mesh, behindLevel);
        expect(behind.boundingSphere!.radius).toBe(-Infinity);
        expect(front.boundingSphere).toBe(mesh.boundingSphere);
        const last = mesh.children[mesh.children.length - 1] as Object3D & { isLOD?: boolean; update(camera: unknown): void };
        expect(last.isLOD).toBe(true);
        expect(typeof last.update).toBe("function");

        last.update(camera);
        expect(behind.boundingSphere).toBe(mesh.boundingSphere);
        expect(front.boundingSphere).toBe(mesh.boundingSphere);
        expect(mesh.children.some((c) => (c as { isLOD?: boolean }).isLOD === true)).toBe(false);
      });

      it(`hides and adds nothing for a manual update with a height (front ${frontZ}, behind ${behindZ})`, () => {
        const { mesh, camera } = setup(frontZ, behindZ);
        mesh.update(camera, H);
        expect(mesh.stats.levelDraws).toBe(2);
        for (const lm of mesh.levelMeshes) expect(lm.boundingSphere).toBe(mesh.boundingSphere);
        expect(mesh.children.some((c) => (c as { isLOD?: boolean }).isLOD === true)).toBe(false);
        expect(mesh.children.length).toBe(2);
      });
    }
  });
});

describe("MeshletMesh drawCost on a built hierarchy", () => {
  let knot: MeshletData;

  beforeAll(async () => {
    knot = await buildMeshletsFromGeometry(mergeVertices(new TorusKnotGeometry(1, 0.35, 320, 48)));
  });

  const knotMesh = (options?: MeshletMeshOptions) => new MeshletMesh(knot, new MeshBasicMaterial(), options);

  it("never draws an instance coarser than it asks for, over 40 distances", () => {
    const n = 40;
    const exact = knotMesh({ drawCost: 0 });
    const merged = knotMesh();
    for (const mesh of [exact, merged]) {
      mesh.count = n;
      for (let i = 0; i < n; i++) mesh.setMatrixAt(i, translation(i * 0.2, 0, -(8 + Math.pow(1.22, i) * 4)));
    }
    const camera = makeCamera(0, 0, 0);
    exact.update(camera, H);
    merged.update(camera, H);
    const levels = new Set<number>();
    for (let i = 0; i < n; i++) {
      const asked = exact.getLevelAt(i);
      const drawn = merged.getLevelAt(i);
      levels.add(asked);
      if (asked === -1 || drawn === -1) continue;
      expect(drawn, `instance ${i}`).toBeLessThanOrEqual(asked);
    }
    expect(levels.size).toBeGreaterThanOrEqual(3);
  });

  it("gives the same levels with drawCost 1 as with 0, because the levels of the knot differ by far more", () => {
    const n = 40;
    const zero = knotMesh({ drawCost: 0 });
    const one = knotMesh({ drawCost: 1 });
    for (const mesh of [zero, one]) {
      mesh.count = n;
      for (let i = 0; i < n; i++) mesh.setMatrixAt(i, translation(i * 0.2, 0, -(8 + Math.pow(1.22, i) * 4)));
    }
    const camera = makeCamera(0, 0, 0);
    zero.update(camera, H);
    one.update(camera, H);
    for (let i = 0; i < n; i++) {
      expect(one.getLevelAt(i), `instance ${i}`).toBe(zero.getLevelAt(i));
      expect(one.getTriangleCountAt(i), `instance ${i}`).toBe(zero.getTriangleCountAt(i));
    }
  });

  it("is in the scene only while it has an instance drawn by cluster", () => {
    const mesh = knotMesh();
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(0, 0, -300));
    expect(mesh.batched.parent).toBeNull();

    mesh.update(makeCamera(0, 0, 200), H);
    expect(mesh.getLevelAt(0)).toBeGreaterThanOrEqual(0);
    expect(mesh.getLevelAt(1)).toBeGreaterThanOrEqual(0);
    expect(mesh.batched.parent).toBeNull();

    // right next to instance 0
    mesh.update(makeCamera(0, 0, 4), H);
    expect(mesh.getLevelAt(0)).toBe(-1);
    expect(mesh.batched.parent).toBe(mesh);
    expect(mesh.children.length).toBe(mesh.stats.levelDraws + 1);

    mesh.update(makeCamera(0, 0, 200), H);
    expect(mesh.getLevelAt(0)).toBeGreaterThanOrEqual(0);
    expect(mesh.batched.parent).toBeNull();
    expect(mesh.children.length).toBe(mesh.stats.levelDraws);
  });

  it("keeps children = level draws + the draw mesh while instances change level", () => {
    const mesh = knotMesh({ drawCost: 0 });
    const n = 6;
    mesh.count = n;
    for (let i = 0; i < n; i++) mesh.setMatrixAt(i, translation(i * 5, 0, -i * 40));
    const camera = makeCamera(0, 0, 0);
    const seen = new Set<number>();
    for (const z of [3, 30, 150, 600, 2500, 30, 3, 100]) {
      camera.position.set(0, 0, z);
      camera.updateMatrixWorld(true);
      mesh.update(camera, H);
      expect(mesh.children.length).toBe(mesh.stats.levelDraws + (mesh.batched.parent === mesh ? 1 : 0));
      for (const lm of mesh.levelMeshes) expect(lm.parent).toBe(lm.count > 0 ? mesh : null);
      for (let i = 0; i < n; i++) seen.add(mesh.getLevelAt(i));
    }
    expect(seen.size).toBeGreaterThanOrEqual(3);
  });

  it("gives the draw meshes of both kinds the world matrix of the MeshletMesh", () => {
    const mesh = knotMesh({ drawCost: 0 });
    const scene = new Scene();
    scene.add(mesh);
    mesh.position.set(5, 0, 0);
    const extra = new Object3D();
    extra.position.set(0, 2, 0);
    mesh.add(extra);
    scene.updateMatrixWorld();

    mesh.count = 2;
    mesh.setMatrixAt(1, translation(0, 0, -1000));
    mesh.update(makeCamera(5, 0, 4), H);
    expect(mesh.batched.parent).toBe(mesh);
    expect(mesh.stats.levelDraws).toBeGreaterThanOrEqual(1);
    scene.updateMatrixWorld();

    const drawMeshes = mesh.children.filter((c) => c !== extra);
    expect(drawMeshes.length).toBe(mesh.stats.levelDraws + 1);
    for (const child of drawMeshes) {
      expect(child.matrixWorld).toBe(mesh.matrixWorld);
      expect(child.matrixAutoUpdate).toBe(false);
      expect(child.matrixWorldAutoUpdate).toBe(false);
      expect(child.matrixWorld.elements[12]).toBe(5);
    }
    expect(new Vector3().setFromMatrixPosition(extra.matrixWorld).toArray()).toEqual([5, 2, 0]);
  });

  // =========================================================================
  // incremental equals from scratch
  // =========================================================================

  const INSTANCES = 60;

  /** 60 instances over a few hundred units, a few of them scaled. */
  function makePlacements(rand: () => number): Matrix4[] {
    const out: Matrix4[] = [];
    for (let i = 0; i < INSTANCES; i++) {
      const position = new Vector3((rand() - 0.5) * 400, (rand() - 0.5) * 400, (rand() - 0.5) * 400);
      const s = i % 12 === 5 ? 0.3 + rand() * 3 : 1;
      const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), rand() * 6);
      out.push(new Matrix4().compose(position, q, new Vector3(s, s, s)));
    }
    return out;
  }

  const placementPosition = (m: Matrix4) => new Vector3().setFromMatrixPosition(m);

  /** A random unit direction. */
  function direction(rand: () => number): Vector3 {
    const v = new Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5);
    return v.lengthSq() > 0 ? v.normalize() : v.set(1, 0, 0);
  }

  /** The step sizes of a walk: mostly tiny and small moves, now and then a big jump or a visit to an instance. */
  function stepKind(rand: () => number): "tiny" | "small" | "jump" | "visit" {
    const r = rand();
    return r < 0.35 ? "tiny" : r < 0.7 ? "small" : r < 0.88 ? "jump" : "visit";
  }

  /** Forces `fresh` to evaluate every instance from scratch at its current state (a settings change re-chooses everything). */
  function evaluateFromScratch(fresh: MeshletMesh, camera: PerspectiveCamera): void {
    const threshold = fresh.errorThreshold;
    fresh.errorThreshold = threshold + 1;
    fresh.update(camera, H);
    fresh.errorThreshold = threshold;
    fresh.update(camera, H);
  }

  function expectSame(walker: MeshletMesh, fresh: MeshletMesh, step: number, where: string): void {
    for (let i = 0; i < INSTANCES; i++) {
      const message = `step ${step}, instance ${i}, ${where}`;
      expect(walker.getLevelAt(i), `level, ${message}`).toBe(fresh.getLevelAt(i));
      expect(walker.getTriangleCountAt(i), `triangles, ${message}`).toBe(fresh.getTriangleCountAt(i));
    }
  }

  const costs: Array<[string, number | undefined]> = [
    ["drawCost 0", 0],
    ["the default drawCost", undefined],
  ];

  for (const [name, drawCost] of costs) {
    const options: MeshletMeshOptions = drawCost === undefined ? {} : { drawCost };

    it(`walking the camera gives what a fresh evaluation gives at every step (${name})`, () => {
      const rand = seeded(0xc0ffee);
      const placements = makePlacements(rand);
      const walker = knotMesh(options);
      const fresh = knotMesh(options);
      for (const mesh of [walker, fresh]) {
        mesh.count = INSTANCES;
        placements.forEach((m, i) => mesh.setMatrixAt(i, m));
      }

      const camera = makeCamera(0, 0, 300);
      const position = camera.position;
      const levels = new Set<number>();
      for (let step = 0; step < 300; step++) {
        const kind = stepKind(rand);
        if (kind === "tiny") position.addScaledVector(direction(rand), 1e-3);
        else if (kind === "small") position.addScaledVector(direction(rand), 0.5);
        else if (kind === "jump") position.addScaledVector(direction(rand), 50 + rand() * 250);
        else {
          // close to an instance, or in its very centre
          const target = placementPosition(placements[Math.floor(rand() * INSTANCES)]);
          position.copy(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
        }
        camera.updateMatrixWorld(true);

        walker.update(camera, H);
        evaluateFromScratch(fresh, camera);
        expectSame(walker, fresh, step, `camera at ${position.toArray().join(", ")}`);
        for (let i = 0; i < INSTANCES; i++) levels.add(walker.getLevelAt(i));
      }
      // the walk is a real test only if it reaches by-cluster and several levels
      expect(levels.has(-1)).toBe(true);
      expect(levels.size).toBeGreaterThanOrEqual(4);
    });

    it(`moving the mesh gives what a fresh evaluation gives at every step (${name})`, () => {
      const rand = seeded(0xbadf00d);
      const placements = makePlacements(rand);
      const walker = knotMesh(options);
      const fresh = knotMesh(options);
      for (const mesh of [walker, fresh]) {
        mesh.count = INSTANCES;
        placements.forEach((m, i) => mesh.setMatrixAt(i, m));
      }

      const camera = makeCamera(20, 10, 100);
      const where = new Vector3(0, 0, 0);
      const levels = new Set<number>();
      for (let step = 0; step < 300; step++) {
        const kind = stepKind(rand);
        if (kind === "tiny") where.addScaledVector(direction(rand), 1e-3);
        else if (kind === "small") where.addScaledVector(direction(rand), 0.5);
        else if (kind === "jump") where.addScaledVector(direction(rand), 50 + rand() * 250);
        else {
          // an instance lands next to (or on) the camera
          const target = placementPosition(placements[Math.floor(rand() * INSTANCES)]);
          where.copy(camera.position).sub(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
        }
        for (const mesh of [walker, fresh]) {
          mesh.position.copy(where);
          mesh.updateMatrixWorld();
        }

        walker.update(camera, H);
        evaluateFromScratch(fresh, camera);
        expectSame(walker, fresh, step, `mesh at ${where.toArray().join(", ")}`);
        for (let i = 0; i < INSTANCES; i++) levels.add(walker.getLevelAt(i));
      }
      expect(levels.has(-1)).toBe(true);
      expect(levels.size).toBeGreaterThanOrEqual(4);
    });

    it(`moving the mesh and changing instance matrices every 10 steps gives what a fresh evaluation gives (${name})`, () => {
      const rand = seeded(0x5eed5);
      const placements = makePlacements(rand);
      const walker = knotMesh(options);
      const fresh = knotMesh(options);
      for (const mesh of [walker, fresh]) {
        mesh.count = INSTANCES;
        placements.forEach((m, i) => mesh.setMatrixAt(i, m));
      }

      const camera = makeCamera(20, 10, 100);
      const where = new Vector3(0, 0, 0);
      for (let step = 0; step < 300; step++) {
        const kind = stepKind(rand);
        if (kind === "tiny") where.addScaledVector(direction(rand), 1e-3);
        else if (kind === "small") where.addScaledVector(direction(rand), 0.5);
        else if (kind === "jump") where.addScaledVector(direction(rand), 50 + rand() * 250);
        else {
          const target = placementPosition(placements[Math.floor(rand() * INSTANCES)]);
          where.copy(camera.position).sub(target).addScaledVector(direction(rand), [0, 0.5, 2, 5][Math.floor(rand() * 4)]);
        }
        for (const mesh of [walker, fresh]) {
          mesh.position.copy(where);
          mesh.updateMatrixWorld();
        }

        if (step % 10 === 5) {
          // one instance goes somewhere else (sometimes right next to the camera), maybe with another scale
          const i = Math.floor(rand() * INSTANCES);
          const s = rand() < 0.3 ? 0.3 + rand() * 3 : 1;
          const target = rand() < 0.5 ? camera.position.clone().sub(where).addScaledVector(direction(rand), rand() * 6) : placementPosition(placements[i]).addScaledVector(direction(rand), rand() * 200);
          const m = new Matrix4().compose(target, new Quaternion(), new Vector3(s, s, s));
          placements[i] = m;
          walker.setMatrixAt(i, m);
          fresh.setMatrixAt(i, m);
        }

        walker.update(camera, H);
        evaluateFromScratch(fresh, camera);
        expectSame(walker, fresh, step, `mesh at ${where.toArray().join(", ")}`);
      }
    });
  }
});
