import { beforeAll, describe, expect, it } from "vitest";
import {
  Color,
  Matrix4,
  MeshBasicMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  Quaternion,
  Raycaster,
  Scene,
  TorusKnotGeometry,
  Vector3,
} from "three";
import type { Camera, InstancedMesh, Material, WebGLRenderer } from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
import type { MeshletMeshOptions } from "../src/three/MeshletMesh.js";
import type { MeshletData } from "../src/types.js";
import { makeFixture } from "./helpers/fixture.js";

const H = 1080;

const fakeRenderer = {
  getRenderTarget: () => null,
  getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
} as unknown as WebGLRenderer;

function makeMesh(options?: MeshletMeshOptions): MeshletMesh {
  return new MeshletMesh(makeFixture(), new MeshBasicMaterial(), options);
}

function makeCamera(x: number, y: number, z: number): PerspectiveCamera {
  const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 10000);
  camera.position.set(x, y, z);
  camera.updateMatrixWorld(true);
  return camera;
}

const translation = (x: number, y = 0, z = 0) => new Matrix4().makeTranslation(x, y, z);

/** Translation x of the first `n` positions of a level mesh. */
function xs(lm: InstancedMesh, n = lm.count): number[] {
  const a = lm.instanceMatrix.array;
  const out: number[] = [];
  for (let p = 0; p < n; p++) out.push(a[p * 16 + 12]);
  return out;
}

const asc = (a: number[]) => [...a].sort((p, q) => p - q);

/** The level mesh of fixture level `level` (level 0 draws indices from 0, level 1 from 384). */
function levelMeshOf(mesh: MeshletMesh, level: number): InstancedMesh | undefined {
  return mesh.levelMeshes.find((lm) => lm.geometry.drawRange.start === (level === 0 ? 0 : 384));
}

function colorAt(lm: InstancedMesh, p: number): string {
  return new Color().fromArray(lm.instanceColor!.array, p * 3).getHexString();
}

function beforeRender(lm: InstancedMesh, camera: Camera): void {
  lm.onBeforeRender(fakeRenderer, new Scene(), camera, lm.geometry, lm.material as Material, null as never);
}
function afterRender(lm: InstancedMesh, camera: Camera): void {
  lm.onAfterRender(fakeRenderer, new Scene(), camera, lm.geometry, lm.material as Material, null as never);
}

describe("MeshletMesh instanced levels", () => {
  it("draws a near instance by level and swaps level meshes with distance", () => {
    const mesh = makeMesh();
    mesh.update(makeCamera(0, 0, 5), H);
    expect(mesh.getLevelAt(0)).toBe(0);
    expect(mesh.stats.levelInstances).toBe(1);
    expect(mesh.stats.levelDraws).toBe(1);
    expect(mesh.stats.draws).toBe(1);
    expect(mesh.stats.triangles).toBe(128);
    expect(mesh.stats.clusters).toBe(4);
    expect(mesh.batched.instanceCount).toBe(0);

    expect(mesh.levelMeshes.length).toBe(1);
    const lm = mesh.levelMeshes[0];
    expect(lm.count).toBe(1);
    expect(lm.geometry.index).toBe(mesh.batched.geometry.index);
    expect(lm.geometry.getAttribute("position")).toBe(mesh.batched.geometry.getAttribute("position"));
    expect(lm.geometry.drawRange).toEqual({ start: 0, count: 384 });

    mesh.update(makeCamera(0, 0, 500), H);
    expect(mesh.getLevelAt(0)).toBe(1);
    expect(mesh.stats.triangles).toBe(32);
    const fine = levelMeshOf(mesh, 0)!;
    const coarse = levelMeshOf(mesh, 1)!;
    expect(fine.count).toBe(0);
    expect(fine.parent).toBeNull();
    expect(coarse.count).toBe(1);
    expect(coarse.parent).toBe(mesh);
  });

  it("draws any number of instances on a level as one draw", () => {
    const mesh = makeMesh();
    mesh.count = 100;
    for (let i = 0; i < 100; i++) mesh.setMatrixAt(i, translation(i * 4, 0, -2000));
    mesh.update(makeCamera(0, 0, 0), H);

    expect(mesh.stats.draws).toBe(1);
    expect(mesh.stats.levelDraws).toBe(1);
    expect(mesh.stats.levelInstances).toBe(100);
    expect(mesh.stats.slots).toBe(0);
    expect(mesh.levelMeshes.length).toBe(1);
    const lm = mesh.levelMeshes[0];
    expect(lm.count).toBe(100);
    expect(asc(xs(lm, 100))).toEqual(Array.from({ length: 100 }, (_, i) => i * 4));
  });

  it("gives every instance its own level", () => {
    const mesh = makeMesh({ drawCost: 0 });
    mesh.count = 3;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.setMatrixAt(2, translation(0, 1000, 0));
    mesh.update(makeCamera(0, 0, 5), H);
    expect([0, 1, 2].map((i) => mesh.getLevelAt(i))).toEqual([0, 1, 1]);
    expect(mesh.stats.triangles).toBe(128 + 32 + 32);
    expect([0, 1, 2].map((i) => mesh.getTriangleCountAt(i))).toEqual([128, 32, 32]);
  });

  it("rewrites the level matrix at once on setMatrixAt and reselects just that instance", () => {
    const mesh = makeMesh({ drawCost: 0 });
    mesh.count = 3;
    mesh.setMatrixAt(1, translation(10, 0, 0));
    mesh.setMatrixAt(2, translation(20, 0, 0));
    const camera = makeCamera(0, 0, 500);
    mesh.update(camera, H);
    expect([0, 1, 2].map((i) => mesh.getLevelAt(i))).toEqual([1, 1, 1]);

    mesh.setMatrixAt(1, translation(77, 0, 0));
    // before the next update
    const all = mesh.levelMeshes.flatMap((lm) => xs(lm));
    expect(asc(all)).toEqual([0, 20, 77]);

    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.lastSelected).toBe(1);
    expect(mesh.getLevelAt(1)).toBe(1);

    // right in front of the camera: needs the fine level
    mesh.setMatrixAt(1, translation(0, 0, 497));
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.lastSelected).toBe(1);
    expect(mesh.getLevelAt(1)).toBe(0);
    expect(mesh.getLevelAt(0)).toBe(1);
    expect(mesh.getLevelAt(2)).toBe(1);
    expect(mesh.stats.triangles).toBe(32 + 128 + 32);
  });

  it("takes removed instances out of their level meshes and adds identity ones when count grows", () => {
    const mesh = makeMesh();
    mesh.count = 5;
    for (let i = 0; i < 5; i++) mesh.setMatrixAt(i, translation(i * 10, 0, 0));
    const camera = makeCamera(0, 0, 500);
    mesh.update(camera, H);
    expect(mesh.stats.levelInstances).toBe(5);
    expect(mesh.levelMeshes[0].count).toBe(5);

    mesh.count = 2;
    expect(mesh.stats.instances).toBe(2);
    expect(mesh.stats.levelInstances).toBe(2);
    expect(mesh.stats.triangles).toBe(64);
    expect(mesh.levelMeshes[0].count).toBe(2);
    expect(asc(xs(mesh.levelMeshes[0]))).toEqual([0, 10]);
    expect(mesh.update(camera, H)).toBe(false);

    mesh.count = 4;
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.levelInstances).toBe(4);
    expect(mesh.getLevelAt(2)).toBeGreaterThanOrEqual(0);
    expect(mesh.getLevelAt(3)).toBeGreaterThanOrEqual(0);
    expect(asc(xs(mesh.levelMeshes[0]))).toEqual([0, 0, 0, 10]);

    mesh.count = 0;
    expect(mesh.stats.levelInstances).toBe(0);
    expect(mesh.stats.levelDraws).toBe(0);
    expect(mesh.levelMeshes[0].count).toBe(0);
  });

  it("grows a level mesh past its initial capacity without losing or duplicating instances", () => {
    const mesh = makeMesh();
    mesh.count = 40;
    for (let i = 0; i < 40; i++) mesh.setMatrixAt(i, translation(i * 3, 0, 0));
    mesh.update(makeCamera(0, 0, 500), H);

    expect(mesh.levelMeshes.length).toBe(1);
    const lm = mesh.levelMeshes[0];
    expect(lm.instanceMatrix.count).toBeGreaterThanOrEqual(40);
    expect(lm.count).toBe(40);
    expect(mesh.children.filter((c) => c === lm).length).toBe(1);
    for (const child of mesh.children) {
      if ((child as InstancedMesh).isInstancedMesh) expect(mesh.levelMeshes).toContain(child);
    }
    expect(asc(xs(lm))).toEqual(Array.from({ length: 40 }, (_, i) => i * 3));
  });
});

describe("MeshletMesh instance culling", () => {
  const positions: Array<[number, number, number]> = [
    [1, 0, -50], // in front
    [5, 0, -80], // in front
    [3, 0, 50], // behind
    [2000, 0, -10], // far to the side
  ];

  function setup(options?: MeshletMeshOptions) {
    const mesh = makeMesh({ drawCost: 0, ...options });
    const scene = new Scene();
    scene.add(mesh);
    mesh.count = positions.length;
    positions.forEach(([x, y, z], i) => mesh.setMatrixAt(i, translation(x, y, z)));
    const camera = makeCamera(0, 0, 0);
    mesh.update(camera);
    return { mesh, scene, camera };
  }

  /** x of the instances on `level`, optionally only the ones among the first `limit`. */
  function onLevel(mesh: MeshletMesh, level: number, limit = positions.length): number[] {
    const out: number[] = [];
    for (let i = 0; i < limit; i++) if (mesh.getLevelAt(i) === level) out.push(positions[i][0]);
    return asc(out);
  }

  /** Level of a fixture level mesh. */
  const levelOf = (lm: InstancedMesh) => (lm.geometry.drawRange.start === 0 ? 0 : 1);

  it("culls the instanced draws of the main pass, not the shadow passes", () => {
    const { mesh, scene, camera } = setup();
    expect(mesh.stats.levelInstances).toBe(4);
    expect(mesh.stats.levelVisible).toBe(2);
    // both levels are in use by the chosen positions
    expect(mesh.levelMeshes.filter((lm) => lm.count > 0).length).toBe(2);

    const light = new PerspectiveCamera(60, 1, 0.5, 500);
    for (const lm of mesh.levelMeshes) {
      const level = levelOf(lm);
      const total = onLevel(mesh, level);
      const inView = onLevel(mesh, level, 2);
      expect(lm.count).toBe(total.length);

      // main pass
      beforeRender(lm, camera);
      expect(lm.count).toBe(inView.length);
      expect(asc(xs(lm))).toEqual(inView);
      afterRender(lm, camera);
      expect(lm.count).toBe(total.length);

      // shadow pass: everything
      lm.onBeforeShadow(fakeRenderer, scene, camera, light, lm.geometry, lm.material as Material, null as never);
      expect(lm.count).toBe(total.length);
      // WebGPU calls onBeforeRender in the shadow pass, with the light camera, and it must leave the total
      beforeRender(lm, light);
      expect(lm.count).toBe(total.length);
      // also with the viewer camera while a shadow pass is drawing this mesh
      beforeRender(lm, camera);
      expect(lm.count).toBe(total.length);
      lm.onAfterShadow(fakeRenderer, scene, camera, light, lm.geometry, lm.material as Material, null as never);

      // back to the main pass
      beforeRender(lm, camera);
      expect(lm.count).toBe(inView.length);
      afterRender(lm, camera);
      expect(lm.count).toBe(total.length);
    }
  });

  it("follows the camera when it turns around", () => {
    const { mesh, camera } = setup();
    camera.rotation.y = Math.PI;
    camera.updateMatrixWorld(true);
    mesh.update(camera);
    expect(mesh.stats.levelVisible).toBe(1);
    expect(mesh.stats.levelInstances).toBe(4);

    const visible: number[] = [];
    for (const lm of mesh.levelMeshes) {
      beforeRender(lm, camera);
      visible.push(...xs(lm));
      afterRender(lm, camera);
    }
    expect(visible).toEqual([3]);
  });

  it("never uses stale culling when the camera turned without an update", () => {
    const { mesh, camera } = setup();
    camera.rotation.y = 0.4;
    camera.updateMatrixWorld(true);
    for (const lm of mesh.levelMeshes) {
      beforeRender(lm, camera);
      expect(lm.count).toBe(onLevel(mesh, levelOf(lm)).length);
      afterRender(lm, camera);
    }
  });

  it("draws every instance when cullInstances is off", () => {
    const { mesh, camera } = setup({ cullInstances: false });
    expect(mesh.stats.levelVisible).toBe(mesh.stats.levelInstances);
    expect(mesh.stats.levelInstances).toBe(4);
    for (const lm of mesh.levelMeshes) {
      beforeRender(lm, camera);
      expect(lm.count).toBe(onLevel(mesh, levelOf(lm)).length);
      afterRender(lm, camera);
    }
  });

  it("re-tests visibility when only the camera orientation changed, without reporting a LOD change", () => {
    const { mesh, camera } = setup();
    expect(mesh.stats.levelVisible).toBe(2);
    camera.rotation.y = Math.PI;
    camera.updateMatrixWorld(true);
    expect(mesh.update(camera)).toBe(false);
    expect(mesh.stats.levelVisible).toBe(1);
    camera.rotation.y = 0;
    camera.updateMatrixWorld(true);
    expect(mesh.update(camera)).toBe(false);
    expect(mesh.stats.levelVisible).toBe(2);
  });
});

describe("MeshletMesh level colors", () => {
  it("setColorAt colors one instance of a level mesh", () => {
    const mesh = makeMesh();
    mesh.count = 3;
    for (let i = 0; i < 3; i++) mesh.setMatrixAt(i, translation(i * 10, 0, 0));
    mesh.update(makeCamera(0, 0, 500), H);

    mesh.setColorAt(1, new Color(1, 0, 0));
    // the level meshes may have been replaced: read them again
    expect(mesh.levelMeshes.length).toBe(1);
    const lm = mesh.levelMeshes[0];
    expect(lm.instanceColor).not.toBeNull();
    const x = xs(lm);
    for (let p = 0; p < lm.count; p++) {
      const rgb = Array.from(lm.instanceColor!.array.subarray(p * 3, p * 3 + 3));
      expect(rgb).toEqual(x[p] === 10 ? [1, 0, 0] : [1, 1, 1]);
    }
  });

  it('debugColors "lod" tints by level and "cluster" moves instances to slots', () => {
    const mesh = makeMesh({ drawCost: 0 });
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    const camera = makeCamera(0, 0, 5);
    mesh.update(camera, H);
    expect([mesh.getLevelAt(0), mesh.getLevelAt(1)]).toEqual([0, 1]);

    mesh.debugColors = "lod";
    const fine = levelMeshOf(mesh, 0)!;
    const coarse = levelMeshOf(mesh, 1)!;
    expect(colorAt(fine, 0)).toBe("4e79a7");
    expect(colorAt(coarse, 0)).toBe("f28e2b");

    mesh.debugColors = "cluster";
    mesh.update(camera, H);
    expect(mesh.stats.levelInstances).toBe(0);
    expect(mesh.getLevelAt(0)).toBe(-1);
    expect(mesh.getLevelAt(1)).toBe(-1);
    expect(mesh.stats.slots).toBeGreaterThan(0);

    mesh.debugColors = "none";
    mesh.update(camera, H);
    expect(mesh.stats.levelInstances).toBe(2);
    expect(mesh.getLevelAt(0)).toBeGreaterThanOrEqual(0);
    expect(mesh.getLevelAt(1)).toBeGreaterThanOrEqual(0);
  });
});

describe("MeshletMesh level meshes as scene objects", () => {
  it("raycasts instances drawn by level and reports the instance id", () => {
    const mesh = makeMesh();
    const scene = new Scene();
    scene.add(mesh);
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(10, 0, 0));
    mesh.update(makeCamera(0, 0, 500), H);
    expect(mesh.getLevelAt(0)).toBeGreaterThanOrEqual(0);
    expect(mesh.getLevelAt(1)).toBeGreaterThanOrEqual(0);
    scene.updateMatrixWorld(true);

    const cast = (x: number, y: number) => {
      const raycaster = new Raycaster(new Vector3(x, y, 50), new Vector3(0, 0, -1));
      return raycaster.intersectObject(mesh, true);
    };
    const first = cast(0.3, 0.2);
    expect(first.length).toBeGreaterThan(0);
    expect(first[0].object).toBe(mesh);
    expect(first[0].instanceId).toBe(0);

    const second = cast(10.3, 0.2);
    expect(second.length).toBeGreaterThan(0);
    expect(second[0].object).toBe(mesh);
    expect(second[0].instanceId).toBe(1);

    expect(cast(5, 0.2).length).toBe(0);
  });

  it("forwards shadow, order and material settings to level meshes, old and new", () => {
    const mesh = makeMesh();
    mesh.update(makeCamera(0, 0, 500), H);
    expect(mesh.levelMeshes.length).toBe(1);

    const material = new MeshBasicMaterial();
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.renderOrder = 7;
    mesh.material = material;
    for (const lm of mesh.levelMeshes) {
      expect(lm.castShadow).toBe(true);
      expect(lm.receiveShadow).toBe(true);
      expect(lm.renderOrder).toBe(7);
      expect(lm.material).toBe(material);
    }

    // a level mesh created afterwards
    mesh.update(makeCamera(0, 0, 5), H);
    expect(mesh.levelMeshes.length).toBe(2);
    for (const lm of mesh.levelMeshes) {
      expect(lm.castShadow).toBe(true);
      expect(lm.receiveShadow).toBe(true);
      expect(lm.renderOrder).toBe(7);
      expect(lm.material).toBe(material);
      expect(lm.layers).toBe(mesh.layers);
    }
    expect(mesh.batched.castShadow).toBe(true);
    expect(mesh.batched.material).toBe(material);
  });

  it("reads castShadow back from the draw mesh", () => {
    const mesh = makeMesh();
    expect(mesh.castShadow).toBe(false);
    mesh.batched.castShadow = true;
    expect(mesh.castShadow).toBe(true);
  });

  it("disposes level meshes", () => {
    const mesh = makeMesh({ drawCost: 0 });
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.update(makeCamera(0, 0, 5), H);
    expect(mesh.levelMeshes.length).toBe(2);
    expect(() => mesh.dispose()).not.toThrow();
    expect(mesh.levelMeshes).toEqual([]);
  });

  it("keeps boundingSphere around every instance", () => {
    const mesh = makeMesh();
    const placements: Array<[Vector3, number]> = [
      [new Vector3(10, 0, 0), 1],
      [new Vector3(-5, 3, 2), 2.5],
      [new Vector3(0, -20, 0), 0.5],
    ];
    mesh.count = placements.length;
    placements.forEach(([p, s], i) => mesh.setMatrixAt(i, new Matrix4().compose(p, new Quaternion(), new Vector3(s, s, s))));
    mesh.update(makeCamera(0, 0, 100), H);

    const sphere = mesh.boundingSphere;
    for (const [p, s] of placements) {
      expect(sphere.center.distanceTo(p) + Math.SQRT2 * s).toBeLessThanOrEqual(sphere.radius + 1e-3);
    }
  });
});

describe("MeshletMesh level selection modes", () => {
  it("handles an orthographic camera", () => {
    const mesh = makeMesh();
    const camera = new OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    camera.position.set(0, 0, 10);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    mesh.update(camera, H);
    expect(mesh.getLevelAt(0)).toBe(0);

    camera.zoom = 0.01;
    camera.updateProjectionMatrix();
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.getLevelAt(0)).toBe(1);

    camera.position.set(3, 2, 20);
    camera.updateMatrixWorld(true);
    expect(mesh.update(camera, H)).toBe(false);
    expect(mesh.getLevelAt(0)).toBe(1);
  });

  it("draws nothing by level with instancedLevels: false", () => {
    const mesh = makeMesh({ instancedLevels: false });
    mesh.update(makeCamera(0, 0, 5), H);
    expect(mesh.stats.levelInstances).toBe(0);
    expect(mesh.levelMeshes).toEqual([]);
    mesh.update(makeCamera(0, 0, 500), H);
    expect(mesh.stats.levelInstances).toBe(0);
    expect(mesh.levelMeshes).toEqual([]);
    expect(mesh.getLevelAt(0)).toBe(-1);
  });
});

describe("MeshletMesh levels on a built hierarchy", () => {
  let knot: MeshletData;
  let center: Vector3;
  let radius: number;

  beforeAll(async () => {
    knot = await buildMeshletsFromGeometry(mergeVertices(new TorusKnotGeometry(1, 0.35, 320, 48)));
    center = new Vector3(knot.boundingSphere[0], knot.boundingSphere[1], knot.boundingSphere[2]);
    radius = knot.boundingSphere[3];
  });

  it("moves an instance between drawing by cluster and drawing by level", () => {
    const mesh = new MeshletMesh(knot, new MeshBasicMaterial());
    // The knot's bounding sphere has radius ~2, so 1.5 from the centre is inside it and the whole mesh
    // is at its finest level (a level); 4 is near enough for a mixed cut.
    const camera = makeCamera(0, 0, 4);
    mesh.update(camera, H);
    expect(mesh.getLevelAt(0)).toBe(-1);
    expect(mesh.stats.slots).toBeGreaterThan(0);
    expect(mesh.stats.triangles).toBe(mesh.getTriangleCountAt(0));
    const triangles = mesh.stats.triangles;
    const clusters = mesh.stats.clusters;

    camera.position.set(0, 0, 200);
    camera.updateMatrixWorld(true);
    mesh.update(camera, H);
    expect(mesh.getLevelAt(0)).toBeGreaterThanOrEqual(0);
    expect(mesh.stats.draws).toBe(1);
    expect(mesh.stats.levelDraws).toBe(1);
    expect(mesh.stats.triangles).toBe(mesh.getTriangleCountAt(0));
    for (let s = 0; s < mesh.batched.instanceCount; s++) expect(mesh.batched.getVisibleAt(s)).toBe(false);

    camera.position.set(0, 0, 4);
    camera.updateMatrixWorld(true);
    mesh.update(camera, H);
    expect(mesh.getLevelAt(0)).toBe(-1);
    expect(mesh.stats.triangles).toBe(triangles);
    expect(mesh.stats.clusters).toBe(clusters);
    expect(mesh.stats.triangles).toBe(mesh.getTriangleCountAt(0));
  });

  it("levelDistance puts a nearer instance on a level that is never coarser than its cut", () => {
    const camera = makeCamera(center.x, center.y, center.z + 3 * radius);

    const byLevel = new MeshletMesh(knot, new MeshBasicMaterial(), { levelDistance: 0.5 });
    byLevel.update(camera, H);
    expect(byLevel.getLevelAt(0)).toBeGreaterThanOrEqual(0);

    const byCluster = new MeshletMesh(knot, new MeshBasicMaterial(), { instancedLevels: false });
    byCluster.update(camera, H);
    expect(byCluster.getLevelAt(0)).toBe(-1);

    expect(byLevel.getTriangleCountAt(0)).toBeGreaterThanOrEqual(byCluster.getTriangleCountAt(0));
  });
});
