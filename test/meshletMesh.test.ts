import { describe, expect, it } from "vitest";
import {
  Color,
  Matrix4,
  MeshBasicMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  Raycaster,
  Scene,
  Sphere,
  Vector3,
} from "three";
import type { Camera, WebGLRenderer } from "three";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
import type { MeshletMeshOptions } from "../src/three/MeshletMesh.js";
import { makeFixture } from "./helpers/fixture.js";

const H = 1080;

function makeMesh(options?: MeshletMeshOptions): MeshletMesh {
  return new MeshletMesh(makeFixture(), new MeshBasicMaterial(), { mergeClusters: false, ...options });
}

function makeCamera(z: number, x = 0, y = 0): PerspectiveCamera {
  const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 10000);
  camera.position.set(x, y, z);
  camera.updateMatrixWorld(true);
  return camera;
}

/** Cluster ids currently drawn (visible slots), optionally restricted to one instance, sorted. */
function drawn(mesh: MeshletMesh, instance?: number): number[] {
  const out: number[] = [];
  for (let s = 0; s < mesh.batched.instanceCount; s++) {
    if (!mesh.batched.getVisibleAt(s)) continue;
    if (instance !== undefined && mesh.slotInstance[s] !== instance) continue;
    out.push(mesh.batched.getGeometryIdAt(s)); // geometry id === cluster id
  }
  return out.sort((a, b) => a - b);
}

function visibleSlots(mesh: MeshletMesh): number[] {
  const out: number[] = [];
  for (let s = 0; s < mesh.batched.instanceCount; s++) if (mesh.batched.getVisibleAt(s)) out.push(s);
  return out;
}

const translation = (x: number, y = 0, z = 0) => new Matrix4().makeTranslation(x, y, z);

describe("MeshletMesh construction", () => {
  it("builds one index-only geometry per cluster whose id equals the cluster id", () => {
    const data = makeFixture();
    const mesh = new MeshletMesh(data, new MeshBasicMaterial(), { mergeClusters: false });
    expect(mesh.isMeshletMesh).toBe(true);
    expect(mesh.children).toContain(mesh.batched);
    const sphere = new Sphere();
    for (let c = 0; c < data.clusterCount; c++) {
      const range = mesh.batched.getGeometryRangeAt(c);
      expect(range.vertexCount).toBe(0);
      expect(range.indexStart).toBe(data.clusterIndexStart[c]);
      expect(range.indexCount).toBe(data.clusterIndexCount[c]);
      expect(mesh.batched.getBoundingSphereAt(c, sphere)).not.toBeNull();
      expect(sphere.radius).toBeCloseTo(data.clusterBounds[c * 4 + 3]);
    }
    expect(mesh.batched.frustumCulled).toBe(false);
    expect(mesh.batched.perObjectFrustumCulled).toBe(true);
    expect(mesh.batched.sortObjects).toBe(false);
    expect(mesh.count).toBe(1);
    expect(mesh.errorThreshold).toBe(1);
  });

  it("is zero-copy: the draw geometry wraps the MeshletData arrays", () => {
    const data = makeFixture();
    const mesh = new MeshletMesh(data, new MeshBasicMaterial());
    const geometry = mesh.batched.geometry;
    for (const a of data.attributes) {
      const attribute = geometry.getAttribute(a.name);
      expect(attribute.array).toBe(a.array);
      expect(attribute.itemSize).toBe(a.itemSize);
    }
    expect(geometry.getAttribute("position").array).toBe(data.attributes[0].array);
    const index = geometry.index!;
    expect(index.count).toBe(data.indices.length);
    for (let i = 0; i < data.indices.length; i++) expect(index.array[i]).toBe(data.indices[i]);
  });

  it("derives default capacities from the coarsest cut and respects initialDrawCapacity", () => {
    expect(makeMesh().drawCapacity).toBe(64);
    expect(makeMesh({ initialDrawCapacity: 7 }).drawCapacity).toBe(7);
  });
});

describe("MeshletMesh LOD", () => {
  it("draws the fine clusters up close and the coarse cluster far away", () => {
    const mesh = makeMesh();
    expect(mesh.update(makeCamera(5), H)).toBe(true);
    expect(drawn(mesh)).toEqual([0, 1, 2, 3]);
    expect(mesh.stats.clusters).toBe(4);
    expect(mesh.stats.triangles).toBe(128);

    expect(mesh.update(makeCamera(500), H)).toBe(true);
    expect(drawn(mesh)).toEqual([4]);
    expect(mesh.stats.clusters).toBe(1);
    expect(mesh.stats.triangles).toBe(32);
  });

  it("recycles slots instead of growing", () => {
    const mesh = makeMesh();
    const near = makeCamera(5);
    const far = makeCamera(500);
    mesh.update(near, H);
    const slots = mesh.batched.instanceCount;
    expect(slots).toBe(4);
    for (let i = 0; i < 10; i++) {
      mesh.update(far, H);
      expect(visibleSlots(mesh).length).toBe(1);
      mesh.update(near, H);
      expect(visibleSlots(mesh).length).toBe(4);
    }
    expect(mesh.batched.instanceCount).toBe(slots);
    expect(mesh.stats.slots).toBe(slots);
    // freed slots are reused: switching to the coarse cut must not create a slot
    mesh.update(far, H);
    expect(mesh.batched.instanceCount).toBe(slots);
  });

  it("skips work when nothing changed, also under camera rotation", () => {
    const mesh = makeMesh();
    const camera = makeCamera(5);
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.update(camera, H)).toBe(false);

    camera.lookAt(10, 3, 5);
    camera.updateMatrixWorld(true);
    const before = mesh.stats.lastUpdateMs;
    expect(mesh.update(camera, H)).toBe(false);
    expect(mesh.stats.lastUpdateMs).toBe(before);

    camera.position.set(0, 0, 500);
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.update(camera, H)).toBe(false);
  });

  it("reports false when the camera moves but the cut stays the same", () => {
    const mesh = makeMesh();
    const camera = makeCamera(5);
    mesh.update(camera, H);
    camera.position.set(0.5, 0, 6);
    expect(mesh.update(camera, H)).toBe(false);
    expect(mesh.stats.lastSelected).toBe(1);
    expect(drawn(mesh)).toEqual([0, 1, 2, 3]);
  });

  it("reacts to viewport height and to the error threshold", () => {
    const mesh = makeMesh();
    const camera = makeCamera(30);
    mesh.update(camera, H);
    expect(mesh.stats.triangles).toBe(128);

    mesh.errorThreshold = 3;
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.triangles).toBe(32);

    mesh.errorThreshold = 1;
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.triangles).toBe(128);

    // a smaller viewport allows less detail (same error is fewer pixels)
    expect(mesh.update(camera, 200)).toBe(true);
    expect(mesh.stats.triangles).toBe(32);
  });

  it("gives the coarse cluster for a huge threshold even up close", () => {
    const mesh = makeMesh({ errorThreshold: 1e6 });
    mesh.update(makeCamera(1.6), H);
    expect(drawn(mesh)).toEqual([4]);
    expect(mesh.stats.triangles).toBe(32);
  });
});

describe("MeshletMesh instances", () => {
  it("handles several instances with separate LODs", () => {
    const mesh = makeMesh();
    mesh.count = 3;
    mesh.setMatrixAt(0, translation(0, 0, 0));
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.setMatrixAt(2, translation(0, 1000, 0));
    mesh.update(makeCamera(5), H);

    expect(mesh.stats.instances).toBe(3);
    expect(mesh.stats.clusters).toBe(6);
    expect(mesh.stats.triangles).toBe(128 + 32 + 32);
    expect(drawn(mesh, 0)).toEqual([0, 1, 2, 3]);
    expect(drawn(mesh, 1)).toEqual([4]);
    expect(drawn(mesh, 2)).toEqual([4]);
  });

  it("moves existing slots on setMatrixAt and only reselects the dirty instance", () => {
    const mesh = makeMesh();
    mesh.count = 3;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.setMatrixAt(2, translation(0, 1000, 0));
    const camera = makeCamera(5);
    mesh.update(camera, H);
    expect(mesh.stats.lastSelected).toBe(3);
    expect(mesh.update(camera, H)).toBe(false);

    const slot = visibleSlots(mesh).find((s) => mesh.slotInstance[s] === 1)!;
    mesh.setMatrixAt(1, translation(500, 0, 0));
    const m = new Matrix4();
    mesh.batched.getMatrixAt(slot, m);
    expect(new Vector3().setFromMatrixPosition(m).x).toBe(500);
    expect(mesh.getMatrixAt(1, new Matrix4()).elements[12]).toBe(500);

    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.lastSelected).toBe(1);
    expect(drawn(mesh, 1)).toEqual([4]);

    // bring it next to the camera: it refines
    mesh.setMatrixAt(1, translation(2, 0, 0));
    expect(mesh.update(camera, H)).toBe(true);
    expect(drawn(mesh, 1)).toEqual([0, 1, 2, 3]);
    expect(mesh.stats.clusters).toBe(4 + 4 + 1);
  });

  it("hides the slots of removed instances when count is lowered, and re-adds identity instances", () => {
    const mesh = makeMesh();
    mesh.count = 3;
    mesh.setMatrixAt(1, translation(1000, 0, 0));
    mesh.setMatrixAt(2, translation(0, 1000, 0));
    const camera = makeCamera(5);
    mesh.update(camera, H);
    expect(visibleSlots(mesh).length).toBe(6);

    mesh.count = 1;
    expect(visibleSlots(mesh).length).toBe(4);
    expect(drawn(mesh, 1)).toEqual([]);
    expect(mesh.stats.clusters).toBe(4);
    expect(mesh.stats.instances).toBe(1);
    expect(mesh.update(camera, H)).toBe(false);

    // raising count again gives fresh identity instances, not the old translations
    mesh.count = 2;
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.getMatrixAt(1, new Matrix4()).equals(new Matrix4())).toBe(true);
    expect(drawn(mesh, 1)).toEqual([0, 1, 2, 3]);
    expect(mesh.stats.clusters).toBe(8);
  });

  it("throws on out of range instance ids", () => {
    const mesh = makeMesh();
    expect(() => mesh.setMatrixAt(1, new Matrix4())).toThrow(RangeError);
    expect(() => mesh.setColorAt(1, new Color())).toThrow(RangeError);
  });

  it("refines a scaled-up instance at a proportionally larger distance", () => {
    const mesh = makeMesh();
    mesh.count = 2;
    mesh.setMatrixAt(1, new Matrix4().makeScale(10, 10, 10));
    mesh.update(makeCamera(100), H);
    expect(drawn(mesh, 0)).toEqual([4]);
    expect(drawn(mesh, 1)).toEqual([0, 1, 2, 3]);

    mesh.update(makeCamera(1000), H);
    expect(drawn(mesh, 0)).toEqual([4]);
    expect(drawn(mesh, 1)).toEqual([4]);
  });

  it("respects the parent's own transform", () => {
    const mesh = makeMesh();
    const camera = makeCamera(5);
    mesh.position.set(0, 0, -1000);
    mesh.update(camera, H);
    expect(drawn(mesh)).toEqual([4]);

    mesh.position.set(0, 0, 0);
    expect(mesh.update(camera, H)).toBe(true);
    expect(drawn(mesh)).toEqual([0, 1, 2, 3]);

    // scaling the object by 20 makes a camera at distance 100 behave like distance 5
    const far = makeCamera(100);
    mesh.scale.setScalar(1);
    mesh.update(far, H);
    expect(drawn(mesh)).toEqual([4]);
    mesh.scale.setScalar(20);
    expect(mesh.update(far, H)).toBe(true);
    expect(drawn(mesh)).toEqual([0, 1, 2, 3]);
  });

  it("grows the draw capacity by rebuilding the draw mesh", () => {
    const data = makeFixture();
    const material = new MeshBasicMaterial();
    const mesh = new MeshletMesh(data, material, { initialDrawCapacity: 2, maxInstances: 1, mergeClusters: false });
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.renderOrder = 3;
    mesh.name = "meshlet";
    expect(mesh.drawCapacity).toBe(2);

    const rebuilt: unknown[] = [];
    mesh.onRebuild = (b) => rebuilt.push(b);
    const first = mesh.batched;
    first.userData.keep = 1;

    mesh.count = 5;
    for (let i = 0; i < 5; i++) mesh.setMatrixAt(i, translation(i * 3, 0, 0));
    mesh.update(makeCamera(5, 6), H);

    expect(mesh.batched).not.toBe(first);
    expect(rebuilt.length).toBeGreaterThan(0);
    expect(rebuilt[rebuilt.length - 1]).toBe(mesh.batched);
    expect(mesh.children).toContain(mesh.batched);
    expect(mesh.children).not.toContain(first);
    expect(mesh.drawCapacity).toBeGreaterThanOrEqual(20);
    expect(mesh.stats.clusters).toBe(20);
    expect(mesh.stats.slots).toBe(20);
    expect(visibleSlots(mesh).length).toBe(20);
    expect(mesh.slotInstance.length).toBeGreaterThanOrEqual(20);
    const m = new Matrix4();
    for (const s of visibleSlots(mesh)) {
      mesh.batched.getMatrixAt(s, m);
      expect(m.elements[12]).toBe(mesh.slotInstance[s] * 3);
    }
    for (let i = 0; i < 5; i++) expect(drawn(mesh, i)).toEqual([0, 1, 2, 3]);

    // state survived, buffers are still the shared zero-copy ones
    expect(mesh.castShadow).toBe(true);
    expect(mesh.batched.castShadow).toBe(true);
    expect(mesh.batched.receiveShadow).toBe(true);
    expect(mesh.batched.renderOrder).toBe(3);
    expect(mesh.batched.material).toBe(material);
    expect(mesh.batched.layers).toBe(mesh.layers);
    expect(mesh.batched.userData.keep).toBe(1);
    expect(mesh.batched.geometry.getAttribute("position").array).toBe(data.attributes[0].array);
    expect(mesh.batched.getGeometryRangeAt(4).indexStart).toBe(data.clusterIndexStart[4]);
    // the shared buffers were not detached from anything the new mesh uses
    expect(mesh.batched.geometry.index!.count).toBe(data.indices.length);

    // further growth with colors in use preserves them
    mesh.debugColors = "cluster";
    mesh.count = 40;
    for (let i = 0; i < 40; i++) mesh.setMatrixAt(i, translation(i * 0.01, 0, 0));
    mesh.update(makeCamera(5), H);
    expect(mesh.stats.clusters).toBe(40 * 4);
    expect(mesh.getMatrixAt(4, m).elements[12]).toBeCloseTo(0.04);
    const c = new Color();
    const slot = visibleSlots(mesh).find((s) => mesh.batched.getGeometryIdAt(s) === 1)!;
    expect(mesh.batched.getColorAt(slot, c).getHexString()).not.toBe("ffffff");

    // the grown mesh still shrinks and recycles
    mesh.update(makeCamera(5000), H);
    expect(mesh.stats.clusters).toBe(40);
    expect(mesh.stats.slots).toBe(40 * 4);
  });

  it("keeps a replaced draw mesh drawable for the rest of the frame and disposes it afterwards", async () => {
    const mesh = makeMesh({ initialDrawCapacity: 2 });
    const first = mesh.batched;
    const renderer = {
      getRenderTarget: () => null,
      getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
    } as unknown as WebGLRenderer;
    const camera = makeCamera(5);
    const scene = new Scene();

    // growth happens inside the shadow pass hook ...
    const light = new OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
    first.onBeforeShadow(renderer, scene, camera, light, first.geometry, first.material, null as never);
    expect(mesh.batched).not.toBe(first);
    const clusters = mesh.stats.clusters;

    // ... and the main pass of the same frame still draws the old mesh: it must stay intact and inert
    expect(() => first.onBeforeRender(renderer, scene, camera, first.geometry, first.material, null as never)).not.toThrow();
    expect(first.geometry.getAttribute("position")).toBeDefined();
    expect(mesh.stats.clusters).toBe(clusters);

    // once the frame is over the old mesh is disposed, without touching the shared buffers
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(first.geometry.getAttribute("position")).toBeUndefined();
    const next = mesh.batched;
    expect(next.geometry.getAttribute("position")).toBeDefined();
    expect(() => next.onBeforeRender(renderer, scene, camera, next.geometry, next.material, null as never)).not.toThrow();
  });

  it("never takes LOD from a light camera (WebGPURenderer hook order)", () => {
    const mesh = makeMesh();
    const batched = mesh.batched;
    const renderer = {
      getRenderTarget: () => null,
      getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
    } as unknown as WebGLRenderer;
    const scene = new Scene();
    const viewer = makeCamera(5000); // far away: coarse
    // an orthographic light camera with a tiny frustum would select full detail if it were used
    const light = new OrthographicCamera(-0.01, 0.01, 0.01, -0.01, 0.1, 100);
    light.updateProjectionMatrix();
    light.updateMatrixWorld(true);

    batched.onBeforeRender(renderer, scene, viewer, batched.geometry, batched.material, null as never);
    expect(drawn(mesh)).toEqual([4]);
    // WebGPURenderer: onBeforeShadow(camera === shadowCamera), then onBeforeRender(shadowCamera)
    batched.onBeforeShadow(renderer, scene, light, light, batched.geometry, batched.material, null as never);
    expect(drawn(mesh)).toEqual([4]);
    batched.onBeforeRender(renderer, scene, light, batched.geometry, batched.material, null as never);
    expect(drawn(mesh)).toEqual([4]);
    // WebGLRenderer: onBeforeShadow(camera = viewer, shadowCamera) does update from the viewer
    const near = makeCamera(5);
    batched.onBeforeShadow(renderer, scene, near, light, batched.geometry, batched.material, null as never);
    expect(drawn(mesh)).toEqual([0, 1, 2, 3]);
  });

  it("works with an orthographic camera and reacts to zoom", () => {
    const mesh = makeMesh();
    const camera = new OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    camera.position.set(0, 0, 10);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    expect(mesh.update(camera, H)).toBe(true);
    expect(Number.isFinite(mesh.stats.triangles)).toBe(true);
    expect(mesh.stats.triangles).toBe(128);

    camera.zoom = 0.01;
    camera.updateProjectionMatrix();
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.triangles).toBe(32);

    // moving an orthographic camera never changes the LOD
    camera.position.set(100, 100, 10);
    expect(mesh.update(camera, H)).toBe(false);

    camera.zoom = 1;
    camera.updateProjectionMatrix();
    expect(mesh.update(camera, H)).toBe(true);
    expect(mesh.stats.triangles).toBe(128);
  });
});

describe("MeshletMesh three.js integration", () => {
  it("is raycastable and reports the MeshletMesh and the instance id", () => {
    const mesh = makeMesh();
    mesh.count = 2;
    mesh.setMatrixAt(1, translation(10, 0, 0));
    mesh.update(makeCamera(5), H);
    const scene = new Scene();
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const raycaster = new Raycaster();
    raycaster.set(new Vector3(0.3, 0.2, 5), new Vector3(0, 0, -1));
    let hits = raycaster.intersectObject(mesh, true);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].object).toBe(mesh);
    expect(hits[0].instanceId).toBe(0);
    expect(hits[0].point.z).toBeCloseTo(0);

    raycaster.set(new Vector3(10.3, 0.2, 5), new Vector3(0, 0, -1));
    hits = raycaster.intersectObject(mesh, true);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].object).toBe(mesh);
    expect(hits[0].instanceId).toBe(1);

    raycaster.set(new Vector3(5, 0, 5), new Vector3(0, 0, -1));
    expect(raycaster.intersectObject(mesh, true).length).toBe(0);
  });

  it("forwards castShadow, receiveShadow, renderOrder, layers and material to the BatchedMesh", () => {
    const material = new MeshBasicMaterial();
    const mesh = new MeshletMesh(makeFixture(), material);
    expect(mesh.material).toBe(material);
    expect(mesh.batched.material).toBe(material);
    const other = new MeshBasicMaterial();
    mesh.material = other;
    expect(mesh.batched.material).toBe(other);

    expect(mesh.castShadow).toBe(false);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.renderOrder = 7;
    expect(mesh.batched.castShadow).toBe(true);
    expect(mesh.batched.receiveShadow).toBe(true);
    expect(mesh.batched.renderOrder).toBe(7);
    mesh.batched.castShadow = false;
    expect(mesh.castShadow).toBe(false);
    expect(mesh.batched.layers).toBe(mesh.layers);
  });

  it("colors slots with debugColors", () => {
    const mesh = makeMesh();
    const near = makeCamera(5);
    const c = new Color();
    mesh.update(near, H);

    const colors = (): string[] => visibleSlots(mesh).map((s) => mesh.batched.getColorAt(s, c).getHexString());

    // (no color read before this point: three r180 throws from getColorAt until a color was set)
    mesh.debugColors = "cluster";
    expect(mesh.debugColors).toBe("cluster");
    const clusterColors = colors();
    expect(new Set(clusterColors).size).toBe(4);
    expect(clusterColors.some((h) => h === "ffffff")).toBe(false);

    // stable per cluster: going coarse and back gives the same colors per cluster
    const colorOf = (cluster: number): string => {
      const s = visibleSlots(mesh).find((slot) => mesh.batched.getGeometryIdAt(slot) === cluster)!;
      return mesh.batched.getColorAt(s, c).getHexString();
    };
    const c1 = colorOf(1);
    mesh.update(makeCamera(500), H);
    mesh.update(near, H);
    expect(colorOf(1)).toBe(c1);

    mesh.debugColors = "lod";
    expect(new Set(colors()).size).toBe(1);
    mesh.update(makeCamera(500), H);
    const lodCoarse = colors();
    expect(lodCoarse.length).toBe(1);
    expect(lodCoarse[0]).not.toBe(clusterColors[0]);
    expect(lodCoarse[0]).toBe(new Color(0xf28e2b).getHexString());

    mesh.debugColors = "none";
    expect(colors().every((h) => h === "ffffff")).toBe(true);

    // instance colors
    mesh.setColorAt(0, new Color(1, 0, 0));
    expect(colors()).toEqual(["ff0000"]);
    mesh.update(near, H);
    expect(colors().every((h) => h === "ff0000")).toBe(true);
    expect(mesh.getColorAt(0, new Color()).r).toBe(1);
  });

  it("does not create the BatchedMesh color texture until colors are used", () => {
    const mesh = makeMesh();
    mesh.update(makeCamera(5), H);
    mesh.setColorAt(0, new Color(1, 1, 1));
    expect((mesh.batched as unknown as { _colorsTexture: unknown })._colorsTexture).toBeNull();
  });
});

describe("MeshletMesh automatic updates", () => {
  const fakeRenderer = {
    getRenderTarget: () => null,
    getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
  } as unknown as WebGLRenderer;
  const scene = new Scene();

  const beforeRender = (mesh: MeshletMesh, camera: Camera) =>
    mesh.batched.onBeforeRender(fakeRenderer, scene, camera, mesh.batched.geometry, mesh.batched.material, null as never);

  it("updates from onBeforeRender and respects autoUpdate", () => {
    const mesh = makeMesh();
    expect(mesh.autoUpdate).toBe(true);
    beforeRender(mesh, makeCamera(5));
    expect(mesh.stats.triangles).toBe(128);
    beforeRender(mesh, makeCamera(500));
    expect(mesh.stats.triangles).toBe(32);

    const off = makeMesh();
    off.autoUpdate = false;
    beforeRender(off, makeCamera(5));
    expect(off.stats.clusters).toBe(0);
    expect(off.stats.triangles).toBe(0);
  });

  it("uses lodCamera instead of the rendering camera", () => {
    const mesh = makeMesh();
    mesh.lodCamera = makeCamera(500);
    beforeRender(mesh, makeCamera(5));
    expect(mesh.stats.triangles).toBe(32);
    mesh.lodCamera = makeCamera(5);
    beforeRender(mesh, makeCamera(500));
    expect(mesh.stats.triangles).toBe(128);
  });

  it("takes LOD from the viewer camera in shadow passes and ignores shadow cameras in onBeforeRender", () => {
    const mesh = makeMesh();
    const viewer = makeCamera(500);
    const shadowCamera = makeCamera(5);
    mesh.batched.onBeforeShadow(fakeRenderer, scene, viewer, shadowCamera, mesh.batched.geometry, new MeshBasicMaterial(), null as never);
    expect(mesh.stats.triangles).toBe(32);
    // WebGPURenderer calls onBeforeRender with the light camera: must not change the LOD
    beforeRender(mesh, shadowCamera);
    expect(mesh.stats.triangles).toBe(32);
    // the viewer camera still works
    beforeRender(mesh, viewer);
    expect(mesh.stats.triangles).toBe(32);
  });

  it("uses the render target height when rendering to a target", () => {
    const mesh = makeMesh();
    const renderer = {
      getRenderTarget: () => ({ height: 100 }),
      getDrawingBufferSize: (v: { set(x: number, y: number): unknown }) => v.set(1920, 1080),
    } as unknown as WebGLRenderer;
    // at z=30 a 1080px viewport refines (see the threshold test) but a 100px one does not
    mesh.batched.onBeforeRender(renderer, scene, makeCamera(30), mesh.batched.geometry, mesh.batched.material, null as never);
    expect(mesh.stats.triangles).toBe(32);
  });

  it("can be disposed", () => {
    const mesh = makeMesh();
    mesh.update(makeCamera(5), H);
    expect(() => mesh.dispose()).not.toThrow();
  });
});
