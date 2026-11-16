import { describe, expect, it } from "vitest";
import {
  BoxGeometry,
  BufferAttribute,
  Matrix4,
  MeshBasicMaterial,
  PerspectiveCamera,
  Raycaster,
  Scene,
  TorusKnotGeometry,
  Vector3,
} from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import {
  createLodChain,
  deserializeMeshlets,
  MeshletMesh,
  serializeMeshlets,
  validateMeshletData,
  type MeshletData,
} from "../src/index.js";

/** End to end: geometry -> builder -> file -> runtime mesh, using only the public entry points. */
describe("build -> serialize -> render pipeline", () => {
  const source = mergeVertices(new TorusKnotGeometry(1, 0.35, 320, 48)); // ~30k triangles
  const fullTriangles = source.index!.count / 3;
  let data: MeshletData;

  it("builds a valid hierarchy from a BufferGeometry", async () => {
    data = await buildMeshletsFromGeometry(source);
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.attributes.map((a) => a.name)).toEqual(["position", "normal", "uv"]);
    expect(data.levelCount).toBeGreaterThan(3);
  });

  it("survives a file round trip", () => {
    const restored = deserializeMeshlets(serializeMeshlets(data));
    expect(validateMeshletData(restored)).toEqual([]);
    expect(restored.indices).toEqual(data.indices);
    expect(restored.groupError).toEqual(data.groupError);
    data = restored; // everything below runs on the deserialized copy
  });

  it("draws full detail up close and a small fraction far away", () => {
    const mesh = new MeshletMesh(data, new MeshBasicMaterial());
    const scene = new Scene().add(mesh);
    const camera = new PerspectiveCamera(50, 16 / 9, 0.05, 1000);
    scene.updateMatrixWorld(true);

    const trianglesAt = (distance: number) => {
      camera.position.set(0, 0, distance);
      mesh.update(camera, 1080);
      return mesh.stats.triangles;
    };

    const near = trianglesAt(1.6);
    const mid = trianglesAt(12);
    const far = trianglesAt(120);
    const veryFar = trianglesAt(900);

    expect(near).toBeGreaterThan(fullTriangles * 0.5);
    expect(near).toBeLessThanOrEqual(fullTriangles);
    expect(mid).toBeLessThan(near);
    expect(far).toBeLessThan(mid * 0.5);
    expect(veryFar).toBeLessThan(fullTriangles * 0.02);
    expect(veryFar).toBeGreaterThan(0);
    mesh.dispose();
  });

  it("gives each instance its own level of detail and reports instance ids on raycast", () => {
    const mesh = new MeshletMesh(data, new MeshBasicMaterial(), { maxInstances: 3 });
    mesh.count = 3;
    const m = new Matrix4();
    mesh.setMatrixAt(0, m.makeTranslation(0, 0, 0));
    mesh.setMatrixAt(1, m.makeTranslation(60, 0, 0));
    mesh.setMatrixAt(2, m.makeTranslation(600, 0, 0));
    const scene = new Scene().add(mesh);
    scene.updateMatrixWorld(true);

    const camera = new PerspectiveCamera(50, 16 / 9, 0.05, 2000);
    camera.position.set(0, 0, 4);
    mesh.update(camera, 1080);

    // per-instance triangle counts, recovered from the draw slots
    const perInstance = [0, 0, 0];
    const batched = mesh.batched;
    for (let slot = 0; slot < batched.instanceCount; slot++) {
      if (!batched.getVisibleAt(slot)) continue;
      perInstance[mesh.slotInstance[slot]] += mesh.ranges.indexCount[batched.getGeometryIdAt(slot)] / 3;
    }
    expect(perInstance[0]).toBeGreaterThan(perInstance[1]);
    expect(perInstance[1]).toBeGreaterThan(perInstance[2]);
    expect(perInstance[0] + perInstance[1] + perInstance[2]).toBe(mesh.stats.triangles);

    const raycaster = new Raycaster(new Vector3(60, 1, 10), new Vector3(0, 0, -1));
    const hits = raycaster.intersectObject(mesh, true);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].object).toBe(mesh);
    expect(hits[0].instanceId).toBe(1);
    mesh.dispose();
  });

  it("produces a static LOD chain with strictly decreasing triangle counts", () => {
    const chain = createLodChain(data, { maxLevels: 5 });
    expect(chain.length).toBeGreaterThan(2);
    expect(chain.length).toBeLessThanOrEqual(5);
    expect(chain[0].index!.count / 3).toBe(fullTriangles);
    for (let i = 1; i < chain.length; i++) {
      expect(chain[i].index!.count).toBeLessThan(chain[i - 1].index!.count);
    }
    expect(chain[chain.length - 1].index!.count / 3).toBeLessThan(fullTriangles * 0.02);
  });
});

describe("builder input variety", () => {
  it("handles position-only geometry", async () => {
    const g = mergeVertices(new TorusKnotGeometry(1, 0.3, 128, 24));
    g.deleteAttribute("normal");
    g.deleteAttribute("uv");
    const data = await buildMeshletsFromGeometry(g);
    expect(validateMeshletData(data)).toEqual([]);
    expect(data.attributes).toHaveLength(1);
    expect(data.levelCount).toBeGreaterThan(1);
  });

  it("keeps integer attributes (normalized vertex colors) untouched", async () => {
    const g = mergeVertices(new TorusKnotGeometry(1, 0.3, 128, 24));
    const colors = new Uint8Array(g.attributes.position.count * 4).map((_, i) => (i * 37) & 255);
    g.setAttribute("color", new BufferAttribute(colors, 4, true));
    const data = await buildMeshletsFromGeometry(g, { attributeWeights: { normal: 0.5, color: 0.25 } });
    expect(validateMeshletData(data)).toEqual([]);
    const color = data.attributes.find((a) => a.name === "color")!;
    expect(color.array).toBe(colors);
    expect(color.normalized).toBe(true);
  });

  it("handles a hard-edged box (every vertex is an attribute seam)", async () => {
    const data = await buildMeshletsFromGeometry(new BoxGeometry(1, 1, 1, 24, 24, 24));
    expect(validateMeshletData(data)).toEqual([]);
    const mesh = new MeshletMesh(data, new MeshBasicMaterial());
    const camera = new PerspectiveCamera(50, 1, 0.1, 5000);
    camera.position.set(0, 0, 2000);
    mesh.update(camera, 1080);
    expect(mesh.stats.triangles).toBeLessThan(24 * 24 * 12 * 0.1);
    mesh.dispose();
  });

  it("can restrict which attributes are baked", async () => {
    const data = await buildMeshletsFromGeometry(new TorusKnotGeometry(1, 0.3, 64, 12), { attributes: ["normal"] });
    expect(data.attributes.map((a) => a.name)).toEqual(["position", "normal"]);
  });
});
