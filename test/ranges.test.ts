import { describe, expect, it } from "vitest";
import { MeshBasicMaterial, PerspectiveCamera, Sphere } from "three";
import { ClusterSelector, computeDrawRanges, lodScale, rangesOfCut } from "../src/core/index.js";
import { MeshletMesh } from "../src/three/MeshletMesh.js";
import { buildMeshlets } from "../src/build/index.js";
import { makeFixture } from "./helpers/fixture.js";
import { geometryToInput, torusKnot } from "./helpers/buildMeshes.js";

describe("draw ranges", () => {
  it("fuses the fixture's four fine clusters into one range", () => {
    const data = makeFixture();
    const ranges = computeDrawRanges(data);
    expect(ranges.count).toBe(2);
    expect(Array.from(ranges.ofCluster)).toEqual([0, 0, 0, 0, 1]);
    expect(Array.from(ranges.firstCluster)).toEqual([0, 4]);
    expect(Array.from(ranges.clusterCount)).toEqual([4, 1]);
    expect(Array.from(ranges.indexStart)).toEqual([0, data.clusterIndexStart[4]]);
    expect(Array.from(ranges.indexCount)).toEqual([4 * 96, 96]);
  });

  it("keeps one range per cluster when merging is off", () => {
    const data = makeFixture();
    const ranges = computeDrawRanges(data, false);
    expect(ranges.count).toBe(data.clusterCount);
    expect(Array.from(ranges.indexStart)).toEqual(Array.from(data.clusterIndexStart));
    expect(Array.from(ranges.bounds)).toEqual(Array.from(data.clusterBounds));
  });

  it("covers every cluster exactly once, with bounds around its clusters, on a built hierarchy", async () => {
    const data = await buildMeshlets(geometryToInput(torusKnot(256, 48, true)));
    const ranges = computeDrawRanges(data);
    expect(ranges.count).toBeLessThan(data.clusterCount);
    let clusters = 0;
    let indices = 0;
    for (let r = 0; r < ranges.count; r++) {
      const first = ranges.firstCluster[r];
      expect(first).toBe(clusters);
      expect(ranges.indexStart[r]).toBe(indices);
      for (let c = first; c < first + ranges.clusterCount[r]; c++) {
        expect(ranges.ofCluster[c]).toBe(r);
        expect(data.clusterGroup[c]).toBe(data.clusterGroup[first]);
        expect(data.clusterRefined[c]).toBe(data.clusterRefined[first]);
        const d = Math.hypot(
          data.clusterBounds[c * 4] - ranges.bounds[r * 4],
          data.clusterBounds[c * 4 + 1] - ranges.bounds[r * 4 + 1],
          data.clusterBounds[c * 4 + 2] - ranges.bounds[r * 4 + 2],
        );
        expect(d + data.clusterBounds[c * 4 + 3]).toBeLessThanOrEqual(ranges.bounds[r * 4 + 3] * (1 + 1e-5) + 1e-6);
      }
      clusters += ranges.clusterCount[r];
      indices += ranges.indexCount[r];
    }
    expect(clusters).toBe(data.clusterCount);
    expect(indices).toBe(data.indices.length);
  });

  it("a cut in ranges draws exactly the cut's clusters, from any viewpoint", async () => {
    const data = await buildMeshlets(geometryToInput(torusKnot(256, 48, true)));
    const ranges = computeDrawRanges(data);
    const selector = new ClusterSelector(data);
    const cut = new Uint32Array(data.clusterCount);
    const k = lodScale(1, 1080, 2);
    for (const [x, y, z] of [[0, 0, 0.2], [0.4, 0.1, 1], [3, 0, 2], [0, 0, 20], [0, 0, 500]]) {
      const n = selector.select(cut, x, y, z, k, 0.01);
      const expected = Array.from(cut.subarray(0, n)).sort((a, b) => a - b);
      const m = rangesOfCut(ranges, cut, n);
      const drawn: number[] = [];
      for (let i = 0; i < m; i++) {
        const r = cut[i];
        for (let c = ranges.firstCluster[r]; c < ranges.firstCluster[r] + ranges.clusterCount[r]; c++) drawn.push(c);
      }
      expect(drawn.sort((a, b) => a - b)).toEqual(expected);
      expect(m).toBeLessThanOrEqual(n);
    }
  });
});

describe("MeshletMesh with merged clusters (the default)", () => {
  it("registers one geometry per draw range and draws the fine level in one slot", () => {
    const data = makeFixture();
    // drawn by cluster: whole-mesh levels (the default for a mesh this small) are tested on their own
    const mesh = new MeshletMesh(data, new MeshBasicMaterial(), { instancedLevels: false });
    expect(mesh.ranges.count).toBe(2);
    const sphere = new Sphere();
    for (let r = 0; r < mesh.ranges.count; r++) {
      const range = mesh.batched.getGeometryRangeAt(r)!;
      expect(range.indexStart).toBe(mesh.ranges.indexStart[r]);
      expect(range.indexCount).toBe(mesh.ranges.indexCount[r]);
      expect(mesh.batched.getBoundingSphereAt(r, sphere)).not.toBeNull();
      expect(sphere.radius).toBeCloseTo(mesh.ranges.bounds[r * 4 + 3]);
    }

    const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 10000);
    camera.position.set(0, 0, 3);
    camera.updateMatrixWorld(true);
    mesh.update(camera, 1080);
    expect(mesh.stats.clusters).toBe(4);
    expect(mesh.stats.draws).toBe(1);
    expect(mesh.stats.triangles).toBe(128);
    expect(mesh.batched.instanceCount).toBe(1);

    camera.position.set(0, 0, 5000);
    camera.updateMatrixWorld(true);
    mesh.update(camera, 1080);
    expect(mesh.stats.clusters).toBe(1);
    expect(mesh.stats.draws).toBe(1);
    expect(mesh.stats.triangles).toBe(32);
    expect(mesh.batched.instanceCount).toBe(1); // the slot was reused
  });
});
