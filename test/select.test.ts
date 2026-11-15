import { describe, expect, it } from "vitest";
import { ClusterSelector, countTriangles, lodScale } from "../src/core/select.js";
import { FIXTURE_ERROR, makeFixture } from "./helpers/fixture.js";

const sorted = (out: Uint32Array, n: number) => Array.from(out.subarray(0, n)).sort((a, b) => a - b);

describe("ClusterSelector", () => {
  const data = makeFixture();
  const selector = new ClusterSelector(data);
  const out = new Uint32Array(data.clusterCount);

  it("selectByError(0) returns the original clusters", () => {
    const n = selector.selectByError(out, 0);
    expect(sorted(out, n)).toEqual([0, 1, 2, 3]);
    expect(countTriangles(data, out, n)).toBe(128);
  });

  it("selectByError at or above the group error returns the coarse cluster", () => {
    // groupError is stored as float32, so compare against the stored value
    const n = selector.selectByError(out, data.groupError[0]);
    expect(sorted(out, n)).toEqual([4]);
    expect(sorted(out, selector.selectByError(out, Infinity))).toEqual([4]);
    expect(countTriangles(data, out, 1)).toBe(32);
  });

  it("selectByError just below the group error keeps the fine level", () => {
    const n = selector.selectByError(out, FIXTURE_ERROR * 0.99);
    expect(sorted(out, n)).toEqual([0, 1, 2, 3]);
  });

  it("perspective selection switches with distance", () => {
    // fov 60deg => proj11 = cot(30deg); 1px budget at 1080p
    const k = lodScale(1, 1080, 1 / Math.tan(Math.PI / 6));
    // error / (dist - r) > k  <=>  dist < error / k + r
    const switchDistance = FIXTURE_ERROR / k + Math.SQRT2;

    let n = selector.select(out, 0, 0, switchDistance * 0.9, k, 0.1);
    expect(sorted(out, n)).toEqual([0, 1, 2, 3]);

    n = selector.select(out, 0, 0, switchDistance * 1.1, k, 0.1);
    expect(sorted(out, n)).toEqual([4]);
  });

  it("is direction independent (only distance matters)", () => {
    const k = lodScale(1, 1080, 1.7);
    const far = 500;
    for (const [x, y, z] of [[far, 0, 0], [0, -far, 0], [0, 0, -far], [far, far, far]]) {
      expect(sorted(out, selector.select(out, x, y, z, k, 0.1))).toEqual([4]);
    }
  });

  it("clamps to the near plane when the camera is inside the LOD sphere", () => {
    const k = lodScale(1, 1080, 1.7);
    const n = selector.select(out, 0, 0, 0, k, 0.1);
    expect(sorted(out, n)).toEqual([0, 1, 2, 3]);
  });

  it("a larger pixel budget coarsens the cut", () => {
    const proj = 1.7;
    const tight = lodScale(1, 1080, proj);
    const loose = lodScale(64, 1080, proj);
    expect(sorted(out, selector.select(out, 0, 0, 5, tight, 0.1))).toEqual([0, 1, 2, 3]);
    expect(sorted(out, selector.select(out, 0, 0, 5, loose, 0.1))).toEqual([4]);
  });

  it("orthographic selection ignores position and honours world scale", () => {
    const proj = 2 / 10; // frustum 10 units tall
    // error in pixels = FIXTURE_ERROR * scale * 0.5 * 1080 * proj = 5.4px * scale
    expect(sorted(out, selector.select(out, 9, 9, 9, lodScale(1, 1080, proj, 1, true), 0.1, true))).toEqual([0, 1, 2, 3]);
    expect(sorted(out, selector.select(out, 9, 9, 9, lodScale(10, 1080, proj, 1, true), 0.1, true))).toEqual([4]);
    expect(sorted(out, selector.select(out, 9, 9, 9, lodScale(10, 1080, proj, 4, true), 0.1, true))).toEqual([0, 1, 2, 3]);
  });

  it("can be reused many times without state leaking between calls", () => {
    const k = lodScale(1, 1080, 1.7);
    for (let i = 0; i < 1000; i++) {
      const near = i % 2 === 0;
      const n = selector.select(out, 0, 0, near ? 2 : 1000, k, 0.1);
      expect(n).toBe(near ? 4 : 1);
    }
  });
});
