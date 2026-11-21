import { describe, expect, it } from "vitest";
import { BufferAttribute, BufferGeometry, DataTexture, MeshStandardMaterial } from "three";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { computeLodLevels, prepareCutoutTexture, validateMeshletData } from "../src/index.js";

/** Loose cards (two triangles each, no shared vertices between cards): what a canopy or a meadow is made of. */
function cards(n: number): BufferGeometry {
  const position = new Float32Array(n * 4 * 3);
  const normal = new Float32Array(n * 4 * 3);
  const uv = new Float32Array(n * 4 * 2);
  const index = new Uint32Array(n * 6);
  let s = 7;
  const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let c = 0; c < n; c++) {
    const x = r() * 10;
    const y = r() * 10;
    const z = r() * 10;
    const corners = [
      [0, 0],
      [0.4, 0],
      [0.4, 0.4],
      [0, 0.4],
    ];
    corners.forEach(([u, v], k) => {
      position.set([x + u, y + v, z], (c * 4 + k) * 3);
      normal.set([0, 0, 1], (c * 4 + k) * 3);
      uv.set([u / 0.4, v / 0.4], (c * 4 + k) * 2);
    });
    index.set([c * 4, c * 4 + 1, c * 4 + 2, c * 4, c * 4 + 2, c * 4 + 3], c * 6);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(position, 3));
  g.setAttribute("normal", new BufferAttribute(normal, 3));
  g.setAttribute("uv", new BufferAttribute(uv, 2));
  g.setIndex(new BufferAttribute(index, 1));
  return g;
}

describe("buildMeshletsFromGeometry: dilateBorders follows the material", () => {
  const geometry = cards(1500);
  const vertices = geometry.getAttribute("position").count;
  const added = async (material: MeshStandardMaterial | null, dilateBorders?: boolean) =>
    (await buildMeshletsFromGeometry(geometry, { attributes: ["normal", "uv"], material, dilateBorders })).vertexCount - vertices;

  it("on for an opaque material, off without one", async () => {
    expect(await added(new MeshStandardMaterial())).toBeGreaterThan(0);
    expect(await added(null)).toBe(0);
  }, 60000);

  it("off for a plain cut-out and a transparent material, on for a cut-out that keeps its coverage", async () => {
    expect(await added(new MeshStandardMaterial({ alphaTest: 0.5 }))).toBe(0);
    expect(await added(new MeshStandardMaterial({ transparent: true, opacity: 0.5 }))).toBe(0);
    const pixels = new Uint8Array(8 * 8 * 4).fill(255);
    for (let i = 3; i < pixels.length; i += 8) pixels[i] = 0;
    const map = prepareCutoutTexture(new DataTexture(pixels, 8, 8), 0.5);
    expect(map.userData.keepsCoverage).toBe(true);
    expect(await added(new MeshStandardMaterial({ alphaTest: 0.5, map }))).toBeGreaterThan(0);
  }, 60000);

  it("an explicit value wins", async () => {
    expect(await added(new MeshStandardMaterial(), false)).toBe(0);
    expect(await added(null, true)).toBeGreaterThan(0);
  }, 60000);

  it("a level whose borders were moved counts the move in its error, and the hierarchy stays valid", async () => {
    const plain = await buildMeshletsFromGeometry(geometry, { attributes: ["normal", "uv"], dilateBorders: false });
    const dilated = await buildMeshletsFromGeometry(geometry, { attributes: ["normal", "uv"], dilateBorders: true });
    expect(validateMeshletData(dilated)).toEqual([]);
    expect(computeLodLevels(plain).count).toBeGreaterThan(1);
    // Every vertex the dilation added is at least as far from the input as the nearest input vertex,
    // and a cluster that uses it is only drawn once the step that made it (its refined group) is
    // allowed that much error.
    const position = geometry.getAttribute("position").array as Float32Array;
    const moved = dilated.attributes.find((x) => x.name === "position")!.array as Float32Array;
    const original = position.length / 3;
    expect(dilated.vertexCount).toBeGreaterThan(original);
    const away = new Float32Array(dilated.vertexCount);
    for (let v = original; v < dilated.vertexCount; v++) {
      let best = Infinity;
      for (let o = 0; o < original; o++) {
        const d = Math.hypot(moved[v * 3] - position[o * 3], moved[v * 3 + 1] - position[o * 3 + 1], moved[v * 3 + 2] - position[o * 3 + 2]);
        if (d < best) best = d;
      }
      away[v] = best;
    }
    let checked = 0;
    for (let c = 0; c < dilated.clusterCount; c++) {
      const refined = dilated.clusterRefined[c];
      if (refined < 0) continue;
      let far = 0;
      for (let k = 0; k < dilated.clusterIndexCount[c]; k++) far = Math.max(far, away[dilated.indices[dilated.clusterIndexStart[c] + k]]);
      if (far === 0) continue;
      checked++;
      expect(dilated.groupError[refined], `cluster ${c}`).toBeGreaterThanOrEqual(far * 0.999);
    }
    expect(checked).toBeGreaterThan(0);
  }, 60000);
});
