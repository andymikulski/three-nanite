import { BoxGeometry, BufferAttribute, BufferGeometry } from "three";
import { describe, expect, it } from "vitest";
import { buildMeshlets } from "../src/build/buildMeshlets.js";
import { buildMeshletsFromGeometry } from "../src/build/index.js";
import { colorMapFromMaterial } from "../src/three/colorMap.js";
import { makeColorMap, roofBuilding } from "./helpers/roofBuilding.js";

describe("colorMapFromMaterial", () => {
  it("K: is undefined without a material or a map", () => {
    expect(colorMapFromMaterial(undefined)).toBeUndefined();
    expect(colorMapFromMaterial(null)).toBeUndefined();
    expect(colorMapFromMaterial({})).toBeUndefined();
    expect(colorMapFromMaterial({ map: null })).toBeUndefined();
  });

  it("L: reads a data texture as it is", () => {
    const data = new Uint8Array(2 * 2 * 4).fill(255);
    const material = {
      map: { image: { data, width: 2, height: 2 }, flipY: false, repeat: { x: 2, y: 3 }, offset: { x: 0.5, y: 0.25 } },
      alphaTest: 0.5,
    };
    const map = colorMapFromMaterial(material)!;
    expect(map).toBeDefined();
    expect(map.data).toBe(data);
    const { data: _data, ...rest } = map;
    expect(rest).toEqual({ width: 2, height: 2, flipY: false, repeat: [2, 3], offset: [0.5, 0.25], alphaTest: 0.5 });
  });

  it("M: an image that cannot be read here is undefined, and building with it still works", async () => {
    const material = { map: { image: { width: 4, height: 4 } } };
    expect(colorMapFromMaterial(material)).toBeUndefined();
    const data = await buildMeshletsFromGeometry(new BoxGeometry(1, 1, 1, 4, 4, 4), { material });
    expect(data.clusterCount).toBeGreaterThan(0);
  });

  it("N: buildMeshletsFromGeometry passes the material's colour map through", async () => {
    const roof = roofBuilding();
    const geometry = new BufferGeometry();
    for (const a of roof.input.attributes) geometry.setAttribute(a.name, new BufferAttribute(a.array as Float32Array, a.itemSize, a.normalized));
    geometry.setIndex(new BufferAttribute(roof.indices, 1));

    const cm = makeColorMap();
    const texture = { image: { data: cm.data as Uint8Array, width: cm.width, height: cm.height }, flipY: false };
    // (with the uv seams unprotected the colour map makes a difference to this fixture's build)
    const options = { protectAttributes: [] as string[] };
    const viaMaterial = await buildMeshletsFromGeometry(geometry, { ...options, material: { map: texture } });
    const direct = await buildMeshlets(roof.input, {
      ...options,
      colorMap: { ...cm, flipY: false, repeat: [1, 1], offset: [0, 0], alphaTest: 0 },
    });
    const plain = await buildMeshletsFromGeometry(geometry, options);
    expect(Array.from(viaMaterial.indices)).toEqual(Array.from(direct.indices));
    expect(Array.from(viaMaterial.groupError)).toEqual(Array.from(direct.groupError));
    expect(Array.from(viaMaterial.groupError)).not.toEqual(Array.from(plain.groupError));
  });
});
