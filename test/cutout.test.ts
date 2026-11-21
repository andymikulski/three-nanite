import { describe, expect, it } from "vitest";
import { DataTexture, LinearMipmapLinearFilter, RepeatWrapping, SRGBColorSpace } from "three";
import { cutoutMipmaps, prepareCutoutTexture } from "../src/index.js";

/** A leaf texture as exporters write them: green discs on black, transparent texels. */
function leaves(size: number): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const discs = [
    [0.3, 0.3, 0.12],
    [0.7, 0.35, 0.1],
    [0.5, 0.72, 0.14],
  ];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x + 0.5) / size;
      const v = (y + 0.5) / size;
      const inside = discs.some(([cx, cy, r]) => Math.hypot(u - cx, v - cy) < r);
      const o = (y * size + x) * 4;
      if (inside) data.set([60, 140, 30, 255], o);
      else data.set([0, 0, 0, 0], o);
    }
  }
  return data;
}

const coverage = (data: Uint8Array, threshold: number) => {
  let n = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i] >= threshold) n++;
  return n / (data.length / 4);
};

describe("cutoutMipmaps", () => {
  it("makes every level down to 1 x 1, the first the image itself where it is opaque", () => {
    const image = { data: leaves(64), width: 64, height: 64 };
    const levels = cutoutMipmaps(image, 0.5);
    expect(levels.map((l) => l.width)).toEqual([64, 32, 16, 8, 4, 2, 1]);
    for (let i = 0; i < 64 * 64; i++) {
      if (image.data[i * 4 + 3] === 255) expect([...levels[0].data.subarray(i * 4, i * 4 + 4)]).toEqual([60, 140, 30, 255]);
    }
    // the input is not modified
    expect(image.data[3]).toBe(0);
    expect(image.data[0]).toBe(0);
  });

  it("gives transparent texels the colour of the opaque ones, not black", () => {
    const levels = cutoutMipmaps({ data: leaves(64), width: 64, height: 64 }, 0.5);
    const base = levels[0].data;
    for (let i = 0; i < 64 * 64; i++) {
      expect(base[i * 4 + 3] === 0 || base[i * 4 + 3] === 255).toBe(true);
      expect([base[i * 4], base[i * 4 + 1], base[i * 4 + 2]]).toEqual([60, 140, 30]);
    }
    // so the coarser levels are the leaves' colour throughout too
    const last = levels[levels.length - 1].data;
    expect(Math.abs(last[0] - 60) + Math.abs(last[1] - 140) + Math.abs(last[2] - 30)).toBeLessThanOrEqual(3);
  });

  it("takes the matte out of partly transparent edge texels", () => {
    // a green square on a white matte: its one-texel border is half transparent and half white
    const size = 32;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const o = (y * size + x) * 4;
        const inside = x >= 8 && x < 24 && y >= 8 && y < 24;
        const edge = !inside && x >= 7 && x < 25 && y >= 7 && y < 25;
        if (inside) data.set([40, 160, 20, 255], o);
        else if (edge) data.set([147, 207, 137, 128], o); // half green, half white
        else data.set([255, 255, 255, 0], o);
      }
    }
    const base = cutoutMipmaps({ data, width: size, height: size }, 0.5)[0].data;
    const edge = (7 * size + 16) * 4;
    // the white is gone from the edge: it is close to the green, and its alpha is kept
    expect(Math.abs(base[edge] - 40) + Math.abs(base[edge + 1] - 160) + Math.abs(base[edge + 2] - 20)).toBeLessThan(6);
    expect(base[edge + 3]).toBe(128);
    // and an opaque texel is untouched
    expect([...base.subarray((16 * size + 16) * 4, (16 * size + 16) * 4 + 4)]).toEqual([40, 160, 20, 255]);
  });

  it("keeps the share of texels that pass the cut-out threshold at every level", () => {
    const image = { data: leaves(128), width: 128, height: 128 };
    for (const alphaTest of [0.2, 0.5]) {
      const threshold = Math.round(alphaTest * 255);
      const want = coverage(image.data, threshold);
      const levels = cutoutMipmaps(image, alphaTest);
      // (until a level is too small to have that share at all)
      for (const level of levels.filter((l) => l.width >= 8)) {
        expect(Math.abs(coverage(level.data, threshold) - want), `${alphaTest}: ${level.width}`).toBeLessThan(0.06);
      }
    }
  });

  it("leaves an image with no transparent texels as it is", () => {
    const data = new Uint8Array(16 * 16 * 4).fill(200);
    const levels = cutoutMipmaps({ data, width: 16, height: 16 }, 0.5);
    expect([...levels[0].data]).toEqual([...data]);
    expect([...levels[1].data.subarray(0, 4)]).toEqual([200, 200, 200, 200]);
  });

  it("handles sizes that are not powers of two", () => {
    const levels = cutoutMipmaps({ data: new Uint8Array(12 * 5 * 4).fill(255), width: 12, height: 5 }, 0.5);
    expect(levels.map((l) => [l.width, l.height])).toEqual([
      [12, 5],
      [6, 2],
      [3, 1],
      [1, 1],
    ]);
  });
});

describe("prepareCutoutTexture", () => {
  it("returns a DataTexture with the levels and the source's settings", () => {
    const source = new DataTexture(leaves(32), 32, 32);
    source.colorSpace = SRGBColorSpace;
    source.wrapS = RepeatWrapping;
    source.flipY = false;
    source.repeat.set(2, 3);
    const out = prepareCutoutTexture(source, 0.5) as DataTexture;
    expect(out).not.toBe(source);
    expect(out.mipmaps.length).toBe(6);
    expect(out.generateMipmaps).toBe(false);
    expect(out.minFilter).toBe(LinearMipmapLinearFilter);
    expect(out.colorSpace).toBe(SRGBColorSpace);
    expect(out.wrapS).toBe(RepeatWrapping);
    expect(out.repeat.toArray()).toEqual([2, 3]);
    expect(out.flipY).toBe(false);
  });

  it("returns the texture unchanged when it has no image", () => {
    const source = new DataTexture(null, 0, 0);
    expect(prepareCutoutTexture(source, 0.5)).toBe(source);
  });
});
