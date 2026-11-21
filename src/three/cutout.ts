import { DataTexture, LinearFilter, LinearMipmapLinearFilter, RGBAFormat, UnsignedByteType } from "three";
import type { Texture } from "three";

/** One level of an RGBA8 image: rows from the top. */
export interface MipLevel {
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * Makes an alpha cut-out image (leaves, grass cards, fences) look right at a distance:
 *
 * - **Colour in the transparent texels.** Exporters often leave them black (or any colour), and
 *   partly transparent edge texels over a white or black matte; a filtered texture then mixes that
 *   colour into every edge, and the further away, the more of the cut-out is edge: canopies go dark
 *   or pale and speckled. Each transparent texel takes the colour of the opaque texels nearest it
 *   (by a push-pull fill), and so does a partly transparent one, keeping its alpha, so filtering
 *   mixes like with like.
 * - **Mipmaps that keep their coverage.** Averaged alpha falls below the cut-out threshold as a
 *   level gets coarser, so a plain mipmapped cut-out thins out and is all but gone far away. Each
 *   level's alpha is scaled so that as many of its texels pass the threshold, proportionally, as in
 *   the full image (Castaño, "Computing Alpha Mipmaps").
 *
 * The second is what makes the hierarchy builder's `dilateBorders` the right choice for such
 * foliage: the material then keeps its coverage with distance, as the dilated levels do.
 *
 * @param alphaTest The material's cut-out threshold (0 to 1).
 * @returns Every level, the full image first, down to 1 x 1.
 */
export function cutoutMipmaps(image: MipLevel, alphaTest: number): MipLevel[] {
  const { width, height } = image;
  const threshold = Math.round(Math.min(1, Math.max(0, alphaTest)) * 255);
  const base = new Uint8Array(image.data);
  fillTransparent(base, width, height);

  const levels: MipLevel[] = [{ data: base, width, height }];
  while (levels[levels.length - 1].width > 1 || levels[levels.length - 1].height > 1) {
    levels.push(halve(levels[levels.length - 1]));
  }
  const want = coverage(base, threshold);
  for (let l = 1; l < levels.length; l++) keepCoverage(levels[l].data, want, threshold);
  return levels;
}

/** Fraction of the texels whose alpha passes the threshold. */
function coverage(data: Uint8Array, threshold: number, scale = 1): number {
  const n = data.length >> 2;
  if (n === 0) return 0;
  let passed = 0;
  // (as three's cut-out test: kept when alpha >= alphaTest)
  for (let i = 3; i < data.length; i += 4) if (Math.min(255, data[i] * scale) >= threshold) passed++;
  return passed / n;
}

/** Scales the alpha of a level so that its coverage is as near `want` as it gets. */
function keepCoverage(data: Uint8Array, want: number, threshold: number): void {
  if (threshold <= 0) return;
  // (a level that already has it keeps its own alpha)
  if (Math.abs(coverage(data, threshold) - want) < 0.5 / (data.length >> 2)) return;
  let lo = 0;
  let hi = 4;
  for (let k = 0; k < 14; k++) {
    const mid = (lo + hi) / 2;
    if (coverage(data, threshold, mid) < want) lo = mid;
    else hi = mid;
  }
  const scale = (lo + hi) / 2;
  for (let i = 3; i < data.length; i += 4) data[i] = Math.min(255, Math.round(data[i] * scale));
}

/** A level half the size (a box filter; odd sizes take the last row or column twice). */
function halve(level: MipLevel): MipLevel {
  const { data, width, height } = level;
  const w = Math.max(1, width >> 1);
  const h = Math.max(1, height >> 1);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const y0 = Math.min(height - 1, y * 2);
    const y1 = Math.min(height - 1, y * 2 + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.min(width - 1, x * 2);
      const x1 = Math.min(width - 1, x * 2 + 1);
      const a = (y0 * width + x0) * 4;
      const b = (y0 * width + x1) * 4;
      const c = (y1 * width + x0) * 4;
      const d = (y1 * width + x1) * 4;
      const o = (y * w + x) * 4;
      for (let k = 0; k < 4; k++) out[o + k] = (data[a + k] + data[b + k] + data[c + k] + data[d + k] + 2) >> 2;
    }
  }
  return { data: out, width: w, height: h };
}

/** The alpha from which a texel's colour is taken as it is (see `fillTransparent`). */
const OPAQUE = 250;

/**
 * Gives every texel that is not opaque the colour of the opaque ones around it, keeping its alpha
 * (push-pull: average the opaque texels into ever coarser levels, then fill each level's holes from
 * the next one).
 */
function fillTransparent(data: Uint8Array, width: number, height: number): void {
  // level 0: colour weighted by whether the texel has any alpha
  type Level = { r: Float32Array; g: Float32Array; b: Float32Array; w: Float32Array; width: number; height: number };
  const first: Level = { r: new Float32Array(width * height), g: new Float32Array(width * height), b: new Float32Array(width * height), w: new Float32Array(width * height), width, height };
  // Only (nearly) opaque texels give colour: a partly transparent one is usually an edge exported
  // over a matte (white or black), and its colour cannot be told from the matte's. Should nothing be
  // that opaque, every texel with any alpha gives colour instead.
  let solid = OPAQUE;
  let opaque = 0;
  for (let i = 0; i < width * height; i++) if (data[i * 4 + 3] >= solid) opaque++;
  if (opaque === width * height) return;
  if (opaque === 0) {
    solid = 1;
    for (let i = 0; i < width * height; i++) if (data[i * 4 + 3] >= solid) opaque++;
    if (opaque === 0) return;
  }
  for (let i = 0; i < width * height; i++) {
    const a = data[i * 4 + 3] >= solid ? 1 : 0;
    first.r[i] = data[i * 4] * a;
    first.g[i] = data[i * 4 + 1] * a;
    first.b[i] = data[i * 4 + 2] * a;
    first.w[i] = a;
  }

  // push: sums of colour and weight over 2 x 2 blocks
  const pyramid: Level[] = [first];
  while (pyramid[pyramid.length - 1].width > 1 || pyramid[pyramid.length - 1].height > 1) {
    const p = pyramid[pyramid.length - 1];
    const w = Math.max(1, p.width >> 1);
    const h = Math.max(1, p.height >> 1);
    const q: Level = { r: new Float32Array(w * h), g: new Float32Array(w * h), b: new Float32Array(w * h), w: new Float32Array(w * h), width: w, height: h };
    for (let y = 0; y < p.height; y++) {
      const qy = Math.min(h - 1, y >> 1);
      for (let x = 0; x < p.width; x++) {
        const j = qy * w + Math.min(w - 1, x >> 1);
        const i = y * p.width + x;
        q.r[j] += p.r[i];
        q.g[j] += p.g[i];
        q.b[j] += p.b[i];
        q.w[j] += p.w[i];
      }
    }
    pyramid.push(q);
  }
  // pull, coarsest first: a texel that gave no colour takes the average of its parent
  for (let l = pyramid.length - 2; l >= 0; l--) {
    const p = pyramid[l];
    const q = pyramid[l + 1];
    for (let y = 0; y < p.height; y++) {
      const qy = Math.min(q.height - 1, y >> 1);
      for (let x = 0; x < p.width; x++) {
        const i = y * p.width + x;
        const missing = 1 - p.w[i];
        if (missing <= 0) continue;
        const j = qy * q.width + Math.min(q.width - 1, x >> 1);
        const k = q.w[j] > 0 ? missing / q.w[j] : 0;
        p.r[i] += q.r[j] * k;
        p.g[i] += q.g[j] * k;
        p.b[i] += q.b[j] * k;
        p.w[i] = 1;
      }
    }
    // (every texel now holds a sum of weight at least one; those of weight one are their colour)
  }
  for (let i = 0; i < width * height; i++) {
    if (data[i * 4 + 3] >= solid) continue;
    data[i * 4] = Math.round(first.r[i]);
    data[i * 4 + 1] = Math.round(first.g[i]);
    data[i * 4 + 2] = Math.round(first.b[i]);
  }
}

interface Canvas2D {
  drawImage(image: unknown, x: number, y: number, w: number, h: number): void;
  getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray };
}

/**
 * {@link cutoutMipmaps} for a loaded three.js texture: a `DataTexture` with those levels and the
 * texture's own settings (colour space, wrapping, repeat, `flipY`), for a material with `alphaTest`.
 * Works on `WebGLRenderer` and `WebGPURenderer`. Needs a canvas to read the image (a browser or a
 * worker); returns the texture unchanged where there is none or the image is not loaded.
 *
 * @param maxSize The largest side kept (larger images are scaled down first). Default 2048.
 */
export function prepareCutoutTexture(texture: Texture, alphaTest: number, maxSize = 2048): Texture {
  const image = texture.image as { width?: number; height?: number; naturalWidth?: number; naturalHeight?: number; data?: unknown } | null;
  if (!image) return texture;
  const width0 = Number(image.naturalWidth ?? image.width ?? 0);
  const height0 = Number(image.naturalHeight ?? image.height ?? 0);
  if (!(width0 > 0 && height0 > 0)) return texture;
  const scale = Math.min(1, maxSize / Math.max(width0, height0));
  const width = Math.max(1, Math.round(width0 * scale));
  const height = Math.max(1, Math.round(height0 * scale));

  let pixels: Uint8Array;
  if (image.data instanceof Uint8Array && image.data.length === width0 * height0 * 4 && scale === 1) {
    pixels = image.data;
  } else {
    const g = globalThis as { OffscreenCanvas?: new (w: number, h: number) => { getContext(kind: "2d", o?: object): Canvas2D | null }; document?: { createElement(tag: "canvas"): { width: number; height: number; getContext(kind: "2d", o?: object): Canvas2D | null } } };
    let context: Canvas2D | null = null;
    if (typeof g.OffscreenCanvas === "function") context = new g.OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true });
    else if (g.document) {
      const c = g.document.createElement("canvas");
      c.width = width;
      c.height = height;
      context = c.getContext("2d", { willReadFrequently: true });
    }
    if (!context) return texture;
    try {
      context.drawImage(image, 0, 0, width, height);
      pixels = new Uint8Array(context.getImageData(0, 0, width, height).data.buffer);
    } catch {
      return texture;
    }
  }

  const levels = cutoutMipmaps({ data: pixels, width, height }, alphaTest);
  const out = new DataTexture(levels[0].data, width, height, RGBAFormat, UnsignedByteType);
  out.mipmaps = levels as unknown as typeof out.mipmaps;
  out.generateMipmaps = false;
  out.minFilter = LinearMipmapLinearFilter;
  out.magFilter = LinearFilter;
  out.colorSpace = texture.colorSpace;
  out.wrapS = texture.wrapS;
  out.wrapT = texture.wrapT;
  out.anisotropy = texture.anisotropy;
  out.repeat.copy(texture.repeat);
  out.offset.copy(texture.offset);
  out.center.copy(texture.center);
  out.rotation = texture.rotation;
  // (rows are as the image has them, so the texture is flipped as the original was)
  out.flipY = texture.flipY;
  out.name = texture.name;
  // (read by buildMeshletsFromGeometry: a material with this map keeps its coverage, so its levels should too)
  out.userData = { ...texture.userData, keepsCoverage: true };
  out.needsUpdate = true;
  return out;
}
