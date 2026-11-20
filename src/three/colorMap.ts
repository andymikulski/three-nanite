import type { ColorMap } from "../build/appearance.js";

/** What of a material and its texture is looked at here (any three.js material fits). */
interface MaterialLike {
  map?: TextureLike | null;
  alphaTest?: number;
}
interface TextureLike {
  image?: unknown;
  flipY?: boolean;
  repeat?: { x: number; y: number };
  offset?: { x: number; y: number };
}
interface Pixels {
  data: ArrayLike<number>;
  width: number;
  height: number;
}
interface Context2D {
  drawImage(image: unknown, x: number, y: number, w: number, h: number): void;
  getImageData(x: number, y: number, w: number, h: number): Pixels;
}
interface CanvasLike {
  width: number;
  height: number;
  getContext(kind: "2d", options?: { willReadFrequently?: boolean }): Context2D | null;
}

/** The largest side the pixels are read at: the builder looks at a small copy anyway. */
const LIMIT = 512;
const cache = new WeakMap<object, { data: Uint8ClampedArray | Uint8Array; width: number; height: number }>();

function canvas(width: number, height: number): CanvasLike | null {
  const g = globalThis as {
    OffscreenCanvas?: new (w: number, h: number) => CanvasLike;
    document?: { createElement(tag: "canvas"): CanvasLike };
  };
  if (typeof g.OffscreenCanvas === "function") return new g.OffscreenCanvas(width, height);
  if (g.document) {
    const c = g.document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c;
  }
  return null;
}

/**
 * The base colour of a material as the hierarchy builder takes it (`colorMap`): the pixels of its
 * `map`, with the texture's `flipY`, `repeat` and `offset` and the material's `alphaTest`.
 *
 * Returns undefined when the material has no map or its pixels cannot be read here: the image is
 * not loaded yet, or there is no canvas to draw it to (Node; pass a `ColorMap` of decoded pixels
 * to the builder yourself there). A data texture of RGBA bytes is read as it is.
 */
export function colorMapFromMaterial(material: MaterialLike | null | undefined): ColorMap | undefined {
  const map = material?.map;
  const image = map?.image as (Partial<Pixels> & { naturalWidth?: number; naturalHeight?: number; videoWidth?: number }) | null | undefined;
  if (!map || !image || typeof image !== "object") return undefined;

  let pixels = cache.get(image);
  if (!pixels) {
    const width = Number(image.naturalWidth ?? image.width ?? 0);
    const height = Number(image.naturalHeight ?? image.height ?? 0);
    if (!(width > 0 && height > 0)) return undefined;
    const data = image.data;
    if (data && (data instanceof Uint8Array || data instanceof Uint8ClampedArray) && data.length === width * height * 4) {
      pixels = { data, width, height };
    } else {
      const scale = Math.min(1, LIMIT / Math.max(width, height));
      const w = Math.max(1, Math.round(width * scale));
      const h = Math.max(1, Math.round(height * scale));
      const context = canvas(w, h)?.getContext("2d", { willReadFrequently: true });
      if (!context) return undefined;
      try {
        context.drawImage(image, 0, 0, w, h);
        const read = context.getImageData(0, 0, w, h);
        pixels = { data: read.data as Uint8ClampedArray, width: read.width, height: read.height };
      } catch {
        // an image from another origin, or not something a canvas can draw
        return undefined;
      }
    }
    cache.set(image, pixels);
  }

  return {
    data: pixels.data,
    width: pixels.width,
    height: pixels.height,
    // (row 0 is v = 0 unless the texture is flipped, for an image and for a data texture alike)
    flipY: map.flipY ?? !image.data,
    repeat: [map.repeat?.x ?? 1, map.repeat?.y ?? 1],
    offset: [map.offset?.x ?? 0, map.offset?.y ?? 0],
    alphaTest: material?.alphaTest ?? 0,
  };
}
