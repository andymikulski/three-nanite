import type { MeshletAttribute, TypedArray } from "../types.js";

/**
 * The base colour of a mesh, for judging what a simplification step changes to the eye
 * (`BuildMeshletsOptions.colorMap`).
 */
export interface ColorMap {
  /** RGBA pixels, 4 bytes each, row by row from the top (as a canvas `ImageData` or a decoded PNG has them). */
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  /** The uv attribute the texture is mapped with. Default `"uv"`. */
  attribute?: string;
  /**
   * Whether v = 0 is the bottom row of the image, as for a three.js texture loaded from an image
   * (`flipY`, the default there), or the top row, as in glTF. Default true.
   */
  flipY?: boolean;
  /** The texture's `repeat` and `offset`. Default `[1, 1]` and `[0, 0]`. */
  repeat?: [number, number];
  offset?: [number, number];
  /** A cut-out material's `alphaTest`: where the texture's alpha is below it there is a hole. Default 0 (none). */
  alphaTest?: number;
}

/** Pixels across a view. */
const SIZE = 40;
const PIXELS = SIZE * SIZE;
/** How far two colours may be apart, per channel (0 to 1, as stored), and still count as the same. */
const TOLERANCE = 0.11;
/** Changed pixels below this number in a view are taken for rasterisation noise. */
const NOISE = 4;

/**
 * The views: straight along the axes, and along four diagonals that between them look at every
 * face of a cube at a slant. Each is direction, right, up.
 */
const VIEWS: Float64Array = (() => {
  const dirs: number[][] = [];
  for (const s of [1, -1]) dirs.push([s, 0, 0], [0, s, 0], [0, 0, s]);
  dirs.push([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]);
  const out = new Float64Array(dirs.length * 9);
  dirs.forEach((d, i) => {
    const l = Math.hypot(d[0], d[1], d[2]);
    const dx = d[0] / l;
    const dy = d[1] / l;
    const dz = d[2] / l;
    // any vector not along the direction
    const ax = Math.abs(dx) < 0.9 ? 1 : 0;
    const ay = ax === 1 ? 0 : 1;
    let ux = ay * dz;
    let uy = -ax * dz;
    let uz = ax * dy - ay * dx;
    const ul = Math.hypot(ux, uy, uz);
    ux /= ul;
    uy /= ul;
    uz /= ul;
    out.set([dx, dy, dz, ux, uy, uz, dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux], i * 9);
  });
  return out;
})();
const VIEW_COUNT = VIEWS.length / 9;

/**
 * Measures what a simplification step changes TO THE EYE, as opposed to how far it moves the
 * surface.
 *
 * A simplifier's error is geometric: how far the surface moved, plus a term for how attributes were
 * stretched. That says nothing about what is seen where. Closing the raised rim of a flat roof over
 * its recessed floor moves the surface by the depth of the recess, a few centimetres, and turns the
 * whole roof from the colour of the floor to the colour of the rim. Two surfaces lying close above
 * one another can swap, a collapse across a texture seam can paint a face with its neighbour's
 * texture: all cheap by the geometric measure and plain to see.
 *
 * So the group is drawn before and after the step, small and from ten directions, with its
 * base colour, and the two are compared. Whatever differs and cannot be explained by the surface
 * having moved as far as the geometric error says is a changed region, and its size (the side of a
 * square of its area, in mesh units) is this error. Used as the error of the step, it makes the
 * step wait until that region is about a pixel across.
 */
export class AppearanceCheck {
  private readonly uv: TypedArray | null = null;
  private readonly uvSize: number = 2;
  private readonly vertexColor: TypedArray | null = null;
  private readonly vertexColorSize: number = 3;
  private readonly vertexColorDivisor: number = 1;
  private readonly map: Float32Array | null = null;
  private readonly mapWidth: number = 0;
  private readonly mapHeight: number = 0;
  private readonly flipY: boolean = true;
  private readonly repeatX: number = 1;
  private readonly repeatY: number = 1;
  private readonly offsetX: number = 0;
  private readonly offsetY: number = 0;
  private readonly alphaTest: number = 0;

  // ---- per group
  private centre = [0, 0, 0];
  private radius = 1;
  private sx = new Float32Array(0);
  private sy = new Float32Array(0);
  private sz = new Float32Array(0);
  private input = new Uint32Array(0);
  /** rgb of every pixel of every view before the step; r < 0 where nothing is drawn. */
  private readonly before = new Float32Array(VIEW_COUNT * PIXELS * 3);
  private readonly after = new Float32Array(PIXELS * 3);
  private readonly triangle = new Int32Array(PIXELS);
  private readonly weights = new Float32Array(PIXELS * 2);
  private readonly depth = new Float32Array(PIXELS);
  private readonly sample = new Float32Array(4);
  private positions: Float32Array = new Float32Array(0);
  private count = 0;
  private ready = false;

  /**
   * @param attributes The input's vertex attributes.
   * @param colorMap   The base colour texture, or undefined to go by a `color` attribute.
   * @returns null when there is nothing to go by (no texture with its uv attribute and no vertex colours).
   */
  static create(attributes: MeshletAttribute[], colorMap: ColorMap | undefined): AppearanceCheck | null {
    if (colorMap) {
      const uv = attributes.find((a) => a.name === (colorMap.attribute ?? "uv"));
      if (!uv || uv.itemSize < 2) return null;
      return new AppearanceCheck(uv, null, colorMap);
    }
    const color = attributes.find((a) => a.name === "color");
    return color && color.itemSize >= 3 ? new AppearanceCheck(null, color, undefined) : null;
  }

  private constructor(uv: MeshletAttribute | null, color: MeshletAttribute | null, colorMap: ColorMap | undefined) {
    if (uv && colorMap) {
      this.uv = uv.array;
      this.uvSize = uv.itemSize;
      this.flipY = colorMap.flipY ?? true;
      this.repeatX = colorMap.repeat?.[0] ?? 1;
      this.repeatY = colorMap.repeat?.[1] ?? 1;
      this.offsetX = colorMap.offset?.[0] ?? 0;
      this.offsetY = colorMap.offset?.[1] ?? 0;
      this.alphaTest = colorMap.alphaTest ?? 0;
      // A small copy: what counts is the colour of an area, not the texel a sample happens to hit.
      // (A cut-out keeps more of it, its holes being the shape of the thing.)
      const limit = this.alphaTest > 0 ? 256 : 64;
      const step = Math.max(1, Math.ceil(Math.max(colorMap.width, colorMap.height) / limit));
      const w = Math.max(1, Math.floor(colorMap.width / step));
      const h = Math.max(1, Math.floor(colorMap.height / step));
      const map = new Float32Array(w * h * 4);
      const data = colorMap.data;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let r = 0;
          let g = 0;
          let b = 0;
          let a = 0;
          for (let j = 0; j < step; j++) {
            for (let i = 0; i < step; i++) {
              const o = ((y * step + j) * colorMap.width + x * step + i) * 4;
              const alpha = data[o + 3] / 255;
              // colour counts by how much of it there is
              r += data[o] * alpha;
              g += data[o + 1] * alpha;
              b += data[o + 2] * alpha;
              a += alpha;
            }
          }
          const o = (y * w + x) * 4;
          const scale = a > 0 ? 1 / (a * 255) : 0;
          map[o] = r * scale;
          map[o + 1] = g * scale;
          map[o + 2] = b * scale;
          map[o + 3] = a / (step * step);
        }
      }
      this.map = map;
      this.mapWidth = w;
      this.mapHeight = h;
    } else if (color) {
      this.vertexColor = color.array;
      this.vertexColorSize = color.itemSize;
      const array = color.array;
      this.vertexColorDivisor = !color.normalized ? 1 : array instanceof Uint8Array ? 255 : array instanceof Uint16Array ? 65535 : 1;
    }
  }

  /**
   * Draws a group as it is before the step.
   *
   * @param indices   Triangle list in local vertex ids.
   * @param positions Local positions.
   * @param count     Number of local vertices.
   * @param toInput   Input vertex of each local vertex (whose attributes it has).
   * @returns false when there is nothing to compare: the group is one colour throughout.
   */
  prepare(indices: Uint32Array, positions: Float32Array, count: number, toInput: (local: number) => number): boolean {
    this.ready = false;
    if (this.sx.length < count) {
      this.sx = new Float32Array(count);
      this.sy = new Float32Array(count);
      this.sz = new Float32Array(count);
      this.input = new Uint32Array(count);
    }
    const input = this.input;
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let l = 0; l < count; l++) {
      input[l] = toInput(l);
      const x = positions[l * 3];
      const y = positions[l * 3 + 1];
      const z = positions[l * 3 + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const radius = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2;
    if (!(radius > 0)) return false;
    this.centre = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    this.radius = radius;
    this.positions = positions;
    this.count = count;

    // one colour throughout and no holes: no step can change what is seen where
    if (this.alphaTest === 0 && this.uniform(indices)) return false;

    for (let v = 0; v < VIEW_COUNT; v++) {
      this.project(v);
      this.draw(indices, this.before.subarray(v * PIXELS * 3, (v + 1) * PIXELS * 3));
    }
    this.ready = true;
    return true;
  }

  /**
   * The size of what a step changes to the eye, in mesh units; 0 when `prepare` found nothing to
   * compare.
   *
   * @param indices  The group after the step (local vertex ids, the vertices given to `prepare`).
   * @param geometric The step's geometric error: a difference that a move of that size explains is paid for already.
   */
  error(indices: Uint32Array, geometric: number): number {
    if (!this.ready) return 0;
    const pixel = (2 * this.radius) / SIZE;
    const reach = 1 + Math.min(3, Math.ceil((Number.isFinite(geometric) ? geometric : 0) / pixel));
    let worst = 0;
    for (let v = 0; v < VIEW_COUNT; v++) {
      this.project(v);
      this.draw(indices, this.after);
      const changed = this.compare(this.before.subarray(v * PIXELS * 3, (v + 1) * PIXELS * 3), this.after, reach);
      if (changed > worst) worst = changed;
      // a quarter of a view has changed: no other view can make this more than twice as bad, and
      // it is past mattering
      if (worst >= PIXELS / 4) break;
    }
    // (half: a region two pixels across that changes colour is about what a moved edge amounts to)
    return worst >= NOISE ? Math.sqrt(worst) * pixel * 0.5 : 0;
  }

  // ---------------------------------------------------------------------------------------------

  /**
   * Whether a triangle list is one colour: at four points of every triangle, a little way in from
   * each corner and in the middle (a texture may change colour between the corners of a triangle).
   */
  private uniform(indices: Uint32Array): boolean {
    const s = this.sample;
    const input = this.input;
    let r = -1;
    let g = 0;
    let b = 0;
    for (let t = 0; t < indices.length; t += 3) {
      const a = input[indices[t]];
      const bb = input[indices[t + 1]];
      const c = input[indices[t + 2]];
      for (let k = 0; k < 4; k++) {
        this.colorAt(a, bb, c, k === 0 ? 0.8 : k === 3 ? 1 / 3 : 0.1, k === 1 ? 0.8 : k === 3 ? 1 / 3 : 0.1);
        if (r < 0) {
          r = s[0];
          g = s[1];
          b = s[2];
        } else if (Math.abs(s[0] - r) > TOLERANCE || Math.abs(s[1] - g) > TOLERANCE || Math.abs(s[2] - b) > TOLERANCE) {
          return false;
        }
      }
    }
    return true;
  }

  /** Colour (and alpha) at a point of a triangle of INPUT vertices, into `sample`. */
  private colorAt(a: number, b: number, c: number, wa: number, wb: number): void {
    const wc = 1 - wa - wb;
    const out = this.sample;
    const map = this.map;
    if (map !== null) {
      const uv = this.uv!;
      const n = this.uvSize;
      let u = (uv[a * n] * wa + uv[b * n] * wb + uv[c * n] * wc) * this.repeatX + this.offsetX;
      let v = (uv[a * n + 1] * wa + uv[b * n + 1] * wb + uv[c * n + 1] * wc) * this.repeatY + this.offsetY;
      // (a coordinate of exactly 1 is the last texel, not the first of the next repeat)
      if (u < 0 || u > 1) u -= Math.floor(u);
      if (v < 0 || v > 1) v -= Math.floor(v);
      if (this.flipY) v = 1 - v;
      const w = this.mapWidth;
      const h = this.mapHeight;
      const x = Math.min(w - 1, Math.floor(u * w));
      const y = Math.min(h - 1, Math.floor(v * h));
      const o = (y * w + x) * 4;
      out[0] = map[o];
      out[1] = map[o + 1];
      out[2] = map[o + 2];
      out[3] = map[o + 3];
    } else {
      const color = this.vertexColor!;
      const n = this.vertexColorSize;
      const d = this.vertexColorDivisor;
      for (let k = 0; k < 3; k++) out[k] = (color[a * n + k] * wa + color[b * n + k] * wb + color[c * n + k] * wc) / d;
      out[3] = 1;
    }
  }

  /** Places the group's vertices on the image of view `v`: x and y in pixels, z growing away from the eye. */
  private project(v: number): void {
    const o = v * 9;
    const [cx, cy, cz] = this.centre;
    const scale = SIZE / (2 * this.radius);
    const p = this.positions;
    const { sx, sy, sz } = this;
    for (let l = 0; l < this.count; l++) {
      const x = p[l * 3] - cx;
      const y = p[l * 3 + 1] - cy;
      const z = p[l * 3 + 2] - cz;
      sx[l] = (x * VIEWS[o + 3] + y * VIEWS[o + 4] + z * VIEWS[o + 5]) * scale + SIZE / 2;
      sy[l] = (x * VIEWS[o + 6] + y * VIEWS[o + 7] + z * VIEWS[o + 8]) * scale + SIZE / 2;
      sz[l] = x * VIEWS[o] + y * VIEWS[o + 1] + z * VIEWS[o + 2];
    }
  }

  /** Draws a triangle list (both faces, nearest wins) and writes the colour of every pixel to `out`. */
  private draw(indices: Uint32Array, out: Float32Array): void {
    const { sx, sy, sz, triangle, weights, depth, input, sample } = this;
    const cut = this.alphaTest;
    triangle.fill(-1);
    depth.fill(Infinity);

    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t];
      const b = indices[t + 1];
      const c = indices[t + 2];
      const x0 = sx[a];
      const y0 = sy[a];
      const x1 = sx[b];
      const y1 = sy[b];
      const x2 = sx[c];
      const y2 = sy[c];
      const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
      if (area === 0) continue;
      const inverse = 1 / area;
      const left = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
      const right = Math.min(SIZE - 1, Math.floor(Math.max(x0, x1, x2)));
      const top = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
      const bottom = Math.min(SIZE - 1, Math.floor(Math.max(y0, y1, y2)));
      for (let y = top; y <= bottom; y++) {
        const py = y + 0.5;
        for (let x = left; x <= right; x++) {
          const px = x + 0.5;
          const wa = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inverse;
          const wb = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inverse;
          const wc = 1 - wa - wb;
          if (wa < 0 || wb < 0 || wc < 0) continue;
          const z = sz[a] * wa + sz[b] * wb + sz[c] * wc;
          const p = y * SIZE + x;
          if (z >= depth[p]) continue;
          if (cut > 0) {
            this.colorAt(input[a], input[b], input[c], wa, wb);
            if (sample[3] < cut) continue;
          }
          depth[p] = z;
          triangle[p] = t;
          weights[p * 2] = wa;
          weights[p * 2 + 1] = wb;
        }
      }
    }

    for (let p = 0; p < PIXELS; p++) {
      const t = triangle[p];
      if (t < 0) {
        out[p * 3] = -1;
        continue;
      }
      this.colorAt(input[indices[t]], input[indices[t + 1]], input[indices[t + 2]], weights[p * 2], weights[p * 2 + 1]);
      out[p * 3] = sample[0];
      out[p * 3 + 1] = sample[1];
      out[p * 3 + 2] = sample[2];
    }
  }

  /**
   * Counts the pixels that differ between two images of a view and are not explained by a shift of
   * up to `reach` pixels: what a pixel showed is still to be seen that near, and what it shows now
   * was.
   */
  private compare(a: Float32Array, b: Float32Array, reach: number): number {
    let changed = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const p = (y * SIZE + x) * 3;
        if (alike(a, p, b, p)) continue;
        let stillThere = false;
        let wasThere = false;
        const top = Math.max(0, y - reach);
        const bottom = Math.min(SIZE - 1, y + reach);
        const left = Math.max(0, x - reach);
        const right = Math.min(SIZE - 1, x + reach);
        for (let j = top; j <= bottom && !(stillThere && wasThere); j++) {
          for (let i = left; i <= right; i++) {
            const q = (j * SIZE + i) * 3;
            if (!stillThere && alike(a, p, b, q)) stillThere = true;
            if (!wasThere && alike(a, q, b, p)) wasThere = true;
          }
        }
        if (!(stillThere && wasThere)) changed++;
      }
    }
    return changed;
  }
}

/** Whether pixel `p` of image `a` and pixel `q` of image `b` show the same: nothing both, or colours close together. */
function alike(a: Float32Array, p: number, b: Float32Array, q: number): boolean {
  const empty = a[p] < 0;
  if (empty !== b[q] < 0) return false;
  if (empty) return true;
  return Math.abs(a[p] - b[q]) <= TOLERANCE && Math.abs(a[p + 1] - b[q + 1]) <= TOLERANCE && Math.abs(a[p + 2] - b[q + 2]) <= TOLERANCE;
}
