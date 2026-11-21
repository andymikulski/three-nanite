import { BufferAttribute, BufferGeometry, Color, SRGBColorSpace } from "three";

/** Side of the square of land, in metres, and the number of cells along a side. */
export const SIZE = 2048;
export const CELLS = 512;
export const STEP = SIZE / CELLS;
/** Height of the river's water. */
export const WATER = 0;

// ---------------------------------------------------------------------------
// noise (2D gradient noise, seeded) and a seeded generator
// ---------------------------------------------------------------------------

export function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const perm = new Uint8Array(512);
{
  const r = seeded(1234);
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
}

function grad(h: number, x: number, y: number): number {
  switch (h & 7) {
    case 0: return x + y;
    case 1: return x - y;
    case 2: return -x + y;
    case 3: return -x - y;
    case 4: return x;
    case 5: return -x;
    case 6: return y;
    default: return -y;
  }
}
const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);

/** Gradient noise, about -1 to 1. */
export function noise(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const X = xi & 255;
  const Y = yi & 255;
  const u = fade(xf);
  const v = fade(yf);
  const a = perm[perm[X] + Y];
  const b = perm[perm[X + 1] + Y];
  const c = perm[perm[X] + Y + 1];
  const d = perm[perm[X + 1] + Y + 1];
  const x1 = grad(a, xf, yf) + u * (grad(b, xf - 1, yf) - grad(a, xf, yf));
  const x2 = grad(c, xf, yf - 1) + u * (grad(d, xf - 1, yf - 1) - grad(c, xf, yf - 1));
  return (x1 + v * (x2 - x1)) * 0.7;
}

export function fbm(x: number, y: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  let f = 1;
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise(x * f + o * 17.3, y * f - o * 9.1);
    f *= 2;
    amp *= 0.5;
  }
  return sum;
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------------------
// the shape of the land
// ---------------------------------------------------------------------------

/** The river: its x for a given z, winding down the valley. */
export function riverX(z: number): number {
  return 90 * Math.sin(z / 260) + 40 * Math.sin(z / 97 + 1.3) - 120;
}

/** The trail: points from the overlook in the south-east down to the river and up the far side. */
export const TRAIL: [number, number][] = [
  [420, 520], [360, 430], [300, 360], [215, 300], [150, 210], [60, 160], [-30, 120], [-95, 80],
  [-150, 40], [-210, -20], [-250, -110], [-280, -210], [-330, -300], [-390, -360],
];

function distanceToTrail(x: number, z: number): number {
  let best = Infinity;
  for (let i = 0; i + 1 < TRAIL.length; i++) {
    const [ax, az] = TRAIL[i];
    const [bx, bz] = TRAIL[i + 1];
    const dx = bx - ax;
    const dz = bz - az;
    const t = Math.min(1, Math.max(0, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
    best = Math.min(best, Math.hypot(x - ax - t * dx, z - az - t * dz));
  }
  return best;
}

export interface Ground {
  height: number;
  /** Distances to the river's centre line and to the trail. */
  river: number;
  trail: number;
  /** 0 to 1: woods, wetness, how high and wild. */
  forest: number;
  moisture: number;
  highland: number;
}

/** The land as designed (the drawn terrain samples it at its vertices: use {@link heightAt} to stand on it). */
export function ground(x: number, z: number): Ground {
  const r = Math.hypot(x, z) / (SIZE / 2);
  const mountains = smooth(0.45, 0.98, r) * (150 + 110 * fbm(x / 300, z / 300, 5));
  const hills = 26 * fbm(x / 420 + 3, z / 420 - 7, 4) + 6 * fbm(x / 90, z / 90, 3);
  const river = Math.abs(x - riverX(z));
  const bank = smooth(14, 60, river);
  const trail = distanceToTrail(x, z);
  let height = 8 + (hills + mountains) * (0.25 + 0.75 * bank);
  // the river's bed, below the water, with banks rising from it
  const bed = 1 - smooth(5, 24, river);
  height += (-1.8 - height) * bed;
  // the trail is levelled a little across its width
  height -= (1 - smooth(1, 4, trail)) * 0.25;
  const forest = smooth(-0.05, 0.25, fbm(x / 230 - 11, z / 230 + 5, 4));
  const moisture = smooth(-0.3, 0.3, fbm(x / 500 + 31, z / 500 - 2, 3)) * 0.6 + 0.4 * (1 - smooth(20, 140, river));
  const highland = smooth(40, 110, height);
  return { height, river, trail, forest, moisture, highland };
}

// ---------------------------------------------------------------------------
// the drawn terrain
// ---------------------------------------------------------------------------

const heights = new Float32Array((CELLS + 1) * (CELLS + 1));

const C = (hex: number) => new Color().setHex(hex, SRGBColorSpace);
const GRASS_LUSH = C(0x6fae2d);
const GRASS_DEEP = C(0x4e8a24);
const GRASS_DRY = C(0xb3ad4a);
const DIRT = C(0x9c8550);
const MUD = C(0x6a5a3a);
const ROCK = C(0x7d7a70);
const SNOW = C(0xf2f4f7);

/** The terrain as an indexed grid with normals and vertex colours, centred on the origin. */
export function createTerrain(): BufferGeometry {
  const n = CELLS + 1;
  const position = new Float32Array(n * n * 3);
  const info: Ground[] = new Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = -SIZE / 2 + i * STEP;
      const z = -SIZE / 2 + j * STEP;
      const g = ground(x, z);
      const k = j * n + i;
      info[k] = g;
      heights[k] = g.height;
      position.set([x, g.height, z], k * 3);
    }
  }
  const index = new Uint32Array(CELLS * CELLS * 6);
  let w = 0;
  for (let j = 0; j < CELLS; j++) {
    for (let i = 0; i < CELLS; i++) {
      const a = j * n + i;
      const b = a + 1;
      const c = a + n;
      const d = c + 1;
      index.set([a, c, b, b, c, d], w);
      w += 6;
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(position, 3));
  geometry.setIndex(new BufferAttribute(index, 1));
  geometry.computeVertexNormals();

  const normal = geometry.getAttribute("normal");
  const color = new Float32Array(n * n * 3);
  const c = new Color();
  for (let k = 0; k < n * n; k++) {
    const g = info[k];
    const x = position[k * 3];
    const z = position[k * 3 + 2];
    const slope = 1 - normal.getY(k);
    c.copy(GRASS_LUSH).lerp(GRASS_DEEP, smooth(-0.2, 0.4, fbm(x / 60, z / 60, 3)) * 0.7 + g.forest * 0.3);
    c.lerp(GRASS_DRY, (1 - g.moisture) * 0.6);
    c.lerp(DIRT, 1 - smooth(1.2, 3.2, g.trail));
    c.lerp(MUD, 1 - smooth(8, 20, g.river));
    c.lerp(ROCK, Math.max(smooth(0.25, 0.45, slope), smooth(90, 150, g.height) * 0.8));
    c.lerp(SNOW, smooth(175, 215, g.height + 20 * fbm(x / 80, z / 80, 2)) * (1 - smooth(0.5, 0.7, slope)));
    color.set([c.r, c.g, c.b], k * 3);
  }
  geometry.setAttribute("color", new BufferAttribute(color, 3));
  geometry.computeBoundingSphere();
  return geometry;
}

/** Height of the drawn terrain (the grid's own triangles) at x, z. */
export function heightAt(x: number, z: number): number {
  const n = CELLS + 1;
  const fx = Math.min(CELLS - 1e-6, Math.max(0, (x + SIZE / 2) / STEP));
  const fz = Math.min(CELLS - 1e-6, Math.max(0, (z + SIZE / 2) / STEP));
  const i = Math.floor(fx);
  const j = Math.floor(fz);
  const u = fx - i;
  const v = fz - j;
  const a = heights[j * n + i];
  const b = heights[j * n + i + 1];
  const c = heights[(j + 1) * n + i];
  const d = heights[(j + 1) * n + i + 1];
  // (the triangles a c b and b c d)
  if (u + v <= 1) return a + (b - a) * u + (c - a) * v;
  return d + (c - d) * (1 - u) + (b - d) * (1 - v);
}

/** Steepness at x, z: 0 flat, 1 vertical (from the drawn terrain). */
export function slopeAt(x: number, z: number): number {
  const e = STEP;
  const dx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e);
  const dz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
  return 1 - 1 / Math.sqrt(1 + dx * dx + dz * dz);
}
