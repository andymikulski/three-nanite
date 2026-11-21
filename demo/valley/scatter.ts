import { Euler, Matrix4, Quaternion, Vector3 } from "three";
import { fbm, ground, heightAt, seeded, SIZE, slopeAt, TRAIL, WATER } from "./terrain.js";
import type { Ground } from "./terrain.js";

/** Where everything stands: matrices by model name. */
export type Placements = Map<string, Matrix4[]>;

const range = (prefix: string, n: number) => Array.from({ length: n }, (_, i) => `${prefix}_${i + 1}`);

const TREES = {
  common: range("CommonTree", 5),
  birch: range("Birch", 5),
  pine: range("Pine", 5),
  giant: range("GiantPine", 5),
  tall: range("TallThick", 5),
  twisted: range("TwistedTree", 5),
  dead: range("DeadTree", 5),
  cherry: range("CherryBlossom", 5),
};
const BUSHES = ["Bush_Common", "Bush_Common_Flowers", "Bush_Large", "Bush_Large_Flowers", "Bush_Long_1", "Bush_Long_2"];
const PLANTS = ["Plant_1", "Plant_2", "Plant_3", "Plant_4", "Plant_5", "Plant_6", "Plant_7", "Plant_1_Big", "Plant_2_Big", "Plant_7_Big", "Clover_1", "Clover_2"];
const FERNS = ["Fern_1", "Fern_2"];
const FLOWERS = ["Flower_1_Group", "Flower_1_Single", "Flower_2_Group", "Flower_2_Single", "Flower_3_Group", "Flower_3_Single", "Flower_4_Group", "Flower_4_Single", "Flower_6", "Flower_6_2", "Flower_7_Group", "Flower_7_Single"];
const MUSHROOMS = ["Mushroom_Common", "Mushroom_Laetiporus", "Mushroom_Oyster", "Mushroom_RedCap"];
const ROCKS = ["Rock_Big_1", "Rock_Big_2", "Rock_Medium_1", "Rock_Medium_2", "Rock_Medium_3", "Rock_Medium_4"];
const PATH_STONES = [...range("RockPath_Round_Small", 3), ...range("RockPath_Square_Small", 3), "RockPath_Round_Thin", "RockPath_Square_Thin", "RockPath_Round_Wide", "RockPath_Square_Wide"];
const PEBBLES = [...range("Pebble_Round", 5), ...range("Pebble_Square", 6)];

interface Spot {
  x: number;
  z: number;
  g: Ground;
  slope: number;
  /** A number in [0, 1) of its own, for choices. */
  r: number;
}

/** Picks from a list by a [0, 1) number. */
const pick = (list: string[], r: number) => list[Math.min(list.length - 1, Math.floor(r * list.length))];

/** A layer of things on a jittered grid: `choose` says what stands at a spot, if anything. */
interface Layer {
  cell: number;
  choose(s: Spot): { model: string; scale: number; sink?: number; tilt?: number } | null;
}

const LAYERS: Layer[] = [
  // trees
  {
    cell: 9,
    choose(s) {
      const { g, slope, r } = s;
      if (g.river < 16 || g.trail < 5 || slope > 0.32 || g.height > 175) return null;
      const scale = 0.8 + 0.5 * ((r * 7.31) % 1);
      // groves of cherry blossom on gentle hills
      if (fbm(s.x / 140 + 50, s.z / 140, 2) > 0.32 && slope < 0.12 && g.highland < 0.3) return r < 0.5 ? { model: pick(TREES.cherry, (r * 13.7) % 1), scale } : null;
      if (g.highland > 0.5) return r < 0.55 * g.highland ? { model: pick(r < 0.25 ? TREES.giant : r < 0.42 ? TREES.tall : TREES.pine, (r * 11.3) % 1), scale } : null;
      if (g.moisture < 0.32) return r < 0.06 ? { model: pick(TREES.dead, (r * 17.1) % 1), scale } : null;
      if (r > g.forest * 0.85 + 0.02) return null;
      const near = g.river < 70 && r < 0.35;
      const kind = near ? TREES.birch : g.highland > 0.2 ? (r < 0.4 ? TREES.pine : TREES.common) : r < 0.12 ? TREES.twisted : r < 0.25 ? TREES.birch : TREES.common;
      return { model: pick(kind, (r * 19.9) % 1), scale };
    },
  },
  // bushes, ferns, plants, mushrooms: the undergrowth
  {
    cell: 5,
    choose(s) {
      const { g, slope, r } = s;
      if (g.river < 9 || g.trail < 2.5 || slope > 0.4 || g.height > 160) return null;
      const scale = 0.75 + 0.5 * ((r * 5.17) % 1);
      const woods = g.forest * (1 - g.highland);
      if (r < woods * 0.3) return { model: pick(r < woods * 0.12 ? FERNS : PLANTS, (r * 23.3) % 1), scale };
      if (r < woods * 0.3 + 0.05 * woods) return { model: pick(MUSHROOMS, (r * 29.1) % 1), scale };
      if (r > 0.92) return { model: pick(BUSHES, (r * 31.7) % 1), scale };
      return null;
    },
  },
  // grass
  {
    cell: 2.6,
    choose(s) {
      const { g, slope, r } = s;
      if (g.river < 7 || g.trail < 1.6 || slope > 0.42 || g.height > 150) return null;
      const meadow = (1 - g.forest * 0.7) * (1 - g.highland * 0.6);
      if (r > meadow * 0.9) return null;
      const scale = 0.8 + 0.6 * ((r * 3.7) % 1);
      const kinds = fbm(s.x / 70, s.z / 70 + 40, 2);
      let model: string;
      if (g.moisture < 0.35) model = (r * 9) % 1 < 0.5 ? "Grass_Wheat" : (r * 7) % 1 < 0.5 ? "Grass_Wispy_Tall" : "Grass_Wispy_Short";
      else if (kinds > 0.15) model = (r * 9) % 1 < 0.5 ? "Grass_Wide_Tall" : "Grass_Wide_Short";
      else model = (r * 9) % 1 < 0.6 ? "Grass_Common_Short" : "Grass_Common_Tall";
      return { model, scale };
    },
  },
  // flowers, in patches
  {
    cell: 3.5,
    choose(s) {
      const { g, slope, r } = s;
      if (g.river < 8 || g.trail < 1.6 || slope > 0.3 || g.highland > 0.6 || g.moisture < 0.35) return null;
      const patch = fbm(s.x / 45 - 20, s.z / 45 + 9, 2);
      if (patch < 0.12 || r > 0.7) return null;
      // one kind of flower a patch
      const kind = Math.floor((fbm(s.x / 120 + 70, s.z / 120, 1) + 1) * 6) % 6;
      const pair = FLOWERS.slice(kind * 2, kind * 2 + 2);
      return { model: pick(pair, r / 0.7), scale: 0.8 + 0.4 * ((r * 3.3) % 1) };
    },
  },
  // rocks, more on slopes and up high
  {
    cell: 16,
    choose(s) {
      const { g, slope, r } = s;
      if (g.trail < 4 || g.height > 200) return null;
      const want = 0.04 + slope * 0.9 + g.highland * 0.15 + (g.river < 30 ? 0.08 : 0);
      if (r > want) return null;
      return { model: pick(ROCKS, (r * 37.3) % 1), scale: 0.6 + 1.2 * ((r * 7.7) % 1) ** 2, sink: 0.3, tilt: 0.25 };
    },
  },
];

const _m = new Matrix4();
const _q = new Quaternion();
const _e = new Euler();
const _p = new Vector3();
const _s = new Vector3();

function place(out: Placements, model: string, x: number, y: number, z: number, yaw: number, scale: number, tiltX = 0, tiltZ = 0): void {
  _q.setFromEuler(_e.set(tiltX, yaw, tiltZ, "YXZ"));
  _m.compose(_p.set(x, y, z), _q, _s.set(scale, scale, scale));
  let list = out.get(model);
  if (!list) out.set(model, (list = []));
  list.push(_m.clone());
}

/**
 * Everything that stands in the valley. `density` scales the number of everything (1 is the
 * designed amount). Deterministic for a given density.
 */
export function scatter(density: number): Placements {
  const out: Placements = new Map();
  const rand = seeded(42);
  const half = SIZE / 2 - 8;
  for (const layer of LAYERS) {
    const cell = layer.cell / Math.sqrt(density);
    for (let z = -half; z < half; z += cell) {
      for (let x = -half; x < half; x += cell) {
        const px = x + rand() * cell;
        const pz = z + rand() * cell;
        const r = rand();
        const g = ground(px, pz);
        if (g.height < WATER + 0.3) continue;
        const spot: Spot = { x: px, z: pz, g, slope: slopeAt(px, pz), r };
        const what = layer.choose(spot);
        if (!what) continue;
        const tilt = what.tilt ?? 0;
        place(out, what.model, px, heightAt(px, pz) - (what.sink ?? 0.05) * what.scale, pz, rand() * Math.PI * 2, what.scale, (rand() - 0.5) * tilt, (rand() - 0.5) * tilt);
      }
    }
  }
  // stepping stones and pebbles along the trail
  for (let i = 0; i + 1 < TRAIL.length; i++) {
    const [ax, az] = TRAIL[i];
    const [bx, bz] = TRAIL[i + 1];
    const length = Math.hypot(bx - ax, bz - az);
    const steps = Math.floor((length / 1.6) * Math.min(1, density));
    for (let k = 0; k < steps; k++) {
      const t = (k + rand() * 0.5) / steps;
      const side = (rand() - 0.5) * 2.2;
      const x = ax + (bx - ax) * t + (-(bz - az) / length) * side;
      const z = az + (bz - az) * t + ((bx - ax) / length) * side;
      if (ground(x, z).height < WATER + 0.2) continue;
      const stone = rand() < 0.65;
      place(out, stone ? pick(PATH_STONES, rand()) : pick(PEBBLES, rand()), x, heightAt(x, z) - 0.04, z, rand() * Math.PI * 2, stone ? 0.9 + 0.3 * rand() : 1 + rand());
    }
  }
  return out;
}
