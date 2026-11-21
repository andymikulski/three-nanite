import { BufferAttribute, BufferGeometry, Color, DoubleSide, MeshStandardMaterial, SRGBColorSpace, TextureLoader } from "three";
import type { Texture } from "three";
import { prepareCutoutTexture } from "../../src/index.js";

/**
 * The Stylized Nature MegaKit (Quaternius, CC0) as glTF, served under /nature/ (see vite.config.ts).
 * Every file is one node without a transform, one buffer, and a primitive per material, so a few
 * lines read it; GLTFLoader would decode every shared texture once per file that names it.
 */
export const KIT_URL = "/nature/";

/** One material of one model: the triangles of that model drawn with it. */
export interface KitPart {
  model: string;
  material: string;
  geometry: BufferGeometry;
}

export interface KitModel {
  name: string;
  parts: KitPart[];
  /** Height of the model above its origin, and its largest horizontal extent from it. */
  height: number;
  radius: number;
}

/** What a material needs, from the glTF: its base colour texture and how it is cut out. */
interface MaterialSpec {
  texture: string;
  /** The alpha below which it is cut out, 0 for an opaque material. */
  cutoff: number;
}

/**
 * The kit's grass shader colours a near-white texture with a gradient from the root up, and its
 * glTF export has one "Grass" material for four Unity ones. Each gets its own material here, and
 * the gradient goes into a vertex colour (works on both renderers, no shader patch). Colours are
 * the Unity materials' `_Color_Bottom` / `_Color_Top`.
 */
const GRASS: Record<string, { bottom: [number, number, number]; top: [number, number, number] }> = {
  Grass: { bottom: [0.366, 0.584, 0.047], top: [0.456, 0.922, 0] },
  Grass_Wheat: { bottom: [0.208, 0.094, 0], top: [0.451, 0.192, 0] },
  Grass_Wispy: { bottom: [0.239, 0.188, 0], top: [0.275, 0.18, 0] },
  Grass_Wide: { bottom: [0.129, 0.224, 0], top: [0.294, 0.318, 0] },
};

function grassMaterialOf(model: string): string {
  if (model.startsWith("Grass_Wheat")) return "Grass_Wheat";
  if (model.startsWith("Grass_Wispy")) return "Grass_Wispy";
  if (model.startsWith("Grass_Wide")) return "Grass_Wide";
  return "Grass";
}

const specs = new Map<string, MaterialSpec>();

interface Gltf {
  buffers: { uri: string }[];
  bufferViews: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
  accessors: { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }[];
  meshes: { primitives: { attributes: Record<string, number>; indices: number; material: number }[] }[];
  materials: { name: string; alphaMode?: string; alphaCutoff?: number; pbrMetallicRoughness?: { baseColorTexture?: { index: number } } }[];
  textures: { source: number }[];
  images: { uri: string }[];
}

const SIZES: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function read(gltf: Gltf, bin: ArrayBuffer, index: number): Float32Array | Uint16Array | Uint32Array {
  const a = gltf.accessors[index];
  const view = gltf.bufferViews[a.bufferView];
  const size = SIZES[a.type];
  if (view.byteStride && view.byteStride !== size * (a.componentType === 5126 || a.componentType === 5125 ? 4 : 2))
    throw new Error("interleaved glTF buffers are not read here");
  const offset = (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
  const n = a.count * size;
  // (copies: aligned, and the builder may be handed them in a worker)
  switch (a.componentType) {
    case 5126:
      return new Float32Array(bin.slice(offset, offset + n * 4));
    case 5123:
      return new Uint16Array(bin.slice(offset, offset + n * 2));
    case 5125:
      return new Uint32Array(bin.slice(offset, offset + n * 4));
    default:
      throw new Error(`glTF component type ${a.componentType} is not read here`);
  }
}

export async function loadModel(name: string): Promise<KitModel> {
  const gltf = (await (await fetch(KIT_URL + name + ".gltf")).json()) as Gltf;
  const bin = await (await fetch(KIT_URL + gltf.buffers[0].uri)).arrayBuffer();
  const parts: KitPart[] = [];
  let height = 0;
  let radius = 0;
  for (const mesh of gltf.meshes) {
    for (const p of mesh.primitives) {
      const source = gltf.materials[p.material];
      let material = source.name;
      const geometry = new BufferGeometry();
      const position = read(gltf, bin, p.attributes.POSITION) as Float32Array;
      geometry.setAttribute("position", new BufferAttribute(position, 3));
      geometry.setAttribute("normal", new BufferAttribute(read(gltf, bin, p.attributes.NORMAL), 3));
      geometry.setAttribute("uv", new BufferAttribute(read(gltf, bin, p.attributes.TEXCOORD_0), 2));
      geometry.setIndex(new BufferAttribute(read(gltf, bin, p.indices), 1));
      let top = 0;
      for (let i = 0; i < position.length; i += 3) {
        top = Math.max(top, position[i + 1]);
        radius = Math.max(radius, Math.hypot(position[i], position[i + 2]));
      }
      height = Math.max(height, top);
      if (material === "Grass") {
        material = grassMaterialOf(name);
        const { bottom, top: up } = GRASS[material];
        const lo = new Color().setRGB(...bottom, SRGBColorSpace);
        const hi = new Color().setRGB(...up, SRGBColorSpace);
        const colors = new Float32Array((position.length / 3) * 3);
        const c = new Color();
        for (let v = 0; v < position.length / 3; v++) {
          c.lerpColors(lo, hi, top > 0 ? Math.min(1, Math.max(0, position[v * 3 + 1] / top)) : 1);
          colors.set([c.r, c.g, c.b], v * 3);
        }
        geometry.setAttribute("color", new BufferAttribute(colors, 3));
      }
      if (!specs.has(material)) {
        const t = source.pbrMetallicRoughness?.baseColorTexture;
        specs.set(material, {
          texture: t ? gltf.images[gltf.textures[t.index].source].uri : "",
          // (cherry blossom leaves are BLEND in the file but are cut-out cards like all the others)
          cutoff: source.alphaMode === "MASK" || source.alphaMode === "BLEND" ? (source.alphaCutoff ?? 0.5) : 0,
        });
      }
      geometry.computeBoundingSphere();
      parts.push({ model: name, material, geometry });
    }
  }
  return { name, parts, height, radius };
}

const textures = new Map<string, Promise<Texture>>();

function loadTexture(file: string): Promise<Texture> {
  let t = textures.get(file);
  if (!t) {
    t = new TextureLoader().loadAsync(KIT_URL + file).then((texture) => {
      texture.flipY = false; // (glTF)
      texture.colorSpace = SRGBColorSpace;
      texture.anisotropy = 8;
      return texture;
    });
    textures.set(file, t);
  }
  return t;
}

/** The material of each name the loaded models use, one object per name. */
export async function createMaterials(): Promise<Map<string, MeshStandardMaterial>> {
  const out = new Map<string, MeshStandardMaterial>();
  await Promise.all(
    [...specs].map(async ([name, spec]) => {
      let map = spec.texture ? await loadTexture(spec.texture) : null;
      // (cut-outs: colour in the transparent texels, and mipmaps that keep the leaves' coverage)
      if (map && spec.cutoff > 0) map = prepareCutoutTexture(map, spec.cutoff, 1024);
      const material = new MeshStandardMaterial({
        name,
        map,
        roughness: 0.95,
        metalness: 0,
        side: DoubleSide,
        vertexColors: name in GRASS,
      });
      if (spec.cutoff > 0) {
        // (a hard cut-out, as the kit's own shader does: alpha to coverage let the sky through every
        // card's soft edge and washed canopies out)
        material.alphaTest = spec.cutoff;
      }
      out.set(name, material);
    }),
  );
  return out;
}

/** Every model of the kit the valley uses (the loose petals are left out). */
export const MODELS = (
  "Birch_1 Birch_2 Birch_3 Birch_4 Birch_5 Bush_Common Bush_Common_Flowers Bush_Large Bush_Large_Flowers Bush_Long_1 Bush_Long_2 " +
  "CherryBlossom_1 CherryBlossom_2 CherryBlossom_3 CherryBlossom_4 CherryBlossom_5 Clover_1 Clover_2 CommonTree_1 CommonTree_2 " +
  "CommonTree_3 CommonTree_4 CommonTree_5 DeadTree_1 DeadTree_2 DeadTree_3 DeadTree_4 DeadTree_5 Fern_1 Fern_2 Flower_1_Group " +
  "Flower_1_Single Flower_2_Group Flower_2_Single Flower_3_Group Flower_3_Single Flower_4_Group Flower_4_Single Flower_6 Flower_6_2 " +
  "Flower_7_Group Flower_7_Single GiantPine_1 GiantPine_2 GiantPine_3 GiantPine_4 GiantPine_5 Grass_Common_Short Grass_Common_Tall " +
  "Grass_Wheat Grass_Wide_Short Grass_Wide_Tall Grass_Wispy_Short Grass_Wispy_Tall Mushroom_Common Mushroom_Laetiporus Mushroom_Oyster " +
  "Mushroom_RedCap Pebble_Round_1 Pebble_Round_2 Pebble_Round_3 Pebble_Round_4 Pebble_Round_5 Pebble_Square_1 Pebble_Square_2 " +
  "Pebble_Square_3 Pebble_Square_4 Pebble_Square_5 Pebble_Square_6 Pine_1 Pine_2 Pine_3 Pine_4 Pine_5 Plant_1 Plant_1_Big Plant_2 " +
  "Plant_2_Big Plant_3 Plant_4 Plant_5 Plant_6 Plant_7 Plant_7_Big Rock_Big_1 Rock_Big_2 Rock_Medium_1 Rock_Medium_2 Rock_Medium_3 " +
  "Rock_Medium_4 RockPath_Round_Small_1 RockPath_Round_Small_2 RockPath_Round_Small_3 RockPath_Round_Thin RockPath_Round_Wide " +
  "RockPath_Square_Small_1 RockPath_Square_Small_2 RockPath_Square_Small_3 RockPath_Square_Thin RockPath_Square_Wide TallThick_1 " +
  "TallThick_2 TallThick_3 TallThick_4 TallThick_5 TwistedTree_1 TwistedTree_2 TwistedTree_3 TwistedTree_4 TwistedTree_5"
).split(" ");
