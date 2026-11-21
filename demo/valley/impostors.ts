import { Group, Mesh } from "three";
import type { Material, Matrix4, WebGLRenderer } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { KitModel } from "./kit.js";

/**
 * The comparison with octahedral impostors (`~/dev/three-impostor`): every placement drawn as an
 * impostor, a card of two triangles lit live by the scene, baked from the same models and
 * materials as the hierarchies. On either renderer: the library's main entry for WebGLRenderer, its
 * `/webgpu` entry (TSL) for WebGPURenderer, loaded when first asked for.
 *
 * A set of the library holds up to 64 models, all of whose views share texture arrays as large as
 * its largest tile, and at most 2,048 layers. So the models are sorted by size and baked in sets of
 * similar ones: trees at 128 px a view, bushes and rocks at 64, grass, flowers and pebbles at 32.
 */
const VIEWS = { azimuthFrames: 12, elevationFrames: 3, elevationRange: [-10, 70] as [number, number] };
/** Models in a set, at most: a set's texture arrays have a layer per model and view. */
const SET_SIZE = 48;

function tileFor(model: KitModel): number {
  const size = Math.max(model.height, model.radius * 2);
  return size > 5 ? 128 : size > 1.6 ? 64 : 32;
}

/** Either entry's bakeImpostorSet, given the renderer it is for. */
type BakeSet = (renderer: never, ...rest: Parameters<typeof import("@octa-lab/three-impostor").bakeImpostorSet> extends [unknown, ...infer R] ? R : never) => Promise<import("@octa-lab/three-impostor").ImpostorAtlas>;

export interface ImpostorField {
  group: Group;
  /** Atlas memory, in megabytes, and how long the bake took. */
  memoryMB: number;
  bakeMs: number;
  dispose(): void;
}

/**
 * Bakes the models that are placed and draws every placement as an impostor. Explicit and slow
 * (it renders each model from every view): call it when the comparison is first asked for.
 */
export async function buildImpostors(
  renderer: WebGLRenderer | WebGPURenderer,
  models: KitModel[],
  materials: Map<string, Material>,
  placements: Map<string, Matrix4[]>,
  onProgress: (text: string) => Promise<unknown>,
): Promise<ImpostorField> {
  const t0 = performance.now();
  // an atlas lives on the device of the renderer that baked it: bake with the one that draws
  const gpu = (renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer === true;
  const bakeSet: BakeSet = gpu ? (await import("@octa-lab/three-impostor/webgpu")).bakeImpostorSet : (await import("@octa-lab/three-impostor")).bakeImpostorSet;
  // (WebGPU devices are made with the default limit unless asked for more: 256 layers, 7 models)
  const layers = gpu ? ((renderer as { backend: { device?: { limits: { maxTextureArrayLayers: number } } } }).backend.device?.limits.maxTextureArrayLayers ?? 256) : 2048;
  const setSize = Math.max(1, Math.min(SET_SIZE, Math.floor(layers / (VIEWS.azimuthFrames * VIEWS.elevationFrames))));
  const placed = models.filter((m) => placements.has(m.name)).sort((a, b) => tileFor(b) - tileFor(a) || (a.name < b.name ? -1 : 1));
  // sets of models with the same tile
  const sets: KitModel[][] = [];
  for (const m of placed) {
    const last = sets[sets.length - 1];
    if (last && last.length < setSize && tileFor(last[0]) === tileFor(m)) last.push(m);
    else sets.push([m]);
  }

  const group = new Group();
  const atlases: { dispose(): void }[] = [];
  let memoryMB = 0;
  let done = 0;
  for (const set of sets) {
    const sources = set.map((model) => {
      // (the bake leaves the root's transform out and takes its children's: the parts sit at the origin)
      const root = new Group();
      for (const part of model.parts) root.add(new Mesh(part.geometry, materials.get(part.material)));
      return root;
    });
    const atlas = await bakeSet(renderer as never, sources, {
      ...VIEWS,
      tileSize: tileFor(set[0]),
      supersample: 2,
      batchSize: 8,
      // (the library's default budget is 256 MB a set; 48 trees at 128 px take some 600)
      maxMemoryMB: 1024,
      onProgress: ({ fraction }: { fraction: number }) => {
        void onProgress(`baking impostors: ${Math.round(((done + fraction * set.length) / placed.length) * 100)}% (${placed.length} models)`);
      },
    });
    done += set.length;
    atlases.push(atlas);
    memoryMB += atlas.info.estimatedMemoryBytes / 2 ** 20;

    const count = set.reduce((n, m) => n + placements.get(m.name)!.length, 0);
    const copies = atlas.createInstances(count);
    let i = 0;
    set.forEach((model, asset) => {
      for (const matrix of placements.get(model.name)!) {
        copies.setMatrixAt(i, matrix);
        if (set.length > 1) copies.setAssetAt(i, asset);
        i++;
      }
    });
    copies.castShadow = tileFor(set[0]) >= 64;
    copies.receiveShadow = true;
    group.add(copies);
  }
  return {
    group,
    memoryMB,
    bakeMs: performance.now() - t0,
    dispose() {
      for (const a of atlases) a.dispose();
    },
  };
}
