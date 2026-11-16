# three-nanite

Nanite-style cluster LOD for three.js: a mesh is split into small clusters ("meshlets"), the clusters are organised into a DAG of progressively simplified versions, and at render time each instance draws a crack-free mix of detail levels chosen by its projected error on screen.

- Works with **`WebGLRenderer` and `WebGPURenderer`** (WebGPU back-end and its WebGL 2 fallback), same code path for both.
- Works with **any standard material** (`MeshStandardMaterial`, node materials, `onBeforeCompile` hooks), including shadows, fog and vertex attributes such as `uv1` or `color`.
- **Instancing built in**: one `MeshletMesh` draws many copies, each with its own level of detail, in one multi-draw call on WebGL 2.
- **Bake offline, load at runtime**: the hierarchy is plain typed arrays with a zero-copy binary format. The runtime entry point has no WebAssembly and no dependency besides three.
- Vertex data is **shared by all detail levels** (simplification never creates vertices), so the cost over the original mesh is roughly a second index buffer.

<p align="center">
<img src="./screenshots/demo.png" alt="64 instances of a 393k triangle mesh drawn with 49k triangles">
</p>

64 copies of a 393,216 triangle mesh (25 million triangles at full detail) drawn with 49 thousand, at an error budget of one pixel. Below, the same scene coloured by detail level: parts of one object nearer the camera use finer clusters, with no seams between levels.

<p align="center">
<img src="./screenshots/demo-lod.png" alt="Clusters coloured by LOD level">
</p>

## Install

```sh
npm install three-nanite three
```

`three` is a peer dependency (r170 or newer; tested on 0.180 and 0.187). Node 20+ for the build tools.

## Quick start

### 1. Build the hierarchy (once, ideally offline)

```ts
import { buildMeshletsFromGeometry } from "three-nanite/build";
import { serializeMeshlets } from "three-nanite";

const data = await buildMeshletsFromGeometry(geometry); // indexed, welded BufferGeometry
await writeFile("rock.meshlets", new Uint8Array(serializeMeshlets(data)));
```

`three-nanite/build` runs in browsers, workers and Node (it uses [meshoptimizer](https://github.com/zeux/meshoptimizer)'s WebAssembly build). A 400k triangle mesh takes a bit over a second. The geometry should be indexed and welded; run `BufferGeometryUtils.mergeVertices` first if it is not.

### 2. Draw it

```ts
import { deserializeMeshlets, MeshletMesh } from "three-nanite";

const data = deserializeMeshlets(await (await fetch("rock.meshlets")).arrayBuffer());

const rocks = new MeshletMesh(data, new THREE.MeshStandardMaterial());
rocks.castShadow = rocks.receiveShadow = true;
rocks.count = 500;
for (let i = 0; i < 500; i++) rocks.setMatrixAt(i, matrixOf(i));
scene.add(rocks);
```

That is all: the level of detail is recomputed automatically for the camera being rendered. The same code runs under `WebGPURenderer`.

## API

### `three-nanite` (runtime)

| Export | Purpose |
| --- | --- |
| `MeshletMesh` | The object you add to the scene. |
| `deserializeMeshlets(buffer)` / `serializeMeshlets(data)` | `.meshlets` files. Deserializing creates views on the buffer, without copying. |
| `createLodGeometry(data, maxError)` | A plain, static `BufferGeometry` for one fixed detail level. |
| `createLodChain(data, { maxLevels })` / `getLodErrors(data)` | A conventional discrete LOD chain out of the same hierarchy. |
| `geometryToMeshletInput(geometry)` | `BufferGeometry` to builder input. |
| `ClusterSelector`, `lodScale`, `countTriangles` | The LOD selection itself, free of three.js objects, for custom renderers. |
| `validateMeshletData(data)` | Consistency check; returns a list of problems. |

#### `MeshletMesh`

```ts
new MeshletMesh(data, material?, { maxInstances?, initialDrawCapacity?, errorThreshold? })
```

It behaves like an `InstancedMesh`:

| Member | |
| --- | --- |
| `count` | Number of instances. Starts at 1 with an identity matrix. |
| `setMatrixAt(i, matrix)` / `getMatrixAt(i, target)` | Instance transform, relative to the `MeshletMesh`. |
| `setColorAt(i, color)` / `getColorAt(i, target)` | Instance tint. |
| `errorThreshold` | Allowed geometric error in **pixels** (default 1). The quality/performance dial. |
| `autoUpdate` | Default `true`: LOD is refreshed from the render hooks. Set `false` and call `update(camera, viewportHeight)` yourself for full control. |
| `lodCamera` | Pin LOD to one camera regardless of which camera renders (debugging, or keeping reflections in step with the main view). |
| `debugColors` | `"none"`, `"cluster"` or `"lod"`: tints each drawn cluster, on any material and renderer. |
| `stats` | `{ instances, clusters, triangles, slots, lastUpdateMs, lastSelected }`. |
| `material`, `castShadow`, `receiveShadow`, `renderOrder`, `layers` | Forwarded to the internal draw mesh. |
| `batched`, `drawCapacity`, `onRebuild` | The internal `THREE.BatchedMesh`; see "Draw capacity" below. |
| `dispose()` | Frees GPU buffers and textures (not the material). |

Raycasting works through `raycaster.intersectObject(mesh, true)`: hits report `object === mesh` and `instanceId`, and test the detail level currently drawn.

Things worth knowing:

- **LOD follows the viewer, in every pass.** Shadow passes reuse the cut chosen for the main camera, so a shadow always matches the mesh that casts it. Frustum culling, on the other hand, is done per cluster for each pass's own camera.
- **Camera rotation is free.** The cut depends only on the camera position, projection, viewport height and threshold. If none of those changed and no instance moved, `update` returns immediately. If only some instances moved, only those are re-evaluated.
- **Draw capacity.** Every drawn (instance, cluster) pair occupies a slot of the internal `BatchedMesh`. When slots run out the internal mesh is rebuilt with more capacity and `batched` becomes a new object. The vertex and index buffers are shared, so nothing is re-uploaded, but re-registering the clusters takes a few milliseconds per thousand clusters (about 20 ms for the 6,400 cluster mesh above), a visible hitch if it happens mid-game. Pass `initialDrawCapacity` (watch `stats.clusters` for a realistic number) to avoid it, and use `onRebuild` if you customise `batched` directly.
- **Cost of an update.** Flying a camera through 1,024 instances of that mesh costs about 0.9 ms of LOD selection per frame on a desktop CPU, 0.3 ms for 64 instances.
- **Per-instance data** beyond matrix and color has no slot of its own. Custom shaders can read the per-draw id three exposes for batching; `InstancedBufferAttribute`s do not apply.

#### `serializeMeshlets` options

```ts
serializeMeshlets(data, { attributes?, compactIndices? })
```

| Option | Default | |
| --- | --- | --- |
| `attributes` | `true` | With `false` the file holds the hierarchy only (clusters, groups, indices), without vertex data. It can drive a `ClusterSelector` but not a `MeshletMesh`; use it when the vertices live in your own buffers. |
| `compactIndices` | `false` | Stores indices as 16 bit when the mesh has at most 65,536 vertices. `deserializeMeshlets` widens them to a `Uint32Array` again, so this only makes the file smaller. |

### `three-nanite/core`

The part of the library that imports neither three nor meshoptimizer: the data types, the file format, the LOD selection and validation. A runtime that draws by itself (your own renderer, or an engine other than three.js) should import this entry point rather than `three-nanite`.

| Export | Purpose |
| --- | --- |
| `MeshletData`, `MeshletInput`, `MeshletAttribute`, `TypedArray` | The data types (also exported by the other entry points). |
| `deserializeMeshlets(buffer)` / `serializeMeshlets(data, options?)` | `.meshlets` files, see above. `MESHLET_FILE_EXTENSION` is `".meshlets"`. |
| `ClusterSelector`, `lodScale`, `countTriangles`, `getLodErrors` | LOD selection and error thresholds. |
| `validateMeshletData(data)` | Consistency check; returns a list of problems. |

### `three-nanite/build`

```ts
buildMeshletsFromGeometry(geometry, options?): Promise<MeshletData>
buildMeshlets(input, options?): Promise<MeshletData>     // raw typed arrays, no three.js objects
```

The defaults follow meshoptimizer's `clodDefaultConfig(128)`. The options most likely to matter:

| Option | Default | |
| --- | --- | --- |
| `maxTriangles` | 128 | Cluster size. Smaller clusters give finer-grained LOD and more draws. |
| `attributeWeights` | `{ normal: 0.5 }` | How strongly each attribute resists simplification. Add `color` or `uv` if they carry important detail. |
| `protectAttributes` | `["uv", "uv1", "uv2", "uv3"]` | Attributes whose seams must be preserved. |
| `attributes` | all | Which attributes to bake (`buildMeshletsFromGeometry` only). |
| `lockedVertices` | none | One number per vertex; non-zero keeps that vertex, and every vertex at the same position, where it is at every level. For vertices shared with a neighbouring mesh that is simplified on its own, such as the borders of terrain tiles, so no crack opens between them. Locked vertices limit how coarse the mesh can become. |
| `fallbackSloppy` | `true` | Use sloppy simplification when regular simplification gets stuck. It does not preserve attribute seams; turn it off when they matter. |
| `onProgress` | | Called once per level. |

The builder is deterministic and does not modify or copy the vertex arrays: the result references the arrays you passed in.

#### Unindexed meshes: `weldVertices` and `unweldMeshlets`

```ts
weldVertices(input): { input: MeshletInput, source: Uint32Array }
unweldMeshlets(data, source, original): MeshletData
```

Simplification needs shared vertices, but "triangle soup" buffers (three vertices per triangle, no index) are common in engines and exporters. `weldVertices` merges vertices whose attributes are all bitwise identical and returns an indexed mesh, plus `source`: for each welded vertex, the first vertex of the original it was made from. Build on the welded mesh, then `unweldMeshlets` rewrites the hierarchy's indices through `source`, so the result indexes the soup's own vertices and attributes again. Nothing else is copied.

```ts
import { buildMeshlets, unweldMeshlets, weldVertices } from "three-nanite/build";

const { input, source } = weldVertices(soup);          // soup: MeshletInput without indices
const welded = await buildMeshlets(input);
const data = unweldMeshlets(welded, source, soup);     // same hierarchy, soup numbering
```

Because every attribute takes part in the comparison, vertices that differ in normal, uv or colour stay separate, and welding never changes what is drawn. Triangles that collapse while welding and vertices no triangle uses are dropped.

## Using it in a game

Three ways to adopt it, from least to most invasive. All of them start from the same baked file.

1. **Discrete LODs from the DAG.** `createLodChain(data, { maxLevels: 4 })` returns ordinary geometries that share one set of vertex arrays. Drop them into `THREE.LOD` or your existing `InstancedMesh` switching. No new runtime behaviour, and already better than hand-authored LODs for dense meshes.
2. **`MeshletMesh` per kind of object.** One `MeshletMesh` per model, one instance per placed copy, in place of an `InstancedMesh` plus an impostor. Near copies get full detail, far copies collapse to a few dozen triangles, and the transition has no pop to hide and no dither. It draws through `BatchedMesh`, so materials that patch shaders with `onBeforeCompile` must use three's batching chunks rather than `instanceMatrix`.
3. **Custom drawing.** `ClusterSelector.select` gives you the list of cluster ids for a view; `data.clusterIndexStart/Count` tell you which index ranges to draw. Import it from `three-nanite/core`, which has no dependency on three or meshoptimizer.

   This is the pattern for a game that already has its own merged, lightmapped vertex buffers: the vertex buffer stays exactly as it is, and only the index buffer is dynamic.

   - **At bake time**, per object: weld the object's vertices with `weldVertices` by *all* attributes that are drawn, including the lightmap uv, so two vertices merge only if they draw identically. Build, then map back with `unweldMeshlets` so the hierarchy indexes the object's own vertices.
   - Fragmented texture atlases are the hard case. With the default protected uv seams, simplification stops at roughly half the triangles on atlas-textured props. Build with `protectAttributes: []` and `attributeWeights: { normal: 0.5, uv: 1 }` instead: it goes much further, but a triangle that crosses a seam now counts as error, so the extra reduction only shows far away. Pick per asset.
   - Pass `lockedVertices` for vertices shared between objects that are built separately, and `fallbackSloppy: false` when seams matter.
   - Store hierarchy-only files: `serializeMeshlets(data, { attributes: false, compactIndices: true })`. The vertices are already in the game's buffers.
   - **At run time**, per object: call `ClusterSelector.select` with the camera expressed in the object's own frame, and write the selected clusters' index ranges, plus the object's first-vertex offset, into a dynamic index buffer. Rewrite only when the cut changes.

   ```ts
   const k = lodScale(1, viewportHeight, camera.projectionMatrix.elements[5]);
   local.copy(camera.position).applyMatrix4(object.inverseWorld);        // camera in object space
   const n = selector.select(cut, local.x, local.y, local.z, k, camera.near / object.scale);
   if (!sameCut(object, cut, n)) {
     let w = object.indexStart;
     for (let i = 0; i < n; i++) {
       const c = cut[i], start = data.clusterIndexStart[c], count = data.clusterIndexCount[c];
       for (let j = 0; j < count; j++) indexArray[w++] = data.indices[start + j] + object.firstVertex;
     }
     object.indexCount = w - object.indexStart;                           // draw range of this object
     indexAttribute.needsUpdate = true;
   }
   ```

   Skip alpha-cutout foliage and thin transparent decals: they are not connected surfaces and simplify poorly.

Guidance on where it pays off:

- It needs triangles to remove. Below a few thousand triangles per model there is little to gain over a single mesh.
- It relies on connected surfaces. Alpha-tested foliage made of separate cards simplifies poorly (the same is true of Nanite); impostors remain the better tool there, and the two combine well: meshlets up close and at mid range, an impostor past the distance where the coarsest cut is still too heavy.
- Each drawn cluster is one sub-draw. WebGL 2 submits them all in a single `multiDraw` call; the WebGPU back-end issues one `drawIndexed` per cluster, which is cheap but not free. Raise `errorThreshold` or `maxTriangles` on low-end devices.
- Memory: vertices are stored once. The index buffer is about twice the original, and each slot costs 84 bytes of texture.

## How it works

The builder is a TypeScript port of meshoptimizer's [`clusterlod.h`](https://github.com/zeux/meshoptimizer/blob/master/demo/clusterlod.h), including a port of its cluster partitioner:

1. Split the mesh into clusters of up to 128 triangles.
2. Partition adjacent clusters into groups of about 16.
3. Merge each group, lock the vertices on its outer boundary, simplify it to half the triangles, and split the result into new clusters.
4. Repeat on the new clusters until one remains or nothing simplifies any more.

Because group boundaries are locked while a group is simplified, a group can be swapped for its simplified version independently of its neighbours without opening cracks. Each group records a bounding sphere and an error that only grow up the DAG, which is what guarantees that a per-group decision "is the simplified version good enough from here?" always yields a consistent cut.

At runtime `MeshletMesh` walks the DAG from the coarsest groups, descending only where the error projected on screen exceeds `errorThreshold`. The clusters it ends up with are drawn through a stock `THREE.BatchedMesh` in which every cluster is registered as an index range over the one shared vertex buffer. Only BatchedMesh's public API is used, so three keeps ownership of all renderer-specific code.

Data layout is documented in [`src/types.ts`](./src/types.ts).

## Demo

```sh
npm install
npm run dev          # then open the printed URL; add ?renderer=webgpu for WebGPU
```

The demo builds the hierarchy in a worker, draws a grid of instances, and lets you switch renderer, error threshold and debug colouring, freeze the LOD camera, compare against a full-detail `InstancedMesh`, drop in your own `.glb`/`.obj`, and download the resulting `.meshlets` file. Its source in [`demo/`](./demo) doubles as a usage example.

## Development

```sh
npm test             # vitest: builder invariants (crack-free cuts), selection, runtime, serialization
npm run typecheck
npm run build        # emits dist/ (ESM + .d.ts)
npm run demo:build   # static demo in demo/dist
```

## Credits and references

This project started as [AIFanatic/three-nanite](https://github.com/AIFanatic/three-nanite). Version 2 replaces the vendored METIS and meshoptimizer builds with the `meshoptimizer` package and a port of its cluster LOD reference code by Arseny Kapoulkine.

- [Nanite - A Deep Dive](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)
- [The Nanite System in Unreal Engine 5](https://www.medien.ifi.lmu.de/lehre/ws2122/gp/slides/gp-ws2122-extra-nanite.pdf)
- [Multiresolution structures for interactive visualization of very large 3D datasets](https://vcg.isti.cnr.it/~ponchio/download/ponchio_phd.pdf)
- [Batched Multi Triangulations](https://vcg.isti.cnr.it/Publications/2005/CGGMPS05/Slide_BatchedMT_Vis05.pdf)
- [CS 418 - Streaming, Level of Detail, and Occlusion](https://cs418.cs.illinois.edu/website/text/nanite.html)
- [Real-Time Ray Tracing of Micro-Poly Geometry](https://cdrdv2-public.intel.com/782066/real-time-ray-tracing-micro-poly-geometry.pdf)
- [CARROT - jglrxavpok's blog](https://jglrxavpok.github.io/)
- [meshoptimizer](https://github.com/zeux/meshoptimizer)

## License

MIT
