import {
  Box3,
  BatchedMesh,
  BufferAttribute,
  BufferGeometry,
  Color,
  Matrix4,
  Object3D,
  Sphere,
  Vector2,
  Vector3,
} from "three";
import type { Camera, Material, Raycaster, Intersection } from "three";
import { ClusterSelector, lodScale } from "../core/select.js";
import type { MeshletData, TypedArray } from "../types.js";

/** How drawn clusters are tinted. */
export type DebugColorMode = "none" | "cluster" | "lod";

/** Construction options of {@link MeshletMesh}. */
export interface MeshletMeshOptions {
  /** Initial instance capacity (grows automatically by doubling). Default 16. */
  maxInstances?: number;
  /** Initial number of draw slots (one slot = one drawn cluster of one instance); grows by doubling. Default 1024. */
  initialDrawCapacity?: number;
  /** Allowed geometric error in pixels. Default 1. */
  errorThreshold?: number;
}

/** Runtime statistics of a {@link MeshletMesh}. */
export interface MeshletMeshStats {
  /** Number of instances (`count`). */
  instances: number;
  /** Clusters currently drawn, summed over all instances (before frustum culling). */
  clusters: number;
  /** Triangles currently drawn, summed over all instances (before frustum culling). */
  triangles: number;
  /** Draw slot capacity in use (`batched.instanceCount`). */
  slots: number;
  /** Duration in milliseconds of the last non-skipped `update`. */
  lastUpdateMs: number;
  /** How many instances were reselected in the last non-skipped `update`. */
  lastSelected: number;
}

/** The subset of a renderer that the automatic update needs (WebGLRenderer and WebGPURenderer both provide it). */
interface LodRenderer {
  getRenderTarget(): { height: number } | null;
  getDrawingBufferSize(target: Vector2): Vector2;
}

const DEFAULT_NEAR = 0.1;
const MIN_SCALE = 1e-20;

const _m = new Matrix4();
const _inv = new Matrix4();
const _v3 = new Vector3();
const _v2 = new Vector2();
const _color = new Color();
const _sphere = new Sphere();
const _box = new Box3();
const _center = new Vector3();
const _radiusVec = new Vector3();
const _range = {
  vertexStart: 0,
  vertexCount: 0,
  reservedVertexCount: 0,
  indexStart: 0,
  indexCount: 0,
  reservedIndexCount: 0,
  start: 0,
  count: 0,
};

/** Palette for `debugColors = "lod"`, indexed by group depth. */
const LOD_PALETTE: readonly Color[] = [
  0x4e79a7, 0xf28e2b, 0xe15759, 0x76b7b2, 0x59a14f, 0xedc948, 0xb07aa1, 0xff9da7, 0x9c755f, 0xbab0ac,
].map((hex) => new Color(hex));

function hash32(n: number): number {
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  return (n ^ (n >>> 16)) >>> 0;
}

const EMPTY_U32 = new Uint32Array(0);
const EMPTY_I32 = new Int32Array(0);

function growFloat64(a: Float64Array, length: number) {
  const b = new Float64Array(length);
  b.set(a);
  return b;
}
function growFloat32(a: Float32Array, length: number, fill = 0) {
  const b = new Float32Array(length);
  if (fill !== 0) b.fill(fill);
  b.set(a);
  return b;
}
function growInt32(a: Int32Array, length: number, fill = 0) {
  const b = new Int32Array(length);
  if (fill !== 0) b.fill(fill);
  b.set(a);
  return b;
}
function growUint8(a: Uint8Array, length: number) {
  const b = new Uint8Array(length);
  b.set(a);
  return b;
}

/**
 * Renders a {@link MeshletData} cluster hierarchy with view dependent LOD.
 *
 * It owns a stock `THREE.BatchedMesh` ({@link MeshletMesh.batched}) and only
 * uses its public API, so it works on `WebGLRenderer` and `WebGPURenderer`
 * with any standard material, shadows included. Each drawn
 * `(instance, cluster)` pair is one BatchedMesh instance, called a SLOT here,
 * which is created lazily and recycled through a free list.
 *
 * The public interface is InstancedMesh-like: {@link count},
 * {@link setMatrixAt}, {@link setColorAt}. Instance 0 exists by default with an
 * identity matrix.
 */
export class MeshletMesh extends Object3D {
  /** Type flag. */
  readonly isMeshletMesh = true;
  /** The cluster hierarchy being drawn. */
  readonly data: MeshletData;
  /**
   * Called with the new draw mesh after every rebuild caused by draw capacity growth, so per-mesh
   * customisations of `batched` (extra hooks, user data on the mesh, ...) can be re-applied.
   */
  onRebuild: ((batched: BatchedMesh) => void) | undefined = undefined;
  /** Allowed geometric error in pixels (higher = coarser). */
  errorThreshold: number;
  /** When true (default) the LOD is updated from `onBeforeRender` / `onBeforeShadow`. */
  autoUpdate = true;
  /** When set, LOD is always computed for this camera instead of the rendering camera (freeze LOD and fly around). */
  lodCamera: Camera | null = null;
  /** Live statistics. */
  readonly stats: MeshletMeshStats = {
    instances: 0,
    clusters: 0,
    triangles: 0,
    slots: 0,
    lastUpdateMs: 0,
    lastSelected: 0,
  };
  /** Slot (BatchedMesh instance id) to instance id, or -1 when the slot is free. */
  slotInstance: Int32Array;

  private readonly selector: ClusterSelector;
  private readonly scratch: Uint32Array;
  private readonly newSlots: Int32Array;
  private readonly marks: Int32Array;
  private readonly shadowCameras = new WeakSet<Camera>();
  private _batched: BatchedMesh;
  /** Zero-copy buffers wrapping the MeshletData arrays, shared by every BatchedMesh generation. */
  private readonly sharedAttributes: Array<{ name: string; attribute: BufferAttribute }>;
  private readonly sharedIndex: BufferAttribute;
  /**
   * Replaced draw meshes waiting to be disposed. Disposal is deferred to a later task because a mesh
   * replaced during a render is still in that render's lists (shadow passes and the main pass).
   */
  private retired: BatchedMesh[] = [];
  private retireTimer: ReturnType<typeof setTimeout> | null = null;
  private mainHeight = 0;

  // ---- instances
  private _count = 0;
  private instCap = 0;
  private matrices = new Float64Array(0);
  private inverses = new Float64Array(0);
  private instScale = new Float32Array(0);
  private colors = new Float32Array(0);
  private dirty = new Uint8Array(0);
  private dirtyList = new Int32Array(0);
  private dirtyCount = 0;
  private cutCount = new Int32Array(0);
  private cutClusters: Uint32Array[] = [];
  private cutSlots: Int32Array[] = [];

  // ---- slots
  private freeSlots = new Int32Array(0);
  private freeCount = 0;

  // ---- colors
  private _debugColors: DebugColorMode = "none";
  private colorsInUse = false;

  // ---- update signature / per-update view state
  private sigValid = false;
  private sigX = 0;
  private sigY = 0;
  private sigZ = 0;
  private sigProj = 0;
  private sigHeight = 0;
  private sigThreshold = 0;
  private sigOrtho = false;
  private sigNear = 0;
  private sigScale = 0;
  private vx = 0;
  private vy = 0;
  private vz = 0;
  private vProj = 1;
  private vHeight = 1;
  private vNear = DEFAULT_NEAR;
  private vOrtho = false;
  private vWorldScale = 1;
  /** Instances the running `update` selects in total / has not selected yet (sizes capacity growth). */
  private pendingTotal = 1;
  private pendingInstances = 0;

  /**
   * @param data     Cluster hierarchy (from the builder or a deserialized file).
   * @param material Any standard three.js material (`undefined` = BatchedMesh default).
   * @param options  See {@link MeshletMeshOptions}.
   */
  constructor(data: MeshletData, material?: Material, options: MeshletMeshOptions = {}) {
    super();
    this.data = data;
    this.errorThreshold = options.errorThreshold ?? 1;

    this.selector = new ClusterSelector(data);
    this.scratch = new Uint32Array(data.clusterCount);
    this.newSlots = new Int32Array(data.clusterCount);
    this.marks = new Int32Array(data.clusterCount);

    this.sharedAttributes = data.attributes.map((a) => ({
      name: a.name,
      // `TypedArray` is a subset of the array types BufferAttribute accepts.
      attribute: new BufferAttribute(a.array as TypedArray, a.itemSize, a.normalized),
    }));
    // With at most 65535 vertices a 16-bit index buffer halves the index memory.
    this.sharedIndex = new BufferAttribute(data.vertexCount <= 65535 ? Uint16Array.from(data.indices) : data.indices, 1);

    const maxInstances = Math.max(1, Math.floor(options.maxInstances ?? 16));
    const coarsest = this.selector.selectByError(this.scratch, Infinity);
    const slotCapacity = Math.max(
      coarsest,
      1,
      Math.floor(options.initialDrawCapacity ?? Math.max(64, maxInstances * coarsest * 4)),
    );

    const batched = this.createBatched(slotCapacity, material);
    this._batched = batched;
    batched.layers = this.layers;
    batched.castShadow = this.castShadow;
    batched.receiveShadow = this.receiveShadow;
    batched.renderOrder = this.renderOrder;
    this.forwardProperties();
    this.add(batched);

    this.slotInstance = new Int32Array(slotCapacity).fill(-1);
    this.freeSlots = new Int32Array(slotCapacity);

    this.ensureInstanceCapacity(maxInstances);
    this.count = 1;
  }

  // =========================================================================
  // public surface
  // =========================================================================

  /**
   * The internal draw mesh. It is REPLACED when draw capacity grows: do not hold on to it, and set
   * `initialDrawCapacity` high enough to avoid growth. See also {@link onRebuild}.
   */
  get batched(): BatchedMesh {
    return this._batched;
  }

  /** Number of draw slots the current draw mesh can hold (`batched.maxInstanceCount`). */
  get drawCapacity(): number {
    return this._batched.maxInstanceCount;
  }

  /** Material used by the drawing BatchedMesh. */
  get material(): Material {
    return this._batched.material;
  }
  set material(value: Material) {
    this._batched.material = value;
  }

  /** Number of instances. Raising adds identity instances, lowering removes the last ones. */
  override get count(): number {
    return this._count;
  }
  override set count(value: number) {
    const n = Math.max(0, Math.floor(value));
    const old = this._count;
    if (n === old) return;
    if (n < old) {
      for (let i = n; i < old; i++) this.releaseInstance(i);
    } else {
      this.ensureInstanceCapacity(n);
      for (let i = old; i < n; i++) {
        const o = i * 16;
        const mat = this.matrices;
        const inv = this.inverses;
        for (let k = 0; k < 16; k++) mat[o + k] = inv[o + k] = k % 5 === 0 ? 1 : 0;
        this.instScale[i] = 1;
        this.colors[i * 3] = this.colors[i * 3 + 1] = this.colors[i * 3 + 2] = 1;
        this.cutCount[i] = 0;
        this.markDirty(i);
      }
    }
    this._count = n;
    this.stats.instances = n;
  }

  /** `"cluster"`: random color per cluster. `"lod"`: color by DAG level. `"none"`: instance color. */
  get debugColors(): DebugColorMode {
    return this._debugColors;
  }
  set debugColors(mode: DebugColorMode) {
    if (mode === this._debugColors) return;
    this._debugColors = mode;
    if (mode !== "none") this.colorsInUse = true;
    if (this.colorsInUse) this.recolorAll();
  }

  /** Sets the matrix of instance `i` (local to this object). Takes effect immediately on drawn slots. */
  setMatrixAt(i: number, matrix: Matrix4): void {
    if (!(i >= 0 && i < this._count)) throw new RangeError(`MeshletMesh.setMatrixAt: instance ${i} out of range (count ${this._count})`);
    matrix.toArray(this.matrices, i * 16);
    _inv.copy(matrix).invert().toArray(this.inverses, i * 16);
    this.instScale[i] = matrix.getMaxScaleOnAxis();
    this.markDirty(i);
    const n = this.cutCount[i];
    const slots = this.cutSlots[i];
    for (let j = 0; j < n; j++) this._batched.setMatrixAt(slots[j], matrix);
  }

  /** Reads the matrix of instance `i` into `target`. */
  getMatrixAt(i: number, target: Matrix4): Matrix4 {
    return target.fromArray(this.matrices, i * 16);
  }

  /** Sets the color of instance `i` (used when `debugColors` is `"none"`). */
  setColorAt(i: number, color: Color): void {
    if (!(i >= 0 && i < this._count)) throw new RangeError(`MeshletMesh.setColorAt: instance ${i} out of range (count ${this._count})`);
    this.colors[i * 3] = color.r;
    this.colors[i * 3 + 1] = color.g;
    this.colors[i * 3 + 2] = color.b;
    if (!this.colorsInUse) {
      if (color.r === 1 && color.g === 1 && color.b === 1) return;
      this.colorsInUse = true;
      this.recolorAll();
    } else if (this._debugColors === "none") {
      const n = this.cutCount[i];
      const slots = this.cutSlots[i];
      const clusters = this.cutClusters[i];
      for (let j = 0; j < n; j++) this.applySlotColor(slots[j], clusters[j], i);
    }
  }

  /** Reads the color of instance `i` into `target`. */
  getColorAt(i: number, target: Color): Color {
    return target.setRGB(this.colors[i * 3], this.colors[i * 3 + 1], this.colors[i * 3 + 2]);
  }

  /**
   * Recomputes the cluster cut for the given camera.
   *
   * Cheap when nothing relevant changed: camera rotation, for example, never
   * triggers any work.
   *
   * @param camera         Camera to compute LOD for.
   * @param viewportHeight Height in pixels of the render target the camera renders to.
   * @returns `true` if the set of drawn clusters or an instance matrix changed.
   */
  update(camera: Camera, viewportHeight: number): boolean {
    this.updateWorldMatrix(true, false);
    camera.updateWorldMatrix(true, false);

    const ortho = (camera as { isOrthographicCamera?: boolean }).isOrthographicCamera === true;
    const proj = camera.projectionMatrix.elements[5];
    // `near` exists on Perspective/OrthographicCamera but not on the Camera base type.
    const near = (camera as { near?: number }).near ?? DEFAULT_NEAR;
    const worldScale = this.matrixWorld.getMaxScaleOnAxis();

    let cx = 0;
    let cy = 0;
    let cz = 0;
    if (!ortho) {
      _inv.copy(this.matrixWorld).invert();
      _v3.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(_inv);
      cx = _v3.x;
      cy = _v3.y;
      cz = _v3.z;
    }

    const sameView =
      this.sigValid &&
      cx === this.sigX &&
      cy === this.sigY &&
      cz === this.sigZ &&
      proj === this.sigProj &&
      viewportHeight === this.sigHeight &&
      this.errorThreshold === this.sigThreshold &&
      ortho === this.sigOrtho &&
      near === this.sigNear &&
      worldScale === this.sigScale;

    if (sameView && this.dirtyCount === 0) return false;

    const t0 = performance.now();
    this.vx = cx;
    this.vy = cy;
    this.vz = cz;
    this.vProj = proj;
    this.vHeight = viewportHeight;
    this.vNear = near;
    this.vOrtho = ortho;
    this.vWorldScale = worldScale;

    let changed = false;
    let selected = 0;
    this.pendingTotal = this.pendingInstances = Math.max(1, sameView ? this.dirtyCount : this._count);
    const dirty = this.dirty;
    const list = this.dirtyList;
    const count = this._count;

    if (sameView) {
      for (let d = 0; d < this.dirtyCount; d++) {
        const i = list[d];
        dirty[i] = 0;
        this.pendingInstances--;
        if (i < count) {
          this.reselect(i);
          selected++;
        }
      }
      changed = true; // an instance matrix changed
    } else {
      for (let d = 0; d < this.dirtyCount; d++) dirty[list[d]] = 0;
      for (let i = 0; i < count; i++) {
        this.pendingInstances--;
        if (this.reselect(i)) changed = true;
        selected++;
      }
      if (this.dirtyCount > 0) changed = true;
      this.sigValid = true;
      this.sigX = cx;
      this.sigY = cy;
      this.sigZ = cz;
      this.sigProj = proj;
      this.sigHeight = viewportHeight;
      this.sigThreshold = this.errorThreshold;
      this.sigOrtho = ortho;
      this.sigNear = near;
      this.sigScale = worldScale;
    }
    this.dirtyCount = 0;

    this.stats.lastSelected = selected;
    this.stats.lastUpdateMs = performance.now() - t0;
    return changed;
  }

  /** Releases the GPU resources of the draw mesh, including the shared vertex/index buffers. The material is not disposed. */
  override dispose(): void {
    this.disposeRetired();
    this._batched.dispose();
  }

  // =========================================================================
  // draw mesh (BatchedMesh) creation and growth
  // =========================================================================

  /**
   * Creates a BatchedMesh whose geometry ids equal cluster ids and whose index layout equals `data.indices`.
   *
   * Every cluster is added as an index-only geometry (zero vertices); afterwards the BatchedMesh's own
   * buffers are swapped for buffers that wrap the MeshletData arrays, so nothing is copied.
   */
  private createBatched(capacity: number, material: Material | undefined): BatchedMesh {
    const data = this.data;
    const bm = new BatchedMesh(capacity, 1, data.indices.length, material);

    const temp = new BufferGeometry();
    for (const a of data.attributes) {
      const ctor = a.array.constructor as new (length: number) => TypedArray;
      temp.setAttribute(a.name, new BufferAttribute(new ctor(0), a.itemSize, a.normalized));
    }
    let maxCount = 0;
    for (let c = 0; c < data.clusterCount; c++) maxCount = Math.max(maxCount, data.clusterIndexCount[c]);
    const dummyIndex = new Uint32Array(maxCount);
    temp.boundingSphere = _sphere;
    temp.boundingBox = _box;

    const bounds = data.clusterBounds;
    for (let c = 0; c < data.clusterCount; c++) {
      const count = data.clusterIndexCount[c];
      temp.setIndex(new BufferAttribute(dummyIndex.subarray(0, count), 1));
      _center.set(bounds[c * 4], bounds[c * 4 + 1], bounds[c * 4 + 2]);
      const r = bounds[c * 4 + 3];
      _sphere.set(_center, r);
      _box.setFromCenterAndSize(_center, _radiusVec.set(2 * r, 2 * r, 2 * r));

      const id = bm.addGeometry(temp);
      const range = bm.getGeometryRangeAt(id, _range);
      if (
        id !== c ||
        range.vertexCount !== 0 ||
        range.indexStart !== data.clusterIndexStart[c] ||
        range.indexCount !== count
      ) {
        throw new Error(
          "three-nanite: unsupported three.js version (BatchedMesh geometry ids / ranges do not match the expected layout). Use three r170 or newer.",
        );
      }
    }

    // swap the husk's buffers for the shared, zero-copy ones
    const geometry = bm.geometry;
    for (const { name, attribute } of this.sharedAttributes) geometry.setAttribute(name, attribute);
    geometry.setIndex(this.sharedIndex);

    bm.frustumCulled = false;
    bm.perObjectFrustumCulled = true;
    bm.sortObjects = false;
    this.installHooks(bm);
    return bm;
  }

  /**
   * Replaces the draw mesh by a larger one (capacity doubles until it holds `minCapacity` slots). Three's
   * `setInstanceCount` is deliberately not used: it leaves node materials of WebGPURenderer pointing at
   * the old data textures. Slot ids, geometry ids, matrices, visibility and colors are preserved.
   */
  private grow(minCapacity: number): void {
    const old = this._batched;
    let capacity = old.maxInstanceCount * 2;
    while (capacity < minCapacity) capacity *= 2;
    const next = this.createBatched(capacity, old.material);

    const m = new Matrix4();
    const color = new Color();
    for (let s = 0; s < old.instanceCount; s++) {
      const id = next.addInstance(old.getGeometryIdAt(s));
      if (id !== s) throw new Error("three-nanite: unexpected BatchedMesh slot id while growing (unsupported three.js version).");
      next.setMatrixAt(s, old.getMatrixAt(s, m));
      if (!old.getVisibleAt(s)) next.setVisibleAt(s, false);
      if (this.colorsInUse) next.setColorAt(s, old.getColorAt(s, color));
    }

    next.layers = old.layers;
    next.castShadow = old.castShadow;
    next.receiveShadow = old.receiveShadow;
    next.renderOrder = old.renderOrder;
    next.visible = old.visible;
    next.name = old.name;
    next.userData = old.userData;
    next.customDepthMaterial = old.customDepthMaterial;
    next.customDistanceMaterial = old.customDistanceMaterial;

    if (this.slotInstance.length < next.maxInstanceCount) {
      this.slotInstance = growInt32(this.slotInstance, next.maxInstanceCount, -1);
      this.freeSlots = growInt32(this.freeSlots, next.maxInstanceCount);
    }

    this.remove(old);
    this.add(next);
    this._batched = next;

    // The old mesh may still be drawn by the render in progress (it shows the previous cut for that frame).
    this.retired.push(old);
    if (this.retireTimer === null) this.retireTimer = setTimeout(() => this.disposeRetired(), 0);
    this.onRebuild?.(next);
  }

  /** Disposes a replaced draw mesh without touching the shared geometry buffers. */
  private disposeBatchedKeepingBuffers(old: BatchedMesh): void {
    // Detach the shared buffers so `old.dispose()` (which disposes its geometry and data textures) cannot free them.
    const geometry = old.geometry;
    for (const { name } of this.sharedAttributes) geometry.deleteAttribute(name);
    geometry.setIndex(null);
    old.dispose();
  }

  private disposeRetired(): void {
    if (this.retireTimer !== null) {
      clearTimeout(this.retireTimer);
      this.retireTimer = null;
    }
    for (const old of this.retired) this.disposeBatchedKeepingBuffers(old);
    this.retired.length = 0;
  }

  /** Redefines `castShadow` / `receiveShadow` / `renderOrder` so they live on the current BatchedMesh. */
  private forwardProperties(): void {
    Object.defineProperty(this, "castShadow", {
      get: () => this._batched.castShadow,
      set: (v: boolean) => {
        this._batched.castShadow = v;
      },
      enumerable: true,
      configurable: true,
    });
    Object.defineProperty(this, "receiveShadow", {
      get: () => this._batched.receiveShadow,
      set: (v: boolean) => {
        this._batched.receiveShadow = v;
      },
      enumerable: true,
      configurable: true,
    });
    Object.defineProperty(this, "renderOrder", {
      get: () => this._batched.renderOrder,
      set: (v: number) => {
        this._batched.renderOrder = v;
      },
      enumerable: true,
      configurable: true,
    });
  }

  private installHooks(batched: BatchedMesh): void {
    const shadowCameras = this.shadowCameras;

    // LOD must always come from the viewer's camera, never from a light's:
    //  - WebGLRenderer calls onBeforeShadow(camera = viewer, shadowCamera), and no onBeforeRender in shadow passes;
    //  - WebGPURenderer calls onBeforeShadow(camera = shadowCamera, shadowCamera), then onBeforeRender(shadowCamera).
    // A mesh replaced by `grow` during this very render keeps being drawn for the rest of it, but must not
    // drive updates any more (`batched !== this._batched`).
    batched.onBeforeRender = (renderer, scene, camera, geometry, material, group) => {
      if (batched === this._batched && !shadowCameras.has(camera)) this.autoUpdateFrom(renderer, camera, true);
      BatchedMesh.prototype.onBeforeRender.call(batched, renderer, scene, camera, geometry, material, group);
    };
    batched.onBeforeShadow = (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) => {
      shadowCameras.add(shadowCamera);
      if (batched === this._batched && !shadowCameras.has(camera)) this.autoUpdateFrom(renderer, camera, false);
      BatchedMesh.prototype.onBeforeShadow.call(batched, renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
    };

    const proto = BatchedMesh.prototype.raycast;
    batched.raycast = (raycaster: Raycaster, intersects: Intersection[]) => {
      const first = intersects.length;
      proto.call(batched, raycaster, intersects);
      for (let i = first; i < intersects.length; i++) {
        const hit = intersects[i];
        hit.instanceId = hit.batchId !== undefined ? this.slotInstance[hit.batchId] : undefined;
        hit.object = this;
      }
    };
  }

  private autoUpdateFrom(renderer: LodRenderer, camera: Camera, mainPass: boolean): void {
    if (!this.autoUpdate) return;
    let height: number;
    if (mainPass) {
      const target = renderer.getRenderTarget();
      height = target !== null ? target.height : renderer.getDrawingBufferSize(_v2).y;
      this.mainHeight = height;
    } else {
      height = this.mainHeight > 0 ? this.mainHeight : renderer.getDrawingBufferSize(_v2).y;
    }
    this.update(this.lodCamera ?? camera, height);
  }

  // =========================================================================
  // instances
  // =========================================================================

  private ensureInstanceCapacity(n: number): void {
    if (n <= this.instCap) return;
    let cap = Math.max(1, this.instCap);
    while (cap < n) cap *= 2;
    const old = this.instCap;
    this.instCap = cap;
    this.matrices = growFloat64(this.matrices, cap * 16);
    this.inverses = growFloat64(this.inverses, cap * 16);
    this.instScale = growFloat32(this.instScale, cap, 1);
    this.colors = growFloat32(this.colors, cap * 3, 1);
    this.dirty = growUint8(this.dirty, cap);
    this.dirtyList = growInt32(this.dirtyList, cap);
    this.cutCount = growInt32(this.cutCount, cap);
    for (let i = old; i < cap; i++) {
      this.cutClusters.push(EMPTY_U32);
      this.cutSlots.push(EMPTY_I32);
    }
  }

  private markDirty(i: number): void {
    if (this.dirty[i] === 0) {
      this.dirty[i] = 1;
      this.dirtyList[this.dirtyCount++] = i;
    }
  }

  private releaseInstance(i: number): void {
    const n = this.cutCount[i];
    const slots = this.cutSlots[i];
    const clusters = this.cutClusters[i];
    for (let j = 0; j < n; j++) this.releaseSlot(slots[j], clusters[j]);
    this.cutCount[i] = 0;
  }

  // =========================================================================
  // LOD selection
  // =========================================================================

  /** Reselects the cut of instance `i` for the current view state and diffs it against the old one. */
  private reselect(i: number): boolean {
    const e = this.inverses;
    const o = i * 16;
    let x = 0;
    let y = 0;
    let z = 0;
    if (!this.vOrtho) {
      const cx = this.vx;
      const cy = this.vy;
      const cz = this.vz;
      x = e[o] * cx + e[o + 4] * cy + e[o + 8] * cz + e[o + 12];
      y = e[o + 1] * cx + e[o + 5] * cy + e[o + 9] * cz + e[o + 13];
      z = e[o + 2] * cx + e[o + 6] * cy + e[o + 10] * cz + e[o + 14];
    }
    let scale = this.instScale[i] * this.vWorldScale;
    if (!(scale > MIN_SCALE)) scale = MIN_SCALE;
    const k = lodScale(this.errorThreshold, this.vHeight, this.vProj, scale, this.vOrtho);
    const scratch = this.scratch;
    const n = this.selector.select(scratch, x, y, z, k, this.vNear / scale, this.vOrtho);

    const oldN = this.cutCount[i];
    // Worst case every drawn cluster of this instance is replaced: make sure the slots fit before diffing.
    const needed = this.stats.clusters - oldN + n;
    if (this._batched.maxInstanceCount < needed) {
      // Rebuilding is not free, so aim past this instance: assume the instances still to be
      // selected in this update need about as much as the ones done so far.
      const done = Math.max(1, this.pendingTotal - this.pendingInstances);
      // Capped, because the first instances selected may be the closest (most detailed) ones.
      this.grow(Math.min(needed * 4, Math.ceil((needed * this.pendingTotal) / done)));
    }

    const marks = this.marks;
    const newSlots = this.newSlots;
    const oldClusters = this.cutClusters[i];
    const oldSlots = this.cutSlots[i];
    let changed = false;

    for (let j = 0; j < n; j++) marks[scratch[j]] = 1;
    // releases first, so freed slots are recycled by the acquires below
    for (let j = 0; j < oldN; j++) {
      const c = oldClusters[j];
      if (marks[c] === 1) {
        marks[c] = oldSlots[j] + 2; // kept
      } else {
        this.releaseSlot(oldSlots[j], c);
        changed = true;
      }
    }
    _m.fromArray(this.matrices, o);
    for (let j = 0; j < n; j++) {
      const c = scratch[j];
      const m = marks[c];
      if (m >= 2) {
        newSlots[j] = m - 2;
      } else {
        newSlots[j] = this.acquireSlot(c, i);
        changed = true;
      }
      marks[c] = 0;
    }

    if (changed || n !== oldN) {
      let clusters = this.cutClusters[i];
      let slots = this.cutSlots[i];
      if (clusters.length < n) {
        const cap = Math.max(n, clusters.length * 2);
        clusters = this.cutClusters[i] = new Uint32Array(cap);
        slots = this.cutSlots[i] = new Int32Array(cap);
      }
      for (let j = 0; j < n; j++) {
        clusters[j] = scratch[j];
        slots[j] = newSlots[j];
      }
      this.cutCount[i] = n;
    }
    return changed;
  }

  /** Takes a slot from the free list (or creates one) and points it at cluster `c` of instance `i`. Uses `_m` as the instance matrix. */
  private acquireSlot(c: number, i: number): number {
    let slot: number;
    if (this.freeCount > 0) {
      slot = this.freeSlots[--this.freeCount];
      const batched = this._batched;
      batched.setGeometryIdAt(slot, c); // geometry id === cluster id
      batched.setMatrixAt(slot, _m);
      if (this.colorsInUse) this.applySlotColor(slot, c, i);
      batched.setVisibleAt(slot, true);
    } else {
      // `reselect` normally grows ahead of time; this is the safety net. Never cache the draw mesh across this call.
      if (this._batched.instanceCount >= this._batched.maxInstanceCount) this.grow(0);
      const batched = this._batched;
      slot = batched.addInstance(c);
      batched.setMatrixAt(slot, _m);
      if (this.colorsInUse) this.applySlotColor(slot, c, i);
      this.stats.slots = batched.instanceCount;
    }
    this.slotInstance[slot] = i;
    this.stats.clusters++;
    this.stats.triangles += this.data.clusterIndexCount[c] / 3;
    return slot;
  }

  private releaseSlot(slot: number, c: number): void {
    this._batched.setVisibleAt(slot, false);
    this.slotInstance[slot] = -1;
    this.freeSlots[this.freeCount++] = slot;
    this.stats.clusters--;
    this.stats.triangles -= this.data.clusterIndexCount[c] / 3;
  }

  // =========================================================================
  // colors
  // =========================================================================

  private applySlotColor(slot: number, c: number, i: number): void {
    switch (this._debugColors) {
      case "cluster": {
        const h = hash32(c + 1);
        _color.setRGB(0.25 + 0.75 * ((h & 255) / 255), 0.25 + 0.75 * (((h >>> 8) & 255) / 255), 0.25 + 0.75 * (((h >>> 16) & 255) / 255));
        break;
      }
      case "lod": {
        const depth = this.data.groupDepth[this.data.clusterGroup[c]];
        _color.copy(LOD_PALETTE[depth % LOD_PALETTE.length]);
        break;
      }
      default:
        _color.setRGB(this.colors[i * 3], this.colors[i * 3 + 1], this.colors[i * 3 + 2]);
    }
    this._batched.setColorAt(slot, _color);
  }

  private recolorAll(): void {
    for (let i = 0; i < this._count; i++) {
      const n = this.cutCount[i];
      const slots = this.cutSlots[i];
      const clusters = this.cutClusters[i];
      for (let j = 0; j < n; j++) this.applySlotColor(slots[j], clusters[j], i);
    }
  }
}
