import {
  Box3,
  BatchedMesh,
  BufferAttribute,
  BufferGeometry,
  Color,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Object3D,
  REVISION,
  Sphere,
  Vector2,
  Vector3,
} from "three";
import type { Camera, Material, Raycaster, Intersection } from "three";
import { computeLodLevels, copyLevelIndices, levelForError, levelOfCut } from "../core/levels.js";
import type { LodLevels } from "../core/levels.js";
import { computeDrawRanges, rangesOfCut } from "../core/ranges.js";
import type { DrawRanges } from "../core/ranges.js";
import { ClusterSelector, lodScale } from "../core/select.js";
import { frustumPlanesIn, PLANE_FLOATS } from "./viewPlanes.js";
import { LOD_PALETTE } from "./palette.js";
import type { MeshletData, TypedArray } from "../types.js";

/** How drawn clusters are tinted. */
export type DebugColorMode = "none" | "cluster" | "lod";

/** Construction options of {@link MeshletMesh}. */
export interface MeshletMeshOptions {
  /** Initial instance capacity (grows automatically by doubling). Default 16. */
  maxInstances?: number;
  /**
   * Initial number of draw slots (one slot = one drawn range of one instance drawn by cluster); grows by
   * doubling, each time with a rebuild of the draw mesh. Default: 256, or four per instance of
   * `maxInstances` when every instance is drawn by cluster.
   */
  initialDrawCapacity?: number;
  /**
   * Draw clusters that are always drawn together as one index range (see `computeDrawRanges`): 3 to 16
   * times fewer draws for the same triangles. Default true. Turn it off to draw, cull and colour every
   * cluster by itself, for example to look at the clusters with `debugColors = "cluster"`.
   */
  mergeClusters?: boolean;
  /** Allowed geometric error in pixels. Default 1. */
  errorThreshold?: number;
  /**
   * Draw every instance that needs one detail level throughout (see `LodLevels`) with ordinary
   * instancing: all instances on a level are one draw, whatever their number. That is every instance
   * seen from a few radii away, and every instance of a mesh too small to mix levels. Default true.
   * With `false`, or `mergeClusters: false`, every instance is drawn cluster by cluster.
   */
  instancedLevels?: boolean;
  /**
   * How far away an instance takes a whole-mesh level without its hierarchy being walked: the
   * distance from the camera to its bounding sphere, in units of that sphere's radius. Default 3.
   * Nearer instances are walked, and still use a level when their cut turns out to be one.
   */
  levelDistance?: number;
  /**
   * Leave instances outside the view out of the instanced draws of the main pass (shadow passes
   * draw all of them). Costs one sphere test per instance per frame the camera turns. Default true.
   */
  cullInstances?: boolean;
  /**
   * What one more draw call is worth, in triangles. Every level in use is a draw, so instances are
   * drawn at a finer level than they need when that takes fewer draws and costs less than this many
   * triangles per draw saved. It is what keeps a small mesh with a handful of instances at one draw
   * instead of one per level; a mesh of many triangles is not affected, its levels are further
   * apart than this. `0` draws every instance at exactly the level it needs. Default 4096.
   */
  drawCost?: number;
}

/** Runtime statistics of a {@link MeshletMesh}. */
export interface MeshletMeshStats {
  /** Number of instances (`count`). */
  instances: number;
  /** Clusters currently drawn, summed over all instances (before frustum culling). */
  clusters: number;
  /** Draws a pass takes at most: the cluster ranges of the instances drawn by cluster, plus one per level in use. */
  draws: number;
  /** Triangles currently drawn, summed over all instances (before frustum culling). */
  triangles: number;
  /** Draw slot capacity in use (`batched.instanceCount`). */
  slots: number;
  /** Instances drawn through an instanced level. */
  levelInstances: number;
  /** Of those, the ones inside the view the levels were last culled for. */
  levelVisible: number;
  /** Levels in use: instanced draws per pass. */
  levelDraws: number;
  /** Duration in milliseconds of the last non-skipped `update`. */
  lastUpdateMs: number;
  /** How many instances were reselected in the last non-skipped `update`. */
  lastSelected: number;
}

/** The subset of a renderer that the automatic update needs (WebGLRenderer and WebGPURenderer both provide it). */
interface LodRenderer {
  getRenderTarget(): { height: number } | null;
  getDrawingBufferSize(target: Vector2): Vector2;
  isWebGPURenderer?: boolean;
}

/**
 * The first three.js revision whose WebGPURenderer draws an InstancedMesh correctly when its instance
 * count and instance buffers change between passes and frames, which drawing by level relies on.
 * (Before it the instance data is bound when the mesh is first drawn, in ways that differ from
 * revision to revision: r181 to r186, for one, size it by the count of that first draw.) Older
 * revisions draw every instance by cluster on that renderer. WebGLRenderer has no such limit.
 */
const WEBGPU_LEVELS_REVISION = 187;
const THREE_REVISION = Number.parseInt(REVISION, 10);

/** One whole-mesh level and the instances drawn at it. */
interface LevelBucket {
  readonly level: number;
  /** Whether the level can be drawn as one index range (null until first asked). */
  usable: boolean | null;
  geometry: BufferGeometry | null;
  /** The geometry has an index buffer of its own (the level is not contiguous in the shared one). */
  ownsIndex: boolean;
  mesh: InstancedMesh | null;
  /** Instance drawn at each position of `mesh`; positions below `visible` are inside the view. */
  ids: Int32Array;
  capacity: number;
  count: number;
  visible: number;
  /** A shadow pass is drawing `mesh` right now. */
  inShadow: boolean;
}

const DEFAULT_NEAR = 0.1;
const MIN_SCALE = 1e-20;
/** A bounding sphere no frustum contains: what a draw mesh carries while nothing of it is in view. */
const NEVER_IN_VIEW = new Sphere(new Vector3(), -Infinity);

/**
 * A child whose turn in the renderer's walk of the scene comes after the draw meshes: the
 * renderers call `update` on anything carrying `isLOD` as they reach it, children in order.
 */
class AfterDrawMeshes extends Object3D {
  readonly isLOD = true;
  autoUpdate = true;
  constructor(private readonly run: () => void) {
    super();
    this.matrixAutoUpdate = false;
  }
  update(): void {
    this.run();
  }
}

let _wanted = new Int32Array(64);
let _from = new Int32Array(65);
let _best = new Float64Array(65);

const _m = new Matrix4();
const _inv = new Matrix4();
const _clip = new Matrix4();
const _v3 = new Vector3();
const _v2 = new Vector2();
const _color = new Color();
const _sphere = new Sphere();
const _box = new Box3();
const _center = new Vector3();
const _radiusVec = new Vector3();
const _planes = new Float64Array(PLANE_FLOATS);
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
 * Marks `count` array elements from `start` for upload. The attribute keeps a single range, the
 * union of everything touched since the renderer last uploaded (it empties the list when it does).
 */
function touchRange(attribute: BufferAttribute, start: number, count: number): void {
  const ranges = attribute.updateRanges;
  if (ranges.length === 0) {
    attribute.addUpdateRange(start, count);
  } else {
    const r = ranges[0];
    const end = Math.max(r.start + r.count, start + count);
    if (start < r.start) r.start = start;
    r.count = end - r.start;
  }
  attribute.needsUpdate = true;
}

/**
 * The height of what each camera was last seen rendering to. It is a fact about the camera, not
 * about a mesh, so all meshes share it: one that has not been drawn yet knows it all the same, and
 * one draw a frame notes it for all of them.
 */
const _heights = new WeakMap<Camera, number>();
/** The height last noted, and the camera it was noted for. */
let _height = 0;
let _heightCamera: Camera | null = null;
/** The camera whose viewport has been noted since a mesh was last updated. */
let _notedCamera: Camera | null = null;
/** Whether the three.js in use has the WebGPURenderer that cannot draw by level (see `WEBGPU_LEVELS_REVISION`). */
const LEVELS_NEED_CHECK = THREE_REVISION < WEBGPU_LEVELS_REVISION;

/** Remembers the height of what `camera` renders to, for the automatic updates of the next frame. */
function noteViewport(renderer: LodRenderer, camera: Camera): void {
  _notedCamera = camera;
  const target = renderer.getRenderTarget();
  const height = target !== null ? target.height : renderer.getDrawingBufferSize(_v2).y;
  if (camera !== _heightCamera || height !== _height) {
    _heightCamera = camera;
    _height = height;
    _heights.set(camera, height);
  }
}

/** Height to assume before the first render told us the real one. */
function defaultViewportHeight(): number {
  const g = globalThis as { innerHeight?: number; devicePixelRatio?: number };
  return typeof g.innerHeight === "number" && g.innerHeight > 0 ? Math.round(g.innerHeight * (g.devicePixelRatio ?? 1)) : 1080;
}

/**
 * Renders a {@link MeshletData} cluster hierarchy with view dependent LOD.
 *
 * Every instance is drawn in one of two ways, chosen per instance on every update:
 *
 *  - BY LEVEL. An instance that needs the same detail level throughout is drawn at one of the
 *    mesh's whole-mesh levels ({@link MeshletMesh.levels}). Each level in use is one stock
 *    `THREE.InstancedMesh` holding all instances on it: one draw per level, however many instances.
 *    Far instances find their level from their distance alone, without walking the hierarchy.
 *  - BY CLUSTER. An instance close enough to need different levels in different places gets its own
 *    cut of the DAG, drawn through a stock `THREE.BatchedMesh` ({@link MeshletMesh.batched}) in DRAW
 *    RANGES ({@link MeshletMesh.ranges}: runs of clusters that are always drawn together). Each drawn
 *    `(instance, range)` pair is one BatchedMesh instance, called a SLOT here, which is created
 *    lazily and recycled through a free list.
 *
 * Both are stock three.js objects driven through their public API, so this works on
 * `WebGLRenderer` and `WebGPURenderer` with any standard material, shadows included. All of them
 * share the vertex buffers of the hierarchy.
 *
 * The public interface is InstancedMesh-like: {@link count}, {@link setMatrixAt},
 * {@link setColorAt}. Instance 0 exists by default with an identity matrix.
 */
export class MeshletMesh extends Object3D {
  /** Type flag. */
  readonly isMeshletMesh = true;
  /**
   * Makes both renderers call {@link update} with the camera while they collect the scene, which
   * is before any draw buffer of the frame is uploaded (they do this for every object carrying this
   * flag, as three's own `LOD` does).
   */
  readonly isLOD = true;
  /** The cluster hierarchy being drawn. */
  readonly data: MeshletData;
  /**
   * Called with the new draw mesh after every rebuild caused by draw capacity growth, so per-mesh
   * customisations of `batched` (extra hooks, user data on the mesh, ...) can be re-applied.
   */
  onRebuild: ((batched: BatchedMesh) => void) | undefined = undefined;
  /** Allowed geometric error in pixels (higher = coarser). */
  errorThreshold: number;
  /** When true (default) the renderer updates the LOD for the camera it renders with. */
  autoUpdate = true;
  /** When set, LOD is always computed for this camera instead of the rendering camera (freeze LOD and fly around). */
  lodCamera: Camera | null = null;
  /**
   * Height in pixels of what the camera renders to, for automatic updates. `null` (default) takes
   * it from the renderer, which is known from the first frame on (the very first update assumes the
   * window height).
   */
  viewportHeight: number | null = null;
  /** See {@link MeshletMeshOptions.levelDistance}. */
  levelDistance: number;
  /** See {@link MeshletMeshOptions.cullInstances}. */
  cullInstances: boolean;
  /** See {@link MeshletMeshOptions.drawCost}. */
  drawCost: number;
  /** Sphere around all instances, in the space of this object. Kept up to date by `update`. */
  readonly boundingSphere = new Sphere();
  /** Live statistics. */
  readonly stats: MeshletMeshStats = {
    instances: 0,
    clusters: 0,
    draws: 0,
    triangles: 0,
    slots: 0,
    levelInstances: 0,
    levelVisible: 0,
    levelDraws: 0,
    lastUpdateMs: 0,
    lastSelected: 0,
  };
  /** Slot (BatchedMesh instance id) to instance id, or -1 when the slot is free. */
  slotInstance: Int32Array;
  /** The draw ranges of `data`. A BatchedMesh geometry id is a range id. */
  readonly ranges: DrawRanges;
  /** The whole-mesh levels of `data`. */
  readonly levels: LodLevels;

  private readonly selector: ClusterSelector;
  private readonly scratch: Uint32Array;
  private readonly newSlots: Int32Array;
  private readonly marks: Int32Array;
  private readonly shadowCameras = new WeakSet<Camera>();
  private _batched: BatchedMesh;
  /** Zero-copy buffers wrapping the MeshletData arrays, shared by every draw mesh. */
  private readonly sharedAttributes: Array<{ name: string; attribute: BufferAttribute }>;
  private readonly sharedIndex: BufferAttribute;
  /**
   * Disposers of replaced draw meshes. Disposal is deferred to a later task because a mesh
   * replaced during a render may still be in that render's lists (shadow passes and the main pass).
   */
  private retired: Array<() => void> = [];
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  // ---- instances
  private _count = 0;
  private instCap = 0;
  private matrices = new Float64Array(0);
  private inverses = new Float64Array(0);
  private instScale = new Float32Array(0);
  /** Bounding sphere of each instance in the space of this object, `[x, y, z, r]`. */
  private instSphere = new Float32Array(0);
  private colors = new Float32Array(0);
  private dirty = new Uint8Array(0);
  private dirtyList = new Int32Array(0);
  private dirtyCount = 0;
  private cutCount = new Int32Array(0);
  /** Per instance drawn by cluster: the draw ranges of its cut (`cutSlots` holds the slot each is drawn through). */
  private cutRanges: Uint32Array[] = [];
  private cutSlots: Int32Array[] = [];
  /** Per instance: the level it is drawn at, or -1 when it is drawn by cluster; and its position in that level. */
  private instLevel = new Int32Array(0);
  private instPos = new Int32Array(0);
  /** Per instance: the level it asks for (the coarsest its error allows), or -1 when it is drawn by cluster. */
  private instWant = new Int32Array(0);
  /** Per instance: the length of `travel` at which its choice has to be looked at again. */
  private instDue = new Float64Array(0);
  /** Instances chosen anew by the running `update`. */
  private touched = new Int32Array(0);
  private boundsDirty = true;

  // ---- the camera's way
  /**
   * How far the camera has come, in the space of this object, since everything was last chosen
   * afresh. An instance on a whole-mesh level keeps it until the camera is that much nearer to it
   * or that much further away, which takes at least as much way; so far instances are not looked
   * at again for a long time, and a camera that turns or creeps looks at none.
   */
  private travel = 0;
  /** The least of `instDue`. */
  private minDue = 0;
  /** What depends on the world matrix alone: kept until it changes. */
  private readonly worldKey = new Float64Array(16).fill(Number.NaN);
  private readonly worldInverse = new Matrix4();
  private worldScale = 1;
  private worldIdentity = true;

  // ---- levels
  private readonly useLevels: boolean;
  /** The renderer in use cannot draw by level (see `WEBGPU_LEVELS_REVISION`). */
  private levelsBlocked = false;
  private readonly buckets: LevelBucket[] = [];
  /** Indices the levels with an index buffer of their own may take in total, and have taken. */
  private readonly levelIndexBudget: number;
  private levelIndexUsed = 0;
  /** Instances asking for each level, and the level those are drawn at (see `drawCost`). */
  private readonly wantCount: Int32Array;
  private readonly levelMap: Int32Array;
  private wantDirty = false;
  /**
   * The view the levels were last culled for: the camera's projection and inverse world matrix and
   * the world matrix of this object.
   */
  private readonly cullKey = new Float64Array(48);
  private cullValid = false;
  /** An instance joined a level or moved since the last culling pass. */
  private cullStale = false;
  /** Levels with instances on them. Their draw meshes are the ones in the scene. */
  private readonly live: LevelBucket[] = [];
  /** Levels whose draw mesh is hidden from the scene walk in progress: none of their instances is in view. */
  private readonly unseen: LevelBucket[] = [];
  /** In the scene, behind the draw meshes, only while there are `unseen` levels to bring back. */
  private readonly afterDrawMeshes: AfterDrawMeshes;
  /** The camera of the last update, when that was the renderer's own call for the render it was starting. */
  private autoCamera: Camera | null = null;

  // ---- slots
  private freeSlots = new Int32Array(0);
  private freeCount = 0;
  /** Slots in use: ranges drawn by cluster, summed over the instances. */
  private slotDraws = 0;

  // ---- colors
  private _debugColors: DebugColorMode = "none";
  private colorsInUse = false;

  // ---- update signature / per-update view state
  private sigValid = false;
  private sigProj = 0;
  private sigHeight = 0;
  private sigThreshold = 0;
  private sigOrtho = false;
  private sigNear = 0;
  private sigScale = 0;
  private sigLevels = false;
  private sigLevelDistance = 0;
  private sigDrawCost = Number.NaN;
  private vx = 0;
  private vy = 0;
  private vz = 0;
  private vProj = 1;
  private vHeight = 1;
  private vNear = DEFAULT_NEAR;
  private vOrtho = false;
  private vWorldScale = 1;
  private vLevels = false;
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
    if (data.attributes.length === 0) {
      throw new Error("three-nanite: this MeshletData holds no vertex attributes (hierarchy only)");
    }
    this.data = data;
    this.errorThreshold = options.errorThreshold ?? 1;
    this.levelDistance = options.levelDistance ?? 3;
    this.cullInstances = options.cullInstances ?? true;
    this.drawCost = options.drawCost ?? 4096;

    const mergeClusters = options.mergeClusters ?? true;
    this.selector = new ClusterSelector(data);
    this.ranges = computeDrawRanges(data, mergeClusters);
    this.levels = computeLodLevels(data);
    this.wantCount = new Int32Array(this.levels.count);
    this.levelMap = new Int32Array(this.levels.count);
    for (let j = 0; j < this.levels.count; j++) this.levelMap[j] = j;
    // drawing cluster by cluster is what `mergeClusters: false` asks for
    this.useLevels = (options.instancedLevels ?? true) && mergeClusters;
    for (let j = 0; j < this.levels.count; j++) {
      this.buckets.push({
        level: j,
        usable: null,
        geometry: null,
        ownsIndex: false,
        mesh: null,
        ids: EMPTY_I32,
        capacity: 0,
        count: 0,
        visible: 0,
        inShadow: false,
      });
    }
    // levels that are not one run of the shared index buffer get a copy of their indices: at most
    // as much again as the hierarchy's own
    this.levelIndexBudget = data.indices.length;
    this.scratch = new Uint32Array(data.clusterCount);
    this.newSlots = new Int32Array(this.ranges.count);
    this.marks = new Int32Array(this.ranges.count);

    this.sharedAttributes = data.attributes.map((a) => ({
      name: a.name,
      // `TypedArray` is a subset of the array types BufferAttribute accepts.
      attribute: new BufferAttribute(a.array as TypedArray, a.itemSize, a.normalized),
    }));
    // With at most 65535 vertices a 16-bit index buffer halves the index memory.
    this.sharedIndex = new BufferAttribute(data.vertexCount <= 65535 ? Uint16Array.from(data.indices) : data.indices, 1);

    const maxInstances = Math.max(1, Math.floor(options.maxInstances ?? 16));
    const coarsest = rangesOfCut(this.ranges, this.scratch, this.selector.selectByError(this.scratch, Infinity));
    const slotCapacity = Math.max(
      coarsest,
      1,
      // with levels only the instances near the camera take slots, however many instances there are
      Math.floor(options.initialDrawCapacity ?? (this.useLevels ? Math.max(256, coarsest * 16) : Math.max(64, maxInstances * coarsest * 4))),
    );

    const batched = this.createBatched(slotCapacity, material);
    this._batched = batched;
    batched.layers = this.layers;
    // (in the scene only while it has something to draw: the renderers set up a draw for an empty one too)
    this.afterDrawMeshes = new AfterDrawMeshes(() => this.showLevels());
    this.afterDrawMeshes.layers = this.layers;
    this.afterDrawMeshes.parent = this;

    this.slotInstance = new Int32Array(slotCapacity).fill(-1);
    this.freeSlots = new Int32Array(slotCapacity);

    this.ensureInstanceCapacity(maxInstances);
    this.count = 1;
  }

  // =========================================================================
  // public surface
  // =========================================================================

  /**
   * The draw mesh of the instances drawn by cluster. It is REPLACED when draw capacity grows: do not
   * hold on to it, and set `initialDrawCapacity` high enough to avoid growth. See also {@link onRebuild}.
   */
  get batched(): BatchedMesh {
    return this._batched;
  }

  /** Number of draw slots the current draw mesh can hold (`batched.maxInstanceCount`). */
  get drawCapacity(): number {
    return this._batched.maxInstanceCount;
  }

  /** Material used by every draw mesh. */
  get material(): Material {
    return this._batched.material;
  }
  set material(value: Material) {
    this._batched.material = value;
    for (const b of this.buckets) if (b.mesh !== null) b.mesh.material = value;
  }

  /** The instanced draw meshes currently in existence, one per level that has been used. They are replaced when they grow. */
  get levelMeshes(): InstancedMesh[] {
    const out: InstancedMesh[] = [];
    for (const b of this.buckets) if (b.mesh !== null) out.push(b.mesh);
    return out;
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
        this.instSphere.set(this.data.boundingSphere, i * 4);
        this.colors[i * 3] = this.colors[i * 3 + 1] = this.colors[i * 3 + 2] = 1;
        this.cutCount[i] = 0;
        this.instLevel[i] = -1;
        this.instWant[i] = -1;
        this.markDirty(i);
      }
    }
    this._count = n;
    this.stats.instances = n;
    this.boundsDirty = true;
    this.refreshDrawStats();
  }

  /** `"cluster"`: random color per draw range (per cluster with `mergeClusters: false`). `"lod"`: color by DAG level. `"none"`: instance color. */
  get debugColors(): DebugColorMode {
    return this._debugColors;
  }
  set debugColors(mode: DebugColorMode) {
    if (mode === this._debugColors) return;
    this._debugColors = mode;
    if (mode !== "none") this.colorsInUse = true;
    if (this.colorsInUse) this.recolorAll();
  }

  /** Sets the matrix of instance `i` (local to this object). Takes effect immediately on what is drawn. */
  setMatrixAt(i: number, matrix: Matrix4): void {
    if (!(i >= 0 && i < this._count)) throw new RangeError(`MeshletMesh.setMatrixAt: instance ${i} out of range (count ${this._count})`);
    matrix.toArray(this.matrices, i * 16);
    _inv.copy(matrix).invert().toArray(this.inverses, i * 16);
    const scale = matrix.getMaxScaleOnAxis();
    this.instScale[i] = scale;
    const s = this.data.boundingSphere;
    _v3.set(s[0], s[1], s[2]).applyMatrix4(matrix);
    const o = i * 4;
    this.instSphere[o] = _v3.x;
    this.instSphere[o + 1] = _v3.y;
    this.instSphere[o + 2] = _v3.z;
    this.instSphere[o + 3] = s[3] * scale;
    // grown at once so the draw meshes are never culled wrongly, tightened by the next update
    this.boundingSphere.union(_sphere.set(_v3, s[3] * scale));
    this.boundsDirty = true;
    this.markDirty(i);

    const level = this.instLevel[i];
    if (level >= 0) {
      this.writeInstance(this.buckets[level], this.instPos[i], i);
      this.cullStale = true;
    } else {
      const n = this.cutCount[i];
      const slots = this.cutSlots[i];
      for (let j = 0; j < n; j++) this._batched.setMatrixAt(slots[j], matrix);
    }
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
      const level = this.instLevel[i];
      if (level >= 0) {
        this.writeInstance(this.buckets[level], this.instPos[i], i);
        return;
      }
      const n = this.cutCount[i];
      const slots = this.cutSlots[i];
      const ranges = this.cutRanges[i];
      for (let j = 0; j < n; j++) this.applySlotColor(slots[j], ranges[j], i);
    }
  }

  /** Reads the color of instance `i` into `target`. */
  getColorAt(i: number, target: Color): Color {
    return target.setRGB(this.colors[i * 3], this.colors[i * 3 + 1], this.colors[i * 3 + 2]);
  }

  /** The whole-mesh level instance `i` is drawn at (an index into {@link levels}), or -1 when it is drawn by cluster. */
  getLevelAt(i: number): number {
    return i >= 0 && i < this._count ? this.instLevel[i] : -1;
  }

  /** Triangles instance `i` is currently drawn with (before frustum culling). */
  getTriangleCountAt(i: number): number {
    if (!(i >= 0 && i < this._count)) return 0;
    const level = this.instLevel[i];
    if (level >= 0) return this.levels.triangles[level];
    const ranges = this.cutRanges[i];
    let indices = 0;
    for (let j = 0; j < this.cutCount[i]; j++) indices += this.ranges.indexCount[ranges[j]];
    return indices / 3;
  }

  /**
   * Recomputes what every instance is drawn with for the given camera.
   *
   * Both renderers call this by themselves with the camera they are about to render (see
   * {@link isLOD}; set {@link autoUpdate} to `false` to stop them). Called that way, without a
   * height, it is ignored for the cameras of shadow passes, {@link lodCamera} replaces the camera
   * for the choice of detail, and the height is the one the camera was last seen rendering to.
   *
   * Cheap when nothing relevant changed. The level of detail depends only on the camera position,
   * projection, viewport height and threshold, so turning the camera only re-tests which instanced
   * instances are in view.
   *
   * @param camera         Camera to compute LOD for, and whose view the instanced draws are culled to.
   * @param viewportHeight Height in pixels of the render target the camera renders to.
   * @returns `true` if the set of drawn clusters or an instance matrix changed.
   */
  update(camera: Camera, viewportHeight?: number): boolean {
    let lodCamera = camera;
    let height: number;
    if (viewportHeight === undefined) {
      // a light's camera never decides the detail: shadows are cast by what the viewer sees
      // (the camera of the last frame's main pass is not one, which is all there is to ask most frames)
      if (camera !== this.autoCamera && (this.shadowCameras.has(camera) || this.inShadowPass())) return false;
      if (this.lodCamera !== null) lodCamera = this.lodCamera;
      height =
        this.viewportHeight ??
        (camera === _heightCamera ? _height : _heights.get(camera)) ??
        (_height > 0 ? _height : defaultViewportHeight());
    } else {
      height = viewportHeight;
    }

    const automatic = viewportHeight === undefined;
    if (automatic) {
      // the renderer has brought the scene and its camera up to date; a camera of our own it has not
      if (camera !== lodCamera) lodCamera.updateWorldMatrix(true, false);
    } else {
      this.updateWorldMatrix(true, false);
      lodCamera.updateWorldMatrix(true, false);
      if (camera !== lodCamera) camera.updateWorldMatrix(true, false);
    }

    if (this.unseen.length > 0) this.showLevels(); // (left by a scene walk that did not finish)
    _notedCamera = null;
    this.autoCamera = automatic ? camera : null;

    const ortho = (lodCamera as { isOrthographicCamera?: boolean }).isOrthographicCamera === true;
    const proj = lodCamera.projectionMatrix.elements[5];
    // `near` exists on Perspective/OrthographicCamera but not on the Camera base type.
    const near = (lodCamera as { near?: number }).near ?? DEFAULT_NEAR;
    const useLevels = this.useLevels && !this.levelsBlocked && this._debugColors !== "cluster";

    const world = this.matrixWorld.elements;
    const kept = this.worldKey;
    for (let j = 0; j < 16; j++) {
      if (kept[j] !== world[j]) {
        kept.set(world);
        this.worldInverse.copy(this.matrixWorld).invert();
        this.worldScale = this.matrixWorld.getMaxScaleOnAxis();
        let identity = true;
        for (let e = 0; e < 16; e++) if (world[e] !== (e % 5 === 0 ? 1 : 0)) identity = false;
        this.worldIdentity = identity;
        break;
      }
    }
    const worldScale = this.worldScale;

    let cx = 0;
    let cy = 0;
    let cz = 0;
    if (!ortho) {
      _v3.setFromMatrixPosition(lodCamera.matrixWorld).applyMatrix4(this.worldInverse);
      cx = _v3.x;
      cy = _v3.y;
      cz = _v3.z;
    }

    // everything the choice of detail depends on, except where the camera is
    const sameSettings =
      this.sigValid &&
      proj === this.sigProj &&
      height === this.sigHeight &&
      this.errorThreshold === this.sigThreshold &&
      ortho === this.sigOrtho &&
      near === this.sigNear &&
      worldScale === this.sigScale &&
      useLevels === this.sigLevels &&
      this.levelDistance === this.sigLevelDistance;
    if (this.drawCost !== this.sigDrawCost) {
      this.sigDrawCost = this.drawCost;
      this.wantDirty = true;
    }

    if (sameSettings) {
      const dx = cx - this.vx;
      const dy = cy - this.vy;
      const dz = cz - this.vz;
      if (dx !== 0 || dy !== 0 || dz !== 0) this.travel += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    this.vx = cx;
    this.vy = cy;
    this.vz = cz;
    // the camera has come far enough for some instance to need another look
    const due = !sameSettings || this.travel > this.minDue;

    let changed = false;
    let worked = false;
    let ms = 0;

    if (due || this.dirtyCount > 0 || this.wantDirty) {
      const t0 = performance.now();
      worked = true;
      this.vProj = proj;
      this.vHeight = height;
      this.vNear = near;
      this.vOrtho = ortho;
      this.vWorldScale = worldScale;
      this.vLevels = useLevels;

      const dirty = this.dirty;
      const list = this.dirtyList;
      const count = this._count;
      let selected = 0;
      if (this.dirtyCount > 0) changed = true; // an instance matrix changed

      if (!sameSettings) {
        // everything is chosen afresh
        this.travel = 0;
        this.minDue = Infinity;
        this.pendingTotal = this.pendingInstances = Math.max(1, count);
        for (let d = 0; d < this.dirtyCount; d++) dirty[list[d]] = 0;
        for (let i = 0; i < count; i++) {
          this.pendingInstances--;
          if (this.reselect(i)) changed = true;
        }
        selected = count;
        this.sigValid = true;
        this.sigProj = proj;
        this.sigHeight = height;
        this.sigThreshold = this.errorThreshold;
        this.sigOrtho = ortho;
        this.sigNear = near;
        this.sigScale = worldScale;
        this.sigLevels = useLevels;
        this.sigLevelDistance = this.levelDistance;
        if (this.placeLevels(-1)) changed = true;
      } else {
        this.pendingTotal = this.pendingInstances = 1;
        const touched = this.touched;
        for (let d = 0; d < this.dirtyCount; d++) {
          const i = list[d];
          dirty[i] = 0;
          if (i < count) {
            if (this.reselect(i)) changed = true;
            touched[selected++] = i;
          }
        }
        if (due) {
          const travel = this.travel;
          const dues = this.instDue;
          let least = Infinity;
          for (let i = 0; i < count; i++) {
            if (dues[i] < travel) {
              if (this.reselect(i)) changed = true;
              touched[selected++] = i;
            }
            if (dues[i] < least) least = dues[i];
          }
          this.minDue = least;
        }
        if (this.placeLevels(selected)) changed = true;
      }
      this.dirtyCount = 0;
      this.stats.lastSelected = selected;
      ms = performance.now() - t0;
    }

    if (this.boundsDirty) this.computeBounds();
    const culling = this.cullLevels(camera);
    if (culling >= 0) {
      worked = true;
      ms += culling;
    }
    // the renderer is walking the scene with this very camera, and reaches the draw meshes next
    if (automatic) this.hideUnseenLevels();

    if (worked) {
      this.refreshDrawStats();
      this.stats.lastUpdateMs = ms;
    }
    return changed;
  }

  /**
   * As `Object3D.updateMatrixWorld`, but without the walk over the draw meshes, which use this
   * object's world matrix itself: in a scene of many objects that walk is most of what it costs.
   */
  override updateMatrixWorld(force?: boolean): void {
    if (this.matrixAutoUpdate) this.updateMatrix();
    if (this.matrixWorldNeedsUpdate || force) {
      if (this.matrixWorldAutoUpdate === true) {
        if (this.parent === null) this.matrixWorld.copy(this.matrix);
        else this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
      }
      this.matrixWorldNeedsUpdate = false;
      force = true;
    }
    // whatever else was added to this object
    const children = this.children;
    if (children.length !== this.live.length + (this.slotDraws > 0 ? 1 : 0)) {
      for (let i = 0; i < children.length; i++) children[i].updateMatrixWorld(force);
    }
  }

  /** Releases the GPU resources of the draw meshes, including the shared vertex/index buffers. The material is not disposed. */
  override dispose(): void {
    this.disposeRetired();
    for (const b of this.buckets) {
      if (b.mesh !== null) {
        this.remove(b.mesh);
        b.mesh.dispose();
        b.mesh = null;
      }
      if (b.geometry !== null) {
        this.disposeLevelGeometry(b);
        b.geometry = null;
      }
    }
    this._batched.dispose();
  }

  // =========================================================================
  // draw mesh (BatchedMesh) creation and growth
  // =========================================================================

  /**
   * Creates a BatchedMesh whose geometry ids equal draw range ids and whose index layout equals `data.indices`.
   *
   * Every draw range is added as an index-only geometry (zero vertices); afterwards the BatchedMesh's own
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
    const ranges = this.ranges;
    let maxCount = 0;
    for (let c = 0; c < ranges.count; c++) maxCount = Math.max(maxCount, ranges.indexCount[c]);
    const dummyIndex = new Uint32Array(maxCount);
    temp.boundingSphere = _sphere;
    temp.boundingBox = _box;

    const bounds = ranges.bounds;
    for (let c = 0; c < ranges.count; c++) {
      const count = ranges.indexCount[c];
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
        range.indexStart !== ranges.indexStart[c] ||
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
    // WebGLRenderer before r181 asks a BatchedMesh for `colorTexture`, which it does not have, takes
    // the answer to mean its colours have changed, and works out its program again every frame
    (bm as unknown as { colorTexture: unknown }).colorTexture = null;
    this.place(bm);
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

    if (old.parent === this) {
      this.remove(old);
      this.add(next);
    }
    this._batched = next;
    (next as unknown as { colorTexture: unknown; _colorsTexture: unknown }).colorTexture = (next as unknown as { _colorsTexture: unknown })._colorsTexture;

    // The old mesh may still be drawn by the render in progress (it shows the previous cut for that frame).
    this.retire(() => this.disposeBatchedKeepingBuffers(old));
    this.onRebuild?.(next);
  }

  /** Disposes a replaced draw mesh without touching the shared geometry buffers. */
  private disposeBatchedKeepingBuffers(old: BatchedMesh): void {
    this.disposeSharingGeometry(old.geometry, true);
    // what is left to release are the data textures of the mesh
    old.geometry = new BufferGeometry();
    old.dispose();
  }

  /**
   * Disposes a geometry that uses the shared vertex buffers (and, with `sharesIndex`, the shared index
   * buffer) without freeing them: the renderers free the buffers of the attributes a disposed
   * geometry holds, so the shared ones are detached first.
   */
  private disposeSharingGeometry(geometry: BufferGeometry, sharesIndex: boolean): void {
    for (const { name } of this.sharedAttributes) geometry.deleteAttribute(name);
    if (sharesIndex) geometry.setIndex(null);
    try {
      geometry.dispose();
    } catch {
      // WebGPURenderer of three r171 to r180 looks the detached attributes up again while it
      // disposes and fails on the first one. Nothing of the geometry is left to free by then.
    }
  }

  private retire(dispose: () => void): void {
    this.retired.push(dispose);
    if (this.retireTimer === null) this.retireTimer = setTimeout(() => this.disposeRetired(), 0);
  }

  private disposeRetired(): void {
    if (this.retireTimer !== null) {
      clearTimeout(this.retireTimer);
      this.retireTimer = null;
    }
    for (const dispose of this.retired) dispose();
    this.retired.length = 0;
  }

  private installHooks(batched: BatchedMesh): void {
    const shadowCameras = this.shadowCameras;

    // The detail itself is chosen in `update`, which the renderers call while collecting the scene.
    // These hooks only learn the things `update` cannot see: how tall the target is, and which
    // cameras belong to lights (WebGPURenderer renders a shadow map as a render of its own, with
    // the light's camera, and calls onBeforeRender in it too).
    batched.onBeforeRender = (renderer, scene, camera, geometry, material, group) => {
      if (LEVELS_NEED_CHECK && (renderer as LodRenderer).isWebGPURenderer === true) this.levelsBlocked = true;
      if (_notedCamera !== camera && !shadowCameras.has(camera)) noteViewport(renderer, camera);
      BatchedMesh.prototype.onBeforeRender.call(batched, renderer, scene, camera, geometry, material, group);
    };
    batched.onBeforeShadow = (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) => {
      shadowCameras.add(shadowCamera);
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

  /** WebGPURenderer renders shadow maps as renders of the scene with a shadow material as override. */
  private inShadowPass(): boolean {
    let root: Object3D = this;
    while (root.parent !== null) root = root.parent;
    const override = (root as { overrideMaterial?: { isShadowPassMaterial?: boolean; isShadowNodeMaterial?: boolean } | null })
      .overrideMaterial;
    // (the flag was called isShadowNodeMaterial before r176)
    return override != null && (override.isShadowPassMaterial === true || override.isShadowNodeMaterial === true);
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
    this.instSphere = growFloat32(this.instSphere, cap * 4);
    this.colors = growFloat32(this.colors, cap * 3, 1);
    this.dirty = growUint8(this.dirty, cap);
    this.dirtyList = growInt32(this.dirtyList, cap);
    this.cutCount = growInt32(this.cutCount, cap);
    this.instLevel = growInt32(this.instLevel, cap, -1);
    this.instPos = growInt32(this.instPos, cap);
    this.instWant = growInt32(this.instWant, cap, -1);
    this.instDue = growFloat64(this.instDue, cap);
    this.touched = new Int32Array(cap);
    for (let i = old; i < cap; i++) {
      this.cutRanges.push(EMPTY_U32);
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
    const level = this.instLevel[i];
    if (level >= 0) this.leaveLevel(i);
    else this.releaseCut(i);
    const asked = this.instWant[i];
    if (asked >= 0) {
      this.wantCount[asked]--;
      this.instWant[i] = -1;
      this.wantDirty = true;
    }
  }

  /**
   * Makes a draw mesh be where this object is, for good: it uses this object's world matrix
   * itself. So a mesh made while a renderer is walking the scene is in place in that very frame,
   * and the scene's matrix update has nothing to do for it.
   */
  private place(mesh: Object3D): void {
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    mesh.matrixWorld = this.matrixWorld;
  }

  /**
   * Takes the levels with no instance in view out of the scene walk in progress, which is the
   * renderer collecting what the main pass draws. `afterDrawMeshes` goes into the scene behind
   * them, so that they are back (`showLevels`) for whatever comes after that walk: the shadow
   * passes of the frame, which draw every instance.
   */
  private hideUnseenLevels(): void {
    if (!this.cullValid || this.cullStale) return;
    const live = this.live;
    const unseen = this.unseen;
    for (let j = 0; j < live.length; j++) {
      const b = live[j];
      if (b.visible === 0) {
        b.mesh!.boundingSphere = NEVER_IN_VIEW;
        unseen.push(b);
      }
    }
    if (unseen.length > 0) this.children.push(this.afterDrawMeshes);
  }

  private showLevels(): void {
    const unseen = this.unseen;
    for (let j = 0; j < unseen.length; j++) {
      const mesh = unseen[j].mesh;
      if (mesh !== null) mesh.boundingSphere = this.boundingSphere;
    }
    unseen.length = 0;
    // (the renderer's walk is at this child, the last one, when it calls this)
    const children = this.children;
    const at = children.lastIndexOf(this.afterDrawMeshes);
    if (at >= 0) children.splice(at, 1);
  }

  /** Frees the slots of an instance drawn by cluster. */
  private releaseCut(i: number): void {
    const n = this.cutCount[i];
    const slots = this.cutSlots[i];
    const ranges = this.cutRanges[i];
    for (let j = 0; j < n; j++) this.releaseSlot(slots[j], ranges[j]);
    this.cutCount[i] = 0;
  }

  /** Recomputes {@link boundingSphere}: a sphere around the box of the instance spheres. */
  private computeBounds(): void {
    this.boundsDirty = false;
    const n = this._count;
    const s = this.instSphere;
    if (n === 0) {
      this.boundingSphere.makeEmpty();
      return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const r = s[o + 3];
      if (s[o] - r < minX) minX = s[o] - r;
      if (s[o] + r > maxX) maxX = s[o] + r;
      if (s[o + 1] - r < minY) minY = s[o + 1] - r;
      if (s[o + 1] + r > maxY) maxY = s[o + 1] + r;
      if (s[o + 2] - r < minZ) minZ = s[o + 2] - r;
      if (s[o + 2] + r > maxZ) maxZ = s[o + 2] + r;
    }
    this.boundingSphere.center.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    this.boundingSphere.radius = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2;
  }

  private refreshDrawStats(): void {
    const live = this.live;
    let visible = 0;
    for (let j = 0; j < live.length; j++) visible += live[j].visible;
    this.stats.levelDraws = live.length;
    this.stats.levelVisible = visible;
    this.stats.draws = this.slotDraws + live.length;
  }

  // =========================================================================
  // LOD selection
  // =========================================================================

  /** Chooses what instance `i` is drawn with for the current view state. Returns whether that changed. */
  private reselect(i: number): boolean {
    let scale = this.instScale[i] * this.vWorldScale;
    if (!(scale > MIN_SCALE)) scale = MIN_SCALE;
    const k = lodScale(this.errorThreshold, this.vHeight, this.vProj, scale, this.vOrtho);
    const near = this.vNear / scale;
    const useLevels = this.vLevels;

    // Far enough for one level throughout: no part of the mesh is nearer than its bounding sphere,
    // so the level whose error is fine at that distance is fine everywhere, and from a few radii
    // away it is also about what a walk of the hierarchy would end up with.
    let level = -1;
    // how far the camera may come before this choice can be another one (see `travel`)
    let slack = 0;
    if (useLevels) {
      if (this.vOrtho) {
        level = levelForError(this.levels, k);
        slack = Infinity;
      } else {
        const s = this.instSphere;
        const o = i * 4;
        const dx = s[o] - this.vx;
        const dy = s[o + 1] - this.vy;
        const dz = s[o + 2] - this.vz;
        const r = s[o + 3];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
        const walked = this.levelDistance * r;
        if (d >= walked) {
          // in mesh units, as a lower bound when the instance is scaled unevenly
          const unit = this.instScale[i] > MIN_SCALE ? this.instScale[i] : 1;
          const local = d / unit;
          const levels = this.levels;
          level = levelForError(levels, k * (local > near ? local : near));
          // the level holds while the distance stays between the two at which its neighbours take over
          const finer = levels.errors[level] / k;
          const nearest = finer > near && finer * unit > walked ? finer * unit : walked;
          const furthest = level + 1 < levels.count ? (levels.errors[level + 1] / k) * unit : Infinity;
          slack = Math.min(d - nearest, furthest - d);
        }
      }
      if (level >= 0 && !this.levelUsable(level)) {
        level = -1;
        slack = 0;
      }
    }
    // (a little less than the slack: the distances are rounded)
    const due = slack > 0 ? this.travel + slack * 0.999 : this.travel;
    this.instDue[i] = due;
    if (due < this.minDue) this.minDue = due;

    let n = 0;
    if (level < 0) {
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
      const scratch = this.scratch;
      n = this.selector.select(scratch, x, y, z, k, near, this.vOrtho);
      if (useLevels) {
        // near, but its cut is a whole-mesh level all the same (always so for small meshes)
        level = levelOfCut(this.levels, this.selector.lastWithin, this.selector.lastOver);
        if (level >= 0 && !this.levelUsable(level)) level = -1;
      }
      // the cut in clusters, then in draw ranges: drawing by cluster works in ranges
      if (level < 0) n = rangesOfCut(this.ranges, scratch, n);
    }

    const asked = this.instWant[i];
    if (asked !== level) {
      if (asked >= 0) this.wantCount[asked]--;
      if (level >= 0) this.wantCount[level]++;
      this.instWant[i] = level;
      this.wantDirty = true;
    }
    // an instance on a level is placed by `placeLevels`, once every instance has asked for its own
    return level >= 0 ? false : this.applyCut(i, n);
  }

  /**
   * Draws every instance that asks for a level at the level chosen for it. `n`: how many
   * instances, listed in `touched`, may have asked for another level; -1 when any may have.
   * Returns whether an instance changed level.
   */
  private placeLevels(n: number): boolean {
    let remapped = false;
    if (this.wantDirty) {
      this.wantDirty = false;
      remapped = this.chooseLevels();
    }
    const want = this.instWant;
    const at = this.instLevel;
    const map = this.levelMap;
    let changed = false;
    if (n < 0 || remapped) {
      const count = this._count;
      for (let i = 0; i < count; i++) {
        const w = want[i];
        if (w >= 0 && at[i] !== map[w] && this.enterLevel(i, map[w])) changed = true;
      }
    } else {
      const touched = this.touched;
      for (let t = 0; t < n; t++) {
        const i = touched[t];
        const w = want[i];
        if (w >= 0 && at[i] !== map[w] && this.enterLevel(i, map[w])) changed = true;
      }
    }
    return changed;
  }

  /**
   * Decides which of the levels asked for are drawn, writing the level each one is drawn at to
   * `levelMap`. Returns whether that changed anything.
   *
   * A level in use costs a draw, `drawCost` triangles' worth, whatever the number of instances on
   * it; instances may always be drawn finer than they ask. So the levels asked for, in order, are
   * cut into runs that are each drawn at their finest level, with the cuts where the total of
   * draws and triangles is least.
   */
  private chooseLevels(): boolean {
    const counts = this.wantCount;
    const map = this.levelMap;
    const triangles = this.levels.triangles;
    const n = counts.length;
    if (_wanted.length < n) {
      _wanted = new Int32Array(n);
      _from = new Int32Array(n + 1);
      _best = new Float64Array(n + 1);
    }
    let k = 0;
    for (let j = 0; j < n; j++) if (counts[j] > 0) _wanted[k++] = j;

    let changed = false;
    const cost = this.drawCost;
    if (!(cost > 0) || k <= 1) {
      for (let a = 0; a < k; a++) {
        const j = _wanted[a];
        if (map[j] !== j) {
          map[j] = j;
          changed = true;
        }
      }
      return changed;
    }

    // _best[b]: the least cost of drawing the first b levels asked for; _from[b]: where its last run starts
    _best[0] = 0;
    for (let b = 1; b <= k; b++) {
      let least = Infinity;
      let start = b - 1;
      let instances = 0;
      for (let a = b - 1; a >= 0; a--) {
        instances += counts[_wanted[a]];
        const c = _best[a] + cost + triangles[_wanted[a]] * instances;
        if (c < least) {
          least = c;
          start = a;
        }
      }
      _best[b] = least;
      _from[b] = start;
    }
    for (let b = k; b > 0; ) {
      const a = _from[b];
      const level = _wanted[a];
      for (let j = a; j < b; j++) {
        const w = _wanted[j];
        if (map[w] !== level) {
          map[w] = level;
          changed = true;
        }
      }
      b = a;
    }
    return changed;
  }

  /** Draws instance `i` by cluster with the `n` draw ranges in `scratch`, diffing against its old cut. */
  private applyCut(i: number, n: number): boolean {
    let changed = false;
    if (this.instLevel[i] >= 0) {
      this.leaveLevel(i);
      changed = true;
    }
    const scratch = this.scratch;
    const oldN = this.cutCount[i];
    // Worst case every drawn range of this instance is replaced: make sure the slots fit before diffing.
    const needed = this.slotDraws - oldN + n;
    if (this._batched.maxInstanceCount < needed) {
      // Rebuilding is not free, so aim past this instance: assume the instances still to be
      // selected in this update need about as much as the ones done so far.
      const done = Math.max(1, this.pendingTotal - this.pendingInstances);
      // Capped, because the first instances selected may be the closest (most detailed) ones.
      this.grow(Math.max(needed, Math.min(needed * 4, Math.ceil((needed * this.pendingTotal) / done))));
    }

    const marks = this.marks;
    const newSlots = this.newSlots;
    const oldRanges = this.cutRanges[i];
    const oldSlots = this.cutSlots[i];

    for (let j = 0; j < n; j++) marks[scratch[j]] = 1;
    // releases first, so freed slots are recycled by the acquires below
    for (let j = 0; j < oldN; j++) {
      const c = oldRanges[j];
      if (marks[c] === 1) {
        marks[c] = oldSlots[j] + 2; // kept
      } else {
        this.releaseSlot(oldSlots[j], c);
        changed = true;
      }
    }
    _m.fromArray(this.matrices, i * 16);
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
      let ranges = this.cutRanges[i];
      let slots = this.cutSlots[i];
      if (ranges.length < n) {
        const cap = Math.max(n, ranges.length * 2);
        ranges = this.cutRanges[i] = new Uint32Array(cap);
        slots = this.cutSlots[i] = new Int32Array(cap);
      }
      for (let j = 0; j < n; j++) {
        ranges[j] = scratch[j];
        slots[j] = newSlots[j];
      }
      this.cutCount[i] = n;
    }
    return changed;
  }

  /** Takes a slot from the free list (or creates one) and points it at draw range `c` of instance `i`. Uses `_m` as the instance matrix. */
  private acquireSlot(c: number, i: number): number {
    let slot: number;
    if (this.freeCount > 0) {
      slot = this.freeSlots[--this.freeCount];
      const batched = this._batched;
      batched.setGeometryIdAt(slot, c); // geometry id === draw range id
      batched.setMatrixAt(slot, _m);
      if (this.colorsInUse) this.applySlotColor(slot, c, i);
      batched.setVisibleAt(slot, true);
    } else {
      // `applyCut` normally grows ahead of time; this is the safety net. Never cache the draw mesh across this call.
      if (this._batched.instanceCount >= this._batched.maxInstanceCount) this.grow(0);
      const batched = this._batched;
      slot = batched.addInstance(c);
      batched.setMatrixAt(slot, _m);
      if (this.colorsInUse) this.applySlotColor(slot, c, i);
      this.stats.slots = batched.instanceCount;
    }
    this.slotInstance[slot] = i;
    if (this.slotDraws++ === 0) this.add(this._batched);
    this.stats.clusters += this.ranges.clusterCount[c];
    this.stats.triangles += this.ranges.indexCount[c] / 3;
    return slot;
  }

  private releaseSlot(slot: number, c: number): void {
    this._batched.setVisibleAt(slot, false);
    this.slotInstance[slot] = -1;
    this.freeSlots[this.freeCount++] = slot;
    if (--this.slotDraws === 0) this.remove(this._batched);
    this.stats.clusters -= this.ranges.clusterCount[c];
    this.stats.triangles -= this.ranges.indexCount[c] / 3;
  }

  // =========================================================================
  // levels (instanced draws)
  // =========================================================================

  /**
   * Whether a level can be drawn as one index range: either it is one run of the shared index buffer
   * (coarse levels usually are), or there is room for a copy of its indices.
   */
  private levelUsable(level: number): boolean {
    const b = this.buckets[level];
    if (b.usable === null) {
      const levels = this.levels;
      if (levels.runOffset[level + 1] - levels.runOffset[level] === 1) {
        b.usable = true;
      } else {
        const need = levels.triangles[level] * 3;
        b.usable = this.levelIndexUsed + need <= this.levelIndexBudget;
        if (b.usable) {
          this.levelIndexUsed += need;
          b.ownsIndex = true;
        }
      }
    }
    return b.usable;
  }

  /** Geometry of a level: the shared vertex buffers with either a range of the shared indices or indices of its own. */
  private createLevelGeometry(b: LevelBucket): BufferGeometry {
    const { data, levels } = this;
    const geometry = new BufferGeometry();
    for (const { name, attribute } of this.sharedAttributes) geometry.setAttribute(name, attribute);
    if (b.ownsIndex) {
      const n = levels.triangles[b.level] * 3;
      const indices = data.vertexCount <= 65535 ? new Uint16Array(n) : new Uint32Array(n);
      copyLevelIndices(data, levels, b.level, indices);
      geometry.setIndex(new BufferAttribute(indices, 1));
    } else {
      const r = levels.runOffset[b.level];
      geometry.setIndex(this.sharedIndex);
      geometry.setDrawRange(levels.runStart[r], levels.runCount[r]);
    }
    const s = data.boundingSphere;
    geometry.boundingSphere = new Sphere(new Vector3(s[0], s[1], s[2]), s[3]);
    const box = data.boundingBox;
    geometry.boundingBox = new Box3(new Vector3(box[0], box[1], box[2]), new Vector3(box[3], box[4], box[5]));
    return geometry;
  }

  /** Disposes the geometry of a level without freeing the buffers it shares with the other draw meshes. */
  private disposeLevelGeometry(b: LevelBucket): void {
    if (b.geometry !== null) this.disposeSharingGeometry(b.geometry, !b.ownsIndex);
  }

  /**
   * Gives a level a draw mesh for at least `min` instances. An InstancedMesh cannot grow in place
   * (the renderers bind its buffers when they first draw it), so a larger one replaces it.
   */
  private resizeBucket(b: LevelBucket, min: number): void {
    let capacity = Math.max(16, b.capacity);
    while (capacity < min) capacity *= 2;
    b.geometry ??= this.createLevelGeometry(b);

    const source = this._batched;
    // The instance buffers keep the default (static) usage on purpose: WebGPURenderer takes a dynamic
    // one to mean "upload every frame, changed or not", and uploads a static one whenever its version
    // changes, which is what is wanted here. WebGLRenderer treats both alike.
    const mesh = new InstancedMesh(b.geometry, source.material, capacity);
    if (this.colorsInUse) mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(capacity * 3).fill(1), 3);
    mesh.boundingSphere = this.boundingSphere;
    mesh.layers = this.layers;
    mesh.castShadow = source.castShadow;
    mesh.receiveShadow = source.receiveShadow;
    mesh.renderOrder = source.renderOrder;
    mesh.customDepthMaterial = source.customDepthMaterial;
    mesh.customDistanceMaterial = source.customDistanceMaterial;
    this.place(mesh);
    this.installLevelHooks(b, mesh);

    if (b.ids.length < capacity) b.ids = growInt32(b.ids, capacity);
    const old = b.mesh;
    b.mesh = mesh;
    b.capacity = capacity;
    for (let p = 0; p < b.count; p++) this.writeInstance(b, p, b.ids[p]);
    mesh.count = b.count;

    if (old !== null) {
      this.remove(old);
      this.retire(() => old.dispose());
    }
    // a level is in the scene while it has instances
    if (b.count > 0) this.add(mesh);
  }

  private installLevelHooks(b: LevelBucket, mesh: InstancedMesh): void {
    const shadowCameras = this.shadowCameras;

    // `count` is what a pass draws: the instances in view for the main pass, all of them for
    // shadows (what is outside the view still casts shadows into it) and for everything else.
    mesh.onBeforeRender = (renderer, _scene, camera) => {
      if (b.inShadow || (camera !== this.autoCamera && shadowCameras.has(camera))) return;
      if (LEVELS_NEED_CHECK && (renderer as LodRenderer).isWebGPURenderer === true) this.levelsBlocked = true;
      if (_notedCamera !== camera) noteViewport(renderer, camera);
      mesh.count = this.culledFor(camera) ? b.visible : b.count;
    };
    mesh.onAfterRender = () => {
      if (!b.inShadow) mesh.count = b.count;
    };
    mesh.onBeforeShadow = (_renderer, _object, _camera, shadowCamera) => {
      shadowCameras.add(shadowCamera);
      b.inShadow = true;
      mesh.count = b.count;
    };
    mesh.onAfterShadow = () => {
      b.inShadow = false;
    };

    const proto = InstancedMesh.prototype.raycast;
    mesh.raycast = (raycaster: Raycaster, intersects: Intersection[]) => {
      if (this.boundsDirty) this.computeBounds();
      const first = intersects.length;
      proto.call(mesh, raycaster, intersects);
      for (let i = first; i < intersects.length; i++) {
        const hit = intersects[i];
        hit.instanceId = hit.instanceId !== undefined ? b.ids[hit.instanceId] : undefined;
        hit.object = this;
      }
    };
  }

  /** Writes instance `i` (matrix and color) at position `p` of a level. */
  private writeInstance(b: LevelBucket, p: number, i: number): void {
    const mesh = b.mesh!;
    const matrix = mesh.instanceMatrix;
    const target = matrix.array;
    const source = this.matrices;
    const o = i * 16;
    const w = p * 16;
    for (let k = 0; k < 16; k++) target[w + k] = source[o + k];
    touchRange(matrix, w, 16);

    const colors = mesh.instanceColor;
    if (colors !== null) {
      this.levelColor(b.level, i).toArray(colors.array, p * 3);
      touchRange(colors, p * 3, 3);
    }
    b.ids[p] = i;
    this.instPos[i] = p;
  }

  private swapInstances(b: LevelBucket, p: number, q: number): void {
    const i = b.ids[p];
    const j = b.ids[q];
    this.writeInstance(b, p, j);
    this.writeInstance(b, q, i);
  }

  /** Draws instance `i` at whole-mesh level `level`. Returns whether that is a change. */
  private enterLevel(i: number, level: number): boolean {
    const current = this.instLevel[i];
    if (current === level) return false;
    if (current >= 0) this.leaveLevel(i);
    else this.releaseCut(i);

    const b = this.buckets[level];
    if (b.mesh === null || b.count === b.capacity) this.resizeBucket(b, b.count + 1);
    const mesh = b.mesh!;
    const p = b.count++;
    if (p === 0) {
      this.live.push(b);
      this.add(mesh);
    }
    this.writeInstance(b, p, i);
    // in view until a culling pass says otherwise: the instances in view are kept in front
    if (p !== b.visible) this.swapInstances(b, p, b.visible);
    b.visible++;
    mesh.count = b.count;
    this.instLevel[i] = level;
    this.cullStale = true;

    this.stats.levelInstances++;
    this.stats.clusters += this.levels.clusters[level];
    this.stats.triangles += this.levels.triangles[level];
    return true;
  }

  /** Takes instance `i` out of the level it is drawn at. */
  private leaveLevel(i: number): void {
    const level = this.instLevel[i];
    const b = this.buckets[level];
    let p = this.instPos[i];
    if (p < b.visible) {
      b.visible--;
      if (p !== b.visible) this.swapInstances(b, p, b.visible);
      p = b.visible;
    }
    const last = --b.count;
    if (p !== last) this.writeInstance(b, p, b.ids[last]);
    const mesh = b.mesh!;
    mesh.count = b.count;
    if (b.count === 0) {
      const live = this.live;
      live[live.indexOf(b)] = live[live.length - 1];
      live.pop();
      this.remove(mesh);
    }
    this.instLevel[i] = -1;

    this.stats.levelInstances--;
    this.stats.clusters -= this.levels.clusters[level];
    this.stats.triangles -= this.levels.triangles[level];
  }

  /**
   * Writes the frustum of `camera` to `out` as planes in the space of this object: left, right,
   * bottom, top and near, each `[nx, ny, nz, d]` with a unit normal pointing inwards. The far plane
   * is left out (it rarely culls anything, and the side planes are the same in every clip space
   * convention). Returns false for a degenerate projection.
   */
  private frustumPlanes(camera: Camera, out: Float64Array): boolean {
    return frustumPlanesIn(camera, this.matrixWorld.elements, this.worldIdentity, out);
  }

  /**
   * Sorts the instances of every level into those inside the view of `camera` and the rest.
   * Returns the milliseconds that took, or -1 when there was nothing to do.
   */
  private cullLevels(camera: Camera): number {
    if (this.stats.levelInstances === 0) {
      this.cullValid = false;
      return -1;
    }
    const wanted = this.cullInstances && (camera as { isArrayCamera?: boolean }).isArrayCamera !== true;
    if (wanted && this.culledFor(camera)) return -1;
    const live = this.live;
    const kept = _planes;
    if (!wanted || !this.frustumPlanes(camera, kept)) {
      if (!this.cullValid) return -1;
      // everything counts as in view; positions need no sorting for that
      for (let j = 0; j < live.length; j++) live[j].visible = live[j].count;
      this.cullValid = false;
      return 0;
    }
    const key = this.cullKey;
    const pe = camera.projectionMatrix.elements;
    const ve = camera.matrixWorldInverse.elements;
    const we = this.matrixWorld.elements;
    for (let j = 0; j < 16; j++) {
      key[j] = pe[j];
      key[16 + j] = ve[j];
      key[32 + j] = we[j];
    }
    this.cullValid = true;
    this.cullStale = false;

    // all of the instances at once, when the sphere around them is inside the view or outside it
    const all = this.boundingSphere;
    const c = all.center;
    let inside = true;
    let outside = false;
    for (let o = 0; o < PLANE_FLOATS; o += 4) {
      const d = kept[o] * c.x + kept[o + 1] * c.y + kept[o + 2] * c.z + kept[o + 3];
      if (!(d >= all.radius)) inside = false;
      if (d < -all.radius) outside = true;
    }
    if (inside || outside) {
      // (a level's instances in view are its first `visible` ones: all of them, or none)
      for (let j = 0; j < live.length; j++) live[j].visible = inside ? live[j].count : 0;
      return 0;
    }

    const s = this.instSphere;
    const levels = this.instLevel;
    const places = this.instPos;
    const buckets = this.buckets;
    const n = this._count;
    for (let i = 0; i < n; i++) {
      const level = levels[i];
      if (level < 0) continue;
      const o = i * 4;
      const x = s[o];
      const y = s[o + 1];
      const z = s[o + 2];
      const outside = -s[o + 3];
      const inView =
        kept[0] * x + kept[1] * y + kept[2] * z + kept[3] >= outside &&
        kept[4] * x + kept[5] * y + kept[6] * z + kept[7] >= outside &&
        kept[8] * x + kept[9] * y + kept[10] * z + kept[11] >= outside &&
        kept[12] * x + kept[13] * y + kept[14] * z + kept[15] >= outside &&
        kept[16] * x + kept[17] * y + kept[18] * z + kept[19] >= outside;

      const b = buckets[level];
      const p = places[i];
      if (inView) {
        if (p >= b.visible) {
          if (p !== b.visible) this.swapInstances(b, p, b.visible);
          b.visible++;
        }
      } else if (p < b.visible) {
        b.visible--;
        if (p !== b.visible) this.swapInstances(b, p, b.visible);
      }
    }
    return 0;
  }

  /** Whether the levels are sorted for exactly the view `camera` has right now. */
  private culledFor(camera: Camera): boolean {
    if (!this.cullValid || this.cullStale) return false;
    const key = this.cullKey;
    const p = camera.projectionMatrix.elements;
    const v = camera.matrixWorldInverse.elements;
    const w = this.matrixWorld.elements;
    // (a camera that turned or moved differs in the first of these already)
    for (let j = 0; j < 16; j++) if (key[16 + j] !== v[j] || key[32 + j] !== w[j] || key[j] !== p[j]) return false;
    return true;
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
        const depth = this.data.groupDepth[this.data.clusterGroup[this.ranges.firstCluster[c]]];
        _color.copy(LOD_PALETTE[depth % LOD_PALETTE.length]);
        break;
      }
      default:
        _color.setRGB(this.colors[i * 3], this.colors[i * 3 + 1], this.colors[i * 3 + 2]);
    }
    const bm = this._batched as unknown as { colorTexture: unknown; _colorsTexture: unknown; material: Material };
    const fresh = bm._colorsTexture === null;
    this._batched.setColorAt(slot, _color);
    bm.colorTexture = bm._colorsTexture;
    // WebGPURenderer looks at what a draw object has (a colour texture now) only when its material changes
    if (fresh) bm.material.needsUpdate = true;
  }

  /** Color of instance `i` when it is drawn at a whole-mesh level (`"cluster"` colours are not drawn by level). */
  private levelColor(level: number, i: number): Color {
    if (this._debugColors === "lod") return _color.copy(LOD_PALETTE[this.levels.depth[level] % LOD_PALETTE.length]);
    return _color.setRGB(this.colors[i * 3], this.colors[i * 3 + 1], this.colors[i * 3 + 2]);
  }

  private recolorAll(): void {
    for (const b of this.buckets) {
      if (b.mesh === null) continue;
      // a draw mesh made before colors were in use has no color buffer: replace it by one that has
      if (b.mesh.instanceColor === null) this.resizeBucket(b, b.capacity);
      else for (let p = 0; p < b.count; p++) this.writeInstance(b, p, b.ids[p]);
    }
    for (let i = 0; i < this._count; i++) {
      if (this.instLevel[i] >= 0) continue;
      const n = this.cutCount[i];
      const slots = this.cutSlots[i];
      const ranges = this.cutRanges[i];
      for (let j = 0; j < n; j++) this.applySlotColor(slots[j], ranges[j], i);
    }
  }
}

/**
 * The properties every draw mesh must share live on the draw meshes: reading one on a MeshletMesh
 * reads the BatchedMesh, writing one writes them all. They are accessors on the prototype, not on
 * each object, because redefining an own property turns a JavaScript object into a dictionary and
 * every field access of the update loop into a hash lookup.
 */
type Forwarded = "castShadow" | "receiveShadow" | "renderOrder" | "customDepthMaterial" | "customDistanceMaterial";
type ForwardTarget = { _batched?: BatchedMesh; buckets?: LevelBucket[] };
for (const key of ["castShadow", "receiveShadow", "renderOrder", "customDepthMaterial", "customDistanceMaterial"] as Forwarded[]) {
  Object.defineProperty(MeshletMesh.prototype, key, {
    get(this: ForwardTarget) {
      return this._batched?.[key];
    },
    set(this: ForwardTarget, value: never) {
      // Object3D assigns its defaults before the draw meshes exist; they are the draw meshes own defaults
      if (this._batched === undefined) return;
      this._batched[key] = value;
      // the keys are Mesh properties, of the same type on both kinds of draw mesh
      for (const b of this.buckets ?? []) if (b.mesh !== null) (b.mesh as unknown as BatchedMesh)[key] = value;
    },
    enumerable: true,
    configurable: true,
  });
}
