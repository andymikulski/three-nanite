import {
  BatchedMesh,
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  Matrix4,
  Mesh,
  MeshDepthMaterial,
  MeshDistanceMaterial,
  Object3D,
  REVISION,
  RGBADepthPacking,
  ShaderChunk,
  Sphere,
  Vector2,
  Vector3,
} from "three";
import type { Camera, Intersection, Material, Raycaster, WebGLRenderer } from "three";
import { computeLodLevels, copyLevelIndices, levelForError } from "../core/levels.js";
import type { LodLevels } from "../core/levels.js";
import { lodScale } from "../core/select.js";
import type { MeshletData, TypedArray } from "../types.js";
import { LOD_PALETTE } from "./palette.js";
import { frustumPlanesIn, PLANE_FLOATS } from "./viewPlanes.js";

/** Construction options of {@link MeshletBatch}. */
export interface MeshletBatchOptions {
  /** Initial instance capacity (grows by doubling, each time with a rebuild of the draw mesh). Default 1024. */
  maxInstances?: number;
  /** Allowed geometric error in pixels. Default 1. */
  errorThreshold?: number;
  /**
   * What one more sub-draw is worth, in triangles: instances are drawn at a finer level than they
   * need when that saves a level of their mesh for fewer triangles than this. A sub-draw here costs
   * far less than a draw call, so the default is low: 512. `0` draws every instance at its own level.
   */
  drawCost?: number;
  /**
   * Leave instances outside the view out of the main pass, and those outside both the view and the
   * shadow camera's out of a shadow pass. Default true.
   */
  cullInstances?: boolean;
  /** Most whole-mesh levels kept of each mesh (their index lists are all stored). Default 12. */
  maxLevels?: number;
}

/** Runtime statistics of a {@link MeshletBatch}. */
export interface MeshletBatchStats {
  /** Meshes in the batch. */
  meshes: number;
  /** Instances (`count`). */
  instances: number;
  /** Instances inside the view the batch was last culled for. */
  visible: number;
  /** Sub-draws of the main pass: (mesh, level) pairs with an instance in view. All of them are ONE draw call. */
  draws: number;
  /** Triangles of the main pass, and of a pass that draws every instance (a shadow pass). */
  triangles: number;
  trianglesAll: number;
  /** Duration in milliseconds of the last `update` that had work to do. */
  lastUpdateMs: number;
  /** How many instances that update looked at. */
  lastSelected: number;
}

/** What is used of a renderer (both of three's provide it). */
interface BatchRenderer {
  getRenderTarget(): { height: number } | null;
  getDrawingBufferSize(target: Vector2): Vector2;
  isWebGPURenderer?: boolean;
}

/** The private parts of `THREE.BatchedMesh` this drives (see {@link MeshletBatch} for why). */
interface BatchedInternals {
  _indirectTexture: { image: { data: Uint32Array }; needsUpdate: boolean };
  _multiDrawCount: number;
  _multiDrawBytesPerElement: number;
  _mergedDrawStarts: Int32Array;
  _mergedDrawCounts: Int32Array;
  _mergedDrawInstanceCounts: Int32Array;
  _mergedDrawCount: number;
  _updateDrawParameters: (...args: unknown[]) => void;
}

/** The first three.js revision whose WebGPURenderer draws a BatchedMesh as instanced draws per geometry. */
const WEBGPU_MERGED_REVISION = 187;
const THREE_REVISION = Number.parseInt(REVISION, 10);

const KIND_UNKNOWN = 0;
const KIND_WEBGL = 1;
const KIND_WEBGPU = 2;

const DEFAULT_NEAR = 0.1;
const MIN_SCALE = 1e-20;

/** Instances per chunk of the culling (see `MeshletBatch.buildChunks`). */
const CHUNK = 64;

/** Spreads the 10 bits of `v` to every third bit (for a Morton code). */
function spread10(v: number): number {
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

/** What a pass draws: the instances in view, those and the ones in the shadow camera's view, or all. */
const PASS_ALL = 0;
const PASS_VIEW = 1;
const PASS_SHADOW = 2;

/**
 * The two passes of a counting sort of the instances by group and segment (see
 * `MeshletBatch.layout`): with `order` null, counts the instances of each (group, segment) into
 * `segments`; with it, places each instance at its (group, segment)'s cursor in `segments`.
 * (A small function by itself: it is most of the work of a frame with many instances.)
 */
function sortInstances(
  n: number,
  want: Int32Array,
  hidden: Uint8Array,
  inView: Uint8Array,
  inShadow: Uint8Array,
  drawn: Int32Array,
  view: number,
  shadowing: number,
  segments: Int32Array,
  order: Uint32Array | null,
): void {
  const notView = view ^ 1;
  if (order === null) {
    for (let i = 0; i < n; i++) {
      const w = want[i];
      if (w < 0 || hidden[i] === 1) continue;
      const v = (inView[i] & view) | notView;
      segments[drawn[w] * 3 + 2 - 2 * v - ((v ^ 1) & inShadow[i] & shadowing)]++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const w = want[i];
      if (w < 0 || hidden[i] === 1) continue;
      const v = (inView[i] & view) | notView;
      order[segments[drawn[w] * 3 + 2 - 2 * v - ((v ^ 1) & inShadow[i] & shadowing)]++] = i;
    }
  }
}

const TINT_NONE = -1;
const TINT_WHITE = -2;

/**
 * How a batch finds the matrix of the instance it is drawing, in place of three's
 * `batching_vertex`: the id texture begins with where the ids of each sub-draw begin, and an
 * instance of a sub-draw takes the id that far along.
 */
const INSTANCED_LOOKUP = `#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( int( getIndirectIndex( gl_DrawID ) ) + gl_InstanceID ) );
#endif`;

/** How three's `color_vertex` finds the colour of what it draws (the same in r170 to r187), and the batch's way. */
const COLOR_LOOKUP = "getBatchingColor( getIndirectIndex( gl_DrawID ) )";
const INSTANCED_COLOR_LOOKUP = "getBatchingColor( getIndirectIndex( int( getIndirectIndex( gl_DrawID ) ) + gl_InstanceID ) )";

/** Three's `color_vertex` with the batch's colour lookup (used only where the draw object has colours, `debugColors`). */
function instancedColorVertex(): string {
  return ShaderChunk.color_vertex.split(COLOR_LOOKUP).join(INSTANCED_COLOR_LOOKUP);
}

const _m = new Matrix4();
const _m2 = new Matrix4();
const _v3 = new Vector3();
const _v2 = new Vector2();
const _sphere = new Sphere();
const _white = new Color(1, 1, 1);
const _planes = new Float64Array(PLANE_FLOATS);
const _rayMesh = new Mesh();
const _hits: Intersection[] = [];
let _wanted = new Int32Array(64);
let _from = new Int32Array(65);
let _best = new Float64Array(65);
let _warnedWebGPU = false;

/**
 * A material that IS the given one, since it reads everything from it, except for how a batch
 * looks up its matrices; so the program WebGLRenderer compiles for it is its own, and the given
 * material stays usable anywhere else.
 */
function withInstancedLookup<T extends Material>(material: T): T {
  const derived = Object.create(material) as T & { _listeners?: unknown };
  // (listeners of its own: the renderer registers one for the program it compiles)
  derived._listeners = undefined;
  derived.onBeforeCompile = function (this: Material, shader, renderer) {
    material.onBeforeCompile.call(this, shader, renderer);
    shader.vertexShader = shader.vertexShader
      .replace("#include <batching_vertex>", INSTANCED_LOOKUP)
      .replace("#include <color_vertex>", instancedColorVertex());
  };
  derived.customProgramCacheKey = function (this: Material) {
    return material.customProgramCacheKey.call(this) + "|meshlet batch";
  };
  return derived;
}

/**
 * Draws MANY meshes, each with its levels of detail and any number of instances, as ONE draw call.
 *
 * {@link MeshletMesh} is one mesh: every level of it in use is a draw call, so a scene of hundreds
 * of different meshes is hundreds of draw calls whatever the meshes cost, and on the web a draw
 * call costs microseconds of script. A batch holds every mesh that shares a material. Each frame it
 * chooses a whole-mesh level for every instance, sorts the instances by (mesh, level), and draws
 * all of them at once:
 *
 *  - on WebGLRenderer with one `multiDrawElementsInstancedWEBGL` call, a sub-draw per (mesh, level)
 *    with its instances (an instanced draw each, where the browser has no `WEBGL_multi_draw`);
 *  - on WebGPURenderer (three r187 or newer) as one render object with one `drawIndexed` per
 *    (mesh, level), which is how that renderer draws a `BatchedMesh`.
 *
 * Instance matrices are in a texture, looked up through a list of instance ids in draw order: what
 * changes when instances change level or leave the view is that list, four bytes an instance.
 *
 * It is for meshes that are small against the scene: an instance is drawn at one level throughout,
 * the one its nearest part needs. A mesh the camera gets close to and that is large enough to need
 * different detail in different places belongs in a {@link MeshletMesh}.
 *
 * The draw object is a stock `THREE.BatchedMesh` ({@link MeshletBatch.batched}), so materials,
 * lights and shadows are three's own. Two things about it are not public API: the batch fills in
 * the draw lists and the id texture that `BatchedMesh` keeps for itself, and on WebGLRenderer it
 * issues the draw call itself, right after three has set everything up for one (three r187 has no
 * instanced multi-draw of its own there). The material is given a matrix lookup of its own for
 * that (a derived material: yours is not touched).
 *
 * Not supported: per-instance colours, wireframe materials, and transparent double-sided materials
 * drawn in two passes. Instances are not sorted by depth, which a transparent material would need.
 */
export class MeshletBatch extends Object3D {
  /** Type flag. */
  readonly isMeshletBatch = true;
  /** Makes both renderers call {@link update} with the camera while they collect the scene. */
  readonly isLOD = true;
  /** The hierarchies in the batch. */
  readonly meshes: readonly MeshletData[];
  /** The whole-mesh levels of each. */
  readonly levels: readonly LodLevels[];
  /** Allowed geometric error in pixels (higher = coarser). */
  errorThreshold: number;
  /** See {@link MeshletBatchOptions.drawCost}. */
  drawCost: number;
  /** See {@link MeshletBatchOptions.cullInstances}. */
  cullInstances: boolean;
  /** When true (default) the renderer updates the levels for the camera it renders with. */
  autoUpdate = true;
  /** When set, levels are chosen for this camera instead of the rendering one. */
  lodCamera: Camera | null = null;
  /** Height in pixels of what the camera renders to; `null` (default) takes it from the renderer. */
  viewportHeight: number | null = null;
  private _debugColors: "none" | "lod" = "none";
  /** Sphere around all instances, in the space of this object. */
  readonly boundingSphere = new Sphere();
  /** Live statistics. */
  readonly stats: MeshletBatchStats = { meshes: 0, instances: 0, visible: 0, draws: 0, triangles: 0, trianglesAll: 0, lastUpdateMs: 0, lastSelected: 0 };

  /** The vertices of every mesh and the indices of every level of every mesh. */
  private readonly geometry: BufferGeometry;
  private readonly bytesPerIndex: number;
  private userMaterial: Material | undefined;
  /** What a draw object gets when no material was given. */
  private defaultMaterial: Material | null = null;
  private _batched: BatchedMesh;
  /** The material WebGLRenderer draws with: the given one with the batch's matrix lookup (it has a program of its own to release). */
  private derived: Material | null = null;
  private depthMaterial: MeshDepthMaterial | null = null;
  private distanceMaterial: MeshDistanceMaterial | null = null;
  private kind = KIND_UNKNOWN;

  // ---- groups: one per (mesh, level)
  private readonly groupTotal: number;
  /** First group of each mesh; its levels follow in order, finest first. */
  private readonly meshGroup: Int32Array;
  private readonly meshSphere: Float32Array;
  /** Where each group's indices are in the index buffer, and how many. */
  private readonly groupStart: Uint32Array;
  private readonly groupCount: Uint32Array;
  /** Instances asking for each group, and the group those are drawn with (see `drawCost`). */
  private readonly wantCount: Int32Array;
  private readonly groupDrawn: Int32Array;
  private readonly meshDirty: Uint8Array;
  private readonly dirtyMeshes: Int32Array;
  private dirtyMeshCount = 0;
  private sigDrawCost = Number.NaN;

  // ---- instances
  private _count = 0;
  private capacity: number;
  private matrices: Float64Array;
  private instMesh: Int32Array;
  private instScale: Float32Array;
  /** Bounding sphere of each instance in the space of this object, `[x, y, z, r]`. */
  private instSphere: Float32Array;
  /** The group each instance asks for, or -1 for one that is switched off. */
  private instWant: Int32Array;
  private instHidden: Uint8Array;
  private instInView: Uint8Array;
  private instDue: Float64Array;
  /** The tint each instance was last given: a palette entry, TINT_WHITE, or TINT_NONE (the draw object has no colour for it yet). */
  private instTint: Int16Array;
  /** Whether colours were ever switched on: until then the draw object has none, and nothing is painted. */
  private colorsUsed = false;
  private dirty: Uint8Array;
  private dirtyList: Int32Array;
  private dirtyCount = 0;
  private boundsDirty = true;

  // ---- the camera's way (see MeshletMesh)
  private travel = 0;
  private minDue = 0;
  private readonly worldKey = new Float64Array(16).fill(Number.NaN);
  private readonly worldInverse = new Matrix4();
  private worldScale = 1;
  private worldIdentity = true;
  private sigValid = false;
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

  // ---- culling
  private readonly cullKey = new Float64Array(48);
  private cullValid = false;
  private cullStale = false;
  /** Whether each instance is inside the frustum of the shadow camera last drawn for (see `cullShadow`). */
  private instInShadow: Uint8Array;
  private readonly shadowKey = new Float64Array(48);
  private shadowValid = false;
  private shadowStale = false;
  /**
   * Chunks: the instances in a spatially coherent order (Morton order of their centres), cut into
   * runs of CHUNK instances, each with a box around its instances' spheres, so a frustum test
   * decides a whole chunk at once wherever the chunk is inside or outside. Built when instances are
   * added; a moved instance widens the box of its chunk.
   */
  private chunkItems: Int32Array = new Int32Array(0);
  private chunkBox: Float32Array = new Float32Array(0);
  private instChunk: Int32Array;
  private chunkCount = 0;
  private chunksStale = true;
  private movedSinceChunks = 0;

  // ---- what is drawn: the instance ids in draw order, and a sub-draw per group in use
  private layoutStale = true;
  private order: Uint32Array;
  private readonly groupAll: Int32Array;
  private readonly groupSeen: Int32Array;
  private readonly groupShadow: Int32Array;
  /** Per group, instances (then cursors) of each of its three segments (see `layout`). */
  private readonly segments: Int32Array;
  private readonly groupBase: Int32Array;
  private drawCount = 0;
  private readonly drawGroup: Int32Array;
  private readonly drawCounts: Int32Array;
  private readonly drawOffsets: Int32Array;
  /** Instances of each sub-draw: in view, in view or in the shadow camera's, all. */
  private readonly drawSeen: Int32Array;
  private readonly drawShadow: Int32Array;
  private readonly drawAll: Int32Array;
  /** The same for WebGPURenderer: a pass's sub-draws have to step over the instances it leaves out. */
  private readonly seenStarts: Int32Array;
  private readonly seenCounts: Int32Array;
  private readonly seenInstances: Int32Array;
  private seenEntries = 0;
  private readonly shadowStarts: Int32Array;
  private readonly shadowCounts: Int32Array;
  private readonly shadowInstances: Int32Array;
  private shadowEntries = 0;
  private shadowTriangles = 0;
  private readonly allStarts: Int32Array;
  private readonly allCounts: Int32Array;
  private readonly allInstances: Int32Array;

  private readonly shadowCameras = new WeakSet<Camera>();
  private readonly heights = new WeakMap<Camera, number>();
  private lastHeight = 0;

  /**
   * @param meshes   The hierarchies to draw. All must have the same vertex attributes.
   * @param material Any standard three.js material (`undefined` = BatchedMesh default).
   * @param options  See {@link MeshletBatchOptions}.
   */
  constructor(meshes: MeshletData[], material?: Material, options: MeshletBatchOptions = {}) {
    super();
    if (meshes.length === 0) throw new Error("three-nanite: a MeshletBatch needs at least one mesh");
    const first = meshes[0];
    if (first.attributes.length === 0) throw new Error("three-nanite: this MeshletData holds no vertex attributes (hierarchy only)");
    for (const data of meshes) {
      const same =
        data.attributes.length === first.attributes.length &&
        data.attributes.every((a, k) => {
          const b = first.attributes[k];
          return a.name === b.name && a.itemSize === b.itemSize && a.normalized === b.normalized && a.array.constructor === b.array.constructor;
        });
      if (!same) throw new Error("three-nanite: the meshes of a MeshletBatch must all have the same vertex attributes");
    }
    this.meshes = meshes;
    this.errorThreshold = options.errorThreshold ?? 1;
    this.drawCost = options.drawCost ?? 512;
    this.cullInstances = options.cullInstances ?? true;
    this.userMaterial = material;

    // ---- levels and groups
    const maxLevels = options.maxLevels ?? 12;
    const levels = meshes.map((data) => computeLodLevels(data, { maxLevels }));
    this.levels = levels;
    this.meshGroup = new Int32Array(meshes.length + 1);
    this.meshSphere = new Float32Array(meshes.length * 4);
    let groups = 0;
    let vertices = 0;
    let indexTotal = 0;
    meshes.forEach((data, k) => {
      this.meshGroup[k] = groups;
      this.meshSphere.set(data.boundingSphere, k * 4);
      groups += levels[k].count;
      vertices += data.vertexCount;
      for (let j = 0; j < levels[k].count; j++) indexTotal += levels[k].triangles[j] * 3;
    });
    this.meshGroup[meshes.length] = groups;
    this.groupTotal = groups;
    this.groupStart = new Uint32Array(groups);
    this.groupCount = new Uint32Array(groups);
    this.wantCount = new Int32Array(groups);
    this.groupDrawn = new Int32Array(groups);
    for (let g = 0; g < groups; g++) this.groupDrawn[g] = g;
    this.meshDirty = new Uint8Array(meshes.length);
    this.dirtyMeshes = new Int32Array(meshes.length);

    // ---- one vertex buffer, and the index list of every level behind one another
    const geometry = new BufferGeometry();
    first.attributes.forEach((a, index) => {
      const ctor = a.array.constructor as new (length: number) => TypedArray;
      const array = new ctor(vertices * a.itemSize);
      let at = 0;
      for (const data of meshes) {
        array.set(data.attributes[index].array, at);
        at += data.vertexCount * a.itemSize;
      }
      geometry.setAttribute(a.name, new BufferAttribute(array, a.itemSize, a.normalized));
    });
    const indices = vertices > 65535 ? new Uint32Array(indexTotal) : new Uint16Array(indexTotal);
    this.bytesPerIndex = indices.BYTES_PER_ELEMENT;
    let scratch = new Uint32Array(0);
    let at = 0;
    let base = 0;
    meshes.forEach((data, k) => {
      for (let j = 0; j < levels[k].count; j++) {
        const n = levels[k].triangles[j] * 3;
        if (scratch.length < n) scratch = new Uint32Array(n);
        copyLevelIndices(data, levels[k], j, scratch);
        // (a multi-draw has no base vertex: the indices are those of the shared buffer)
        for (let i = 0; i < n; i++) indices[at + i] = scratch[i] + base;
        const g = this.meshGroup[k] + j;
        this.groupStart[g] = at;
        this.groupCount[g] = n;
        at += n;
      }
      base += data.vertexCount;
    });
    geometry.setIndex(new BufferAttribute(indices, 1));
    // (bounds for whoever asks: the meshes lie on top of one another here, each in its own space)
    const reach = meshes.reduce((r, data) => Math.max(r, Math.hypot(data.boundingSphere[0], data.boundingSphere[1], data.boundingSphere[2]) + data.boundingSphere[3]), 0);
    geometry.boundingSphere = new Sphere(new Vector3(), reach);
    geometry.boundingBox = new Box3(new Vector3(-reach, -reach, -reach), new Vector3(reach, reach, reach));
    this.geometry = geometry;

    // ---- draw lists
    this.groupAll = new Int32Array(groups);
    this.groupSeen = new Int32Array(groups);
    this.groupShadow = new Int32Array(groups);
    this.segments = new Int32Array(groups * 3);
    this.groupBase = new Int32Array(groups);
    this.drawGroup = new Int32Array(groups);
    this.drawCounts = new Int32Array(groups);
    this.drawOffsets = new Int32Array(groups);
    this.drawSeen = new Int32Array(groups);
    this.drawShadow = new Int32Array(groups);
    this.drawAll = new Int32Array(groups);
    this.seenStarts = new Int32Array(groups * 2);
    this.seenCounts = new Int32Array(groups * 2);
    this.seenInstances = new Int32Array(groups * 2);
    this.shadowStarts = new Int32Array(groups * 2);
    this.shadowCounts = new Int32Array(groups * 2);
    this.shadowInstances = new Int32Array(groups * 2);
    this.allStarts = new Int32Array(groups);
    this.allCounts = new Int32Array(groups);
    this.allInstances = new Int32Array(groups);

    // ---- instances
    const capacity = Math.max(1, Math.floor(options.maxInstances ?? 1024));
    this.capacity = capacity;
    this.matrices = new Float64Array(capacity * 16);
    this.instMesh = new Int32Array(capacity);
    this.instScale = new Float32Array(capacity);
    this.instSphere = new Float32Array(capacity * 4);
    this.instWant = new Int32Array(capacity).fill(-1);
    this.instHidden = new Uint8Array(capacity);
    this.instInView = new Uint8Array(capacity);
    this.instInShadow = new Uint8Array(capacity);
    this.instChunk = new Int32Array(capacity);
    this.instDue = new Float64Array(capacity);
    this.instTint = new Int16Array(capacity).fill(TINT_NONE);
    this.dirty = new Uint8Array(capacity);
    this.dirtyList = new Int32Array(capacity);
    this.order = new Uint32Array(capacity);

    this._batched = this.createBatched(capacity);
    this.applyMaterial();
    this.add(this._batched);
    this.stats.meshes = meshes.length;
  }

  // =========================================================================
  // public surface
  // =========================================================================

  /** The draw object. It is REPLACED when the instance capacity grows. */
  get batched(): BatchedMesh {
    return this._batched;
  }

  /** Number of instances. */
  override get count(): number {
    return this._count;
  }

  /** The material given to the batch. */
  get material(): Material | undefined {
    return this.userMaterial;
  }
  set material(value: Material | undefined) {
    this.userMaterial = value;
    this.applyMaterial();
  }

  /**
   * Adds an instance of mesh `mesh` (an index into {@link meshes}) and returns its id.
   * Its matrix is relative to this object.
   */
  addInstance(mesh: number, matrix?: Matrix4): number {
    if (!(mesh >= 0 && mesh < this.meshes.length)) throw new RangeError(`MeshletBatch.addInstance: no mesh ${mesh}`);
    if (this._count === this.capacity) this.grow();
    const i = this._count++;
    this.instMesh[i] = mesh;
    this.instWant[i] = -1;
    this.instHidden[i] = 0;
    this.instInView[i] = 1;
    this.instInShadow[i] = 1;
    this.chunksStale = true;
    const id = this._batched.addInstance(0);
    if (id !== i) throw new Error("three-nanite: unexpected BatchedMesh instance id (unsupported three.js version)");
    this.stats.instances = this._count;
    this.setMatrixAt(i, matrix ?? _m.identity());
    return i;
  }

  /** Sets the matrix of instance `i` (relative to this object). */
  setMatrixAt(i: number, matrix: Matrix4): void {
    if (!(i >= 0 && i < this._count)) throw new RangeError(`MeshletBatch.setMatrixAt: instance ${i} out of range (count ${this._count})`);
    matrix.toArray(this.matrices, i * 16);
    this._batched.setMatrixAt(i, matrix);
    const scale = matrix.getMaxScaleOnAxis();
    this.instScale[i] = scale;
    const s = this.meshSphere;
    const k = this.instMesh[i] * 4;
    _v3.set(s[k], s[k + 1], s[k + 2]).applyMatrix4(matrix);
    const o = i * 4;
    this.instSphere[o] = _v3.x;
    this.instSphere[o + 1] = _v3.y;
    this.instSphere[o + 2] = _v3.z;
    this.instSphere[o + 3] = s[k + 3] * scale;
    this.boundsDirty = true;
    this.cullStale = true;
    this.shadowStale = true;
    if (!this.chunksStale) {
      // (its chunk grows to take it in; after many moves the chunks are made again)
      const box = this.chunkBox;
      const b = this.instChunk[i] * 6;
      const r = this.instSphere[o + 3];
      for (let a = 0; a < 3; a++) {
        const v = this.instSphere[o + a];
        if (v - r < box[b + a]) box[b + a] = v - r;
        if (v + r > box[b + 3 + a]) box[b + 3 + a] = v + r;
      }
      if (++this.movedSinceChunks > (this._count >> 3) + 64) this.chunksStale = true;
    }
    if (this.dirty[i] === 0) {
      this.dirty[i] = 1;
      this.dirtyList[this.dirtyCount++] = i;
    }
  }

  /** Reads the matrix of instance `i` into `target`. */
  getMatrixAt(i: number, target: Matrix4): Matrix4 {
    return target.fromArray(this.matrices, i * 16);
  }

  /** The mesh instance `i` is an instance of. */
  getMeshAt(i: number): number {
    return i >= 0 && i < this._count ? this.instMesh[i] : -1;
  }

  /** Switches instance `i` off (not drawn in any pass) or back on. */
  setVisibleAt(i: number, visible: boolean): void {
    if (!(i >= 0 && i < this._count)) throw new RangeError(`MeshletBatch.setVisibleAt: instance ${i} out of range (count ${this._count})`);
    const hidden = visible ? 0 : 1;
    if (this.instHidden[i] === hidden) return;
    this.instHidden[i] = hidden;
    this.layoutStale = true;
  }

  getVisibleAt(i: number): boolean {
    return i >= 0 && i < this._count && this.instHidden[i] === 0;
  }

  /** The level of its mesh instance `i` is drawn at (0 is the finest), or -1 before the first update. */
  getLevelAt(i: number): number {
    if (!(i >= 0 && i < this._count) || this.instWant[i] < 0) return -1;
    return this.groupDrawn[this.instWant[i]] - this.meshGroup[this.instMesh[i]];
  }

  /**
   * `"lod"`: every instance is tinted by the level it is drawn at, coloured by depth in its hierarchy
   * as `MeshletMesh` colours its levels, on both renderers. `"none"` (default): untinted. The colour
   * multiplies the material's, as an instance colour does.
   */
  get debugColors(): "none" | "lod" {
    return this._debugColors;
  }
  set debugColors(mode: "none" | "lod") {
    if (mode === this._debugColors) return;
    this._debugColors = mode;
    if (mode !== "none") this.colorsUsed = true;
    this.layoutStale = true;
  }

  /** Triangles instance `i` is drawn with. */
  getTriangleCountAt(i: number): number {
    if (!(i >= 0 && i < this._count) || this.instWant[i] < 0) return 0;
    return this.groupCount[this.groupDrawn[this.instWant[i]]] / 3;
  }

  /**
   * Chooses the level of every instance for the given camera and sorts them for drawing.
   *
   * Both renderers call this by themselves with the camera they are about to render with (set
   * {@link autoUpdate} to `false` to stop them); it then ignores the cameras of shadow passes.
   * Cheap when nothing relevant changed, and when the camera moves only the instances near enough
   * to a change of level are looked at (see `MeshletMesh.update`).
   *
   * @returns `true` if what is drawn changed.
   */
  update(camera: Camera, viewportHeight?: number): boolean {
    let lodCamera = camera;
    let height: number;
    const automatic = viewportHeight === undefined;
    if (automatic) {
      if (this.shadowCameras.has(camera) || this.inShadowPass()) return false;
      if (this.lodCamera !== null) lodCamera = this.lodCamera;
      height = this.viewportHeight ?? this.heights.get(camera) ?? (this.lastHeight > 0 ? this.lastHeight : defaultViewportHeight());
      if (camera !== lodCamera) lodCamera.updateWorldMatrix(true, false);
    } else {
      height = viewportHeight;
      this.updateWorldMatrix(true, false);
      lodCamera.updateWorldMatrix(true, false);
      if (camera !== lodCamera) camera.updateWorldMatrix(true, false);
    }

    const ortho = (lodCamera as { isOrthographicCamera?: boolean }).isOrthographicCamera === true;
    const proj = lodCamera.projectionMatrix.elements[5];
    const near = (lodCamera as { near?: number }).near ?? DEFAULT_NEAR;

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

    const sameSettings =
      this.sigValid &&
      proj === this.sigProj &&
      height === this.sigHeight &&
      this.errorThreshold === this.sigThreshold &&
      ortho === this.sigOrtho &&
      near === this.sigNear &&
      worldScale === this.sigScale;
    if (sameSettings) {
      const dx = cx - this.vx;
      const dy = cy - this.vy;
      const dz = cz - this.vz;
      if (dx !== 0 || dy !== 0 || dz !== 0) this.travel += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    this.vx = cx;
    this.vy = cy;
    this.vz = cz;
    const due = !sameSettings || this.travel > this.minDue;

    let changed = false;
    let worked = false;
    const t0 = performance.now();

    if (due || this.dirtyCount > 0) {
      worked = true;
      this.vProj = proj;
      this.vHeight = height;
      this.vNear = near;
      this.vOrtho = ortho;
      this.vWorldScale = worldScale;
      const count = this._count;
      const dirty = this.dirty;
      const list = this.dirtyList;
      let selected = 0;

      if (!sameSettings) {
        this.travel = 0;
        this.minDue = Infinity;
        for (let d = 0; d < this.dirtyCount; d++) dirty[list[d]] = 0;
        for (let i = 0; i < count; i++) if (this.choose(i)) changed = true;
        selected = count;
        this.sigValid = true;
        this.sigProj = proj;
        this.sigHeight = height;
        this.sigThreshold = this.errorThreshold;
        this.sigOrtho = ortho;
        this.sigNear = near;
        this.sigScale = worldScale;
      } else {
        for (let d = 0; d < this.dirtyCount; d++) {
          const i = list[d];
          dirty[i] = 0;
          if (i < count) {
            if (this.choose(i)) changed = true;
            selected++;
          }
        }
        if (due) {
          const travel = this.travel;
          const dues = this.instDue;
          let least = Infinity;
          for (let i = 0; i < count; i++) {
            if (dues[i] < travel) {
              if (this.choose(i)) changed = true;
              selected++;
            }
            if (dues[i] < least) least = dues[i];
          }
          this.minDue = least;
        }
      }
      this.dirtyCount = 0;
      this.stats.lastSelected = selected;
    }

    // which levels of each mesh are drawn, where what its instances ask for has changed
    if (this.drawCost !== this.sigDrawCost) {
      this.sigDrawCost = this.drawCost;
      for (let k = 0; k < this.meshes.length; k++) this.markMesh(k);
    }
    if (this.dirtyMeshCount > 0) {
      worked = true;
      for (let d = 0; d < this.dirtyMeshCount; d++) {
        const k = this.dirtyMeshes[d];
        this.meshDirty[k] = 0;
        if (this.chooseLevels(k)) changed = true;
      }
      this.dirtyMeshCount = 0;
      this.layoutStale = true;
    }

    if (this.boundsDirty) this.computeBounds();
    if (this.cull(camera)) worked = true;
    // (the draw lists are made when a pass is about to draw: a shadow pass, culled for its own camera, may come first)

    if (worked) this.stats.lastUpdateMs = performance.now() - t0;
    return changed || worked;
  }

  /** Releases the GPU resources of the batch. The material is not disposed. */
  override dispose(): void {
    this.disposeBatched(this._batched);
    this.geometry.dispose();
    this.derived?.dispose();
    this.depthMaterial?.dispose();
    this.distanceMaterial?.dispose();
  }

  // =========================================================================
  // the draw object
  // =========================================================================

  /**
   * A BatchedMesh that draws `geometry` with the batch's own draw lists: it is given one empty
   * geometry of its own, which every instance is an instance of as far as it knows.
   */
  private createBatched(capacity: number): BatchedMesh {
    // (room in the id texture for the table of where each sub-draw's ids begin)
    const bm = new BatchedMesh(capacity + this.groupTotal, 1, 3, this.userMaterial ?? this.defaultMaterial ?? undefined);
    this.defaultMaterial ??= bm.material as Material;
    const husk = new BufferGeometry();
    for (const name of Object.keys(this.geometry.attributes)) {
      const a = this.geometry.getAttribute(name) as BufferAttribute;
      const ctor = a.array.constructor as new (length: number) => TypedArray;
      husk.setAttribute(name, new BufferAttribute(new ctor(0), a.itemSize, a.normalized));
    }
    husk.setIndex(new BufferAttribute(new Uint32Array(3), 1));
    husk.boundingSphere = new Sphere();
    husk.boundingBox = new Box3(new Vector3(), new Vector3());
    if (bm.addGeometry(husk) !== 0) throw new Error("three-nanite: unexpected BatchedMesh geometry id (unsupported three.js version)");
    const own = bm.geometry;
    bm.geometry = this.geometry;
    own.dispose();

    bm.frustumCulled = false;
    bm.perObjectFrustumCulled = false;
    bm.sortObjects = false;
    bm.layers = this.layers;
    bm.matrixAutoUpdate = false;
    bm.matrixWorldAutoUpdate = false;
    bm.matrixWorld = this.matrixWorld;

    const internals = bm as unknown as BatchedInternals;
    // the draw lists are the batch's: three's own way of making them is switched off
    internals._updateDrawParameters = () => {};
    internals._multiDrawCount = 0;
    // WebGLRenderer before r181 asks a BatchedMesh for `colorTexture`, which it does not have, takes
    // the answer to mean its colours have changed, and works out its program again every frame
    (bm as unknown as { colorTexture: null }).colorTexture = null;
    this.installHooks(bm);
    return bm;
  }

  private installHooks(bm: BatchedMesh): void {
    bm.onBeforeRender = (renderer, scene, camera) => {
      const r = renderer as unknown as BatchRenderer;
      this.meet(r);
      const override = (scene as { overrideMaterial?: { isShadowPassMaterial?: boolean; isShadowNodeMaterial?: boolean } | null }).overrideMaterial;
      const shadow = this.shadowCameras.has(camera) || (override != null && (override.isShadowPassMaterial === true || override.isShadowNodeMaterial === true));
      if (!shadow) {
        const target = r.getRenderTarget();
        const height = target !== null ? target.height : r.getDrawingBufferSize(_v2).y;
        if (height !== this.lastHeight || !this.heights.has(camera)) this.heights.set(camera, height);
        this.lastHeight = height;
      }
      if (this.layoutStale) this.layout();
      if (this.kind === KIND_WEBGPU) this.present(shadow ? this.shadowPass(camera) : this.culledFor(camera) ? PASS_VIEW : PASS_ALL);
    };
    bm.onAfterRender = (renderer, _scene, camera, _geometry, material) => {
      if (this.kind === KIND_WEBGL) this.drawGL(renderer as WebGLRenderer, material, this.culledFor(camera) ? PASS_VIEW : PASS_ALL);
    };
    bm.onBeforeShadow = (renderer, _object, _camera, shadowCamera) => {
      this.shadowCameras.add(shadowCamera);
      this.meet(renderer as unknown as BatchRenderer);
      this.cullShadow(shadowCamera);
      if (this.layoutStale) this.layout();
    };
    bm.onAfterShadow = (renderer, _object, _camera, shadowCamera, _geometry, depthMaterial) => {
      if (this.kind === KIND_WEBGL) this.drawGL(renderer as WebGLRenderer, depthMaterial, this.shadowPass(shadowCamera));
    };
    bm.raycast = (raycaster: Raycaster, intersects: Intersection[]) => this.raycastInstances(raycaster, intersects);
  }

  /** Learns which renderer draws the batch: the id texture and the material differ between the two. */
  private meet(renderer: BatchRenderer): void {
    const kind = renderer.isWebGPURenderer === true ? KIND_WEBGPU : KIND_WEBGL;
    if (kind === this.kind) return;
    this.kind = kind;
    this.layoutStale = true;
    this.applyMaterial();
    if (kind === KIND_WEBGPU && THREE_REVISION < WEBGPU_MERGED_REVISION && !_warnedWebGPU) {
      _warnedWebGPU = true;
      console.warn(`three-nanite: MeshletBatch needs three r${WEBGPU_MERGED_REVISION} or newer on WebGPURenderer (this is r${REVISION}); nothing is drawn.`);
    }
  }

  /**
   * Gives the draw object its material: on WebGLRenderer one with the batch's matrix lookup, and
   * depth materials with the same for the shadow passes (three's own would not have it).
   */
  private applyMaterial(): void {
    const bm = this._batched;
    const material = (this.userMaterial ?? this.defaultMaterial) as Material;
    this.depthMaterial?.dispose();
    this.distanceMaterial?.dispose();
    this.depthMaterial = this.distanceMaterial = null;
    this.derived?.dispose();
    this.derived = null;
    if (this.kind === KIND_WEBGPU) {
      bm.material = material;
      bm.customDepthMaterial = undefined;
      bm.customDistanceMaterial = undefined;
      return;
    }
    // (WebGLRenderer, or not known yet: a material with the lookup draws correctly on either)
    this.derived = withInstancedLookup(material);
    bm.material = this.derived;
    const cut = material as { map?: unknown; alphaMap?: unknown; alphaTest?: number; side?: number; shadowSide?: number | null };
    const depth = new MeshDepthMaterial({ depthPacking: RGBADepthPacking });
    const distance = new MeshDistanceMaterial();
    for (const m of [depth, distance] as Array<MeshDepthMaterial | MeshDistanceMaterial>) {
      // a cut-out casts the shadow of what is left of it
      if (cut.alphaTest !== undefined && cut.alphaTest > 0) {
        m.alphaTest = cut.alphaTest;
        m.map = (cut.map ?? null) as typeof m.map;
        m.alphaMap = (cut.alphaMap ?? null) as typeof m.alphaMap;
      }
      const side = cut.shadowSide ?? cut.side;
      if (side !== undefined) m.side = side as typeof m.side;
      m.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader.replace("#include <batching_vertex>", INSTANCED_LOOKUP);
      };
    }
    this.depthMaterial = depth;
    this.distanceMaterial = distance;
    bm.customDepthMaterial = depth;
    bm.customDistanceMaterial = distance;
  }

  /** Doubles the instance capacity: a larger draw object takes the place of the old one. */
  private grow(): void {
    const capacity = this.capacity * 2;
    const grown = <T extends { set(from: T): void }>(to: T, from: T): T => {
      to.set(from);
      return to;
    };
    this.matrices = grown(new Float64Array(capacity * 16), this.matrices);
    this.instMesh = grown(new Int32Array(capacity), this.instMesh);
    this.instScale = grown(new Float32Array(capacity), this.instScale);
    this.instSphere = grown(new Float32Array(capacity * 4), this.instSphere);
    this.instWant = grown(new Int32Array(capacity).fill(-1), this.instWant);
    this.instHidden = grown(new Uint8Array(capacity), this.instHidden);
    this.instInView = grown(new Uint8Array(capacity), this.instInView);
    this.instInShadow = grown(new Uint8Array(capacity), this.instInShadow);
    this.instChunk = grown(new Int32Array(capacity), this.instChunk);
    this.instDue = grown(new Float64Array(capacity), this.instDue);
    // (the new draw object has no colours: every instance is painted again)
    this.instTint = new Int16Array(capacity).fill(TINT_NONE);
    this.dirty = grown(new Uint8Array(capacity), this.dirty);
    this.dirtyList = grown(new Int32Array(capacity), this.dirtyList);
    this.order = new Uint32Array(capacity);
    this.capacity = capacity;

    const old = this._batched;
    const next = this.createBatched(capacity);
    for (let i = 0; i < this._count; i++) {
      next.addInstance(0);
      next.setMatrixAt(i, _m.fromArray(this.matrices, i * 16));
    }
    next.castShadow = old.castShadow;
    next.receiveShadow = old.receiveShadow;
    next.renderOrder = old.renderOrder;
    next.visible = old.visible;
    this.remove(old);
    this.add(next);
    this._batched = next;
    this.applyMaterial();
    this.layoutStale = true;
    // (the old one may be in the lists of a render in progress)
    setTimeout(() => this.disposeBatched(old), 0);
  }

  /** Disposes a draw object without the geometry it shares. */
  private disposeBatched(bm: BatchedMesh): void {
    bm.geometry = new BufferGeometry();
    bm.dispose();
  }

  private inShadowPass(): boolean {
    let root: Object3D = this;
    while (root.parent !== null) root = root.parent;
    const override = (root as { overrideMaterial?: { isShadowPassMaterial?: boolean; isShadowNodeMaterial?: boolean } | null }).overrideMaterial;
    return override != null && (override.isShadowPassMaterial === true || override.isShadowNodeMaterial === true);
  }

  // =========================================================================
  // choosing levels
  // =========================================================================

  private markMesh(k: number): void {
    if (this.meshDirty[k] === 0) {
      this.meshDirty[k] = 1;
      this.dirtyMeshes[this.dirtyMeshCount++] = k;
    }
  }

  /**
   * Chooses the level instance `i` asks for: the coarsest one that is fine at the distance of the
   * nearest point its bounding sphere has, so fine everywhere on it. Returns whether that changed.
   */
  private choose(i: number): boolean {
    const mesh = this.instMesh[i];
    const levels = this.levels[mesh];
    let scale = this.instScale[i] * this.vWorldScale;
    if (!(scale > MIN_SCALE)) scale = MIN_SCALE;
    const k = lodScale(this.errorThreshold, this.vHeight, this.vProj, scale, this.vOrtho);
    const near = this.vNear / scale;

    let level: number;
    let slack: number;
    if (this.vOrtho) {
      level = levelForError(levels, k);
      slack = Infinity;
    } else {
      const s = this.instSphere;
      const o = i * 4;
      const dx = s[o] - this.vx;
      const dy = s[o + 1] - this.vy;
      const dz = s[o + 2] - this.vz;
      const away = Math.sqrt(dx * dx + dy * dy + dz * dz) - s[o + 3];
      const d = away > 0 ? away : 0;
      const unit = this.instScale[i] > MIN_SCALE ? this.instScale[i] : 1;
      const local = d / unit;
      level = levelForError(levels, k * (local > near ? local : near));
      // the level holds while the distance stays between the two at which its neighbours take over
      const finer = levels.errors[level] / k;
      const nearest = finer > near ? finer * unit : -Infinity;
      const furthest = level + 1 < levels.count ? (levels.errors[level + 1] / k) * unit : Infinity;
      // (inside the sphere the distance counts as none: it has to come out first, which only adds)
      slack = Math.min(d - nearest, furthest - d);
    }
    const due = slack > 0 ? this.travel + slack * 0.999 : this.travel;
    this.instDue[i] = due;
    if (due < this.minDue) this.minDue = due;

    const g = this.meshGroup[mesh] + level;
    const asked = this.instWant[i];
    if (asked === g) return false;
    if (asked >= 0) this.wantCount[asked]--;
    this.wantCount[g]++;
    this.instWant[i] = g;
    this.markMesh(mesh);
    return true;
  }

  /**
   * Decides which of the levels asked for of mesh `k` are drawn (see `MeshletMesh`'s method of the
   * same name: the levels asked for, in order, are cut into runs that are each drawn at their finest
   * level, with the cuts where sub-draws and triangles together cost least).
   */
  private chooseLevels(k: number): boolean {
    const first = this.meshGroup[k];
    const n = this.meshGroup[k + 1] - first;
    const counts = this.wantCount;
    const drawn = this.groupDrawn;
    const size = this.groupCount;
    if (_wanted.length < n) {
      _wanted = new Int32Array(n);
      _from = new Int32Array(n + 1);
      _best = new Float64Array(n + 1);
    }
    let w = 0;
    for (let j = 0; j < n; j++) if (counts[first + j] > 0) _wanted[w++] = first + j;

    let changed = false;
    const cost = this.drawCost;
    if (!(cost > 0) || w <= 1) {
      for (let a = 0; a < w; a++) {
        const g = _wanted[a];
        if (drawn[g] !== g) {
          drawn[g] = g;
          changed = true;
        }
      }
      return changed;
    }
    _best[0] = 0;
    for (let b = 1; b <= w; b++) {
      let least = Infinity;
      let start = b - 1;
      let instances = 0;
      for (let a = b - 1; a >= 0; a--) {
        instances += counts[_wanted[a]];
        const c = _best[a] + cost + (size[_wanted[a]] / 3) * instances;
        if (c < least) {
          least = c;
          start = a;
        }
      }
      _best[b] = least;
      _from[b] = start;
    }
    for (let b = w; b > 0; ) {
      const a = _from[b];
      const g = _wanted[a];
      for (let j = a; j < b; j++) {
        if (drawn[_wanted[j]] !== g) {
          drawn[_wanted[j]] = g;
          changed = true;
        }
      }
      b = a;
    }
    return changed;
  }

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

  // =========================================================================
  // culling and the draw lists
  // =========================================================================

  /** Whether the instances are marked for exactly the view `camera` has right now. */
  private culledFor(camera: Camera): boolean {
    if (!this.cullValid || this.cullStale) return false;
    const key = this.cullKey;
    const p = camera.projectionMatrix.elements;
    const v = camera.matrixWorldInverse.elements;
    const w = this.matrixWorld.elements;
    for (let j = 0; j < 16; j++) if (key[16 + j] !== v[j] || key[32 + j] !== w[j] || key[j] !== p[j]) return false;
    return true;
  }

  /** Marks every instance as inside the view of `camera` or not. Returns whether it had work to do. */
  private cull(camera: Camera): boolean {
    const n = this._count;
    if (n === 0) return false;
    const wanted = this.cullInstances && (camera as { isArrayCamera?: boolean }).isArrayCamera !== true;
    if (wanted && this.culledFor(camera)) return false;
    const inView = this.instInView;
    const planes = _planes;
    if (!wanted || !frustumPlanesIn(camera, this.matrixWorld.elements, this.worldIdentity, planes)) {
      if (!this.cullValid) return false;
      inView.fill(1, 0, n);
      this.cullValid = false;
      this.layoutStale = true;
      return true;
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

    if (this.cullInto(planes, inView)) this.layoutStale = true;
    return true;
  }

  /**
   * Marks every instance as inside the frustum of the shadow camera about to be drawn for, or not.
   * A shadow pass then draws the instances in view and those (`layout`).
   */
  private cullShadow(shadowCamera: Camera): void {
    const n = this._count;
    if (n === 0) return;
    if (!this.cullInstances || !frustumPlanesIn(shadowCamera, this.matrixWorld.elements, this.worldIdentity, _planes)) {
      if (this.shadowValid) {
        this.shadowValid = false;
        this.layoutStale = true;
      }
      return;
    }
    const key = this.shadowKey;
    const pe = shadowCamera.projectionMatrix.elements;
    const ve = shadowCamera.matrixWorldInverse.elements;
    const we = this.matrixWorld.elements;
    let same = this.shadowValid && !this.shadowStale;
    for (let j = 0; same && j < 16; j++) if (key[j] !== pe[j] || key[16 + j] !== ve[j] || key[32 + j] !== we[j]) same = false;
    if (same) return;
    for (let j = 0; j < 16; j++) {
      key[j] = pe[j];
      key[16 + j] = ve[j];
      key[32 + j] = we[j];
    }
    const changed = this.cullInto(_planes, this.instInShadow);
    if (changed || !this.shadowValid) this.layoutStale = true;
    this.shadowValid = true;
    this.shadowStale = false;
  }

  /** Sets `flags[i]` to whether instance `i` is inside the planes. Returns whether any flag changed. */
  private cullInto(planes: Float64Array, flags: Uint8Array): boolean {
    const n = this._count;
    let changed = false;
    // all of the instances at once, when the sphere around them is inside the view or outside it
    const whole = this.boundingSphere;
    const c = whole.center;
    let inside = true;
    let outside = false;
    for (let o = 0; o < PLANE_FLOATS; o += 4) {
      const d = planes[o] * c.x + planes[o + 1] * c.y + planes[o + 2] * c.z + planes[o + 3];
      if (!(d >= whole.radius)) inside = false;
      if (d < -whole.radius) outside = true;
    }
    if (inside || outside) {
      const value = inside ? 1 : 0;
      for (let i = 0; i < n; i++) {
        if (flags[i] !== value) {
          flags[i] = value;
          changed = true;
        }
      }
      return changed;
    }
    if (this.chunksStale) this.buildChunks();
    const s = this.instSphere;
    const items = this.chunkItems;
    const box = this.chunkBox;
    for (let k = 0; k < this.chunkCount; k++) {
      const first = k * CHUNK;
      const last = Math.min(n, first + CHUNK);
      // the box against each plane: its corner furthest along the plane's normal, and the nearest
      const b = k * 6;
      const cx = (box[b] + box[b + 3]) * 0.5;
      const cy = (box[b + 1] + box[b + 4]) * 0.5;
      const cz = (box[b + 2] + box[b + 5]) * 0.5;
      const ex = (box[b + 3] - box[b]) * 0.5;
      const ey = (box[b + 4] - box[b + 1]) * 0.5;
      const ez = (box[b + 5] - box[b + 2]) * 0.5;
      let side = 1; // 1 inside, 0 outside, -1 across
      for (let o = 0; o < PLANE_FLOATS; o += 4) {
        const nx = planes[o];
        const ny = planes[o + 1];
        const nz = planes[o + 2];
        const d = nx * cx + ny * cy + nz * cz + planes[o + 3];
        const reach = Math.abs(nx) * ex + Math.abs(ny) * ey + Math.abs(nz) * ez;
        if (d < -reach) {
          side = 0;
          break;
        }
        if (d < reach) side = -1;
      }
      if (side >= 0) {
        for (let j = first; j < last; j++) {
          const i = items[j];
          if (flags[i] !== side) {
            flags[i] = side;
            changed = true;
          }
        }
        continue;
      }
      for (let j = first; j < last; j++) {
        const i = items[j];
        const o = i * 4;
        const x = s[o];
        const y = s[o + 1];
        const z = s[o + 2];
        const out = -s[o + 3];
        const value =
          planes[0] * x + planes[1] * y + planes[2] * z + planes[3] >= out &&
          planes[4] * x + planes[5] * y + planes[6] * z + planes[7] >= out &&
          planes[8] * x + planes[9] * y + planes[10] * z + planes[11] >= out &&
          planes[12] * x + planes[13] * y + planes[14] * z + planes[15] >= out &&
          planes[16] * x + planes[17] * y + planes[18] * z + planes[19] >= out
            ? 1
            : 0;
        if (flags[i] !== value) {
          flags[i] = value;
          changed = true;
        }
      }
    }
    return changed;
  }

  /** Puts the instances in Morton order of their centres and cuts them into chunks with a box each. */
  private buildChunks(): void {
    this.chunksStale = false;
    this.movedSinceChunks = 0;
    const n = this._count;
    if (this.boundsDirty) this.computeBounds();
    const s = this.instSphere;
    const c = this.boundingSphere.center;
    const r = this.boundingSphere.radius > 0 ? this.boundingSphere.radius : 1;
    const scale = 1023 / (2 * r);
    // (keys: a 30 bit Morton code above the instance's number, exact in a double for up to 2^22 instances)
    const keys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const qx = Math.max(0, Math.min(1023, Math.floor((s[o] - c.x + r) * scale)));
      const qy = Math.max(0, Math.min(1023, Math.floor((s[o + 1] - c.y + r) * scale)));
      const qz = Math.max(0, Math.min(1023, Math.floor((s[o + 2] - c.z + r) * scale)));
      keys[i] = (spread10(qx) | (spread10(qy) << 1) | (spread10(qz) << 2)) * 4194304 + i;
    }
    keys.sort();
    const chunks = Math.ceil(n / CHUNK);
    if (this.chunkItems.length < this.capacity) this.chunkItems = new Int32Array(this.capacity);
    if (this.chunkBox.length < chunks * 6) this.chunkBox = new Float32Array(Math.ceil(this.capacity / CHUNK) * 6);
    const items = this.chunkItems;
    const box = this.chunkBox;
    for (let k = 0; k < chunks; k++) {
      let minX = Infinity;
      let minY = Infinity;
      let minZ = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      let maxZ = -Infinity;
      for (let j = k * CHUNK; j < Math.min(n, (k + 1) * CHUNK); j++) {
        const i = keys[j] % 4194304;
        items[j] = i;
        this.instChunk[i] = k;
        const o = i * 4;
        const rad = s[o + 3];
        if (s[o] - rad < minX) minX = s[o] - rad;
        if (s[o] + rad > maxX) maxX = s[o] + rad;
        if (s[o + 1] - rad < minY) minY = s[o + 1] - rad;
        if (s[o + 1] + rad > maxY) maxY = s[o + 1] + rad;
        if (s[o + 2] - rad < minZ) minZ = s[o + 2] - rad;
        if (s[o + 2] + rad > maxZ) maxZ = s[o + 2] + rad;
      }
      box.set([minX, minY, minZ, maxX, maxY, maxZ], k * 6);
    }
    this.chunkCount = chunks;
  }

  /**
   * Sorts the instances by the group they are drawn with and, within each, puts first those in
   * view, then those only in the shadow camera's view, then the rest; and writes the id texture and
   * the sub-draws of each kind of pass (main: the first part of each group, shadow: the first two,
   * neither culled: all of it).
   */
  private layout(): void {
    this.layoutStale = false;
    const n = this._count;
    const groups = this.groupTotal;
    const all = this.groupAll;
    const seen = this.groupSeen;
    const shade = this.groupShadow;
    const base = this.groupBase;
    const want = this.instWant;
    const drawn = this.groupDrawn;
    const hidden = this.instHidden;
    const inView = this.instInView;
    const inShadow = this.instInShadow;
    const culled = this.cullValid;
    // (a shadow pass not culled for its camera draws everything)
    const shadowCulled = culled && this.shadowValid;
    // Each instance goes to a segment of its group: 0 in view, 1 only in the shadow camera's view,
    // 2 neither. Without branches on the flags: instances come in no particular order, and a
    // branch on whether each is in view is mispredicted half the time (it was most of the cost).
    const view = culled ? 1 : 0;
    const shadowing = shadowCulled ? 1 : 0;
    const segments = this.segments;
    segments.fill(0);
    sortInstances(n, want, hidden, inView, inShadow, drawn, view, shadowing, segments, null);
    for (let g = 0; g < groups; g++) {
      seen[g] = segments[g * 3];
      shade[g] = segments[g * 3 + 1];
      all[g] = seen[g] + shade[g] + segments[g * 3 + 2];
    }

    const bytes = this.bytesPerIndex;
    let D = 0;
    let at = 0;
    let visible = 0;
    let draws = 0;
    let triangles = 0;
    let trianglesAll = 0;
    let shadowTriangles = 0;
    for (let g = 0; g < groups; g++) {
      const count = all[g];
      if (count === 0) continue;
      const indexCount = this.groupCount[g];
      const start = this.groupStart[g] * bytes;
      base[g] = at;
      this.drawGroup[D] = g;
      this.drawCounts[D] = indexCount;
      this.drawOffsets[D] = start;
      this.drawSeen[D] = seen[g];
      this.drawShadow[D] = shadowCulled ? seen[g] + shade[g] : count;
      shadowTriangles += (indexCount / 3) * this.drawShadow[D];
      this.drawAll[D] = count;
      this.allStarts[D] = start;
      this.allCounts[D] = indexCount;
      this.allInstances[D] = count;
      at += count;
      D++;
      visible += seen[g];
      if (seen[g] > 0) draws++;
      triangles += (indexCount / 3) * seen[g];
      trianglesAll += (indexCount / 3) * count;
    }
    this.drawCount = D;
    if (this.kind === KIND_WEBGPU) {
      this.seenEntries = this.passEntries(this.drawSeen, this.seenStarts, this.seenCounts, this.seenInstances);
      this.shadowEntries = this.passEntries(this.drawShadow, this.shadowStarts, this.shadowCounts, this.shadowInstances);
    }

    const order = this.order;
    // (the counts become cursors: where the next instance of each segment goes)
    for (let g = 0; g < groups; g++) {
      if (all[g] === 0) continue;
      segments[g * 3] = base[g];
      segments[g * 3 + 1] = base[g] + seen[g];
      segments[g * 3 + 2] = base[g] + seen[g] + shade[g];
    }
    sortInstances(n, want, hidden, inView, inShadow, drawn, view, shadowing, segments, order);
    if (this.colorsUsed) this.paint();

    const internals = this._batched as unknown as BatchedInternals;
    const table = internals._indirectTexture.image.data;
    if (this.kind === KIND_WEBGPU) {
      table.set(order.subarray(0, at));
    } else {
      // WebGLRenderer: a sub-draw knows its number, not where its instances are in the list
      for (let d = 0; d < D; d++) table[d] = D + base[this.drawGroup[d]];
      table.set(order.subarray(0, at), D);
    }
    internals._indirectTexture.needsUpdate = true;

    this.stats.visible = visible;
    this.stats.draws = draws;
    this.stats.triangles = triangles;
    this.stats.trianglesAll = trianglesAll;
    this.shadowTriangles = shadowTriangles;
  }

  /**
   * WebGPURenderer: the sub-draws of a pass that draws the first `drawn[d]` instances of each
   * sub-draw's block. Its instances are numbered on from draw to draw, so the ones a pass leaves
   * out are stepped over with an empty draw. Returns how many entries were written.
   */
  private passEntries(drawn: Int32Array, starts: Int32Array, counts: Int32Array, instances: Int32Array): number {
    let entries = 0;
    let skipped = 0;
    for (let d = 0; d < this.drawCount; d++) {
      const k = drawn[d];
      if (k > 0) {
        if (skipped > 0) {
          starts[entries] = this.drawOffsets[d];
          counts[entries] = 0;
          instances[entries] = skipped;
          entries++;
          skipped = 0;
        }
        starts[entries] = this.drawOffsets[d];
        counts[entries] = this.drawCounts[d];
        instances[entries] = k;
        entries++;
      }
      skipped += this.drawAll[d] - k;
    }
    return entries;
  }

  /** Gives every instance whose tint is not what `debugColors` asks for its colour in the draw object. */
  private paint(): void {
    const bm = this._batched;
    const lod = this._debugColors === "lod";
    const tint = this.instTint;
    const want = this.instWant;
    const fresh = (bm as unknown as { _colorsTexture: unknown })._colorsTexture === null;
    let painted = false;
    for (let i = 0; i < this._count; i++) {
      let t = TINT_WHITE;
      if (lod && want[i] >= 0) {
        const mesh = this.instMesh[i];
        const levels = this.levels[mesh];
        t = levels.depth[this.groupDrawn[want[i]] - this.meshGroup[mesh]] % LOD_PALETTE.length;
      }
      if (tint[i] === t) continue;
      tint[i] = t;
      bm.setColorAt(i, t === TINT_WHITE ? _white : LOD_PALETTE[t]);
      painted = true;
    }
    // (WebGLRenderer before r181 looks for the colours under this name, see createBatched)
    if (painted) (bm as unknown as { colorTexture: unknown }).colorTexture = (bm as unknown as { _colorsTexture: unknown })._colorsTexture;
    // WebGPURenderer looks at what a draw object has (a colour texture now) only when its material changes
    if (fresh && painted) (bm.material as Material).needsUpdate = true;
  }

  /** WebGPURenderer: hands the draw object the sub-draws of the pass about to be drawn. */
  private present(pass: number): void {
    const internals = this._batched as unknown as BatchedInternals;
    if (THREE_REVISION < WEBGPU_MERGED_REVISION) {
      internals._mergedDrawCount = 0;
      return;
    }
    internals._multiDrawBytesPerElement = this.bytesPerIndex; // (the starts are in bytes, as three keeps them)
    if (pass === PASS_VIEW) {
      internals._mergedDrawStarts = this.seenStarts;
      internals._mergedDrawCounts = this.seenCounts;
      internals._mergedDrawInstanceCounts = this.seenInstances;
      internals._mergedDrawCount = this.seenEntries;
    } else if (pass === PASS_SHADOW) {
      internals._mergedDrawStarts = this.shadowStarts;
      internals._mergedDrawCounts = this.shadowCounts;
      internals._mergedDrawInstanceCounts = this.shadowInstances;
      internals._mergedDrawCount = this.shadowEntries;
    } else {
      internals._mergedDrawStarts = this.allStarts;
      internals._mergedDrawCounts = this.allCounts;
      internals._mergedDrawInstanceCounts = this.allInstances;
      internals._mergedDrawCount = this.drawCount;
    }
  }

  /**
   * WebGLRenderer: the draw call. three has just drawn the (empty) BatchedMesh, so its program,
   * its textures and its vertex array are bound.
   */
  private drawGL(renderer: WebGLRenderer, material: Material, pass: number): void {
    const D = this.drawCount;
    if (D === 0) return;
    const instances = pass === PASS_VIEW ? this.drawSeen : pass === PASS_SHADOW ? this.drawShadow : this.drawAll;
    const gl = renderer.getContext() as WebGL2RenderingContext;
    const type = this.bytesPerIndex === 2 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;
    const extension = renderer.extensions.get("WEBGL_multi_draw") as WEBGL_multi_draw | null;
    if (extension) {
      extension.multiDrawElementsInstancedWEBGL(gl.TRIANGLES, this.drawCounts, 0, type, this.drawOffsets, 0, instances, 0, D);
    } else {
      // no multi-draw (Firefox): an instanced draw per sub-draw, telling the shader its number as three does
      const program = (renderer.properties.get(material) as { currentProgram?: { getUniforms(): { setValue(gl: WebGL2RenderingContext, name: string, value: number): void } } }).currentProgram;
      if (!program) return;
      const uniforms = program.getUniforms();
      for (let d = 0; d < D; d++) {
        if (instances[d] === 0) continue;
        uniforms.setValue(gl, "_gl_DrawID", d);
        gl.drawElementsInstanced(gl.TRIANGLES, this.drawCounts[d], type, this.drawOffsets[d], instances[d]);
      }
    }
    const info = renderer.info.render;
    info.calls++;
    info.triangles += pass === PASS_VIEW ? this.stats.triangles : pass === PASS_SHADOW ? this.shadowTriangles : this.stats.trianglesAll;
  }

  /** The kind of pass a shadow pass for `shadowCamera` is: culled for it if the batch was last culled for it. */
  private shadowPass(shadowCamera: Camera): number {
    if (!this.cullValid || !this.shadowValid) return PASS_ALL;
    const key = this.shadowKey;
    const p = shadowCamera.projectionMatrix.elements;
    const v = shadowCamera.matrixWorldInverse.elements;
    for (let j = 0; j < 16; j++) if (key[j] !== p[j] || key[16 + j] !== v[j]) return PASS_ALL;
    return PASS_SHADOW;
  }

  // =========================================================================
  // raycasting
  // =========================================================================

  /** Tests every instance at the level it is drawn at. Hits report this object, `instanceId` and `batchId` (the mesh). */
  private raycastInstances(raycaster: Raycaster, intersects: Intersection[]): void {
    const geometry = this.geometry;
    _rayMesh.geometry = geometry;
    _rayMesh.material = this._batched.material;
    const s = this.instSphere;
    for (let i = 0; i < this._count; i++) {
      const w = this.instWant[i];
      if (w < 0 || this.instHidden[i] === 1) continue;
      _sphere.center.set(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]);
      _sphere.radius = s[i * 4 + 3];
      _sphere.applyMatrix4(this.matrixWorld);
      if (!raycaster.ray.intersectsSphere(_sphere)) continue;
      const g = this.groupDrawn[w];
      geometry.setDrawRange(this.groupStart[g], this.groupCount[g]);
      _rayMesh.matrixWorld.multiplyMatrices(this.matrixWorld, _m2.fromArray(this.matrices, i * 16));
      _rayMesh.raycast(raycaster, _hits);
      for (const hit of _hits) {
        hit.object = this;
        hit.instanceId = i;
        hit.batchId = this.instMesh[i];
        intersects.push(hit);
      }
      _hits.length = 0;
    }
    geometry.setDrawRange(0, Infinity);
  }
}

/** Height to assume before the first render told us the real one. */
function defaultViewportHeight(): number {
  const g = globalThis as { innerHeight?: number; devicePixelRatio?: number };
  return typeof g.innerHeight === "number" && g.innerHeight > 0 ? Math.round(g.innerHeight * (g.devicePixelRatio ?? 1)) : 1080;
}

// `castShadow`, `receiveShadow` and `renderOrder` are those of the draw object. They are accessors
// on the prototype (not per instance, which would turn every batch into a dictionary-mode object).
type Forwarded = "castShadow" | "receiveShadow" | "renderOrder";
for (const key of ["castShadow", "receiveShadow", "renderOrder"] as Forwarded[]) {
  Object.defineProperty(MeshletBatch.prototype, key, {
    get(this: { _batched?: BatchedMesh }) {
      return this._batched?.[key];
    },
    set(this: { _batched?: BatchedMesh }, value: never) {
      // (Object3D assigns its defaults before the draw object exists; they are the draw object's own)
      if (this._batched !== undefined) this._batched[key] = value;
    },
    enumerable: true,
    configurable: true,
  });
}
