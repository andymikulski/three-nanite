import { InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, Vector3 } from "three";
import type { BufferGeometry, Camera, Object3D } from "three";
import { MeshletMesh } from "../src/index.js";
import type { DebugColorMode, MeshletData } from "../src/index.js";

/** Deterministic hash of (index, salt) to [0, 1). */
function hash01(i: number, salt: number): number {
  let h = Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

const _position = new Vector3();
const _quaternion = new Quaternion();
const _scale = new Vector3();
const _up = new Vector3(0, 1, 0);
const _matrix = new Matrix4();

/** Everything the controls can change; kept here so a rebuilt mesh gets the same state. */
export interface ContentSettings {
  grid: number;
  errorThreshold: number;
  debugColors: DebugColorMode;
  wireframe: boolean;
  shadows: boolean;
  lodCamera: Camera | null;
  compare: boolean;
}

/**
 * The model on screen: one {@link MeshletMesh} with a grid of instances and, on demand, an
 * `InstancedMesh` of the original geometry with the very same matrices for comparison.
 */
export class Content {
  mesh: MeshletMesh | null = null;
  /** Triangles of the original geometry (one instance). */
  triangles = 0;
  /** Half-size of the square instance grid, and the model height, for framing. */
  half = 1;
  height = 1;

  private readonly material = new MeshStandardMaterial({ color: 0xd9cbb4, roughness: 0.55, metalness: 0 });
  private readonly referenceMaterial = this.material.clone();
  private reference: InstancedMesh | null = null;
  private geometry: BufferGeometry | null = null;
  private radius = 1;

  constructor(
    private readonly parent: Object3D,
    readonly settings: ContentSettings,
  ) {}

  /** Replaces the model. Takes ownership of `geometry` (the original, for the comparison). */
  setModel(geometry: BufferGeometry, data: MeshletData): void {
    this.clear();
    this.geometry = geometry;
    this.triangles = geometry.getIndex()!.count / 3;
    this.radius = geometry.boundingSphere!.radius;
    this.height = geometry.boundingBox!.max.y;

    const s = this.settings;
    const mesh = new MeshletMesh(data, this.material, { maxInstances: s.grid * s.grid, errorThreshold: s.errorThreshold });
    mesh.castShadow = mesh.receiveShadow = true;
    this.mesh = mesh;
    this.parent.add(mesh);
    this.layout();
    this.applySettings();
  }

  /** Changes the instance count in place; the hierarchy and draw buffers are reused. */
  setGrid(grid: number): void {
    this.settings.grid = grid;
    this.layout();
    this.applySettings();
  }

  /** Pushes the current settings onto the meshes (also toggles the reference mesh). */
  applySettings(): void {
    const { mesh } = this;
    if (!mesh) return;
    const s = this.settings;
    mesh.errorThreshold = s.errorThreshold;
    mesh.debugColors = s.debugColors;
    mesh.lodCamera = s.lodCamera;
    mesh.castShadow = s.shadows;
    this.material.wireframe = this.referenceMaterial.wireframe = s.wireframe;
    mesh.visible = !s.compare;
    if (s.compare && !this.reference) this.createReference();
    if (this.reference) {
      this.reference.visible = s.compare;
      this.reference.castShadow = s.shadows;
    }
  }

  dispose(): void {
    this.clear();
    this.material.dispose();
    this.referenceMaterial.dispose();
  }

  private clear(): void {
    this.removeReference();
    if (this.mesh) {
      this.parent.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
    }
    this.geometry?.dispose();
    this.geometry = null;
  }

  private removeReference(): void {
    if (!this.reference) return;
    this.parent.remove(this.reference);
    this.reference.dispose();
    this.reference = null;
  }

  /** A grid of `grid x grid` instances, spaced by the bounding sphere, with seeded yaw and scale. */
  private layout(): void {
    const { mesh } = this;
    if (!mesh) return;
    const n = this.settings.grid;
    const spacing = this.radius * 2.4;
    mesh.count = n * n;
    for (let i = 0; i < n * n; i++) {
      _position.set(((i % n) - (n - 1) / 2) * spacing, 0, (Math.floor(i / n) - (n - 1) / 2) * spacing);
      _quaternion.setFromAxisAngle(_up, hash01(i, 1) * Math.PI * 2);
      _scale.setScalar(0.85 + 0.3 * hash01(i, 2));
      mesh.setMatrixAt(i, _matrix.compose(_position, _quaternion, _scale));
    }
    this.half = (n * spacing) / 2;
    this.removeReference(); // rebuilt on demand with the new matrices
  }

  private createReference(): void {
    const { mesh, geometry } = this;
    if (!mesh || !geometry) return;
    const reference = new InstancedMesh(geometry, this.referenceMaterial, mesh.count);
    for (let i = 0; i < mesh.count; i++) reference.setMatrixAt(i, mesh.getMatrixAt(i, _matrix));
    reference.castShadow = reference.receiveShadow = true;
    this.reference = reference;
    this.parent.add(reference);
  }
}
