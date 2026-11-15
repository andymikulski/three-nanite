import * as THREE from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { MeshletAttribute, MeshletData, MeshletInput } from "../../src/types.js";
import { ClusterSelector } from "../../src/core/select.js";
import { MeshoptSimplifier } from "meshoptimizer/simplifier";

/** Converts a BufferGeometry into builder input (position first, then normal, uv). */
export function geometryToInput(geometry: THREE.BufferGeometry): MeshletInput {
  const attributes: MeshletAttribute[] = [];
  for (const name of ["position", "normal", "uv"]) {
    const a = geometry.getAttribute(name);
    if (!a) continue;
    attributes.push({
      name,
      array: a.array as Float32Array,
      itemSize: a.itemSize,
      normalized: a.normalized,
    });
  }
  const index = geometry.getIndex();
  return { attributes, indices: index ? (index.array as Uint32Array | Uint16Array) : null };
}

export function torusKnot(tubularSegments = 256, radialSegments = 48, welded = false): THREE.BufferGeometry {
  const g = snapPositions(new THREE.TorusKnotGeometry(1, 0.35, tubularSegments, radialSegments));
  return welded ? weld(g) : g;
}

/** Non-indexed icosphere (3 unique vertices per triangle). */
export function icosphere(detail = 40, welded = false): THREE.BufferGeometry {
  const g = snapPositions(new THREE.IcosahedronGeometry(1, detail));
  return welded ? weld(g) : g;
}

/** Open plane with a border (n x n quads, 2 n^2 triangles). */
export function plane(n = 40): THREE.BufferGeometry {
  return snapPositions(new THREE.PlaneGeometry(2, 2, n, n));
}

export function octahedron(): THREE.BufferGeometry {
  return snapPositions(new THREE.OctahedronGeometry(1, 0));
}

/**
 * three.js generates seam vertices with float noise in the last bits, so the
 * "same" position can differ by ~1e-7. Make such positions bit-identical (other
 * attributes are untouched, so normal/uv seams survive) so the mesh is
 * topologically closed under an exact position remap.
 */
export function snapPositions(g: THREE.BufferGeometry, eps = 1e-5): THREE.BufferGeometry {
  const out = g.clone();
  const pos = out.getAttribute("position") as THREE.BufferAttribute;
  const a = pos.array as Float32Array;
  const cell = eps * 10;
  const grid = new Map<string, number>();
  const cellKey = (x: number, y: number, z: number): string => `${x},${y},${z}`;
  for (let i = 0; i < a.length; i += 3) {
    const cx = Math.floor(a[i] / cell);
    const cy = Math.floor(a[i + 1] / cell);
    const cz = Math.floor(a[i + 2] / cell);
    let found = -1;
    for (let dx = -1; dx <= 1 && found < 0; dx++)
      for (let dy = -1; dy <= 1 && found < 0; dy++)
        for (let dz = -1; dz <= 1 && found < 0; dz++) {
          const j = grid.get(cellKey(cx + dx, cy + dy, cz + dz));
          if (j === undefined) continue;
          if (Math.abs(a[j] - a[i]) < eps && Math.abs(a[j + 1] - a[i + 1]) < eps && Math.abs(a[j + 2] - a[i + 2]) < eps) found = j;
        }
    if (found >= 0) {
      a[i] = a[found];
      a[i + 1] = a[found + 1];
      a[i + 2] = a[found + 2];
    } else {
      grid.set(cellKey(cx, cy, cz), i);
    }
  }
  return out;
}

/** Merges vertices by position only (drops normal/uv so the mesh becomes properly closed). */
export function weld(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const clone = g.clone();
  clone.deleteAttribute("uv");
  clone.deleteAttribute("normal");
  return mergeVertices(clone, 1e-6);
}

/** Canonical form of a triangle (rotated to start at the smallest index) as a string key. */
export function triKey(a: number, b: number, c: number): string {
  if (a <= b && a <= c) return `${a},${b},${c}`;
  if (b <= a && b <= c) return `${b},${c},${a}`;
  return `${c},${a},${b}`;
}

/** Multiset of input triangles (degenerates removed), keyed by canonical triangle. */
export function triangleMultiset(indices: ArrayLike<number>, count = indices.length): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < count; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    if (a === b || b === c || a === c) continue;
    const k = triKey(a, b, c);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

export function sameMultiset(a: Map<string, number>, b: Map<string, number>): boolean {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}

export function identityIndices(n: number): Uint32Array {
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/** Gathers the triangles of the given clusters, mapping vertex ids through a position remap. */
export function gatherRemappedTriangles(data: MeshletData, clusters: ArrayLike<number>, count: number, remap: Uint32Array): Uint32Array {
  let total = 0;
  for (let i = 0; i < count; i++) total += data.clusterIndexCount[clusters[i]];
  const out = new Uint32Array(total);
  let w = 0;
  for (let i = 0; i < count; i++) {
    const c = clusters[i];
    const s = data.clusterIndexStart[c];
    const n = data.clusterIndexCount[c];
    for (let j = 0; j < n; j++) out[w++] = remap[data.indices[s + j]];
  }
  return out;
}

export interface EdgeStats {
  /** Edges (undirected, by remapped position id) used exactly once. */
  open: number;
  /** Edges used exactly twice. */
  paired: number;
  /** Edges used more than twice. */
  overused: number;
  /** Directed edges used more than once (same winding twice => overlap or flipped surface). */
  duplicateDirected: number;
  triangles: number;
}

/** Counts how often each undirected edge is used by a triangle soup. */
export function edgeStats(tris: Uint32Array): EdgeStats {
  const undirected = new Map<number, number>();
  const directed = new Map<number, number>();
  const key = (a: number, b: number): number => a * 4294967296 + b;
  let degenerate = 0;
  for (let i = 0; i < tris.length; i += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[i + e];
      const b = tris[i + ((e + 1) % 3)];
      if (a === b) {
        degenerate++;
        continue;
      }
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      undirected.set(key(lo, hi), (undirected.get(key(lo, hi)) ?? 0) + 1);
      directed.set(key(a, b), (directed.get(key(a, b)) ?? 0) + 1);
    }
  }
  let open = 0;
  let paired = 0;
  let overused = 0;
  for (const n of undirected.values()) {
    if (n === 1) open++;
    else if (n === 2) paired++;
    else overused++;
  }
  let duplicateDirected = 0;
  for (const n of directed.values()) if (n > 1) duplicateDirected++;
  return { open, paired, overused, duplicateDirected: duplicateDirected + degenerate, triangles: tris.length / 3 };
}

/** Max finite group error. */
export function maxFiniteError(data: MeshletData): number {
  let m = 0;
  for (let g = 0; g < data.groupCount; g++) {
    const e = data.groupError[g];
    if (Number.isFinite(e) && e > m) m = e;
  }
  return m;
}

export interface CutResult {
  label: string;
  stats: EdgeStats;
}

/** Runs a spread of error-based and view-dependent cuts and returns edge statistics per cut. */
export async function sampleCuts(data: MeshletData): Promise<CutResult[]> {
  await MeshoptSimplifier.ready;
  const remap = MeshoptSimplifier.generatePositionRemap(data.attributes[0].array as Float32Array, 3);
  const selector = new ClusterSelector(data);
  const out = new Uint32Array(data.clusterCount);
  const results: CutResult[] = [];

  const run = (label: string, n: number): void => {
    results.push({ label, stats: edgeStats(gatherRemappedTriangles(data, out, n, remap)) });
  };

  const maxErr = maxFiniteError(data);
  const steps = 10;
  for (let i = 0; i <= steps; i++) run(`error ${(maxErr * i) / steps}`, selector.selectByError(out, (maxErr * i) / steps));
  run("error Infinity", selector.selectByError(out, Infinity));

  const r = data.boundingSphere[3];
  const cx = data.boundingSphere[0];
  const cy = data.boundingSphere[1];
  const cz = data.boundingSphere[2];
  const cams: [number, number, number][] = [
    [cx, cy, cz + r * 3],
    [cx + r * 2, cy + r, cz - r * 6],
    [cx - r * 0.3, cy + r * 0.2, cz + r * 1.2],
    [cx + r * 20, cy, cz],
    [cx, cy - r * 100, cz],
  ];
  const kScales = [0, 1e-4, 1e-3, 5e-3, 2e-2, 0.1];
  for (const cam of cams) {
    for (const k of kScales) {
      run(`view ${cam.map((v) => v.toFixed(2)).join(",")} k=${k}`, selector.select(out, cam[0], cam[1], cam[2], k, 0.01));
    }
  }
  return results;
}
