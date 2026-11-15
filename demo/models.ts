import { BufferGeometry, TorusKnotGeometry, Vector3 } from "three";
import type { Mesh, Object3D } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import type { ModelName } from "./params.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";

/** Returns the geometry of the first mesh in `root`, baked into world space. */
function firstMeshGeometry(root: Object3D): BufferGeometry {
  root.updateMatrixWorld(true);
  let found: Mesh | undefined;
  root.traverse((o) => {
    if (!found && (o as Mesh).isMesh) found = o as Mesh;
  });
  if (!found) throw new Error("The file contains no mesh");
  return found.geometry.clone().applyMatrix4(found.matrixWorld);
}

/** Parses a dropped `.glb` / `.gltf` / `.obj` file. */
async function loadFile(file: File): Promise<BufferGeometry> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".obj")) return firstMeshGeometry(new OBJLoader().parse(await file.text()));
  if (name.endsWith(".glb")) return firstMeshGeometry((await new GLTFLoader().parseAsync(await file.arrayBuffer(), "")).scene);
  if (name.endsWith(".gltf")) return firstMeshGeometry((await new GLTFLoader().parseAsync(await file.text(), "")).scene);
  throw new Error(`Unsupported file type: ${file.name}`);
}

async function loadBunny(): Promise<BufferGeometry> {
  const response = await fetch("bunny.obj");
  if (!response.ok) throw new Error(`bunny.obj: HTTP ${response.status}`);
  return firstMeshGeometry(new OBJLoader().parse(await response.text()));
}


/**
 * Turns any loaded geometry into a unit-sized, welded, indexed one that stands on y = 0, centred on x/z.
 * Only the positions are kept; normals are recomputed (smooth) once the vertices are welded.
 */
export function prepareGeometry(source: BufferGeometry): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", source.getAttribute("position").clone());
  if (source.index) geometry.setIndex(source.index.clone());

  // normalise before welding so that the weld tolerance is relative to the model size
  geometry.computeBoundingBox();
  const box = geometry.boundingBox!;
  const size = box.getSize(new Vector3());
  const scale = 1 / Math.max(size.x, size.y, size.z, 1e-9);
  geometry.translate(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
  geometry.scale(scale, scale, scale);

  const welded = mergeVertices(geometry);
  welded.computeVertexNormals();
  welded.computeBoundingBox();
  welded.computeBoundingSphere();
  return welded;
}

export async function loadModel(model: ModelName, file: File | null): Promise<BufferGeometry> {
  switch (model) {
    case "bunny":
      return prepareGeometry(await loadBunny());
    case "knot":
      return prepareGeometry(new TorusKnotGeometry(1, 0.35, 1024, 192));
    case "file":
      if (!file) throw new Error("No file was dropped");
      return prepareGeometry(await loadFile(file));
  }
}
