import { Matrix4 } from "three";
import type { Camera } from "three";

/** Frustum planes of a culling pass: left, right, bottom, top, near; four numbers each. */
export const PLANE_FLOATS = 20;

/**
 * The frustum of the camera last asked about, as planes in world space: left, right, bottom, top
 * and near, each `[nx, ny, nz, d]` with a unit normal pointing inwards. The far plane is left out
 * (it rarely culls anything, and the side planes are the same in every clip space convention).
 * It is kept because the objects of a scene all ask about the same camera.
 */
const _planes = new Float64Array(PLANE_FLOATS);
const _key = new Float64Array(32);
const _clip = new Matrix4();
let _camera: Camera | null = null;
let _ok = false;

/** Brings the world-space planes up to date for `camera`. Returns false for a degenerate projection. */
function viewPlanes(camera: Camera): boolean {
  const p = camera.projectionMatrix.elements;
  const v = camera.matrixWorldInverse.elements;
  const key = _key;
  let same = camera === _camera;
  for (let j = 0; same && j < 16; j++) if (key[j] !== v[j] || key[16 + j] !== p[j]) same = false;
  if (same) return _ok;
  _camera = camera;
  for (let j = 0; j < 16; j++) {
    key[j] = v[j];
    key[16 + j] = p[j];
  }

  const out = _planes;
  const c = _clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
  // clip space: -w <= x <= w and -w <= y <= w
  for (let side = 0; side < 4; side++) {
    const axis = side >> 1;
    const sign = (side & 1) === 0 ? 1 : -1;
    const o = side * 4;
    out[o] = c[3] + sign * c[axis];
    out[o + 1] = c[7] + sign * c[4 + axis];
    out[o + 2] = c[11] + sign * c[8 + axis];
    out[o + 3] = c[15] + sign * c[12 + axis];
  }
  // view space: in front of the near plane (three's cameras look down -z)
  const near = (camera as { near?: number }).near ?? 0;
  out[16] = -v[2];
  out[17] = -v[6];
  out[18] = -v[10];
  out[19] = -v[14] - near;

  _ok = true;
  for (let o = 0; o < PLANE_FLOATS; o += 4) {
    const length = Math.hypot(out[o], out[o + 1], out[o + 2]);
    if (!(length > 0) || !Number.isFinite(length)) {
      _ok = false;
      break;
    }
    out[o] /= length;
    out[o + 1] /= length;
    out[o + 2] /= length;
    out[o + 3] /= length;
  }
  return _ok;
}

/**
 * Writes the frustum of `camera` to `out` as planes in the space of an object: left, right, bottom,
 * top and near, each `[nx, ny, nz, d]` with a unit normal pointing inwards, so that a plane value
 * is a distance in the units of that object. Returns false for a degenerate projection.
 *
 * @param world    Elements of the object's world matrix.
 * @param identity Whether that matrix is the identity (the planes are then the world's own).
 */
export function frustumPlanesIn(camera: Camera, world: ArrayLike<number>, identity: boolean, out: Float64Array): boolean {
  if (!viewPlanes(camera)) return false;
  const w = _planes;
  if (identity) {
    for (let j = 0; j < PLANE_FLOATS; j++) out[j] = w[j];
    return true;
  }
  // a point x of the object is at M x in the world, so a world plane (n, d) is (M^T n, n.t + d) there
  for (let o = 0; o < PLANE_FLOATS; o += 4) {
    const nx = w[o];
    const ny = w[o + 1];
    const nz = w[o + 2];
    const x = world[0] * nx + world[1] * ny + world[2] * nz;
    const y = world[4] * nx + world[5] * ny + world[6] * nz;
    const z = world[8] * nx + world[9] * ny + world[10] * nz;
    const length = Math.sqrt(x * x + y * y + z * z);
    if (!(length > 0) || !Number.isFinite(length)) return false;
    out[o] = x / length;
    out[o + 1] = y / length;
    out[o + 2] = z / length;
    out[o + 3] = (world[12] * nx + world[13] * ny + world[14] * nz + w[o + 3]) / length;
  }
  return true;
}
