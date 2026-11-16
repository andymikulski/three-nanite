import type { MeshletData } from "../types.js";

/** Relative slack for sphere nesting checks (float32 rounding of merged bounds). */
const SPHERE_EPS = 1e-3;

/**
 * Structural validation of a {@link MeshletData}. Returns a list of
 * human-readable problems; an empty list means the data is valid.
 *
 * Intentionally free of three.js / meshoptimizer imports so it can run
 * anywhere (loaders, tests, tools).
 */
export function validateMeshletData(data: MeshletData): string[] {
  const problems: string[] = [];
  const fail = (msg: string): void => {
    if (problems.length < 100) problems.push(msg);
    else if (problems.length === 100) problems.push("... further problems omitted");
  };

  const { vertexCount, clusterCount, groupCount } = data;

  // ---- array lengths ----------------------------------------------------------
  const lengths: [string, number, number][] = [
    ["clusterIndexStart", data.clusterIndexStart.length, clusterCount],
    ["clusterIndexCount", data.clusterIndexCount.length, clusterCount],
    ["clusterBounds", data.clusterBounds.length, clusterCount * 4],
    ["clusterGroup", data.clusterGroup.length, clusterCount],
    ["clusterRefined", data.clusterRefined.length, clusterCount],
    ["groupClusterStart", data.groupClusterStart.length, groupCount],
    ["groupClusterCount", data.groupClusterCount.length, groupCount],
    ["groupBounds", data.groupBounds.length, groupCount * 4],
    ["groupError", data.groupError.length, groupCount],
    ["groupDepth", data.groupDepth.length, groupCount],
    ["boundingSphere", data.boundingSphere.length, 4],
    ["boundingBox", data.boundingBox.length, 6],
  ];
  let lengthsOk = true;
  for (const [name, actual, expected] of lengths) {
    if (actual !== expected) {
      fail(`${name} has length ${actual}, expected ${expected}`);
      lengthsOk = false;
    }
  }
  if (data.version !== 1) fail(`unsupported version ${String(data.version)}`);
  if (!Number.isInteger(vertexCount) || vertexCount <= 0) fail(`invalid vertexCount ${vertexCount}`);
  if (!Number.isInteger(clusterCount) || clusterCount <= 0) fail(`invalid clusterCount ${clusterCount}`);
  if (!Number.isInteger(groupCount) || groupCount <= 0) fail(`invalid groupCount ${groupCount}`);

  // ---- attributes (an empty list means hierarchy only) --------------------------------------------------------------
  const attrs = data.attributes;
  if (!attrs) {
    fail("no attributes");
  } else if (attrs.length > 0) {
    const p = attrs[0];
    if (p.name !== "position") fail(`attributes[0] is '${p.name}', expected 'position'`);
    if (!(p.array instanceof Float32Array)) fail("position attribute is not a Float32Array");
    if (p.itemSize !== 3) fail(`position itemSize is ${p.itemSize}, expected 3`);
    if (p.normalized) fail("position attribute must not be normalized");
    for (const a of attrs) {
      if (a.array.length !== vertexCount * a.itemSize) {
        fail(`attribute '${a.name}' has length ${a.array.length}, expected ${vertexCount * a.itemSize}`);
      }
    }
  }

  // bail out early if the basic shape is wrong; the checks below index freely
  if (!lengthsOk || problems.length > 0) return problems;

  // ---- indices -----------------------------------------------------------------
  const indices = data.indices;
  if (indices.length % 3 !== 0) fail(`indices.length ${indices.length} is not divisible by 3`);
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] >= vertexCount) {
      fail(`indices[${i}] = ${indices[i]} is out of range (vertexCount ${vertexCount})`);
      break;
    }
  }

  // ---- clusters: contiguous, gap-free ------------------------------------------
  let expectedStart = 0;
  for (let c = 0; c < clusterCount; c++) {
    const start = data.clusterIndexStart[c];
    const count = data.clusterIndexCount[c];
    if (start !== expectedStart) {
      fail(`cluster ${c} starts at ${start}, expected ${expectedStart} (clusters must be contiguous)`);
      expectedStart = start;
    }
    if (count % 3 !== 0) fail(`cluster ${c} index count ${count} is not divisible by 3`);
    if (count === 0) fail(`cluster ${c} is empty`);
    expectedStart += count;
  }
  if (expectedStart !== indices.length) {
    fail(`clusters cover ${expectedStart} indices, but indices.length is ${indices.length}`);
  }

  // ---- groups tile [0, clusterCount) ---------------------------------------------
  let nextCluster = 0;
  for (let g = 0; g < groupCount; g++) {
    const start = data.groupClusterStart[g];
    const count = data.groupClusterCount[g];
    if (start !== nextCluster) {
      fail(`group ${g} starts at cluster ${start}, expected ${nextCluster} (groups must tile the clusters)`);
    }
    if (count === 0) fail(`group ${g} has no clusters`);
    nextCluster = start + count;

    if (start + count <= clusterCount) {
      for (let c = start; c < start + count; c++) {
        if (data.clusterGroup[c] !== g) {
          fail(`clusterGroup[${c}] = ${data.clusterGroup[c]}, but cluster ${c} lies in the range of group ${g}`);
        }
      }
    }
  }
  if (nextCluster !== clusterCount) fail(`groups cover ${nextCluster} clusters, expected ${clusterCount}`);

  // ---- NaN checks ------------------------------------------------------------------
  for (let i = 0; i < data.clusterBounds.length; i++) {
    if (Number.isNaN(data.clusterBounds[i])) {
      fail(`clusterBounds[${i}] is NaN`);
      break;
    }
  }
  for (let i = 0; i < data.groupBounds.length; i++) {
    if (Number.isNaN(data.groupBounds[i])) {
      fail(`groupBounds[${i}] is NaN`);
      break;
    }
  }
  for (let g = 0; g < groupCount; g++) {
    if (Number.isNaN(data.groupError[g])) fail(`groupError[${g}] is NaN`);
  }
  for (let i = 0; i < 4; i++) if (!Number.isFinite(data.boundingSphere[i])) fail(`boundingSphere[${i}] is not finite`);
  for (let i = 0; i < 6; i++) if (!Number.isFinite(data.boundingBox[i])) fail(`boundingBox[${i}] is not finite`);

  // ---- DAG: refinement, depth, error and sphere monotonicity -------------------------
  const { groupBounds, groupError, groupDepth } = data;

  for (let c = 0; c < clusterCount; c++) {
    const refined = data.clusterRefined[c];
    if (refined === -1) continue;

    const g = data.clusterGroup[c];
    if (!Number.isInteger(refined) || refined < 0 || refined >= groupCount) {
      fail(`clusterRefined[${c}] = ${refined} is not a valid group id`);
      continue;
    }
    if (g >= groupCount) continue; // already reported above
    if (!(groupDepth[refined] < groupDepth[g])) {
      fail(`cluster ${c}: depth of refined group ${refined} (${groupDepth[refined]}) is not below depth of group ${g} (${groupDepth[g]})`);
    }
    if (!(groupError[g] >= groupError[refined])) {
      fail(`cluster ${c}: error of group ${g} (${groupError[g]}) is below error of refined group ${refined} (${groupError[refined]})`);
    }

    // the parent LOD sphere must enclose the finer one
    const dx = groupBounds[g * 4] - groupBounds[refined * 4];
    const dy = groupBounds[g * 4 + 1] - groupBounds[refined * 4 + 1];
    const dz = groupBounds[g * 4 + 2] - groupBounds[refined * 4 + 2];
    const rg = groupBounds[g * 4 + 3];
    const rr = groupBounds[refined * 4 + 3];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d + rr > rg + SPHERE_EPS * Math.max(rg, rr, 1e-30)) {
      fail(`cluster ${c}: sphere of group ${g} does not enclose sphere of refined group ${refined}`);
    }
  }

  // ---- roots and levels -------------------------------------------------------------
  const expectedRoots: number[] = [];
  let maxDepth = 0;
  for (let g = 0; g < groupCount; g++) {
    if (groupError[g] === Infinity) expectedRoots.push(g);
    if (groupDepth[g] > maxDepth) maxDepth = groupDepth[g];
  }
  const roots = data.rootGroups;
  if (expectedRoots.length === 0) fail("no terminal group (groupError === Infinity) exists");
  if (roots.length === 0) fail("rootGroups is empty");
  if (roots.length !== expectedRoots.length) {
    fail(`rootGroups has ${roots.length} entries, expected ${expectedRoots.length} (groups with error Infinity)`);
  } else {
    const sorted = Array.from(roots).sort((a, b) => a - b);
    for (let i = 0; i < sorted.length; i++) {
      if (sorted[i] !== expectedRoots[i]) {
        fail("rootGroups does not match the set of groups with error Infinity");
        break;
      }
    }
  }
  if (data.levelCount !== maxDepth + 1) fail(`levelCount is ${data.levelCount}, expected ${maxDepth + 1}`);

  return problems;
}
