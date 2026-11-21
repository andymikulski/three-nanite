import { Color } from "three";

/** Palette for `debugColors = "lod"`, indexed by group depth (shared by `MeshletMesh` and `MeshletBatch`). */
export const LOD_PALETTE: readonly Color[] = [
  0x4e79a7, 0xf28e2b, 0xe15759, 0x76b7b2, 0x59a14f, 0xedc948, 0xb07aa1, 0xff9da7, 0x9c755f, 0xbab0ac,
].map((hex) => new Color(hex));
