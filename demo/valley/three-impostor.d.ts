// The part of @octa-lab/three-impostor (and of its /webgpu entry, the same API for WebGPURenderer) (~/dev/three-impostor, aliased in vite.config.ts) the valley
// uses. Declared here rather than type-checked from its sources, which are written against three
// r180's types and their own compiler settings.
declare module "@octa-lab/three-impostor" {
  import type { InstancedMesh, Object3D, WebGLRenderer } from "three";

  export interface ImpostorInstances extends InstancedMesh {
    setAssetAt(index: number, asset: number): void;
    setFadeAt(index: number, fade: number): void;
  }
  export interface ImpostorAtlas {
    readonly info: { estimatedMemoryBytes: number; bakeMilliseconds: number; layers: number; layerSize: number };
    createInstances(count: number, options?: Record<string, unknown>): ImpostorInstances;
    dispose(): void;
  }
  export function bakeImpostorSet(
    renderer: WebGLRenderer,
    sources: Array<Object3D | { source: Object3D; options?: Record<string, unknown> }>,
    options?: {
      azimuthFrames?: number;
      elevationFrames?: number;
      elevationRange?: [number, number];
      tileSize?: number;
      supersample?: number;
      batchSize?: number;
      onProgress?: (progress: { fraction: number }) => void;
      maxMemoryMB?: number;
    },
  ): Promise<ImpostorAtlas>;
}

declare module "@octa-lab/three-impostor/webgpu" {
  import type { Object3D } from "three";
  import type { WebGPURenderer } from "three/webgpu";
  import type { ImpostorAtlas } from "@octa-lab/three-impostor";

  export function bakeImpostorSet(
    renderer: WebGPURenderer,
    sources: Array<Object3D | { source: Object3D; options?: Record<string, unknown> }>,
    options?: Parameters<typeof import("@octa-lab/three-impostor").bakeImpostorSet>[2],
  ): Promise<ImpostorAtlas>;
}
