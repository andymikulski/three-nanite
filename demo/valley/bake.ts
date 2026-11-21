import type { BufferGeometry } from "three";
import { deserializeMeshlets, serializeMeshlets } from "../../src/index.js";
import type { MeshletData } from "../../src/index.js";
import type { BuildFromGeometryOptions } from "../../src/build/index.js";
import type { BakeReply, BakeRequest } from "./bake.worker.js";

/**
 * Builds hierarchies on a few workers, and keeps them in IndexedDB so a reload skips the build.
 * Bump BAKE_VERSION when the builder or the inputs change (or use the panel's "rebuild levels").
 */
const BAKE_VERSION = "valley-5";
const DB = "three-nanite-valley";
const STORE = "meshlets";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

const db = openDb().catch(() => null);

async function cached(key: string): Promise<MeshletData | null> {
  const d = await db;
  if (!d) return null;
  return new Promise((resolve) => {
    const request = d.transaction(STORE).objectStore(STORE).get(BAKE_VERSION + ":" + key);
    request.onsuccess = () => resolve(request.result instanceof ArrayBuffer ? deserializeMeshlets(request.result) : null);
    request.onerror = () => resolve(null);
  });
}

async function keep(key: string, data: MeshletData): Promise<void> {
  const d = await db;
  if (!d) return;
  d.transaction(STORE, "readwrite").objectStore(STORE).put(serializeMeshlets(data), BAKE_VERSION + ":" + key);
}

export async function clearBakeCache(): Promise<void> {
  const d = await db;
  if (!d) return;
  await new Promise<void>((resolve) => {
    const t = d.transaction(STORE, "readwrite");
    t.objectStore(STORE).clear();
    t.oncomplete = () => resolve();
    t.onerror = () => resolve();
  });
}

interface Job {
  request: BakeRequest;
  resolve(data: MeshletData): void;
  reject(error: Error): void;
}

/** A few workers; the machine keeps some cores for itself. */
class Pool {
  private readonly idle: Worker[] = [];
  private readonly queue: Job[] = [];
  private readonly running = new Map<number, Job>();
  private readonly workers: Worker[] = [];
  private next = 1;

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const worker = new Worker(new URL("./bake.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (event: MessageEvent<BakeReply>) => {
        const reply = event.data;
        const job = this.running.get(reply.id);
        this.running.delete(reply.id);
        this.idle.push(worker);
        this.pump();
        if (!job) return;
        if ("error" in reply) job.reject(new Error(reply.error));
        else job.resolve(reply.data);
      };
      this.workers.push(worker);
      this.idle.push(worker);
    }
  }

  run(request: Omit<BakeRequest, "id">): Promise<MeshletData> {
    return new Promise((resolve, reject) => {
      this.queue.push({ request: { ...request, id: this.next++ }, resolve, reject });
      this.pump();
    });
  }

  private pump(): void {
    while (this.idle.length > 0 && this.queue.length > 0) {
      const job = this.queue.shift()!;
      const worker = this.idle.pop()!;
      this.running.set(job.request.id, job);
      worker.postMessage(job.request);
    }
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
  }
}

let pool: Pool | null = null;

/** The hierarchy of `geometry` under `key`: from the cache, or built on a worker. */
export async function bake(key: string, geometry: BufferGeometry, options: BuildFromGeometryOptions): Promise<{ data: MeshletData; built: boolean }> {
  const hit = await cached(key);
  if (hit) return { data: hit, built: false };
  pool ??= new Pool(Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 2)));
  const attributes: BakeRequest["attributes"] = {};
  for (const [name, a] of Object.entries(geometry.attributes)) {
    attributes[name] = { array: a.array as Float32Array, itemSize: a.itemSize };
  }
  const data = await pool.run({ attributes, index: geometry.getIndex()!.array as Uint16Array | Uint32Array, options });
  void keep(key, data);
  return { data, built: true };
}

/** Stops the workers (once everything is built). */
export function endBake(): void {
  pool?.dispose();
  pool = null;
}
