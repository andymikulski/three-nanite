import { defineConfig } from "vite";

export default defineConfig({
  server: {
    // the demo imports the library straight from ../src
    fs: { allow: [import.meta.dirname + "/.."] },
  },
  build: {
    // top-level await (renderer selection) and the meshoptimizer wasm need modern output
    target: "esnext",
    chunkSizeWarningLimit: 700, // three.js itself is ~600 kB
  },
  worker: { format: "es" },
});
