import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { defineConfig } from "vite";
import type { Plugin } from "vite";

/**
 * The Stylized Nature MegaKit (Quaternius, CC0), glTF folder, for the valley page. Not in this
 * repository: set NATURE_KIT to wherever the kit's `glTF` folder is.
 */
const NATURE_KIT = process.env.NATURE_KIT ?? "C:/Users/Andy/dev/flipcab/assets/StylizedNature/glTF";

/**
 * three-impostor (Andy's octahedral impostor library, `~/dev/three-impostor`), compiled from its
 * source for the valley's "impostors" comparison. Set IMPOSTOR_LIB to its folder if it is elsewhere.
 */
const IMPOSTOR_LIB = process.env.IMPOSTOR_LIB ?? "C:/Users/Andy/dev/three-impostor";

const TYPES: Record<string, string> = { ".gltf": "model/gltf+json", ".bin": "application/octet-stream", ".png": "image/png", ".jpg": "image/jpeg" };

/** Serves `dir` under `prefix` (Vite's /@fs/ does not serve every folder on Windows). */
function mount(prefix: string, dir: string): Plugin {
  const base = normalize(dir);
  return {
    name: "mount " + prefix,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        if (!url.startsWith(prefix)) return next();
        let file = "";
        try {
          file = normalize(join(base, decodeURIComponent(url.slice(prefix.length))));
        } catch {
          file = "";
        }
        let size = -1;
        if (file.startsWith(base + sep)) {
          try {
            const stat = statSync(file);
            if (stat.isFile()) size = stat.size;
          } catch {
            size = -1;
          }
        }
        if (size < 0) {
          res.statusCode = 404;
          res.end("not found");
          return;
        }
        res.setHeader("Content-Type", TYPES[extname(file).toLowerCase()] ?? "application/octet-stream");
        res.setHeader("Content-Length", size);
        createReadStream(file).pipe(res);
      });
    },
  };
}

/** `/valley` without the slash would fall back to the main page: send it to the valley's. */
function valleySlash(): Plugin {
  return {
    name: "valley slash",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if ((req.url ?? "").split("?")[0] !== "/valley") return next();
        res.statusCode = 302;
        res.setHeader("Location", "/valley/");
        res.end();
      });
    },
  };
}

export default defineConfig({
  plugins: [mount("/nature/", NATURE_KIT), valleySlash()],
  resolve: {
    alias: [
      // (exact matches: the WebGL entry and the WebGPURenderer one)
      { find: /^@octa-lab\/three-impostor\/webgpu$/, replacement: IMPOSTOR_LIB + "/src/webgpu/index.ts" },
      { find: /^@octa-lab\/three-impostor$/, replacement: IMPOSTOR_LIB + "/src/index.ts" },
    ],
    // (its sources would otherwise find the three in their own folder: one three for everything)
    dedupe: ["three"],
  },
  server: {
    // the demo imports the library straight from ../src, and three-impostor from its folder
    fs: { allow: [import.meta.dirname + "/..", IMPOSTOR_LIB] },
  },
  build: {
    // top-level await (renderer selection) and the meshoptimizer wasm need modern output
    target: "esnext",
    chunkSizeWarningLimit: 700, // three.js itself is ~600 kB
    rollupOptions: {
      input: {
        main: import.meta.dirname + "/index.html",
        valley: import.meta.dirname + "/valley/index.html",
      },
    },
  },
  worker: { format: "es" },
});
