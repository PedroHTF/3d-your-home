import { defineConfig } from "vite";

// Build vai para dentro do pacote Python: a API serve em /viewer/
export default defineConfig({
  base: "/viewer/",
  build: { outDir: "../src/tdyh/viewer_dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
  server: { proxy: { "/projects": "http://localhost:8000", "/buildings": "http://localhost:8000" } },
});
