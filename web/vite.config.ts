import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  base: './',
  plugins: [viteSingleFile()],
  worker: { format: 'es' },
  build: { target: 'es2022', assetsInlineLimit: 100_000_000, chunkSizeWarningLimit: 5000, modulePreload: { polyfill: false } },
  test: { environment: 'node' },
});
