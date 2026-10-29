import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The production bundle goes straight into Spring Boot's static resources so the
// jar serves UI and API from one port. In dev, Vite proxies /api to Spring on :8080.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../target/classes/static',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1600,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
      '/actuator': 'http://localhost:8080',
    },
  },
});
