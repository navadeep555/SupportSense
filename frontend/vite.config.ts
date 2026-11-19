import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev proxy: the browser calls http://localhost:5173/api/..., and Vite forwards
    // it to FastAPI. The browser only ever sees ONE origin, so the frontend code
    // never contains a backend URL and needs no CORS in development. In Phase 9
    // the same relative /api paths are served behind one web server.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom', // a simulated browser DOM, so components can render in Node
    setupFiles: './src/test/setup.ts',
  },
})
