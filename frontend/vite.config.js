import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend address for the dev-server proxy (override with PROMPTSHIELD_API_URL)
const apiTarget = process.env.PROMPTSHIELD_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
    css: false,
  },
})
