import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://localhost:8080',
      '/generate-report': 'http://localhost:8080',
      '/find-hospitals': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/features': 'http://localhost:8080',
    }
  }
})
