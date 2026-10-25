import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so dist/ works from any static host (GitHub Pages, nginx in Docker)
export default defineConfig(({ mode }) => {
  const cabinetMode = ['server', 'full'].includes(mode) ? mode : process.env.VITE_CABINET_MODE || 'full';
  if (!['server', 'full'].includes(cabinetMode)) throw new Error('VITE_CABINET_MODE must be server or full');
  return {
    plugins: [react()], base: './',
    define: { 'import.meta.env.VITE_CABINET_MODE': JSON.stringify(cabinetMode) },
    server: { proxy: {
      '/v1': process.env.VITE_BACKEND_TARGET || 'http://127.0.0.1:18000',
      '/feedback': { target: process.env.VITE_FEEDBACK_TARGET || 'http://127.0.0.1:18001',
        rewrite: path => path.replace(/^\/feedback/, '') },
    } },
  };
});
