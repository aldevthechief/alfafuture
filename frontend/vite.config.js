import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so dist/ works from any static host (GitHub Pages, nginx in Docker)
export default defineConfig({ plugins: [react()], base: './' });
