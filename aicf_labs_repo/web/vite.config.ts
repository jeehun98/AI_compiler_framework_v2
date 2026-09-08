import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  test: { include: ['src/App.test.tsx'], fileParallelism: false, maxWorkers: 1 },
});
