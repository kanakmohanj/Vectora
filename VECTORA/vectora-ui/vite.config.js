import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load environment variables based on the current mode (development/production)
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    server: {
      // Use the port from .env, or fallback to 5173 if not found
      port: parseInt(env.VITE_PORT) || 5173,
      // Optional: Set to true if you want Vite to auto-open your browser on start
      open: true,
    },
  };
});
