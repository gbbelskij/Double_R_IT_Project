import { defineConfig } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  base: "/",
  plugins: [react()],
  server: {
    proxy: {
      // Проксируем все запросы, начинающиеся с /api, на Flask-бэкенд
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => {
          const cleanedPath = path.replace(/^\/api\//, "");
          return cleanedPath;
        },
      },
    },
  },
  resolve: {
    alias: {
      "@components": path.resolve(__dirname, "src/components"),
      "@hooks": path.resolve(__dirname, "src/hooks"),
      "@mocks": path.resolve(__dirname, "src/mocks"),
      "@utils": path.resolve(__dirname, "src/utils"),
      "@schemas": path.resolve(__dirname, "src/schemas"),
      "@data": path.resolve(__dirname, "src/data"),
      "@pages": path.resolve(__dirname, "src/pages/index.ts"),
      src: path.resolve(__dirname, "src"),
      assets: path.resolve(__dirname, "src/assets"),
      types: path.resolve(__dirname, "src/types"),
    },
  },
});
