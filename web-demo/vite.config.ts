import { defineConfig } from "vite";

export default defineConfig(() => {
  const repoBase = "/birdnet-V3.0-dev/";
  const base = process.env.GITHUB_ACTIONS ? repoBase : "/";
  return {
    base,
    build: {
      outDir: "dist"
    }
  };
});
