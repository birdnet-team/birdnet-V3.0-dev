import { mkdirSync, copyFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(__dirname, "..");
const distDir = resolve(rootDir, "node_modules/onnxruntime-web/dist");
const outDir = resolve(rootDir, "public/ort");

const files = readdirSync(distDir).filter((name) => name.endsWith(".wasm"));

mkdirSync(outDir, { recursive: true });

for (const file of files) {
  copyFileSync(resolve(distDir, file), resolve(outDir, file));
}

console.log(`Copied ${files.length} onnxruntime-web wasm files to public/ort`);
