#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="$ROOT_DIR/public/assets"

MODEL_URL="https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx?download=1"
LABELS_URL="https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_Labels.csv?download=1"

MODEL_PATH="$ASSETS_DIR/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx"
LABELS_PATH="$ASSETS_DIR/BirdNET+_V3.0-preview3_Global_11K_Labels.csv"

mkdir -p "$ASSETS_DIR"

echo "Downloading model..."
curl -L "$MODEL_URL" -o "$MODEL_PATH"

echo "Downloading labels..."
curl -L "$LABELS_URL" -o "$LABELS_PATH"

echo "Done:"
ls -lh "$MODEL_PATH" "$LABELS_PATH"
