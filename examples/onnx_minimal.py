"""Smallest useful BirdNET+ V3.0 inference with ONNX Runtime (no helper code).

    pip install numpy scipy soundfile onnxruntime
    curl -L --create-dirs -o models/BirdNET+_V3.0_Global_12K_FP16.onnx \
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP16.onnx?download=1"
    curl -L --create-dirs -o models/BirdNET+_V3.0_Global_12K_Labels.csv \
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_Labels.csv?download=1"
    python examples/onnx_minimal.py

(``python analyze.py examples/audio/soundscape.wav`` also downloads both files.)
"""

import csv

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

MODEL = "models/BirdNET+_V3.0_Global_12K_FP16.onnx"
LABELS = "models/BirdNET+_V3.0_Global_12K_Labels.csv"

audio, sr = sf.read("examples/audio/soundscape.wav", dtype="float32", always_2d=True)
audio = resample_poly(audio.mean(axis=1), 32000, sr).astype(np.float32)  # average channels, 32 kHz

# Input: float32 [batch, samples]. Any length works; 3 s chunks match training best.
x = audio[: 10 * 3 * 32000].reshape(10, -1)  # first 30 s as 10 chunks of 3 s

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])  # or CUDAExecutionProvider
scores, embeddings = session.run(["predictions", "embeddings_out"], {"input": x})
# scores: [10, 11725], independent per class (0-1); embeddings: [10, 1280]

names = [row["common_name"] for row in csv.DictReader(open(LABELS, encoding="utf-8"))]
for i, row in enumerate(scores):
    top = np.argsort(-row)[:3]
    print(f"{3 * i:3d}s  " + ", ".join(f"{names[j]} {row[j]:.2f}" for j in top))
