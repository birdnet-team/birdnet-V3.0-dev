"""BirdNET+ V3.0 inference with PyTorch (TorchScript), on CPU or GPU.

    pip install numpy scipy soundfile torch
    curl -L --create-dirs -o models/BirdNET+_V3.0_Global_12K_FP32.pt \
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP32.pt?download=1"
    python examples/torchscript_minimal.py
"""

import csv

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

MODEL = "models/BirdNET+_V3.0_Global_12K_FP32.pt"
LABELS = "models/BirdNET+_V3.0_Global_12K_Labels.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

audio, sr = sf.read("example/soundscape.wav", dtype="float32", always_2d=True)
audio = resample_poly(audio.mean(axis=1), 32000, sr).astype(np.float32)
x = torch.from_numpy(audio[: 10 * 3 * 32000].reshape(10, -1)).to(device)  # 10 chunks of 3 s

model = torch.jit.load(MODEL, map_location=device).eval()
with torch.inference_mode():
    scores, embeddings = model(x)  # [10, 11725], [10, 1280]

names = [row["common_name"] for row in csv.DictReader(open(LABELS, encoding="utf-8"))]
for i, row in enumerate(scores.cpu().numpy()):
    top = np.argsort(-row)[:3]
    print(f"{3 * i:3d}s  " + ", ".join(f"{names[j]} {row[j]:.2f}" for j in top))
