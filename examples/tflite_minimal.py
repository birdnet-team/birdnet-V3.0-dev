"""BirdNET+ V3.0 inference with LiteRT (TFLite), e.g., for Android, Raspberry Pi, or edge.

    pip install numpy scipy soundfile ai-edge-litert
    curl -L --create-dirs -o models/BirdNET+_V3.0_Global_12K_FP16_pruned.tflite \
      "https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP16_pruned.tflite?download=1"
    python examples/tflite_minimal.py
"""

import csv

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import resample_poly

MODEL = "models/BirdNET+_V3.0_Global_12K_FP16_pruned.tflite"
LABELS = "models/BirdNET+_V3.0_Global_12K_Labels.csv"

audio, sr = sf.read("example/soundscape.wav", dtype="float32", always_2d=True)
audio = resample_poly(audio.mean(axis=1), 32000, sr).astype(np.float32)

interpreter = Interpreter(MODEL, num_threads=4)
inp = interpreter.get_input_details()[0]
interpreter.resize_tensor_input(inp["index"], [1, 3 * 32000])  # batch size is fixed to 1
interpreter.allocate_tensors()
# Two outputs, (scores, embeddings); tell them apart by size (embeddings are 1280-d).
outs = sorted(interpreter.get_output_details(), key=lambda o: o["shape"][-1] == 1280)

names = [row["common_name"] for row in csv.DictReader(open(LABELS, encoding="utf-8"))]
for start in range(0, 30, 3):
    interpreter.set_tensor(inp["index"], audio[None, start * 32000 : (start + 3) * 32000])
    interpreter.invoke()
    scores = interpreter.get_tensor(outs[0]["index"])[0]
    top = np.argsort(-scores)[:3]
    print(f"{start:3d}s  " + ", ".join(f"{names[j]} {scores[j]:.2f}" for j in top))
