"""Minimal reference code for running BirdNET+ V3.0 models.

One small module, no framework: load audio, split it into chunks, run any of the
released model formats (ONNX, LiteRT/TFLite, TorchScript), and optionally filter
species with the BirdNET+ Geomodel. ``analyze.py``, ``app.py`` and the examples
build on it. For a maintained, pip-installable API, use the ``birdnet`` package:
https://github.com/birdnet-team/birdnet
"""

from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 32000
EMBEDDING_DIM = 1280

MODEL_NAME = "BirdNET+_V3.0_Global_12K"
MODEL_URL = "https://zenodo.org/records/24162518/files/{}?download=1"
MODEL_DIR = Path(os.environ.get("BIRDNET_MODEL_DIR", "models"))
DEFAULT_MODEL = MODEL_DIR / f"{MODEL_NAME}_FP16.onnx"
DEFAULT_LABELS = MODEL_DIR / f"{MODEL_NAME}_Labels.csv"

GEO_NAME = "BirdNET+_Geomodel_V3.0.4_Global_14K"
GEO_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/{}"
DEFAULT_GEO_MODEL = MODEL_DIR / f"{GEO_NAME}_FP16.onnx"
DEFAULT_GEO_LABELS = MODEL_DIR / f"{GEO_NAME}_Labels.txt"


def ensure_file(path: str | Path) -> Path:
    """Return ``path``, downloading it first if it is a missing release file."""
    path = Path(path)
    if path.is_file():
        return path
    name = path.name
    if name.startswith(MODEL_NAME):
        url = MODEL_URL.format(name)
    elif name.startswith(GEO_NAME):
        url = GEO_URL.format(name.replace("+", "%2B"))
    else:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def load_labels(path: str | Path = DEFAULT_LABELS) -> list[dict[str, str]]:
    """Read the labels CSV: one row per model output, in output order."""
    with open(ensure_file(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as float32 at ``sr`` Hz; channels are averaged."""
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        audio = resample_poly(audio, sr // g, file_sr // g).astype(np.float32)
    return audio


def chunk(
    audio: np.ndarray, length: float = 3.0, overlap: float = 0.0, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """Split audio into fixed-length chunks; the last one is zero-padded.

    Returns ``(chunks [N, samples], start times in seconds [N])``.
    """
    size, step = int(length * sr), int((length - overlap) * sr)
    if step <= 0:
        raise ValueError("overlap must be smaller than the chunk length")
    starts = np.arange(0, max(len(audio) - size, 0) + step, step)
    starts = starts[starts < max(len(audio), 1)]
    chunks = np.zeros((len(starts), size), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = audio[s : s + size]
        chunks[i, : len(seg)] = seg
    return chunks, starts / sr


class Model:
    """BirdNET+ V3.0 acoustic model; the backend is picked from the file extension.

    ``predict`` returns ``(scores [N, classes], embeddings [N, 1280])``. Scores are
    independent per class (0..1), so several species can score high at once.
    """

    def __init__(self, path: str | Path = DEFAULT_MODEL, threads: int | None = None, device: str = "cpu"):
        self.path = ensure_file(path)
        ext = self.path.suffix.lower()
        if ext == ".onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(self.path), opts, providers=providers)
            self._run = self._run_onnx
        elif ext == ".tflite":
            from ai_edge_litert.interpreter import Interpreter

            self._interp = Interpreter(str(self.path), num_threads=threads or os.cpu_count())
            self._input = self._interp.get_input_details()[0]["index"]
            self._shape = None
            self._run = self._run_tflite
        elif ext == ".pt":
            import torch

            if threads:
                torch.set_num_threads(threads)
            self._torch = torch
            self._device = torch.device(device)
            self._module = torch.jit.load(str(self.path), map_location=self._device).eval()
            self._run = self._run_torch
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    @staticmethod
    def _split(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Outputs are (scores, embeddings) in some order; the embedding is 1280-d.
        emb = next(o for o in outputs if o.shape[-1] == EMBEDDING_DIM)
        scores = next(o for o in outputs if o is not emb)
        return scores, emb

    def _run_onnx(self, x: np.ndarray) -> list[np.ndarray]:
        return self._session.run(None, {self._session.get_inputs()[0].name: x})

    def _run_tflite(self, x: np.ndarray) -> list[np.ndarray]:
        # The LiteRT models have a fixed batch size of 1 but accept any length.
        it, results = self._interp, []
        for row in x:
            if self._shape != row.shape:
                it.resize_tensor_input(self._input, (1, len(row)))
                it.allocate_tensors()
                self._shape = row.shape
            it.set_tensor(self._input, row[None])
            it.invoke()
            results.append([it.get_tensor(o["index"])[0] for o in it.get_output_details()])
        return [np.stack(r) for r in zip(*results)]

    def _run_torch(self, x: np.ndarray) -> list[np.ndarray]:
        with self._torch.inference_mode():
            out = self._module(self._torch.from_numpy(x).to(self._device))
        return [o.float().cpu().numpy() for o in out]

    def predict(self, chunks: np.ndarray, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Run the model on ``chunks`` ([N, samples] float32 at 32 kHz)."""
        scores, embs = [], []
        for i in range(0, len(chunks), batch_size):
            s, e = self._split(self._run(np.ascontiguousarray(chunks[i : i + batch_size], dtype=np.float32)))
            scores.append(s)
            embs.append(e)
        if not scores:
            return np.zeros((0, 0), np.float32), np.zeros((0, EMBEDDING_DIM), np.float32)
        # Digital silence (zero variance) yields NaN; treat it as "nothing detected".
        return (
            np.nan_to_num(np.concatenate(scores)).astype(np.float32),
            np.nan_to_num(np.concatenate(embs)).astype(np.float32),
        )


class GeoModel:
    """BirdNET+ Geomodel: species occurrence probability for a location and week.

    See https://github.com/birdnet-team/geomodel. Weeks run from 1 to 48
    (four per month); week 0 or ``None`` returns the maximum over the year.
    """

    def __init__(self, path: str | Path = DEFAULT_GEO_MODEL, labels: str | Path = DEFAULT_GEO_LABELS):
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(ensure_file(path)), providers=["CPUExecutionProvider"])
        with open(ensure_file(labels), encoding="utf-8") as f:
            self.species = [line.rstrip("\n").split("\t")[1] for line in f]

    def predict(self, lat: float, lon: float, week: int | None = None) -> dict[str, float]:
        """Return ``{scientific name: probability}`` for all geomodel species."""
        weeks = range(1, 49) if not week else [week]
        x = np.array([[lat, lon, w] for w in weeks], dtype=np.float32)
        probs = self._session.run(None, {self._session.get_inputs()[0].name: x})[0].max(axis=0)
        return dict(zip(self.species, probs.tolist()))

    def mask(
        self, labels: list[dict[str, str]], lat: float, lon: float, week: int | None = None, threshold: float = 0.03
    ) -> np.ndarray:
        """Boolean mask over acoustic labels: True where the species is plausible.

        Classes the geomodel does not cover (e.g., non-animal sounds) stay True.
        """
        probs = self.predict(lat, lon, week)
        return np.array([probs.get(r["scientific_name"], 1.0) >= threshold for r in labels])
