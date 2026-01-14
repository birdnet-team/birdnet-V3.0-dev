#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from typing import List, Tuple, Optional
import numpy as np
import torch
import librosa
import urllib.request
import urllib.error
import tempfile
import shutil

SR = 32000  # model expects 32 kHz

# Defaults and hardcoded URLs (replace with actual links)
DEFAULT_MODEL_PATH = "models/BirdNET+_V3.0-preview3_Global_11K_FP32.pt"
DEFAULT_LABELS_PATH = "models/BirdNET+_V3.0-preview3_Global_11K_Labels.csv"
DEFAULT_MODEL_URL = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_FP32.pt?download=1"
DEFAULT_LABELS_URL = "https://zenodo.org/records/18247420/files/BirdNET+_V3.0-preview3_Global_11K_Labels.csv?download=1"


def load_labels(labels_csv: str) -> List[str]:
    # CSV is semicolon-delimited with columns: id;sci_name;com_name;gbif;class;order
    labels = []
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sci = row.get("sci_name", "").strip()
            com = row.get("com_name", "").strip()
            labels.append(f"{sci}_{com}")
    if not labels:
        raise ValueError(f"No labels found in {labels_csv}")
    return labels


def chunk_audio(y: np.ndarray, chunk_length: float, overlap: float = 0.0, sr: int = SR) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Split audio into chunks with optional temporal overlap.

    Args:
        y: 1D numpy array (mono audio).
        chunk_length: Length of each chunk in seconds (>0).
        overlap: Overlap between consecutive chunks in seconds (0 <= overlap < chunk_length).
        sr: Sample rate.

    Returns:
        chunks: Float32 array of shape [N, chunk_samples]
        spans: List of (start_sec, end_sec) for each chunk (end_sec truncated to original audio length).
    """
    chunk_len = int(round(chunk_length * sr))
    if chunk_len <= 0:
        raise ValueError("chunk_length must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_length:
        raise ValueError("overlap must be < chunk_length")

    step = chunk_len - int(round(overlap * sr))
    if step <= 0:
        raise ValueError("Invalid step size (adjust overlap/chunk_length).")

    n = len(y)
    if n == 0:
        return np.zeros((0, chunk_len), dtype=np.float32), []

    starts = np.arange(0, n, step)
    chunks = []
    spans = []
    for s in starts:
        e = min(s + chunk_len, n)
        seg = y[s:e]
        if len(seg) < chunk_len:
            pad = np.zeros(chunk_len - len(seg), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        chunks.append(seg.astype(np.float32, copy=False))
        spans.append((s / sr, min(e, n) / sr))
        if e >= n:
            break
    return np.stack(chunks, axis=0), spans


def run_inference(
    model: torch.jit.ScriptModule,
    chunks: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run inference with the new model that returns (embeddings, predictions).

    Args:
        model: TorchScript model returning (embeddings, predictions).
        chunks: [N, T] float32 mono audio.
        device: torch.device for inference.
        batch_size: batch size.
        return_embeddings: if True, also return stacked embeddings [N, D].

    Returns:
        predictions: [N, C] float32
        embeddings: [N, D] float32 or None if return_embeddings=False
    """
    if chunks.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32), (np.zeros((0, 0), dtype=np.float32) if return_embeddings else None)

    preds_out: List[np.ndarray] = []
    embs_out: List[np.ndarray] = []

    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            x = torch.from_numpy(batch).to(device)  # [B, T]
            out = model(x)
            # Expect (embeddings, predictions)
            if not (isinstance(out, (tuple, list)) and len(out) == 2):
                raise RuntimeError("Model is expected to return (embeddings, predictions).")
            emb, pred = out

            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            preds_out.append(pred.detach().cpu().numpy().astype(np.float32))

            if return_embeddings:
                if emb.ndim == 1:
                    emb = emb.unsqueeze(0)
                embs_out.append(emb.detach().cpu().numpy().astype(np.float32))

    predictions = np.concatenate(preds_out, axis=0)
    embeddings = np.concatenate(embs_out, axis=0) if return_embeddings and embs_out else None
    return predictions, embeddings


def save_per_chunk_csv(
    audio_path: str,
    spans: List[Tuple[float, float]],
    probs_chunks: np.ndarray,
    labels: List[str],
    out_csv: str,
    min_conf: float,
    export_embeddings: bool = False,
    embeddings: Optional[np.ndarray] = None,
):
    """Save rows for every (chunk,label) with confidence >= min_conf, sorted by descending confidence per chunk.
    Columns:
      - name,start_sec,end_sec,confidence,label
      - if export_embeddings=True: add a single "embeddings" column containing the whole embedding vector
        serialized as a comma-separated string wrapped in quotes (handled by csv writer).
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(audio_path)
    rows = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)  # default delimiter=",", quotechar='"', QUOTE_MINIMAL
        header = ["name", "start_sec", "end_sec", "confidence", "label"]
        if export_embeddings:
            header += ["embeddings"]  # single column with the full vector as a quoted string
        w.writerow(header)

        for ci, ((start, end), probs) in enumerate(zip(spans, probs_chunks)):
            if probs.ndim != 1:
                probs = probs.ravel()
            idx = np.where(probs >= min_conf)[0]
            if idx.size == 0:
                continue
            sort_order = np.argsort(-probs[idx])
            # Prepare embedding string once per chunk if requested
            emb_str = None
            if export_embeddings and embeddings is not None and ci < len(embeddings):
                vec = embeddings[ci].ravel().astype(np.float32)
                # Comma-separated to force quoting in CSV
                emb_str = ",".join(f"{v}" for v in vec)

            for j in idx[sort_order]:
                conf = float(probs[j])
                row = [base, round(start, 3), round(end, 3), round(conf, 6), labels[j]]
                if export_embeddings:
                    row.append("" if emb_str is None else emb_str)
                w.writerow(row)
                rows += 1
    return rows


def _download_file(url: str, dst: str) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp:
            tmp_path = tmp.name
            with urllib.request.urlopen(url) as r:
                shutil.copyfileobj(r, tmp)
        os.replace(tmp_path, dst)
        return True
    except Exception as e:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        print(f"Error downloading {url} -> {dst}: {e}", file=sys.stderr)
        return False


def _download_defaults(model_path: str, labels_path: str) -> None:
    # Download default model if missing
    if model_path == DEFAULT_MODEL_PATH and not os.path.isfile(model_path):
        print(f"Default model not found. Downloading:\n  {DEFAULT_MODEL_URL}\n  -> {model_path}")
        if not _download_file(DEFAULT_MODEL_URL, model_path):
            print("Failed to download default model.", file=sys.stderr)
            sys.exit(1)
    # Download default labels if missing
    if labels_path == DEFAULT_LABELS_PATH and not os.path.isfile(labels_path):
        print(f"Default labels not found. Downloading:\n  {DEFAULT_LABELS_URL}\n  -> {labels_path}")
        if not _download_file(DEFAULT_LABELS_URL, labels_path):
            print("Failed to download default labels.", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run BirdNET+ V3.0 preview model on an audio file.")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to TorchScript .pt model")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Path to labels CSV")
    parser.add_argument("--chunk_length", type=float, default=3.0, help="Chunk length in seconds (default: 3.0)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap in seconds (default: 0.0; must be < chunk_length)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--out-csv", help="Path to write per-chunk CSV (default: <audio>.results.csv)")
    parser.add_argument("--min-conf", type=float, default=0.15, help="Minimum confidence threshold for exporting a detection (default: 0.15)")
    parser.add_argument("--export-embeddings", action="store_true", help="Include per-chunk embedding vector columns in the output CSV")
    args = parser.parse_args()
    
    print(f"BirdNET+ V3.0 developer preview run on {args.audio}")

    # Auto-download defaults if missing
    _download_defaults(args.model, args.labels)

    if not os.path.isfile(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.labels):
        print(f"Error: labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)

    try:
        labels = load_labels(args.labels)
    except Exception as e:
        print(f"Error loading labels: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        device = torch.device(args.device)
        model = torch.jit.load(args.model, map_location=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        y, sr = librosa.load(args.audio, sr=SR, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)

    chunks, spans = chunk_audio(y, args.chunk_length, overlap=args.overlap, sr=SR)
    if len(chunks) == 0:
        print("No audio samples to process.", file=sys.stderr)
        sys.exit(1)

    probs_chunks, embeddings = run_inference(
        model, chunks, device=device, return_embeddings=args.export_embeddings
    )

    out_csv = args.out_csv if args.out_csv else (os.path.splitext(args.audio)[0] + ".results.csv")
    rows = save_per_chunk_csv(
        args.audio,
        spans,
        probs_chunks,
        labels,
        out_csv,
        args.min_conf,
        export_embeddings=args.export_embeddings,
        embeddings=embeddings,
    )

    print(f"Chunks processed: {len(chunks)}; detections exported: {rows} (min_conf={args.min_conf}, overlap={args.overlap}, export_embeddings={args.export_embeddings})")
    print(f"CSV: {out_csv}")
    print(f"SR={SR}, Device={device}")


if __name__ == "__main__":
    main()