#!/usr/bin/env python3
import os
import io
import sys
import csv
import tempfile
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import time

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from analyze import (
    load_labels,
    chunk_audio,
    run_inference,
    _download_defaults,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
    SR,
)

st.set_page_config(page_title="BirdNET+ V3.0 Preview", layout="wide")

# ---------------------------
# Model Paths
# ---------------------------
MODEL_DIR = "models"
MODEL_BASE = "BirdNET+_V3.0-preview3.1_Global_11K"

# Available model variants (format: (display_name, extension, framework))
# Note: INT8 removed due to unreliable results with 11K-class softmax
MODEL_VARIANTS = [
    ("FP32 (PyTorch)", "_FP32.pt", "pytorch"),
    ("FP16 (ONNX)", "_FP16_pruned.onnx", "onnx"),
]

def get_model_path(suffix: str) -> str:
    """Get full path for a model variant."""
    if suffix.startswith("_"):
        return os.path.join(MODEL_DIR, f"{MODEL_BASE}{suffix}")
    return os.path.join(MODEL_DIR, f"{MODEL_BASE}{suffix}")

def get_available_models() -> List[Tuple[str, str, str]]:
    """Return list of (display_name, path, framework) for existing model files."""
    available = []
    for name, suffix, framework in MODEL_VARIANTS:
        path = get_model_path(suffix)
        if os.path.isfile(path):
            available.append((name, path, framework))
    return available

# ---------------------------
# Caching
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device_str: str = "cpu"):
    """Load PyTorch TorchScript model."""
    device = torch.device(device_str)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device

@st.cache_resource(show_spinner=False)
def load_onnx_model(model_path: str, device_str: str = "cpu"):
    """Load ONNX model with onnxruntime."""
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
    
    # Select execution provider based on device
    if device_str == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    session = ort.InferenceSession(model_path, providers=providers)
    return session

def run_onnx_inference(
    session: "ort.InferenceSession",
    chunks: np.ndarray,
    batch_size: int = 16,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run inference with ONNX model.
    
    Returns:
        predictions: [N, C] float32
        embeddings: [N, D] float32 or None
    """
    if chunks.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32), None
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    output_names = [o.name for o in session.get_outputs()]
    
    # Determine input dtype (handle FP16 models)
    if "float16" in input_type:
        input_dtype = np.float16
    else:
        input_dtype = np.float32
    
    preds_out: List[np.ndarray] = []
    embs_out: List[np.ndarray] = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size].astype(input_dtype)
        outputs = session.run(output_names, {input_name: batch})
        
        # Model outputs: predictions, embeddings (two outputs) or just predictions
        if len(outputs) == 2:
            pred, emb = outputs
            if return_embeddings:
                embs_out.append(emb.astype(np.float32))
        else:
            pred = outputs[0]
        
        preds_out.append(pred.astype(np.float32))
    
    predictions = np.concatenate(preds_out, axis=0)
    embeddings = np.concatenate(embs_out, axis=0) if return_embeddings and embs_out else None
    return predictions, embeddings

@st.cache_data(show_spinner=False)
def cache_labels(labels_path: str) -> List[str]:
    return load_labels(labels_path)

@st.cache_data(show_spinner=False)
def load_audio_bytes(b: bytes, sr: int = SR):
    y, _sr = librosa.load(io.BytesIO(b), sr=sr, mono=True)
    return y

@st.cache_data(show_spinner=False)
def compute_mel_spectrogram(y: np.ndarray, sr: int = SR):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# ---------------------------
# Default paths
# ---------------------------
labels_path = DEFAULT_LABELS_PATH

# ---------------------------
# UI Sidebar
# ---------------------------
logo_path = "img/logo-birdnet-circle.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width='content')

st.sidebar.header("Settings")

# Model format selector
available_models = get_available_models()
if available_models:
    model_options = [name for name, _, _ in available_models]
    selected_model_idx = st.sidebar.selectbox(
        "Model format",
        options=range(len(model_options)),
        format_func=lambda i: model_options[i],
        index=0,
        help="Select model precision. FP16 is recommended (half the size, same accuracy)."
    )
    selected_model_name, model_path, model_framework = available_models[selected_model_idx]
    
    # Show model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    st.sidebar.caption(f"Model size: {model_size_mb:.1f} MB")
else:
    st.sidebar.warning("No models found. Run analyze.py first to download the default model.")
    model_path = DEFAULT_MODEL_PATH
    model_framework = "pytorch"
    selected_model_name = "FP32 (PyTorch)"

chunk_length = st.sidebar.number_input("Chunk length (s)", min_value=0.5, max_value=30.0, value=3.0, step=1.0)
overlap = st.sidebar.number_input("Overlap (s)", min_value=0.0, max_value=29.9, value=0.0, step=0.5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=512, value=16, step=1)
min_conf = st.sidebar.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
agg_method = st.sidebar.selectbox("Aggregate method", options=["mean", "max"], index=0)
top_n = st.sidebar.number_input("Top-N overall", min_value=1, max_value=25, value=10, step=1)

# Device selector (PyTorch can use CUDA, ONNX can too if onnxruntime-gpu is installed)
if model_framework == "pytorch":
    device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
else:
    # ONNX - check for CUDA provider
    if ONNX_AVAILABLE:
        available_providers = ort.get_available_providers()
        device_options = ["cpu", "cuda"] if "CUDAExecutionProvider" in available_providers else ["cpu"]
    else:
        device_options = ["cpu"]

device_choice = st.sidebar.selectbox("Device", options=device_options, index=len(device_options) - 1)
playback_enabled = st.sidebar.checkbox("Show audio player", value=True)
show_spectrogram = st.sidebar.checkbox("Show spectrogram", value=True)

# Download species list (labels CSV)
if os.path.isfile(labels_path):
    with open(labels_path, "rb") as f:
        st.sidebar.download_button(
            "Download species list (CSV)",
            data=f.read(),
            file_name=os.path.basename(labels_path),
            mime="text/csv",
            type="secondary",
        )
    

st.title("BirdNET+ V3.0 Developer Preview 3.1 - 11K Species")
st.caption("Developer preview; models, labels, and outputs may change with future releases.")

# ---------------------------
# File Uploader
# ---------------------------
audio_bytes = None
uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"], )
if uploaded:
    st.write(f"File: {uploaded.name} ({uploaded.type}, {uploaded.size/1024:.1f} KB)")
    try:
        audio_bytes = uploaded.getvalue()
        y = load_audio_bytes(audio_bytes, sr=SR)
        if show_spectrogram:
            start_time = time.time()
            S_db = compute_mel_spectrogram(y, sr=SR)
            fig, ax = plt.subplots(figsize=(15, 3))
            librosa.display.specshow(S_db, sr=SR, hop_length=256, ax=ax)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            st.pyplot(fig, width="stretch")
            end_time = time.time()
            st.write(f"Spectrogram computation time: {end_time - start_time:.2f} s")
        if playback_enabled and audio_bytes:
            st.audio(audio_bytes, format=uploaded.type or "audio/wav")
    except Exception as e:
        st.error(f"Audio load error: {e}")
        y = None
else:
    y = None

# Auto-download defaults if needed
if uploaded:
    _download_defaults(model_path, labels_path)

# ---------------------------
# Main logic (auto-runs on any change)
# ---------------------------

# Helper to produce Audacity labels TSV
def make_tsv(csv_bytes: bytes) -> bytes:
    import math
    # Accept either bytes or str
    if isinstance(csv_bytes, (bytes, bytearray)):
        text = csv_bytes.decode("utf-8")
    else:
        text = str(csv_bytes)

    output = io.StringIO()
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        # Extract fields
        start_sec = row.get("start_sec", "")
        end_sec = row.get("end_sec", "")
        common_name = row.get("common_name", "")
        confidence_raw = row.get("confidence", "")

        # Convert confidence → percent with 2 decimals
        confidence = ""
        try:
            if confidence_raw not in ("", None, "nan", "NaN"):
                val = float(confidence_raw)
                if not math.isnan(val):
                    confidence = f"{val * 100:.2f}%"
        except Exception:
            confidence = ""

        values = [
            str(start_sec),
            str(end_sec),
            str(common_name) + ' ' + confidence,
        ]
        output.write("\t".join(values) + "\n")

    # Return bytes so the download button behaves like your CSV download
    return output.getvalue().encode("utf-8")

if uploaded and y is not None and len(y) > 0:
    try:
        labels = cache_labels(labels_path)
    except Exception as e:
        st.error(f"Label load error: {e}")
        st.stop()

    if not os.path.isfile(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()

    # Load model based on framework
    try:
        if model_framework == "pytorch":
            model, device = load_model(model_path, device_str=device_choice)
        else:
            model = load_onnx_model(model_path, device_str=device_choice)
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    duration = len(y) / SR
    st.write(f"Audio duration: {duration:.2f} s, Sample rate: {SR} Hz")
    st.write(f"Model: {selected_model_name}")

    # Chunking
    # Time inference
    start_time = time.time()
    try:
        chunks, spans = chunk_audio(y, chunk_length=chunk_length, overlap=overlap, sr=SR)
    except Exception as e:
        st.error(f"Chunking error: {e}")
        st.stop()

    st.write(f"Chunks: {len(chunks)} (length={chunk_length:.2f}s, overlap={overlap:.2f}s)")

    # Inference
    with st.spinner("Running inference..."):
        if model_framework == "pytorch":
            probs_chunks, _ = run_inference(model, chunks, device=device, batch_size=batch_size, return_embeddings=False)
        else:
            probs_chunks, _ = run_onnx_inference(model, chunks, batch_size=batch_size, return_embeddings=False)

    if probs_chunks.shape[-1] != len(labels):
        c = min(probs_chunks.shape[-1], len(labels))
        probs_chunks = probs_chunks[:, :c]
        labels = labels[:c]
        st.warning("Adjusted label count to match model output.")

    # Aggregate
    if agg_method == "mean":
        agg_scores = probs_chunks.mean(axis=0)
    else:
        agg_scores = probs_chunks.max(axis=0)

    # Top-N overall
    top_idx = np.argsort(-agg_scores)[:top_n]
    top_labels = [labels[i] for i in top_idx]
    top_values = agg_scores[top_idx]

    # Only show common names in chart
    common_names = []
    for lbl in top_labels:
        parts = lbl.split("_", 1)
        common_names.append(parts[1] if len(parts) == 2 else lbl)

    overall_df = pd.DataFrame({"common_name": common_names, "score": top_values})
    # Sort by score (desc) and preserve order via categorical index
    overall_df = overall_df.sort_values("score", ascending=False).reset_index(drop=True)
    overall_df["common_name"] = pd.Categorical(
        overall_df["common_name"], categories=overall_df["common_name"], ordered=True
    )

    st.subheader("Overall Aggregated Scores")
    st.bar_chart(overall_df.set_index("common_name"), width="stretch")

    # Per-chunk detections (thresholded)
    records = []
    for (start, end), probs in zip(spans, probs_chunks):
        idx = np.where(probs >= min_conf)[0]
        if idx.size == 0:
            continue
        sort_order = np.argsort(-probs[idx])
        for j in idx[sort_order]:
            lbl = labels[j]
            sci, com = (lbl.split("_", 1) + [""])[:2] if "_" in lbl else (lbl, "")
            records.append({
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "scientific_name": sci,
                "common_name": com,
                "confidence": float(probs[j]),
            })

    if records:
        per_chunk_df = pd.DataFrame(records)
        per_chunk_df = per_chunk_df.sort_values(["start_sec", "confidence"], ascending=[True, False])
        st.subheader("Per-Chunk Detections")
        st.dataframe(per_chunk_df, width="stretch")
        csv_bytes = per_chunk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name=f"{os.path.splitext(uploaded.name)[0]}_detections.csv",
            mime="text/csv",
        )
        tsv_bytes = make_tsv(csv_bytes)  # now returns bytes
        st.download_button(
            "Download Audacity Labels",
            data=tsv_bytes,
            file_name=f"{os.path.splitext(uploaded.name)[0]}_labels.txt",
            mime="text/tab-separated-values",  # or 'text/plain'
        )

    else:
        st.info("No detections above threshold.")
    end_time = time.time()
    st.write(f"Inference time: {end_time - start_time:.2f} s")
else:
    if not uploaded:
        st.info("Upload an audio file to run inference automatically.")

# Footer
st.markdown("---")
st.caption("BirdNET+ V3.0 Preview.")

