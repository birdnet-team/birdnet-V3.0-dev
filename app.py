#!/usr/bin/env python3
import os
import io
import sys
import csv
import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch

from analyze import (
    load_labels,
    chunk_audio,
    run_inference,
    _download_defaults,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
    SR,
)

st.set_page_config(page_title="BirdNET+ V3.0 Preview 1", layout="wide")

# ---------------------------
# Caching
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device_str: str = "cpu"):
    device = torch.device(device_str)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device

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
# Default paths only for now
# ---------------------------
model_path = DEFAULT_MODEL_PATH
labels_path = DEFAULT_LABELS_PATH

# ---------------------------
# UI Sidebar
# ---------------------------
logo_path = "logo-birdnet-circle.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width='content')

st.sidebar.header("Settings")
chunk_length = st.sidebar.number_input("Chunk length (s)", min_value=0.5, max_value=30.0, value=3.0, step=1.0)
overlap = st.sidebar.number_input("Overlap (s)", min_value=0.0, max_value=29.9, value=0.0, step=0.5)
min_conf = st.sidebar.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
agg_method = st.sidebar.selectbox("Aggregate method", options=["mean", "max"], index=0)
top_n = st.sidebar.number_input("Top-N overall", min_value=1, max_value=25, value=10, step=1)
device_choice = st.sidebar.selectbox("Device", options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0)
playback_enabled = st.sidebar.checkbox("Show audio player", value=True)
show_spectrogram = st.sidebar.checkbox("Show spectrogram", value=True)

st.title("BirdNET+ V3.0 Developer Preview")
st.caption("Developer preview; models, labels, and outputs may change with future releases.")

# ---------------------------
# File Uploader
# ---------------------------
audio_bytes = None
uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
if uploaded:
    st.write(f"File: {uploaded.name} ({uploaded.type}, {uploaded.size/1024:.1f} KB)")
    try:
        audio_bytes = uploaded.getvalue()
        y = load_audio_bytes(audio_bytes, sr=SR)
        if show_spectrogram:
            S_db = compute_mel_spectrogram(y, sr=SR)
            fig, ax = plt.subplots(figsize=(15, 3))
            librosa.display.specshow(S_db, sr=SR, hop_length=256, ax=ax)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            st.pyplot(fig, width="stretch")
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
if uploaded and y is not None and len(y) > 0:
    try:
        labels = cache_labels(labels_path)
    except Exception as e:
        st.error(f"Label load error: {e}")
        st.stop()

    if not os.path.isfile(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()

    try:
        model, device = load_model(model_path, device_str=device_choice)
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    duration = len(y) / SR
    st.write(f"Audio duration: {duration:.2f} s, Sample rate: {SR} Hz")

    # Chunking
    try:
        chunks, spans = chunk_audio(y, chunk_length=chunk_length, overlap=overlap, sr=SR)
    except Exception as e:
        st.error(f"Chunking error: {e}")
        st.stop()

    st.write(f"Chunks: {len(chunks)} (length={chunk_length:.2f}s, overlap={overlap:.2f}s)")

    # Inference
    with st.spinner("Running inference..."):
        probs_chunks = run_inference(model, chunks, device=device)

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
    else:
        st.info("No detections above threshold.")
else:
    if not uploaded:
        st.info("Upload an audio file to run inference automatically.")

# Footer
st.markdown("---")
st.caption("BirdNET+ V3.0 Preview.")