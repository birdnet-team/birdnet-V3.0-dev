#!/usr/bin/env python3
"""Streamlit demo for BirdNET+ V3.0.  Run with:  streamlit run app.py"""

import io
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import birdnet_v3 as bn

ROOT = Path(__file__).resolve().parent
LOGO = str(ROOT / "img" / "logo-birdnet-circle.png")
EXAMPLE = ROOT / "example" / "soundscape.wav"

st.set_page_config(page_title="BirdNET+ V3.0", page_icon=LOGO, layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str) -> bn.Model:
    return bn.Model(path)


@st.cache_resource(show_spinner="Loading Geomodel...")
def get_geomodel() -> bn.GeoModel:
    return bn.GeoModel()


@st.cache_data
def get_labels() -> pd.DataFrame:
    return pd.DataFrame(bn.load_labels())


@st.cache_data(show_spinner="Running BirdNET+...")
def analyze(data: bytes, model_path: str, chunk_len: float, overlap: float):
    audio = bn.load_audio(io.BytesIO(data))
    chunks, starts = bn.chunk(audio, chunk_len, overlap)
    scores, _ = get_model(model_path).predict(chunks)
    return audio, scores, starts


def plot_spectrogram(audio: np.ndarray, n_fft: int = 1024, max_frames: int = 3000) -> plt.Figure:
    # At most ``max_frames`` frames, so long recordings stay fast and small in memory.
    audio = np.pad(audio, (0, max(0, n_fft - len(audio))))
    hop = max(n_fft // 4, (len(audio) - n_fft) // max_frames + 1)
    starts = np.arange(0, len(audio) - n_fft + 1, hop)
    frames = audio[starts[:, None] + np.arange(n_fft)] * np.hanning(n_fft).astype(np.float32)
    db = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-6)
    fig, ax = plt.subplots(figsize=(14, 2.5))
    extent = (0, len(audio) / bn.SAMPLE_RATE, 0, bn.SAMPLE_RATE / 2000)
    ax.imshow(db, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=np.percentile(db, 50))
    ax.set(xlabel="Time (s)", ylabel="kHz", ylim=(0, 12))
    fig.tight_layout()
    return fig


# --- Sidebar -------------------------------------------------------------------
st.sidebar.image(LOGO, width=120)
models = sorted(str(p) for ext in ("onnx", "tflite", "pt") for p in bn.MODEL_DIR.glob(f"{bn.MODEL_NAME}*.{ext}"))
if str(bn.DEFAULT_MODEL) not in models:
    models.insert(0, str(bn.DEFAULT_MODEL))  # downloaded on first use
model_path = st.sidebar.selectbox(
    "Model",
    models,
    index=models.index(str(bn.DEFAULT_MODEL)),
    format_func=lambda p: Path(p).name.removeprefix(bn.MODEL_NAME + "_"),
)
labels = get_labels()
langs = [c.removeprefix("common_name_") for c in labels.columns if c.startswith("common_name_")]
lang = st.sidebar.selectbox("Common names", langs)
min_conf = st.sidebar.slider("Minimum confidence", 0.05, 1.0, 0.25, 0.05)
chunk_len = st.sidebar.number_input("Chunk length (s)", 1.0, 30.0, 3.0, 0.5)
overlap = st.sidebar.number_input("Overlap (s)", 0.0, chunk_len - 0.5, 0.0, 0.5)
groups = st.sidebar.multiselect("Taxon groups", sorted(labels["taxon_group"].unique()), placeholder="All")

use_geo = st.sidebar.toggle("Filter by location (Geomodel)")
if use_geo:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 42.48, format="%.4f")
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, -76.45, format="%.4f")
    today = date.today()  # Geomodel weeks: four per month, 1-48
    week = st.sidebar.slider("Week (0 = whole year)", 0, 48, (today.month - 1) * 4 + min(3, (today.day - 1) // 7) + 1)

# --- Main ------------------------------------------------------------------------
st.title("BirdNET+ V3.0")
st.caption(
    "11,709 species and 16 non-animal sound classes · 32 kHz audio · "
    "[Models on Zenodo](https://zenodo.org/records/24162518)"
)

uploaded = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "mp3"])
if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif st.button("Try the example recording", on_click=lambda: st.session_state.update(example=True)) or (
    st.session_state.get("example") or "example" in st.query_params
):
    data, name = EXAMPLE.read_bytes(), EXAMPLE.name
else:
    st.stop()

audio, scores, starts = analyze(data, model_path, chunk_len, overlap)
st.audio(data)
st.pyplot(plot_spectrogram(audio))

keep = np.ones(len(labels), dtype=bool)
if groups:
    keep &= labels["taxon_group"].isin(groups).to_numpy()
if use_geo:
    keep &= get_geomodel().mask(labels.to_dict("records"), lat, lon, week)
scores = np.where(keep, scores, 0)

name_col = "common_name" if lang == "en" else f"common_name_{lang}"
names = labels[name_col].where(labels[name_col] != "", labels["common_name"])
c, j = np.nonzero(scores >= min_conf)
duration = len(audio) / bn.SAMPLE_RATE
det = pd.DataFrame(
    {
        "start_sec": starts[c],
        "end_sec": np.minimum(starts[c] + chunk_len, duration),
        "common_name": names.to_numpy()[j],
        "scientific_name": labels["scientific_name"].to_numpy()[j],
        "confidence": scores[c, j].round(4),
    }
).sort_values(["start_sec", "confidence"], ascending=[True, False])

if det.empty:
    st.info("No detections above the confidence threshold.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Top species")
    top = det.groupby("common_name")["confidence"].max().sort_values(ascending=False).head(15)
    st.bar_chart(top, horizontal=True)
with col2:
    st.subheader(f"Detections ({len(det)})")
    st.dataframe(det, hide_index=True, height=380)

stem = Path(name).stem
audacity = "\n".join(f"{r.start_sec}\t{r.end_sec}\t{r.common_name} {r.confidence:.0%}" for r in det.itertuples())
b1, b2, _ = st.columns([1, 1, 4])
b1.download_button("Download CSV", det.to_csv(index=False), f"{stem}.BirdNET.csv", "text/csv")
b2.download_button("Audacity labels", audacity, f"{stem}.BirdNET.txt", "text/plain")
st.caption(f"{duration:.1f} s · {len(starts)} chunks · {Path(model_path).name}")
