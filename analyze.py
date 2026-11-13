#!/usr/bin/env python3
"""Analyze audio files or folders with BirdNET+ V3.0 and write detections to CSV.

Examples:
    python analyze.py example/soundscape.wav
    python analyze.py recordings/ --min-conf 0.5 --lang de -o results.csv
    python analyze.py recordings/ --lat 42.48 --lon -76.45 --week 20
    python analyze.py example/soundscape.wav --model models/BirdNET+_V3.0_Global_12K_FP16.tflite
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import birdnet_v3 as bn

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def find_audio(inputs: list[str]) -> list[Path]:
    files = []
    for p in map(Path, inputs):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT) if p.is_dir() else [p]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Audio files and/or folders (searched recursively)")
    ap.add_argument("-o", "--output", default="results.csv", help="Output CSV (default: results.csv)")
    ap.add_argument("--model", default=bn.DEFAULT_MODEL, help="Model file: .onnx, .tflite or .pt (default: FP16 ONNX)")
    ap.add_argument("--labels", default=bn.DEFAULT_LABELS, help="Labels CSV")
    ap.add_argument("--min-conf", type=float, default=0.25, help="Minimum score to report (default: 0.25)")
    ap.add_argument("--chunk", type=float, default=3.0, help="Chunk length in seconds (default: 3.0)")
    ap.add_argument("--overlap", type=float, default=0.0, help="Overlap between chunks in seconds (default: 0)")
    ap.add_argument("--lang", default="en", help="Common name language, e.g. de, fr, es_MX (default: en)")
    ap.add_argument("--species", help="Text file with scientific names to keep, one per line")
    ap.add_argument("--groups", nargs="+", help="Keep only these taxon groups, e.g. Aves Amphibia")
    ap.add_argument("--lat", type=float, help="Latitude for the Geomodel species filter")
    ap.add_argument("--lon", type=float, help="Longitude for the Geomodel species filter")
    ap.add_argument("--week", type=int, default=0, help="Week 1-48 for the Geomodel filter (0 = whole year)")
    ap.add_argument("--geo-threshold", type=float, default=0.03, help="Geomodel occurrence threshold (default: 0.03)")
    ap.add_argument("--embeddings", help="Also save per-chunk embeddings to this .npz file")
    ap.add_argument("--batch-size", type=int, default=16, help="Chunks per inference batch (default: 16)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: all)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="ONNX/TorchScript device")
    args = ap.parse_args()

    missing = [p for p in args.inputs if not Path(p).exists()]
    if missing:
        ap.error(f"not found: {', '.join(missing)}")
    if (args.lat is None) != (args.lon is None):
        ap.error("--lat and --lon must be given together")
    if not 0 <= args.week <= 48:
        ap.error("--week must be between 0 and 48")
    files = find_audio(args.inputs)
    if not files:
        ap.error(f"no audio files ({', '.join(sorted(AUDIO_EXT))}) found")

    labels = bn.load_labels(args.labels)
    name_col = f"common_name_{args.lang}" if args.lang != "en" else "common_name"
    if name_col not in labels[0]:
        langs = [k.removeprefix("common_name_") for k in labels[0] if k.startswith("common_name_")]
        ap.error(f"unknown language '{args.lang}', choose from: {', '.join(langs)}")

    # Which classes may be reported: species list, taxon groups, and location.
    keep = np.ones(len(labels), dtype=bool)
    if args.species:
        lines = Path(args.species).read_text(encoding="utf-8").splitlines()
        wanted = {line.strip() for line in lines if line.strip()}
        unknown = wanted - {r["scientific_name"] for r in labels}
        if unknown:
            print(f"Warning: {len(unknown)} names in {args.species} are not in the labels, e.g. {sorted(unknown)[0]}")
        keep &= np.array([r["scientific_name"] in wanted for r in labels])
    if args.groups:
        all_groups = sorted({r["taxon_group"] for r in labels})
        if set(args.groups) - set(all_groups):
            ap.error(f"unknown taxon group in {args.groups}, choose from: {', '.join(all_groups)}")
        keep &= np.array([r["taxon_group"] in args.groups for r in labels])
    if args.lat is not None:
        keep &= bn.GeoModel().mask(labels, args.lat, args.lon, args.week, args.geo_threshold)
    print(f"Reporting {keep.sum():,} of {len(labels):,} classes")

    model = bn.Model(args.model, threads=args.threads, device=args.device)
    all_embs, all_meta, n_rows = [], [], 0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["file", "start_sec", "end_sec", "scientific_name", "common_name", "confidence"])
        for i, path in enumerate(files, 1):
            try:
                audio = bn.load_audio(path)
            except Exception as e:  # unreadable or unsupported file
                print(f"[{i}/{len(files)}] skipped {path}: {e}")
                continue
            chunks, starts = bn.chunk(audio, args.chunk, args.overlap)
            scores, embs = model.predict(chunks, args.batch_size)
            if scores.shape[1] != len(labels):
                ap.error(f"{args.model} has {scores.shape[1]} outputs but {args.labels} has {len(labels)} rows")
            scores[:, ~keep] = 0
            for c, row in enumerate(scores):
                end = min(starts[c] + args.chunk, len(audio) / bn.SAMPLE_RATE)
                hits = np.flatnonzero(row >= args.min_conf)
                for j in hits[np.argsort(-row[hits])]:
                    r = labels[j]
                    out.writerow(
                        [
                            path,
                            f"{starts[c]:.2f}",
                            f"{end:.2f}",
                            r["scientific_name"],
                            r[name_col] or r["common_name"],
                            f"{row[j]:.4f}",
                        ]
                    )
                n_rows += len(hits)
            if args.embeddings:
                all_embs.append(embs)
                all_meta += [(str(path), s) for s in starts]
            print(f"[{i}/{len(files)}] {path}: {len(chunks)} chunks")

    if args.embeddings and all_embs:
        files_col, starts_col = zip(*all_meta)
        np.savez_compressed(
            args.embeddings, embeddings=np.concatenate(all_embs), files=np.array(files_col), starts=np.array(starts_col)
        )
        print(f"Embeddings: {args.embeddings}")
    print(f"{n_rows} detections -> {args.output}")


if __name__ == "__main__":
    main()
