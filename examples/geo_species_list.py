"""Species list for a location and week from the BirdNET+ Geomodel.

Prints the species BirdNET+ V3.0 can detect that are likely at a place and time,
and writes them to a file you can pass to ``analyze.py --species``.

    python examples/geo_species_list.py --lat 42.48 --lon -76.45 --week 20 --lang de
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import birdnet_v3 as bn  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--week", type=int, default=0, help="1-48, four weeks per month (0 = whole year)")
ap.add_argument("--threshold", type=float, default=0.03, help="Minimum occurrence probability")
ap.add_argument("--lang", default="en", help="Common name language (default: en)")
ap.add_argument("-o", "--output", default="species_list.txt")
args = ap.parse_args()

labels = bn.load_labels()
name = "common_name" if args.lang == "en" else f"common_name_{args.lang}"
if name not in labels[0]:
    ap.error(f"unknown language '{args.lang}'")

probs = bn.GeoModel().predict(args.lat, args.lon, args.week)
hits = sorted(
    ((probs[r["scientific_name"]], r) for r in labels if probs.get(r["scientific_name"], 0) >= args.threshold),
    key=lambda h: -h[0],
)

for p, r in hits[:25]:
    print(f"{p:.2f}  {r['taxon_group']:<10} {r[name] or r['common_name']} ({r['scientific_name']})")
Path(args.output).write_text("\n".join(r["scientific_name"] for _, r in hits) + "\n", encoding="utf-8")
print(f"... {len(hits)} species -> {args.output}")
