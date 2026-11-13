"""Find chunks that sound like a query, using BirdNET+ V3.0 embeddings.

Embeddings (1280-d) are useful beyond the 11,725 classes: similarity search,
clustering, or training a small classifier for sounds the model does not know.

    python examples/embeddings_search.py example/soundscape.wav --query-start 18 --folder example/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import birdnet_v3 as bn  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("query", help="Audio file containing the query sound")
ap.add_argument("--query-start", type=float, default=0.0, help="Start of the 3 s query in seconds")
ap.add_argument("--folder", required=True, help="Folder of audio files to search")
ap.add_argument("--top", type=int, default=10, help="Number of matches to show")
args = ap.parse_args()

model = bn.Model()
start = int(args.query_start * bn.SAMPLE_RATE)
query = bn.load_audio(args.query)[None, start : start + 3 * bn.SAMPLE_RATE]
_, q = model.predict(query)

embs, where = [], []
for path in sorted(Path(args.folder).rglob("*.wav")):
    chunks, starts = bn.chunk(bn.load_audio(path), 3.0, overlap=1.5)
    embs.append(model.predict(chunks)[1])
    where += [(path, s) for s in starts]
embs = np.concatenate(embs)

# Cosine similarity: normalize, then dot product.
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)  # noqa: E731
sims = unit(embs) @ unit(q)[0]
for i in np.argsort(-sims)[: args.top]:
    print(f"{sims[i]:.3f}  {where[i][0]}  {where[i][1]:.1f}s")
