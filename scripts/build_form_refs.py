"""Build form_refs.npz, the demo's second centroids for species with a color form.

``--members`` is a JSON object mapping a species name to the indices (into
``--rows``) of its photos of the form. Run this after every gallery rebuild.

    python scripts/build_form_refs.py --rows rows.npy --members forms.json \
        --gallery image_gallery.npz --tower-dims 768 1024 --out form_refs.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from orchid_clip.references import build_form_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="[N, D] L2-normed embeddings (.npy)")
    ap.add_argument("--members", required=True, help="JSON: species -> row indices")
    ap.add_argument("--gallery", required=True, help="centroid gallery (.npz)")
    ap.add_argument("--tower-dims", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Species names are stored as an object array, hence allow_pickle.
    gal = np.load(args.gallery, allow_pickle=True)
    known = {str(b) for b in gal["binomials"]}
    with open(args.members) as fh:
        members = json.load(fh)
    unknown = sorted(set(members) - known)
    if unknown:
        raise SystemExit(f"species not in the gallery: {unknown[:5]}")
    out = build_form_refs(np.load(args.rows, mmap_mode="r"), members, args.tower_dims)
    np.savez(args.out, ckpt_sha256_prefix=gal["ckpt_sha256_prefix"], **out)
    print(f"wrote {args.out}: {len(out['binomials'])} species")


if __name__ == "__main__":
    main()
