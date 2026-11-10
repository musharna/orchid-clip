"""Build form_refs.npz, the demo's extra centroids.

``--members`` is a JSON object mapping a species name to the indices (into
``--rows``) of the photos to average: its photos of a color form, or its photos
that show no flower. Run this after every gallery rebuild. ``--append-to`` adds
the new centroids to an existing file, so a species can have one of each kind.

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

from orchid_clip.references import build_form_refs, concat_form_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="[N, D] L2-normed embeddings (.npy)")
    ap.add_argument("--members", required=True, help="JSON: species -> row indices")
    ap.add_argument("--gallery", required=True, help="centroid gallery (.npz)")
    ap.add_argument("--tower-dims", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-to", help="existing form_refs.npz to add to")
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
    n_new = len(out["binomials"])
    if args.append_to:
        base = np.load(args.append_to, allow_pickle=False)
        if str(base["ckpt_sha256_prefix"]) != str(gal["ckpt_sha256_prefix"]):
            raise SystemExit(
                f"{args.append_to} was built for {base['ckpt_sha256_prefix']} but "
                f"the gallery for {gal['ckpt_sha256_prefix']}"
            )
        out = concat_form_refs(dict(base), out)
    np.savez(args.out, ckpt_sha256_prefix=gal["ckpt_sha256_prefix"], **out)
    print(f"wrote {args.out}: {len(out['binomials'])} centroids ({n_new} new)")


if __name__ == "__main__":
    main()
