"""Build photo_rows.npz, the demo's photo-level references.

Inputs are the embeddings the centroid gallery was built from: one row per
reference photo, and for each row the index of its species in the gallery. Run
this after every gallery rebuild; the demo refuses to start on a file built for
another checkpoint or species list.

    python scripts/build_photo_rows.py --rows rows.npy --species-index sp.npy \
        --gallery image_gallery.npz --out photo_rows.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from orchid_clip.references import build_photo_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="[N, D] L2-normed embeddings (.npy)")
    ap.add_argument(
        "--species-index", required=True, help="[N] gallery row per photo (.npy)"
    )
    ap.add_argument("--gallery", required=True, help="centroid gallery (.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int, default=512)
    args = ap.parse_args()

    # The gallery stores species names as an object array, hence allow_pickle;
    # it is a file you built yourself.
    gal = np.load(args.gallery, allow_pickle=True)
    n_species = len(gal["binomials"])
    rows = np.load(args.rows, mmap_mode="r")
    sp = np.load(args.species_index)
    if "n_rows_used" in gal.files:
        counts = np.bincount(sp, minlength=n_species)
        bad = np.where(counts != np.asarray(gal["n_rows_used"]))[0]
        if len(bad):
            raise SystemExit(
                f"{len(bad)} species have a different number of rows here than the "
                f"gallery used (first: index {bad[0]}, {counts[bad[0]]} vs "
                f"{gal['n_rows_used'][bad[0]]}); these are not the gallery's rows"
            )
    out = build_photo_rows(rows, sp, n_species, dims=args.dims)
    out.pop("order")
    np.savez(args.out, ckpt_sha256_prefix=gal["ckpt_sha256_prefix"], **out)
    print(
        f"wrote {args.out}: {len(out['rows']):,} photos, {args.dims} dims, {n_species:,} species"
    )


if __name__ == "__main__":
    main()
