"""Add reference photos to an existing photo_rows.npz.

Use this to add photos of another kind (the demo added photos that show no
flower) while leaving the rows already in the file byte-for-byte unchanged. The
new rows use the file's projection and scale. To rebuild everything, use
build_photo_rows.py instead.

    python scripts/append_photo_rows.py --base photo_rows.npz --rows new.npy \
        --species-index new_sp.npy --out photo_rows_new.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from orchid_clip.references import append_photo_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="existing photo_rows.npz")
    ap.add_argument(
        "--rows",
        required=True,
        help="[N, D] L2-normed embeddings (.npy); float64 is projected in float64",
    )
    ap.add_argument(
        "--species-index", required=True, help="[N] gallery row per photo (.npy)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-kept-energy",
        type=float,
        default=0.95,
        help="refuse if the projection keeps less than this share of the new rows",
    )
    args = ap.parse_args()

    base = dict(np.load(args.base, allow_pickle=False))
    out = append_photo_rows(base, np.load(args.rows), np.load(args.species_index))
    kept = float(out.pop("kept_energy"))
    n_new = int(out.pop("is_new").sum())
    if kept < args.min_kept_energy:
        raise SystemExit(
            f"the file's projection keeps only {kept:.3f} of the new rows "
            f"(minimum {args.min_kept_energy}); rebuild from all rows instead"
        )
    np.savez(args.out, ckpt_sha256_prefix=base["ckpt_sha256_prefix"], **out)
    print(
        f"wrote {args.out}: {len(out['rows']):,} photos ({n_new:,} new), "
        f"projection keeps {kept:.3f} of the new rows"
    )


if __name__ == "__main__":
    main()
