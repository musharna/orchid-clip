"""Head-to-head: orchid-CLIP (any version) vs BioCLIP 2 on the orchid val set.

Loads both models, embeds the same val images, embeds the same species labels
through the same 7-template prompt ensemble, and reports:

  - top-1 species classification accuracy (image -> nearest species text)
  - top-5 species accuracy
  - per-genus breakdown for genera with >= MIN_PER_GENUS val examples
  - genus-level top-1 accuracy (predicted_species -> predicted_genus match)

Val set comes from the same split logic the trainer uses (bucket < 0.02 of
md5(source_id)). Filters identical to dataset.py except we don't enforce
min_images_per_species at val time — we want the long tail too.

Run on desktop (v8 / open_clip):
    python \\
        scripts/eval_bioclip_vs_orchid_clip.py \\
        --orchid-clip ./orchid-clip-v8 \\
        --bioclip ~/bioclip2/open_clip_pytorch_model.bin \\
        --db ./orchid_images.db \\
        --image-root ./images \\
        --max-val 4000 \\
        --use-species-accepted \\
        --out ./eval_out/bioclip_vs_v8.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchid_clip.embedder import load_embedder  # noqa: E402

ENSEMBLE_TEMPLATES = [
    "a photograph of {} orchid flower",
    "a close-up photo of {} orchid",
    "a macro photograph of a {} flower",
    "{}, an orchid species",
    "a picture of {} in bloom",
    "a botanical photograph of {}",
    "{} orchid, flowering",
]

MIN_PER_GENUS = 20  # only report per-genus accuracy when val has enough examples


def split_bucket(source_id: str) -> float:
    h = hashlib.md5(source_id.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") / (1 << 32)


def load_val_rows(
    db: str,
    image_root: str,
    val_fraction: float,
    max_val: int,
    use_species_accepted: bool = False,
    bucket_lo: float = 0.0,
    cap_per_species: int = 0,
    min_imgs_per_species: int = 0,
) -> list[dict]:
    """Load rows whose hash bucket lies in [bucket_lo, val_fraction).

    Default (bucket_lo=0.0, val_fraction=0.02) reproduces the §5.1 holdout.
    To score a held-back slice disjoint from §5.1 (e.g. for v13 P_conf
    construction), set bucket_lo=0.02, val_fraction=0.04.
    """
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    sql = """
      SELECT source_id, genus, species, species_accepted, local_path
      FROM images
      WHERE downloaded = 1 AND is_flower = 1
        AND species IS NOT NULL AND species != ''
        AND genus IS NOT NULL AND local_path IS NOT NULL
        AND LOWER(species) NOT IN
            ('hybrid','species','sp','sp.','spp','spp.','hybr','hyb','unknown','indet')
        AND (quality_score IS NULL OR quality_score >= 0.3)
    """
    rows = []
    relabeled = 0
    for r in conn.execute(sql):
        b = split_bucket(r["source_id"])
        if b < bucket_lo or b >= val_fraction:
            continue
        p = r["local_path"]
        if not p.startswith("/"):
            p = os.path.join(image_root, p)
        if not os.path.exists(p):
            continue
        sp_clean = r["species"].strip().strip("'").strip('"')
        genus = r["genus"].strip()
        binomial = f"{genus} {sp_clean}"
        if use_species_accepted:
            accepted = (r["species_accepted"] or "").strip()
            if accepted and accepted != binomial:
                parts = accepted.split(" ", 1)
                if len(parts) == 2:
                    genus, sp_clean = parts[0], parts[1]
                    binomial = accepted
                    relabeled += 1
        rows.append(
            {
                "path": p,
                "genus": genus,
                "species": sp_clean,
                "binomial": binomial,
            }
        )
        if max_val and not cap_per_species and len(rows) >= max_val:
            break
    conn.close()
    if cap_per_species:
        rng = random.Random(0)
        by_sp: dict[str, list] = defaultdict(list)
        for r in rows:
            by_sp[r["binomial"]].append(r)
        capped: list[dict] = []
        for sp, rs in by_sp.items():
            if min_imgs_per_species and len(rs) < min_imgs_per_species:
                continue
            if len(rs) > cap_per_species:
                rs = rng.sample(rs, cap_per_species)
            capped.extend(rs)
        rows = capped
        print(
            f"  cap_per_species={cap_per_species} min_imgs={min_imgs_per_species}: "
            f"{len(by_sp)} val species -> {len(rows)} rows after cap/min-filter"
        )
    if use_species_accepted:
        print(f"  species_accepted relabeled {relabeled} val rows")
    return rows


# ---------- orchid-CLIP (HF SigLIP2 OR open_clip, via embedder dispatch) ----------


class OrchidCLIP:
    """Generic wrapper over either HF SigLIP2 (v3, v5.2) or open_clip (v6+, v7).

    Dispatch is based on presence of model_config.json with framework=open_clip
    at the checkpoint root (see orchid_clip.embedder.is_openclip_ckpt).
    """

    def __init__(self, ckpt: str, device: str):
        self.device = device
        self.emb = load_embedder(ckpt, device)

    @torch.no_grad()
    def embed_images(self, paths: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except (OSError, ValueError):
                    imgs.append(Image.new("RGB", (224, 224)))
            pixels = torch.stack([self.emb.preprocess_image(im) for im in imgs], dim=0)
            ie = self.emb.encode_image(pixels)
            out.append(ie.cpu())
            if (i // batch) % 10 == 0:
                print(f"  orchid img {i + len(imgs)}/{len(paths)}")
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def embed_labels(self, labels: list[str]) -> np.ndarray:
        out = []
        for name in labels:
            chunks = []
            for tpl in ENSEMBLE_TEMPLATES:
                te = self.emb.encode_text([tpl.format(name)])
                chunks.append(te.cpu())
            avg = torch.stack(chunks, dim=0).mean(dim=0)
            avg = avg / avg.norm(dim=-1, keepdim=True)
            out.append(avg.numpy()[0])
        return np.stack(out, axis=0)


# ---------- BioCLIP 2 (open_clip ViT-L/14) ----------


class BioCLIP2:
    def __init__(self, ckpt_path: str, device: str):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt_path
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    @torch.no_grad()
    def embed_images(self, paths: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                try:
                    imgs.append(self.preprocess(Image.open(p).convert("RGB")))
                except (OSError, ValueError):
                    imgs.append(self.preprocess(Image.new("RGB", (224, 224))))
            x = torch.stack(imgs, dim=0).to(self.device)
            ie = self.model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            out.append(ie.cpu())
            if (i // batch) % 10 == 0:
                print(f"  bioclip img {i + len(imgs)}/{len(paths)}")
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def embed_labels(self, labels: list[str]) -> np.ndarray:
        out = []
        for name in labels:
            chunks = []
            for tpl in ENSEMBLE_TEMPLATES:
                tok = self.tokenizer([tpl.format(name)]).to(self.device)
                te = self.model.encode_text(tok)
                te = te / te.norm(dim=-1, keepdim=True)
                chunks.append(te.cpu())
            avg = torch.stack(chunks, dim=0).mean(dim=0)
            avg = avg / avg.norm(dim=-1, keepdim=True)
            out.append(avg.numpy()[0])
        return np.stack(out, axis=0)


def score_model(
    name: str,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    labels: list[str],
    rows: list[dict],
) -> dict:
    sims = img_emb @ txt_emb.T  # [N_val, N_labels]
    pred1 = np.argsort(-sims, axis=1)[:, 0]
    pred5 = np.argsort(-sims, axis=1)[:, :5]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    n = len(rows)
    correct1 = 0
    correct5 = 0
    correct_genus = 0
    per_genus_total: dict[str, int] = defaultdict(int)
    per_genus_correct: dict[str, int] = defaultdict(int)
    per_species_total: dict[str, int] = defaultdict(int)
    per_species_correct: dict[str, int] = defaultdict(int)

    for i, r in enumerate(rows):
        true_label = r["binomial"]
        if true_label not in label_to_idx:
            continue
        true_idx = label_to_idx[true_label]
        if pred1[i] == true_idx:
            correct1 += 1
            per_genus_correct[r["genus"]] += 1
            per_species_correct[true_label] += 1
        if true_idx in pred5[i]:
            correct5 += 1
        if labels[pred1[i]].split()[0] == r["genus"]:
            correct_genus += 1
        per_genus_total[r["genus"]] += 1
        per_species_total[true_label] += 1

    per_genus = {
        g: {"n": per_genus_total[g], "top1": per_genus_correct[g] / per_genus_total[g]}
        for g in per_genus_total
        if per_genus_total[g] >= MIN_PER_GENUS
    }
    per_species = {
        s: {
            "n": per_species_total[s],
            "top1": per_species_correct[s] / per_species_total[s],
        }
        for s in per_species_total
    }
    return {
        "model": name,
        "n_val": n,
        "n_labels": len(labels),
        "top1": correct1 / n,
        "top5": correct5 / n,
        "genus_top1": correct_genus / n,
        "per_genus": dict(sorted(per_genus.items(), key=lambda kv: -kv[1]["n"])),
        "per_species": dict(sorted(per_species.items(), key=lambda kv: -kv[1]["n"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchid-clip", required=True)
    ap.add_argument("--bioclip", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--image-root", default="")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument(
        "--cap-per-species",
        type=int,
        default=0,
        help="If >0, tail-balance the val set: scan all val rows, keep up to N per species "
        "(seeded sample), drop species below --min-imgs-per-species. Disables --max-val early-break.",
    )
    ap.add_argument(
        "--min-imgs-per-species",
        type=int,
        default=0,
        help="With --cap-per-species, drop species with fewer than this many val images.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--use-species-accepted",
        action="store_true",
        help="Relabel val rows via images.species_accepted (POWO dedup). Use for v8 native-label-space metrics.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading val rows...")
    rows = load_val_rows(
        args.db,
        args.image_root,
        args.val_fraction,
        args.max_val,
        use_species_accepted=args.use_species_accepted,
        cap_per_species=args.cap_per_species,
        min_imgs_per_species=args.min_imgs_per_species,
    )
    print(f"  {len(rows)} val rows")
    labels = sorted({r["binomial"] for r in rows})
    print(f"  {len(labels)} unique species labels")
    paths = [r["path"] for r in rows]

    oc_tag = (
        os.path.basename(os.path.dirname(args.orchid_clip.rstrip("/"))) or "orchid_clip"
    )

    print(f"\n=== orchid-CLIP ({oc_tag}) ===")
    t0 = time.time()
    oc = OrchidCLIP(args.orchid_clip, device)
    oc_img = oc.embed_images(paths)
    oc_txt = oc.embed_labels(labels)
    oc_score = score_model(oc_tag, oc_img, oc_txt, labels, rows)
    print(
        f"  top1={oc_score['top1']:.3f}  top5={oc_score['top5']:.3f}  genus_top1={oc_score['genus_top1']:.3f}  ({time.time() - t0:.0f}s)"
    )
    del oc
    torch.cuda.empty_cache()

    print("\n=== BioCLIP 2 ===")
    t0 = time.time()
    bc = BioCLIP2(args.bioclip, device)
    bc_img = bc.embed_images(paths)
    bc_txt = bc.embed_labels(labels)
    bc_score = score_model("bioclip-2", bc_img, bc_txt, labels, rows)
    print(
        f"  top1={bc_score['top1']:.3f}  top5={bc_score['top5']:.3f}  genus_top1={bc_score['genus_top1']:.3f}  ({time.time() - t0:.0f}s)"
    )

    out = {oc_tag: oc_score, "bioclip2": bc_score}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n=== summary ===")
    print(f"{'metric':18s} {oc_tag:>10s} {'bioclip2':>10s}  delta")
    for k in ("top1", "top5", "genus_top1"):
        print(
            f"{k:18s} {oc_score[k]:>10.3f} {bc_score[k]:>10.3f}  "
            f"{oc_score[k] - bc_score[k]:+.3f}"
        )


if __name__ == "__main__":
    main()
