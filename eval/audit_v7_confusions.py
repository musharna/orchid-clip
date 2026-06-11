"""orchid-CLIP hard-negative audit: systematic top-1 confusions on the holdout.

Loads the same val set + label ensemble as eval_bioclip_vs_orchid_clip.py, runs
image→text classification for the given checkpoint, and buckets the WRONG
predictions by (true_genus, pred_genus). Splits errors into:

  - cross-genus: predicted genus != true genus (fixable with more data / better
    priors)
  - intra-genus: predicted genus == true genus but wrong species (often genuine
    taxonomic ambiguity — e.g. Ophrys x Ophrys, Pleurothallis x Pleurothallis)

For each top confusion pair, reports the most frequent (true_species →
pred_species) triples so the user can decide whether it's a data gap or a
legit taxonomic boundary problem.

Run on a CUDA host (current default = v8):

    python \\
        scripts/audit_v7_confusions.py \\
        --orchid-clip ./orchid-clip-v8 \\
        --db ./orchid_images.db \\
        --image-root ./images \\
        --max-val 4000 \\
        --use-species-accepted \\
        --out ./eval_out/confusions_v8.json \\
        --markdown ./eval_out/confusions_v8.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from eval_bioclip_vs_orchid_clip import (  # noqa: E402
    OrchidCLIP,
    load_val_rows,
)


TOP_N_CONFUSION_PAIRS = 30  # top pairs to report
TOP_N_EXAMPLES_PER_PAIR = 5  # species-level examples per (genus, genus) pair


def audit(
    rows: list[dict],
    labels: list[str],
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    *,
    top_n_pairs: int = TOP_N_CONFUSION_PAIRS,
    top_n_examples: int = TOP_N_EXAMPLES_PER_PAIR,
) -> dict:
    sims = img_emb @ txt_emb.T  # [N_val, N_labels]
    pred1 = np.argsort(-sims, axis=1)[:, 0]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Track errors
    cross_genus_pairs: Counter[tuple[str, str]] = Counter()
    intra_genus_pairs: Counter[str] = Counter()  # key = genus only
    species_errors: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(
        Counter
    )  # (true_genus, pred_genus) -> Counter((true_sp, pred_sp))
    # For cross-genus errors, how confident was the model (sim gap pred−true)?
    confidence_gap: dict[tuple[str, str], list[float]] = defaultdict(list)

    per_genus_total: Counter[str] = Counter()
    per_genus_correct: Counter[str] = Counter()

    n_skipped = 0
    for i, r in enumerate(rows):
        true_label = r["binomial"]
        if true_label not in label_to_idx:
            n_skipped += 1
            continue
        true_idx = label_to_idx[true_label]
        per_genus_total[r["genus"]] += 1

        if pred1[i] == true_idx:
            per_genus_correct[r["genus"]] += 1
            continue

        pred_label = labels[pred1[i]]
        pred_genus = pred_label.split()[0]
        pred_species = " ".join(pred_label.split()[1:])

        sp_true = r["binomial"].split()[1] if " " in r["binomial"] else ""
        key = (r["genus"], pred_genus)
        species_errors[key][(sp_true, pred_species)] += 1

        if pred_genus == r["genus"]:
            intra_genus_pairs[r["genus"]] += 1
        else:
            cross_genus_pairs[key] += 1
            confidence_gap[key].append(float(sims[i, pred1[i]] - sims[i, true_idx]))

    n_used = sum(per_genus_total.values())
    n_correct = sum(per_genus_correct.values())
    n_wrong = n_used - n_correct

    # Build report payload
    top_cross = []
    for (tg, pg), cnt in cross_genus_pairs.most_common(top_n_pairs):
        examples = species_errors[(tg, pg)].most_common(top_n_examples)
        gaps = confidence_gap[(tg, pg)]
        top_cross.append(
            {
                "true_genus": tg,
                "pred_genus": pg,
                "count": cnt,
                "frac_of_true_genus_total": cnt / max(1, per_genus_total[tg]),
                "mean_sim_gap": float(np.mean(gaps)) if gaps else 0.0,
                "median_sim_gap": float(np.median(gaps)) if gaps else 0.0,
                "examples": [
                    {"true_species": ts, "pred_species": ps, "count": c}
                    for (ts, ps), c in examples
                ],
            }
        )

    # Intra-genus summary: rank by count, report with denominator
    top_intra = []
    for g, cnt in intra_genus_pairs.most_common(top_n_pairs):
        examples = species_errors[(g, g)].most_common(top_n_examples)
        top_intra.append(
            {
                "genus": g,
                "count": cnt,
                "total": per_genus_total[g],
                "frac": cnt / max(1, per_genus_total[g]),
                "examples": [
                    {"true_species": ts, "pred_species": ps, "count": c}
                    for (ts, ps), c in examples
                ],
            }
        )

    return {
        "n_val": n_used,
        "n_skipped_missing_label": n_skipped,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "top1": n_correct / max(1, n_used),
        "cross_genus_errors": sum(cross_genus_pairs.values()),
        "intra_genus_errors": sum(intra_genus_pairs.values()),
        "top_cross_genus_pairs": top_cross,
        "top_intra_genus_pairs": top_intra,
    }


def format_markdown(report: dict, model_tag: str) -> str:
    lines = [
        f"# v7 hard-negative audit — {model_tag}",
        "",
        f"N_val = {report['n_val']}, correct = {report['n_correct']} "
        f"({report['top1']:.1%}), wrong = {report['n_wrong']}",
        "",
        f"- cross-genus errors: {report['cross_genus_errors']} "
        f"({report['cross_genus_errors'] / max(1, report['n_wrong']):.1%} of wrong)",
        f"- intra-genus errors: {report['intra_genus_errors']} "
        f"({report['intra_genus_errors'] / max(1, report['n_wrong']):.1%} of wrong)",
        "",
        "## Top cross-genus confusions",
        "",
        "Cross-genus errors are the fixable ones — more/better training data for "
        "the *true* genus typically collapses these. `sim_gap` is how much more "
        "confident v7 was about the wrong genus (higher = more entrenched).",
        "",
        "| true_genus → pred_genus | count | % of true_genus val | mean sim_gap | examples |",
        "|-------------------------|-------|---------------------|--------------|----------|",
    ]
    for row in report["top_cross_genus_pairs"]:
        ex = "; ".join(
            f"{e['true_species']} → {e['pred_species']} ({e['count']}x)"
            for e in row["examples"]
        )
        lines.append(
            f"| {row['true_genus']} → {row['pred_genus']} "
            f"| {row['count']} "
            f"| {row['frac_of_true_genus_total']:.1%} "
            f"| {row['mean_sim_gap']:+.3f} "
            f"| {ex} |"
        )

    lines.extend(
        [
            "",
            "## Top intra-genus confusions",
            "",
            "Intra-genus errors are often taxonomic ambiguity (sister species, "
            "hybrids, ID disagreement in the val set). Data alone rarely fixes "
            "these — worth checking whether the confused pairs are real problems "
            "or known-ambiguous complexes.",
            "",
            "| genus | count | total val | err rate | examples |",
            "|-------|-------|-----------|----------|----------|",
        ]
    )
    for row in report["top_intra_genus_pairs"]:
        ex = "; ".join(
            f"{e['true_species']} → {e['pred_species']} ({e['count']}x)"
            for e in row["examples"]
        )
        lines.append(
            f"| {row['genus']} "
            f"| {row['count']} "
            f"| {row['total']} "
            f"| {row['frac']:.1%} "
            f"| {ex} |"
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orchid-clip", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--image-root", default="")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument(
        "--bucket-lo",
        type=float,
        default=0.0,
        help=(
            "Lower bound of hash bucket to score (default 0.0). Set to 0.02 + "
            "--val-fraction 0.04 to score the held-back P_conf slice disjoint "
            "from §5.1 eval (v13 leakage fix)."
        ),
    )
    ap.add_argument(
        "--top-n-pairs",
        type=int,
        default=TOP_N_CONFUSION_PAIRS,
        help=(
            "Genera to report in top_intra_genus_pairs / top_cross_genus_pairs "
            f"(default {TOP_N_CONFUSION_PAIRS}). v13 P_conf builds want this "
            "raised so the JSON captures more of the long tail."
        ),
    )
    ap.add_argument(
        "--top-n-examples",
        type=int,
        default=TOP_N_EXAMPLES_PER_PAIR,
        help=(
            "Species-level examples per (genus, genus) pair "
            f"(default {TOP_N_EXAMPLES_PER_PAIR}). v13 P_conf builds need this "
            "much higher (e.g. 100) so individual species pairs aren't truncated."
        ),
    )
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--markdown", default=None, help="Optional markdown summary path")
    ap.add_argument(
        "--use-species-accepted",
        action="store_true",
        help="Relabel val rows via images.species_accepted (POWO dedup). Use for v8 native-label-space metrics.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading val rows (bucket [{args.bucket_lo}, {args.val_fraction}))...")
    rows = load_val_rows(
        args.db,
        args.image_root,
        args.val_fraction,
        args.max_val,
        use_species_accepted=args.use_species_accepted,
        bucket_lo=args.bucket_lo,
    )
    print(f"  {len(rows)} val rows")
    labels = sorted({r["binomial"] for r in rows})
    print(f"  {len(labels)} unique species labels")
    paths = [r["path"] for r in rows]

    model_tag = (
        os.path.basename(os.path.dirname(args.orchid_clip.rstrip("/"))) or "orchid_clip"
    )

    t0 = time.time()
    oc = OrchidCLIP(args.orchid_clip, device)
    oc_img = oc.embed_images(paths)
    oc_txt = oc.embed_labels(labels)
    print(f"  embedded in {time.time() - t0:.0f}s")

    report = audit(
        rows,
        labels,
        oc_img,
        oc_txt,
        top_n_pairs=args.top_n_pairs,
        top_n_examples=args.top_n_examples,
    )
    report["model"] = model_tag
    report["n_labels"] = len(labels)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(format_markdown(report, model_tag))
        print(f"wrote {args.markdown}")

    # Console summary
    print(
        f"\ntop1 = {report['top1']:.3f}  "
        f"(cross-genus err {report['cross_genus_errors']}, "
        f"intra-genus err {report['intra_genus_errors']})"
    )
    print("\ntop 10 cross-genus confusion pairs:")
    for row in report["top_cross_genus_pairs"][:10]:
        print(
            f"  {row['true_genus']:>18s} -> {row['pred_genus']:<18s} "
            f"{row['count']:>4d}  ({row['frac_of_true_genus_total']:.1%}, "
            f"gap {row['mean_sim_gap']:+.3f})"
        )
    print("\ntop 10 intra-genus confusions:")
    for row in report["top_intra_genus_pairs"][:10]:
        print(
            f"  {row['genus']:>24s} {row['count']:>4d}/{row['total']:<4d}  "
            f"({row['frac']:.1%})"
        )


if __name__ == "__main__":
    main()
