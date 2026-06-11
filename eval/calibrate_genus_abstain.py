"""A3 genus-abstain calibration — CPU, zero-GPU.

Scores leakage-safe in-vocab positives (g1_positives.image_emb @ v8_text_embeddings.T,
the exact live fp.v8_scores signal) to build a risk-coverage curve, picks the
better of {max_cosine, margin} by coverage at the target precision, selects the
smallest tau whose covered-species precision >= target, validates the genus floor
on the genuine-novel negatives, and writes orchid_clip_fusion/genus_abstain.json.

Pure core (score_signals / risk_coverage_curve / select_tau / pick_signal /
genus_floor) is unit-tested; main() is real-execution. No torch, no sklearn.

Spec: specs/stage12_a3_genus_abstain.md section 4.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def score_signals(image_emb, text_emb, vocab_binoms, true_binoms, chunk=512):
    """Per-image: max_cosine (top1), margin (top1-top2), species_correct,
    genus_correct (argmax over the vocab vs the true label). Chunked over images.
    image_emb (M,D) and text_emb (V,D) must be L2-normed; cosine == dot."""
    image_emb = np.asarray(image_emb, np.float32)
    text_emb = np.asarray(text_emb, np.float32)
    vocab = [str(b) for b in vocab_binoms]
    vocab_genus = [b.split()[0] for b in vocab]
    true = [str(b) for b in true_binoms]
    true_genus = [b.split()[0] for b in true]
    n = image_emb.shape[0]
    max_cos = np.empty(n, np.float32)
    margin = np.empty(n, np.float32)
    sp_correct = np.zeros(n, bool)
    gn_correct = np.zeros(n, bool)
    for i in range(0, n, chunk):
        sims = image_emb[i : i + chunk] @ text_emb.T  # (c, V)
        part = np.partition(sims, -2, axis=1)
        max_cos[i : i + chunk] = part[:, -1]
        margin[i : i + chunk] = part[:, -1] - part[:, -2]
        arg = sims.argmax(axis=1)
        for j, a in enumerate(arg):
            idx = i + j
            sp_correct[idx] = vocab[a] == true[idx]
            gn_correct[idx] = vocab_genus[a] == true_genus[idx]
    return {
        "max_cosine": max_cos,
        "margin": margin,
        "species_correct": sp_correct,
        "genus_correct": gn_correct,
    }


def risk_coverage_curve(signal_values, species_correct):
    """Sweep tau over the unique signal values; per tau report coverage (frac with
    signal >= tau) and precision (frac species-correct among covered). Ascending tau."""
    sv = np.asarray(signal_values, float)
    sc = np.asarray(species_correct, bool)
    if sv.shape[0] != sc.shape[0]:
        raise ValueError("signal_values and species_correct length mismatch")
    if sv.shape[0] == 0:
        raise ValueError("empty signal array")
    out = []
    for tau in np.unique(sv):
        covered = sv >= tau
        out.append(
            {
                "tau": float(tau),
                "coverage": float(covered.mean()),
                "precision": float(sc[covered].mean()) if covered.any() else 0.0,
            }
        )
    return out


def select_tau(curve, target_precision):
    """Correctness-first: smallest tau whose covered-set precision >= target (= the
    qualifying point with the highest coverage). If none qualify, achievable=False
    and the highest-precision point is returned as best-achievable."""
    qualifying = [
        p for p in curve if p["precision"] >= target_precision and p["coverage"] > 0.0
    ]
    if qualifying:
        best = min(qualifying, key=lambda p: p["tau"])
        achievable = True
    else:
        best = max(curve, key=lambda p: (p["precision"], p["coverage"]))
        achievable = False
    return {
        "tau": best["tau"],
        "achieved_precision": best["precision"],
        "achieved_coverage": best["coverage"],
        "achievable": achievable,
    }


def pick_signal(curves, target_precision):
    """Run select_tau on each named curve; pick the signal with the highest achieved
    coverage among achievable ones (else highest achieved precision). Ties -> the
    signal named 'max_cosine'. Returns (chosen_name, {name: select_tau result})."""
    results = {n: select_tau(c, target_precision) for n, c in curves.items()}
    achievable = {n: r for n, r in results.items() if r["achievable"]}
    pool = achievable or results
    key = "achieved_coverage" if achievable else "achieved_precision"
    best_name = max(pool, key=lambda n: (pool[n][key], n == "max_cosine"))
    return best_name, results


def genus_floor(neg_signal_values, neg_genus_correct, tau):
    """On novel-species negatives: fraction routed to genus_only (signal < tau) and
    genus-correctness of the rollup top-1 (regardless of tier)."""
    sv = np.asarray(neg_signal_values, float)
    gc = np.asarray(neg_genus_correct, bool)
    if sv.shape[0] != gc.shape[0]:
        raise ValueError("neg signal/genus_correct length mismatch")
    n = int(sv.shape[0])
    return {
        "frac_genus_only": float((sv < tau).mean()) if n else 0.0,
        "genus_top1": float(gc.mean()) if n else 0.0,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="data/osr/g1_positives.npz")
    ap.add_argument("--negatives", default="data/osr/g1_negatives.npz")
    ap.add_argument("--text-emb", default="data/fusion/v8_text_embeddings.npz")
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--per-genus-min", type=int, default=20)
    ap.add_argument("--out", default="orchid_clip_fusion/genus_abstain.json")
    args = ap.parse_args()

    tz = np.load(args.text_emb, allow_pickle=True)
    text_emb = np.asarray(tz["text_embeddings"], np.float32)
    vocab_binoms = tz["binomials"]
    ckpt_sha = str(tz["ckpt_sha256_prefix"]) if "ckpt_sha256_prefix" in tz.files else ""
    assert text_emb.shape[0] == len(vocab_binoms), "text emb / vocab size mismatch"

    pz = np.load(args.positives, allow_pickle=True)
    pos_emb = np.asarray(pz["image_emb"], np.float32)
    pos_bino = pz["binomials"]
    nz = np.load(args.negatives, allow_pickle=True)
    neg_emb = np.asarray(nz["image_emb"], np.float32)
    neg_bino = nz["binomials"]
    print(
        f"text {text_emb.shape}, positives {pos_emb.shape}, "
        f"negatives {neg_emb.shape}, ckpt {ckpt_sha}"
    )

    pos = score_signals(pos_emb, text_emb, vocab_binoms, pos_bino)
    curves = {
        "max_cosine": risk_coverage_curve(pos["max_cosine"], pos["species_correct"]),
        "margin": risk_coverage_curve(pos["margin"], pos["species_correct"]),
    }
    chosen, results = pick_signal(curves, args.target_precision)
    sel = results[chosen]
    tau = sel["tau"]

    neg = score_signals(neg_emb, text_emb, vocab_binoms, neg_bino)
    floor = genus_floor(neg[chosen], neg["genus_correct"], tau)

    sigvals = pos[chosen]
    genera = np.array([str(b).split()[0] for b in pos_bino])
    per_genus = []
    for g in sorted(set(genera.tolist())):
        m = genera == g
        if int(m.sum()) < args.per_genus_min:
            continue
        cov_mask = sigvals[m] >= tau
        cov = float(cov_mask.mean())
        prec = (
            float(pos["species_correct"][m][cov_mask].mean()) if cov_mask.any() else 0.0
        )
        per_genus.append(
            {"genus": g, "n": int(m.sum()), "precision": prec, "coverage": cov}
        )

    full = curves[chosen]
    # Subsample the curve for compact storage; the tau / achieved_precision /
    # achieved_coverage scalars above are authoritative (computed on the FULL
    # curve) and must not be re-derived from this downsampled curve.
    stride = max(1, len(full) // 200)
    report = {
        "signal": chosen,
        "tau": float(tau),
        "target_precision": float(args.target_precision),
        "achieved_precision": float(sel["achieved_precision"]),
        "achieved_coverage": float(sel["achieved_coverage"]),
        "achievable": bool(sel["achievable"]),
        "enabled": bool(sel["achievable"]),
        "n_calib": int(pos_emb.shape[0]),
        "ckpt_sha256_prefix": ckpt_sha,
        "signals_compared": {
            n: {
                "achieved_coverage": r["achieved_coverage"],
                "achieved_precision": r["achieved_precision"],
                "achievable": r["achievable"],
            }
            for n, r in results.items()
        },
        "curve": full[::stride],
        "genus_floor": floor,
        "per_genus_diag": per_genus,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(
        f"[calibrate] signal={chosen} tau={tau:.4f} "
        f"precision={sel['achieved_precision']:.3f} "
        f"coverage={sel['achieved_coverage']:.3f} achievable={sel['achievable']} | "
        f"genus_floor genus_top1={floor['genus_top1']:.3f} "
        f"frac_genus_only={floor['frac_genus_only']:.3f}"
    )


if __name__ == "__main__":
    main()
