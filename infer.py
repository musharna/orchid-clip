"""Gradio-free inference core for the orchid genus-ID Space.

Kept separate from app.py so the scoring + abstain + verdict formatting can be
imported and exercised on real data (desktop, with the real v8 weights) without
pulling in gradio. app.py is the thin UI shell over this.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from orchid_clip.abstain import AbstainConfig, decide, load_config


def load_assets(assets_dir: str | Path):
    """Load the shipped text bank, abstain config, and taxonomy.

    Returns (text_emb [V,D] float32 L2-normed, binomials [V], AbstainConfig, tax dict).
    """
    assets = Path(assets_dir)
    z = np.load(assets / "v8_text_embeddings.npz", allow_pickle=True)
    text_emb = np.asarray(z["text_embeddings"], np.float32)
    binomials = [str(b) for b in z["binomials"]]
    if text_emb.shape[0] != len(binomials):
        raise ValueError("text_embeddings / binomials size mismatch")
    abstain = load_config(str(assets / "genus_abstain.json"))
    tax_path = assets / "taxonomy.json"
    tax = json.load(open(tax_path)) if tax_path.exists() else {}
    return text_emb, binomials, abstain, tax


def embed_image(embedder, image) -> np.ndarray:
    """One RGB image -> L2-normalized v8 image embedding (768,) as float32."""
    import torch

    pix = embedder.preprocess_image(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        ie = embedder.encode_image(pix)
    return ie[0].detach().cpu().numpy().astype(np.float32)


def score_and_format(
    query_emb: np.ndarray,
    topk: int,
    text_emb: np.ndarray,
    binomials: list[str],
    abstain: AbstainConfig,
    tax: dict,
):
    """Cosine-rank a query embedding, apply the abstain, format the result card.

    Returns (verdict_markdown, rows[[rank, species, cosine]], note_markdown).
    """
    sims = text_emb @ np.asarray(query_emb, np.float32)  # (V,)
    k = max(int(topk), 2)
    top = np.argsort(-sims)[:k]
    cand_ids = [binomials[i] for i in top]
    cand_scores = [float(sims[i]) for i in top]

    decision = decide(cand_ids, cand_scores, abstain)
    # Genus readouts are concrete counts over the top-k candidates, NOT the
    # softmax-over-cosine "probabilities" (misleadingly flat at this cosine
    # scale). The abstain itself is margin-based and unaffected.
    gcounts = Counter(c.split()[0] for c in cand_ids)
    genera_str = " · ".join(
        f"{g} ({n}/{len(cand_ids)})" for g, n in gcounts.most_common(3)
    )

    if decision.tier == "genus_only":
        # Abstaining from species -> headline the dominant genus (the rollup
        # top), not the unreliable top-1's genus.
        headline_genus = decision.genus_rollup[0][0]
        verdict = (
            f"### 🌿 Genus: {headline_genus}\n"
            f"**Species uncertain** — top species margin "
            f"`{decision.signal_value:.4f}` < calibrated τ `{abstain.tau:.4f}`, "
            f"so the model abstains from a species call. Candidate genera in the "
            f"top {len(cand_ids)}: {genera_str}. Read the species list below as "
            f"*“this genus, probably one of these.”*"
        )
    else:
        # Confident species call -> genus is simply that species' genus.
        headline_genus = cand_ids[0].split()[0]
        verdict = (
            f"### ✅ {cand_ids[0]}\n"
            f"Genus **{headline_genus}** · confident species call "
            f"(margin `{decision.signal_value:.4f}` ≥ τ `{abstain.tau:.4f}`)."
        )
    # Subfamily of the *headline* genus, read off a candidate of that genus.
    rep = next((c for c in cand_ids if c.split()[0] == headline_genus), cand_ids[0])
    subfam = tax.get(rep, {}).get("subfamily", "") if tax else ""
    if subfam:
        verdict += f"\n\n*Subfamily: {subfam}*"

    rows = [
        [r, sp, f"{sc:.4f}"] for r, (sp, sc) in enumerate(zip(cand_ids, cand_scores), 1)
    ]
    note = (
        f"Ranked against {len(binomials):,} orchid species with orchid-clip-v8. "
        f"Abstain calibrated to {abstain.target_precision:.0%} shown-species "
        f"precision (margin signal). Cosines are in [-1, 1]; in-distribution "
        f"positives are typically 0.4–0.8."
    )
    return verdict, rows, note
