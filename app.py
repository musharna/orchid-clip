"""HF Space: Orchid genus identification with a calibrated species abstain.

Thin gradio shell over ``infer.py``. Upload an orchid photo -> orchid-clip-v8
image embedding -> cosine vs the 18,858 v8 species text embeddings -> top-k
species, gated by a calibrated genus-abstain (show a species only when the
top1-top2 cosine margin clears tau; else "Genus X (species uncertain)").

Self-contained: the v8 image tower is pulled from the public model
``mjarnold/orchid-clip-v8`` whose checkpoint sha256 (81ae2b09...) is identical
to the checkpoint the shipped text embeddings were built from, so the live
margin signal IS the signal the abstain threshold was calibrated on.

Run locally:
    bash build_assets.sh        # populate assets/ from the upstream training repo
    ORCHID_CLIP_CHECKPOINT=/path/to/orchid_clip_v8/epoch_5 python app.py

Set ORCHID_SPACE_LAZY=1 to skip the eager model load at import (for tests).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from infer import embed_image, load_assets, score_and_format
from orchid_clip.embedder import load_embedder

_HERE = Path(__file__).resolve().parent
ASSETS = _HERE / "assets"
HF_MODEL = os.environ.get("ORCHID_V8_REPO", "mjarnold/orchid-clip-v8")

TITLE = "🌿 Orchid Genus ID — with calibrated species abstain"
DESCRIPTION = (
    "Upload an orchid photo. The model ([orchid-clip-v8]"
    "(https://huggingface.co/mjarnold/orchid-clip-v8), a BioCLIP-2 ViT-L/14 "
    "fine-tune) embeds it and ranks it against **18,858 orchid species**. "
    "Genus is the level it reliably nails (genus top-1 ≈ 0.94 on this cross-modal "
    "image↔text path); species within a "
    "genus is genuinely hard. So a **calibrated abstain** shows a species only "
    "when it is confident — otherwise it gives you the genus and lists the "
    "species candidates as possibilities. The abstain holds shown-species "
    "precision at ~90% while still naming a species on ~60% of photos."
)

if not (ASSETS / "v8_text_embeddings.npz").exists():
    raise SystemExit(
        f"assets missing under {ASSETS}; run `bash build_assets.sh` first "
        "(copies v8_text_embeddings.npz + genus_abstain.json + taxonomy.json "
        "from the upstream training repo)."
    )

TEXT_EMB, BINOMIALS, ABSTAIN, TAX = load_assets(ASSETS)
print(
    f"[space] {len(BINOMIALS):,} species; abstain enabled={ABSTAIN.enabled} "
    f"signal={ABSTAIN.signal} tau={ABSTAIN.tau:.4f}"
)


def _resolve_ckpt() -> str:
    """Local checkpoint dir if provided, else download the public HF model."""
    local = os.environ.get("ORCHID_CLIP_CHECKPOINT")
    if local and os.path.isfile(os.path.join(local, "open_clip_pytorch_model.bin")):
        print(f"[space] using local checkpoint {local}")
        return local
    from huggingface_hub import snapshot_download

    print(f"[space] downloading {HF_MODEL} from the HuggingFace Hub...")
    return snapshot_download(HF_MODEL)


_EMB = None
if os.environ.get("ORCHID_SPACE_LAZY") != "1":
    _EMB = load_embedder(_resolve_ckpt())
    print("[space] image tower ready")


def _get_embedder():
    global _EMB
    if _EMB is None:
        _EMB = load_embedder(_resolve_ckpt())
    return _EMB


def identify(image, topk: int):
    if image is None:
        return "Upload an orchid photo to identify.", [], ""
    q = embed_image(_get_embedder(), image)
    return score_and_format(q, topk, TEXT_EMB, BINOMIALS, ABSTAIN, TAX)


with gr.Blocks(title="Orchid Genus ID") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Orchid photo", type="pil")
            topk = gr.Slider(2, 15, value=8, step=1, label="Candidates to show")
            btn = gr.Button("Identify", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            table = gr.Dataframe(
                headers=["rank", "species", "cosine"],
                datatype=["number", "str", "str"],
                label="Top species candidates",
                interactive=False,
            )
            note = gr.Markdown()

    btn.click(identify, [image_in, topk], [verdict, table, note])

    gr.Markdown(
        "---\n"
        "**Why abstain?** Across six independent extension attempts, genus "
        "structure in this embedding transfers and stays decodable while "
        "within-genus *species* identity hits a wall. Rather than emit a "
        "confident wrong binomial, this card serves the granularity the model "
        "actually earns. See the "
        "[write-up](https://musharna.github.io/projects/OrchidCLIP/)."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
