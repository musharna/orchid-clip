"""Gradio demo: orchid genus identification with a species abstain.

An earlier, text-embedding version of the Space's logic; the live Space now
ranks against image centroids.

Thin gradio shell over ``infer.py``. Upload an orchid photo -> orchid-clip-v8
image embedding -> cosine vs the 18,858 v8 species text embeddings -> top-k
species, gated by a genus-abstain (show a species only when the top1-top2
cosine margin clears tau; else "Genus X (species uncertain)"). tau was chosen
on a 7,137-image calibration set; its 0.90 precision at 0.60 coverage was
measured on that same set, so it is an in-sample figure.

The v8 image tower is pulled from the public model ``musharna/orchid-clip-v8``,
whose checkpoint sha256 (81ae2b09...) matches the checkpoint the text
embeddings were built from.

Run locally:
    # assets/ (v8_text_embeddings.npz, genus_abstain.json, taxonomy.json)
    # come from the training pipeline, which is not included in this repo.
    ORCHID_CLIP_CHECKPOINT=/path/to/checkpoint_dir python app.py

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
HF_MODEL = os.environ.get("ORCHID_V8_REPO", "musharna/orchid-clip-v8")
# Pinned to a commit so a rewritten Hub repo cannot swap the weights under a
# deployed Space (bandit B615). Bump deliberately: main sha as of 2026-09-17.
HF_REVISION = os.environ.get(
    "ORCHID_V8_REVISION", "2e33008ac00f88389345ca863c4b319589f29827"
)

TITLE = "🌿 Orchid Genus ID — with calibrated species abstain"
DESCRIPTION = (
    "Upload an orchid photo. The model ([orchid-clip-v8]"
    "(https://huggingface.co/musharna/orchid-clip-v8), a BioCLIP-2 ViT-L/14 "
    "fine-tune) embeds it and ranks it against **18,858 orchid species**. "
    "Genus is more reliable than species, but not guaranteed: on species not "
    "seen in training, genus top-1 is 0.48 (n = 187 images). A species is "
    "shown only when the top-1/top-2 score margin clears a threshold; "
    "otherwise you get the genus and a list of candidate species. On the "
    "7,137-image set used to choose the threshold, shown species were correct "
    "90% of the time and a species was named for 60% of photos (in-sample)."
)

if not (ASSETS / "v8_text_embeddings.npz").exists():
    raise SystemExit(
        f"assets missing under {ASSETS}: this needs v8_text_embeddings.npz, "
        "genus_abstain.json and taxonomy.json, which come from the training "
        "pipeline (not included in this repo)."
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
    return snapshot_download(HF_MODEL, revision=HF_REVISION)


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
        "**Why abstain?** Within-genus species identification is the model's "
        "weak point, so the card names a species only when the margin is large "
        "and otherwise reports the genus. See the "
        "[project page](https://musharna.github.io/projects/OrchidCLIP/)."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))  # nosec B104 - Space container; must bind all interfaces
