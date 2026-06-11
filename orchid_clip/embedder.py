"""Unified embedder dispatch for orchid-CLIP checkpoints.

Resolves the HF SigLIP2-style vs open_clip (BioCLIP 2, v6+) divergence so
consumer scripts don't have to branch. Detection is via `model_config.json`
at the checkpoint root: presence of `"framework": "open_clip"` routes to
the open_clip backend.

Both backends expose the same surface:
    preprocess_image(pil) -> torch.Tensor[3, H, W]
    encode_image(batch)   -> torch.Tensor[B, D]  (L2-normalized)
    encode_text(texts)    -> torch.Tensor[B, D]  (L2-normalized)

Use in consumer scripts:
    emb = load_embedder(ckpt, device)
    pixels = torch.stack([emb.preprocess_image(img) for img in pils]).to(device)
    img_feat = emb.encode_image(pixels)
    txt_feat = emb.encode_text(["Cattleya mossiae, an orchid species"])
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import torch
from PIL import Image


def is_openclip_ckpt(ckpt: str) -> bool:
    cfg_path = os.path.join(ckpt, "model_config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            return json.load(f).get("framework") == "open_clip"
    except (OSError, ValueError):
        return False


class Embedder(Protocol):
    backend: str
    device: str

    def preprocess_image(self, image: Image.Image) -> torch.Tensor: ...
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor: ...
    def encode_text(self, texts: list[str]) -> torch.Tensor: ...


class HFEmbedder:
    backend = "hf"

    def __init__(self, ckpt: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.model = AutoModel.from_pretrained(ckpt).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        enc = self.processor(images=image, return_tensors="pt")
        return enc["pixel_values"][0]

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        ie = self.model.get_image_features(pixel_values=pixels.to(self.device))
        if not torch.is_tensor(ie):
            ie = getattr(ie, "image_embeds", None) or ie.pooler_output
        return ie / ie.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        enc = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        te = self.model.get_text_features(**enc)
        if not torch.is_tensor(te):
            te = getattr(te, "text_embeds", None) or te.pooler_output
        return te / te.norm(dim=-1, keepdim=True)


class OpenCLIPEmbedder:
    backend = "openclip"

    def __init__(self, ckpt: str, device: str):
        import open_clip

        self.device = device
        with open(os.path.join(ckpt, "model_config.json")) as f:
            cfg = json.load(f)
        self.model_name = cfg["model_name"]
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=None
        )
        state = torch.load(
            os.path.join(ckpt, "open_clip_pytorch_model.bin"),
            map_location="cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state["state_dict"])
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        ie = self.model.encode_image(pixels.to(self.device))
        return ie / ie.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts).to(self.device)
        te = self.model.encode_text(tok)
        return te / te.norm(dim=-1, keepdim=True)


def load_embedder(ckpt: str, device: str | None = None) -> Embedder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_openclip_ckpt(ckpt):
        print(f"[embedder] open_clip backend for {ckpt}")
        return OpenCLIPEmbedder(ckpt, device)
    print(f"[embedder] HF transformers backend for {ckpt}")
    return HFEmbedder(ckpt, device)
