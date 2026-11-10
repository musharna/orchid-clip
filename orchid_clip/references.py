"""Extra references for the demo's gallery: color-form centroids and single photos.

The demo scores a photo against one image centroid per species. Two files add to
that, and this module builds, loads and scores with both:

- ``form_refs.npz``: a second centroid for species with a known color form (for
  example white flowers in a species that is normally purple). The better of the
  two centroids counts.
- ``photo_rows.npz``: every reference photo's embedding, reduced by PCA and stored
  as 8-bit integers. A species' score becomes the mean of its centroid score and
  the cosine to its single closest photo.

Both files must come from the same checkpoint(s) and species list as the centroid
gallery. The loaders raise if they do not, so a gallery rebuilt without rebuilding
these files fails at startup instead of scoring against stale references.

Embeddings may be the concatenation of several towers' L2-normed embeddings,
scaled by 1/sqrt(n_towers); a cosine is then the mean of the towers' cosines.
Pure numpy, no torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def tower_centroid(rows: np.ndarray, tower_dims: list[int]) -> np.ndarray:
    """Centroid of fused rows: each tower's mean is L2-normed on its own, then the
    towers are concatenated and scaled by 1/sqrt(n_towers)."""
    rows = np.asarray(rows, np.float32)
    if rows.ndim != 2 or rows.shape[1] != sum(tower_dims):
        raise ValueError(f"rows must be [N, {sum(tower_dims)}], got {rows.shape}")
    parts, lo = [], 0
    for d in tower_dims:
        parts.append(_norm(rows[:, lo : lo + d].mean(0)))
        lo += d
    return (np.concatenate(parts) / np.sqrt(len(tower_dims))).astype(np.float32)


def build_form_refs(
    rows: np.ndarray, members: dict[str, list[int]], tower_dims: list[int]
) -> dict[str, np.ndarray]:
    """One form centroid per species from the listed row indices.

    ``members`` maps a species name to the indices (into ``rows``) of its photos of
    the color form. Species with fewer than two photos are skipped.
    """
    names = sorted(b for b, ix in members.items() if len(ix) >= 2)
    if not names:
        raise ValueError("no species with at least two form photos")
    return {
        "binomials": np.array(names),
        "centroids": np.stack(
            [tower_centroid(rows[members[b]], tower_dims) for b in names]
        ),
        "n_rows_used": np.array([len(members[b]) for b in names], np.int64),
    }


def build_photo_rows(
    rows: np.ndarray,
    species_index: np.ndarray,
    n_species: int,
    dims: int = 512,
    chunk: int = 50_000,
) -> dict[str, np.ndarray]:
    """Compress reference-photo embeddings for photo-level scoring.

    ``rows`` [N, D] are L2-normed embeddings and ``species_index`` [N] the gallery
    row of each photo's species. Rows are grouped by species (stable sort), projected
    onto the top ``dims`` principal directions of the uncentered rows, L2-normed
    again, and stored as int8 with one scale per dimension (the 99.9th percentile of
    the absolute value over every 7th row).
    Because of that sampling the stored integers depend slightly on the order of
    the input rows; the decoded rows do not (cosine 0.9999 between two orders).
    """
    rows = np.asarray(rows, np.float32)
    sp = np.asarray(species_index, np.int64)
    if rows.ndim != 2 or rows.shape[0] != sp.shape[0]:
        raise ValueError("rows / species_index size mismatch")
    if sp.min() < 0 or sp.max() >= n_species:
        raise ValueError("species_index outside the species list")
    if dims > rows.shape[1]:
        raise ValueError("dims exceeds the embedding dimension")
    order = np.argsort(sp, kind="stable")
    rows, sp = rows[order], sp[order]
    gram = np.zeros((rows.shape[1], rows.shape[1]), np.float64)
    for lo in range(0, len(rows), chunk):
        block = rows[lo : lo + chunk]
        gram += (block.T @ block).astype(np.float64)
    _, vecs = np.linalg.eigh(gram)  # ascending eigenvalues
    proj = np.ascontiguousarray(vecs[:, ::-1][:, :dims]).astype(np.float32)
    low = _norm(rows @ proj)
    scale = np.percentile(np.abs(low[::7]), 99.9, axis=0).astype(np.float32)
    quant = np.clip(np.rint(low / scale * 127), -127, 127).astype(np.int8)
    return {
        "rows": quant,
        "scale": scale,
        "proj": proj,
        "species_index": sp.astype(np.int32),
        "order": order,
        "n_species": np.int64(n_species),
    }


def load_form_refs(
    path: str | Path, binomials: list[str], ckpt_sha256_prefix: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load form centroids: (centroids [F, D] float32 L2-normed, gallery row [F]).

    Returns None when the file is absent. Raises if it was built for another
    checkpoint or names a species that is not in ``binomials``.
    """
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["ckpt_sha256_prefix"]) != str(ckpt_sha256_prefix):
        raise ValueError(
            f"{path.name} was built for {z['ckpt_sha256_prefix']} but the gallery "
            f"for {ckpt_sha256_prefix}"
        )
    index = {b: i for i, b in enumerate(binomials)}
    missing = [str(b) for b in z["binomials"] if str(b) not in index]
    if missing:
        raise ValueError(f"{path.name} names species not in the gallery: {missing[:5]}")
    emb = _norm(np.asarray(z["centroids"], np.float32))
    return emb, np.array([index[str(b)] for b in z["binomials"]], np.int64)


def load_photo_rows(
    path: str | Path, binomials: list[str], ckpt_sha256_prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load photo-level references.

    Returns (rows [N, dims] float32, projection [D, dims], species that have rows,
    offset of each such species' first row), or None when the file is absent.
    Raises if the file was built for another checkpoint or species list, or if its
    rows are not grouped by species.
    """
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["ckpt_sha256_prefix"]) != str(ckpt_sha256_prefix):
        raise ValueError(
            f"{path.name} was built for {z['ckpt_sha256_prefix']} but the gallery "
            f"for {ckpt_sha256_prefix}"
        )
    sp = np.asarray(z["species_index"], np.int64)
    if (
        int(z["n_species"]) != len(binomials)
        or sp.min() < 0
        or sp.max() >= len(binomials)
    ):
        raise ValueError(f"{path.name} was built for a different species list")
    if (np.diff(sp) < 0).any():
        raise ValueError(f"{path.name} rows are not grouped by species")
    rows = z["rows"].astype(np.float32) * (np.asarray(z["scale"], np.float32) / 127.0)
    if rows.shape[0] != sp.shape[0]:
        raise ValueError(f"{path.name} rows / species_index size mismatch")
    present, start = np.unique(sp, return_index=True)
    return rows, np.asarray(z["proj"], np.float32), present, start


def species_sims(
    gallery_emb: np.ndarray, query_emb: np.ndarray, forms=None, rows=None
) -> np.ndarray:
    """Score of one query embedding for every species.

    Centroid score: cosine to the species' centroid, or to its form centroid if
    that is closer. With photo-level references the score is the mean of the
    centroid score and the cosine to the species' closest reference photo; a
    species with no photos in the file keeps its centroid score.
    """
    q = np.asarray(query_emb, np.float32)
    sims = gallery_emb @ q
    if forms is not None:
        emb, idx = forms
        np.maximum.at(sims, idx, emb @ q)
    if rows is not None:
        photos, proj, present, start = rows
        qp = q @ proj
        qp /= np.linalg.norm(qp) + 1e-12
        best = sims.copy()
        best[present] = np.maximum.reduceat(photos @ qp, start)
        sims = 0.5 * sims + 0.5 * best
    return sims
