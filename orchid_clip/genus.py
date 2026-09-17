# orchid_clip/genus.py
"""Genus-level rollup over species candidates — the reliable taxonomic level.

Pure, deterministic, model-free. `genus_rollup` aggregates a ranked species
candidate list (binomials + scores) up to genus via softmax-then-sum-within-genus.
`format_genus_lead` renders the rollup as a single human-readable lead line.
"""

from __future__ import annotations

import math


def genus_rollup(
    candidate_ids: list[str],
    scores: list[float],
    top_k: int = 5,
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """Roll species candidates up to genus.

    genus = first whitespace token of the binomial (standard orchid taxonomy).
    softmax(scores / temperature) over the candidate set gives per-species
    probability; sum within genus; sort by confidence desc (genus name asc on
    ties); return up to ``top_k`` (genus, confidence) pairs.

    Fail-loud: empty input, length mismatch, or non-positive temperature is a
    caller bug, not a silent [].
    """
    assert len(candidate_ids) == len(scores), (
        f"candidate_ids ({len(candidate_ids)}) and scores ({len(scores)}) "
        "length mismatch"
    )
    assert len(candidate_ids) > 0, "genus_rollup called with empty candidate list"
    assert temperature > 0, f"temperature must be > 0, got {temperature}"

    # Numerically-stable softmax over the candidate scores.
    hi = max(scores)
    exps = [math.exp((s - hi) / temperature) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]

    # Sum probability within genus (first token of the binomial).
    genus_prob: dict[str, float] = {}
    for cid, p in zip(candidate_ids, probs):
        tokens = cid.split()
        genus = tokens[0] if tokens else cid
        genus_prob[genus] = genus_prob.get(genus, 0.0) + p

    # Confidence desc, then genus name asc for deterministic tie-breaking.
    ranked = sorted(genus_prob.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


def format_genus_lead(rollup: list[tuple[str, float]]) -> str:
    """Render a genus rollup as a single lead line.

    e.g. [("Cattleya", 0.94), ("Guarianthe", 0.04)]
         -> "Genus: Cattleya (0.94) · Guarianthe (0.04)"
    """
    assert rollup, "format_genus_lead called with empty rollup"
    parts = [f"{g} ({c:.2f})" for g, c in rollup]
    return "Genus: " + " · ".join(parts)
