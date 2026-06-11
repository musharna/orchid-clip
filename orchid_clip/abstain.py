"""Genus-abstain decision for the ID card — show species only when confident.

Pure, model-free. ``decide`` applies a calibrated threshold tau to a per-image
species-confidence signal (max-cosine or top1-minus-top2 margin) over the ranked
species candidates; above tau it shows the top species, otherwise it falls back to
the genus rollup ("Genus X, species uncertain"). The genus rollup is always
computed (reusing orchid_clip.genus). ``load_config`` reads the committed
calibration artifact; a missing artifact disables the layer (current behavior).

Spec: specs/stage12_a3_genus_abstain.md sections 5-7.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from orchid_clip.genus import genus_rollup

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "orchid_clip_fusion", "genus_abstain.json"
)


@dataclass(frozen=True)
class AbstainConfig:
    signal: str  # "max_cosine" | "margin"
    tau: float
    target_precision: float
    enabled: bool = True


@dataclass(frozen=True)
class AbstainDecision:
    tier: str  # "species" | "genus_only"
    species: str | None  # the shown species when tier == "species", else None
    signal_value: float
    genus_rollup: list[tuple[str, float]]


def _signal_value(signal: str, scores_desc: list[float]) -> float:
    if signal == "max_cosine":
        return scores_desc[0]
    if signal == "margin":
        if len(scores_desc) < 2:
            raise ValueError("margin signal requires >= 2 candidates")
        return scores_desc[0] - scores_desc[1]
    raise ValueError(f"unknown signal: {signal!r}")


def decide(candidate_ids, scores, cfg: AbstainConfig) -> AbstainDecision:
    """Apply the abstain threshold; return the tier + the always-computed rollup."""
    if len(candidate_ids) != len(scores):
        raise ValueError(
            f"candidate_ids ({len(candidate_ids)}) and scores ({len(scores)}) "
            "length mismatch"
        )
    if len(candidate_ids) == 0:
        raise ValueError("decide called with empty candidate list")
    scores = [float(s) for s in scores]
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("non-finite score in candidate list")

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ids_desc = [candidate_ids[i] for i in order]
    scores_desc = [scores[i] for i in order]

    rollup = genus_rollup(candidate_ids, scores)

    if not cfg.enabled:
        # Disabled = pure pass-through (current behavior); never interfere or raise on
        # signal shape. signal_value is informational: report the top cosine.
        return AbstainDecision("species", ids_desc[0], scores_desc[0], rollup)

    signal_value = _signal_value(cfg.signal, scores_desc)
    if signal_value >= cfg.tau:
        return AbstainDecision("species", ids_desc[0], signal_value, rollup)
    return AbstainDecision("genus_only", None, signal_value, rollup)


def load_config(path: str | None = None) -> AbstainConfig:
    """Load the calibration artifact; missing file -> disabled (current behavior)."""
    resolved = path or os.environ.get("ORCHID_ABSTAIN_CONFIG") or _DEFAULT_CONFIG
    if not os.path.exists(resolved):
        return AbstainConfig(
            signal="max_cosine", tau=-1.0, target_precision=0.0, enabled=False
        )
    raw = open(resolved).read().strip()
    if not raw:  # empty file == "missing/empty config -> layer off" (spec section 7)
        return AbstainConfig(
            signal="max_cosine", tau=-1.0, target_precision=0.0, enabled=False
        )
    try:
        blob = json.loads(raw)
        return AbstainConfig(
            signal=str(blob["signal"]),
            tau=float(blob["tau"]),
            target_precision=float(blob["target_precision"]),
            enabled=bool(blob.get("enabled", True)),
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"genus_abstain.json at {resolved!r} is malformed: {e}") from e
    except KeyError as e:
        raise ValueError(
            f"genus_abstain.json at {resolved!r} missing required key: {e}"
        ) from e
