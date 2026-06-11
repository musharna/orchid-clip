"""Interactive risk-coverage curve for the genus-abstain card.

Renders the precision-vs-coverage trade-off the abstain threshold rides, straight
from the shipped calibration artifact ``orchid_clip_fusion/genus_abstain.json``
(its stored ``curve`` + authoritative operating-point scalars), so the figure is
byte-faithful to the deployed threshold. Every point is one tau on the sweep;
hover reads off (margin tau, coverage, shown-species precision).

The marked operating point is the live card's setting: margin tau = 0.0149 →
shown-species precision 0.90 at 60% coverage. The right end of the curve
(coverage = 1.0) is the no-abstain baseline (precision 0.71).

Plotly.js from the CDN; self-contained, iframe-embeddable HTML.

    python3 scripts/viz_risk_coverage.py \
        --abstain orchid_clip_fusion/genus_abstain.json \
        --out /path/to/site/assets/plotly/orchidclip_risk_coverage.html
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abstain", default="orchid_clip_fusion/genus_abstain.json")
    p.add_argument("--out", required=True, help="Destination .html")
    return p.parse_args()


def main() -> None:
    import plotly.graph_objects as go

    args = parse_args()
    with open(args.abstain) as f:
        d = json.load(f)

    curve = sorted(d["curve"], key=lambda r: r["coverage"])
    cov = [r["coverage"] for r in curve]
    prec = [r["precision"] for r in curve]
    tau = [r["tau"] for r in curve]

    op_cov = d["achieved_coverage"]
    op_prec = d["achieved_precision"]
    op_tau = d["tau"]
    # no-abstain baseline = the full-coverage point on the swept curve.
    base = min(curve, key=lambda r: abs(r["coverage"] - 1.0))
    base_prec = base["precision"]

    fig = go.Figure()

    # target-precision guide line.
    fig.add_hline(
        y=d["target_precision"],
        line=dict(color="#bbbbbb", width=1, dash="dot"),
        annotation_text=f"target {d['target_precision']:.2f}",
        annotation_position="bottom left",
        annotation_font_size=11,
    )

    # the risk-coverage curve (precision vs coverage), colored by margin tau.
    fig.add_trace(
        go.Scatter(
            x=cov,
            y=prec,
            mode="lines+markers",
            line=dict(color="#2c5282", width=2),
            marker=dict(
                size=6,
                color=tau,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="margin τ", thickness=12, len=0.6, x=1.02),
            ),
            customdata=tau,
            hovertemplate=(
                "coverage %{x:.1%}<br>shown-species precision %{y:.3f}"
                "<br>margin τ %{customdata:.4f}<extra></extra>"
            ),
            name="risk–coverage",
            showlegend=False,
        )
    )

    # operating point (the deployed threshold).
    fig.add_trace(
        go.Scatter(
            x=[op_cov],
            y=[op_prec],
            mode="markers+text",
            marker=dict(
                size=15,
                color="#e8590c",
                symbol="star",
                line=dict(color="white", width=1),
            ),
            text=[f"  operating point<br>  τ={op_tau:.4f}"],
            textposition="middle right",
            textfont=dict(size=12, color="#e8590c"),
            hovertemplate=(
                f"operating point<br>coverage {op_cov:.1%}"
                f"<br>precision {op_prec:.3f}<br>τ {op_tau:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # no-abstain baseline marker.
    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[base_prec],
            mode="markers+text",
            marker=dict(
                size=11,
                color="#868e96",
                symbol="circle",
                line=dict(color="white", width=1),
            ),
            text=[f"no abstain: {base_prec:.2f}  "],
            textposition="middle left",
            textfont=dict(size=11, color="#495057"),
            hovertemplate=(
                f"no abstain (show every photo)<br>coverage 100%"
                f"<br>precision {base_prec:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Shown-species precision vs coverage — the abstain trade-off"
                f"<br><sup>orchid-clip-v8 image→text margin, n={d['n_calib']:,} "
                "leakage-safe in-vocab holdout</sup>"
            ),
            font=dict(size=15),
            x=0.02,
        ),
        xaxis=dict(
            title="coverage (fraction of photos given a species)",
            tickformat=".0%",
            range=[-0.02, 1.05],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="shown-species precision",
            tickformat=".2f",
            range=[min(prec) - 0.03, 1.02],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=70, b=55),
        autosize=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.write_html(
        args.out,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(
        f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB); "
        f"op=({op_cov:.3f},{op_prec:.3f}) baseline_prec={base_prec:.3f}"
    )


if __name__ == "__main__":
    main()
