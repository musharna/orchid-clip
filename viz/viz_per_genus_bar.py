"""Interactive per-genus top-1 bar chart — orchid-clip-v8 vs BioCLIP 2.

The browser-explorable twin of ``paper/figures/fig1_per_genus.png``. Same numbers
(``~/orchid_clip_v8/eval_v8_vs_v7.json``, n=4000, native eval), kept
in sync with ``paper/figures/render_fig1_per_genus.py`` (ROWS below mirror it).
Grouped horizontal bars sorted by long-tail delta; hover reads off n, both
accuracies, and Δ. Plotly.js from the CDN; self-contained, iframe-embeddable.

    python3 scripts/viz_per_genus_bar.py \
        --out /path/to/site/assets/plotly/orchidclip_per_genus.html
"""

from __future__ import annotations

import argparse
import os

# (genus, n, v8_top1, bioclip2_top1) — mirrors render_fig1_per_genus.py ROWS,
# sourced from eval_v8_vs_v7.json (n=4000, 547 species, native eval pipeline).
ROWS = [
    ("Lepanthes", 40, 0.800, 0.525),
    ("Stelis", 25, 0.640, 0.400),
    ("Bulbophyllum", 41, 0.732, 0.585),
    ("Maxillaria", 94, 0.787, 0.649),
    ("Pleurothallis", 100, 0.800, 0.690),
    ("Habenaria", 232, 0.922, 0.845),
    ("Masdevallia", 64, 0.859, 0.781),
    ("Oncidium", 57, 0.860, 0.825),
    ("Prosthechea", 145, 0.890, 0.855),
    ("Ophrys", 2754, 0.933, 0.905),
    ("Dendrobium", 161, 0.919, 0.907),
    ("Encyclia", 101, 0.832, 0.842),
    ("Cymbidium", 99, 0.879, 0.939),
    ("Laelia", 24, 0.958, 1.000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Destination .html")
    return p.parse_args()


def main() -> None:
    import plotly.graph_objects as go

    args = parse_args()
    # sort by delta ascending so the largest lift sits at the TOP of a horizontal bar.
    rows = sorted(ROWS, key=lambda r: r[2] - r[3])
    labels = [f"{g}  (n={n})" for g, n, _, _ in rows]
    v8 = [r[2] for r in rows]
    bc2 = [r[3] for r in rows]
    delta = [r[2] - r[3] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=bc2,
            orientation="h",
            name="BioCLIP 2",
            marker_color="#9aa5b1",
            customdata=delta,
            hovertemplate="BioCLIP 2 top-1 %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=v8,
            orientation="h",
            name="orchid-clip-v8",
            marker_color="#2c5282",
            customdata=delta,
            hovertemplate=(
                "orchid-clip-v8 top-1 %{x:.3f}<br>Δ vs BioCLIP 2 "
                "%{customdata:+.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        title=dict(
            text=(
                "Per-genus top-1 accuracy — orchid-clip-v8 vs BioCLIP 2"
                "<br><sup>sorted by long-tail Δ; lift concentrates on the "
                "smallest, longest-tailed Pleurothallidinae genera</sup>"
            ),
            font=dict(size=15),
            x=0.02,
        ),
        xaxis=dict(
            title="top-1 accuracy",
            range=[0, 1.02],
            tickformat=".1f",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(title="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        plot_bgcolor="white",
        margin=dict(l=10, r=20, t=70, b=45),
        autosize=True,
        height=560,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.write_html(
        args.out,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(
        f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB), {len(rows)} genera"
    )


if __name__ == "__main__":
    main()
