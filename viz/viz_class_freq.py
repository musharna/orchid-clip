"""Interactive log-log species-frequency (Zipf) plot of the v8 training pool.

The browser-explorable twin of ``paper/figures/fig2_class_freq.png``. Reads the
per-species row counts (``paper/figures/v8_class_counts.json``: a descending list
of ``[binomial, n_rows]``) and plots rank vs count on log-log axes — every point
is one species you can hover to read its exact image count. Guide lines mark the
per-species cap (2,000) and the long-tail floor (3 rows).

Plotly.js from the CDN; self-contained, iframe-embeddable.

    python3 scripts/viz_class_freq.py \
        --counts paper/figures/v8_class_counts.json \
        --out /path/to/site/assets/plotly/orchidclip_class_freq.html
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", default="paper/figures/v8_class_counts.json")
    p.add_argument("--out", required=True, help="Destination .html")
    return p.parse_args()


def main() -> None:
    import numpy as np
    import plotly.graph_objects as go

    args = parse_args()
    with open(args.counts) as f:
        rows = json.load(f)
    # rows: list of [binomial, count], descending. Be order-robust anyway.
    rows = sorted(rows, key=lambda r: -int(r[1]))
    species = [str(r[0]) for r in rows]
    counts = np.array([int(r[1]) for r in rows])
    rank = np.arange(1, len(counts) + 1)

    n = len(counts)
    median = int(np.median(counts))
    floor = int(counts.min())
    peak = int(counts.max())
    # v8's long-tail sampler caps each species at 2,000 rows for training; the
    # raw pool plotted here runs far higher, so the cap is a guide line, not the
    # data max. Species above the line are down-sampled at train time.
    SAMPLER_CAP = 2000
    n_above_cap = int((counts > SAMPLER_CAP).sum())

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=rank,
            y=counts,
            mode="markers",
            marker=dict(size=4, color="#2c5282", opacity=0.7),
            hovertext=species,
            hovertemplate="rank %{x}<br><b>%{hovertext}</b><br>%{y:,} images<extra></extra>",
            name="species",
        )
    )
    # sampler-cap guide (fixed at 2,000, not the data max).
    fig.add_hline(
        y=SAMPLER_CAP,
        line=dict(color="#e8590c", width=1, dash="dot"),
        annotation_text=f"sampler cap {SAMPLER_CAP:,}  ·  {n_above_cap} species above",
        annotation_position="top right",
        annotation_font_size=11,
    )
    # median guide.
    fig.add_hline(
        y=median,
        line=dict(color="#999999", width=1, dash="dot"),
        annotation_text=f"median {median} rows",
        annotation_position="bottom right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Training-pool species-frequency distribution (log-log)"
                f"<br><sup>{n:,} species; median {median} rows, floor {floor}, "
                f"peak {peak:,}; {n_above_cap} species above the 2,000 sampler cap</sup>"
            ),
            font=dict(size=15),
            x=0.02,
        ),
        xaxis=dict(
            title="species rank (most-photographed → rarest)",
            type="log",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="images in training pool",
            type="log",
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
        f"n={n} median={median} floor={floor} peak={peak} n_above_2000={n_above_cap}"
    )


if __name__ == "__main__":
    main()
