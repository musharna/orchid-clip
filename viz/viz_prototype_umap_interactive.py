"""Interactive Plotly UMAP of orchid-clip-v8 image prototypes.

The browser-explorable twin of ``paper/figures/fig4_umap_subfamily.png`` (rendered
by ``scripts/embedding_topology_eval.py``). Loads the per-binomial v8 image
centroids, projects them with the *same* UMAP hyper-parameters as the static
cover, and writes a single self-describing HTML page where every point is a
species you can hover to identify.

The interactive HTML carries a **color-by dropdown** (subfamily ↔ tribe): the
same projection recolored at two taxonomic depths. Genus is intentionally not a
toggle option — there are thousands of genera, so a categorical legend is
useless; the hover already names each point's genus.

Inputs (defaults are repo-relative; the .npz is git-ignored / DVC-tracked):
  --prototypes  prototypes_full.npz with arrays ``binomials`` (str, N),
                ``centroids`` (float32, N x 768), ``n_rows_used`` (int32, N).
  --taxonomy    JSON dict ``{binomial: {genus, tribe, subfamily}}`` (WCVP-backed).

Only the five accepted Orchidaceae subfamilies are plotted; binomials whose
subfamily is empty or non-orchid are dropped (a handful of label-noise rows).

Plotly.js is loaded from the CDN (``include_plotlyjs="cdn"``) so the page stays
light (~a few hundred KB of point data) and embeds cleanly in an <iframe>.
WebGL rendering keeps ~18k points interactive.

A single UMAP fit (deterministic at a fixed seed) drives both the interactive
``--out-html`` (subfamily/tribe toggle, default subfamily) and the static
``--out-png`` (matplotlib, subfamily, same palette as the cover), so the two
stay pixel-consistent. At least one output is required.

Run locally (CPU, ~2-4 min):

    python3 scripts/viz_prototype_umap_interactive.py \
        --prototypes data/osr/prototypes_full.npz \
        --taxonomy   data/label_audit/taxonomy.json \
        --out-html   /path/to/site/assets/plotly/orchidclip_umap_subfamily.html \
        --out-png    /path/to/site/assets/img/orchidclip/orchidclip_card.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# The five accepted Orchidaceae subfamilies, in (roughly) phylogenetic order,
# with a palette matched to the static cover figure for visual continuity.
SUBFAMILY_COLORS = {
    "Epidendroideae": "#1f77b4",  # blue  — megadiverse
    "Orchidoideae": "#ff7f0e",  # orange — megadiverse
    "Cypripedioideae": "#2ca02c",  # green — slipper orchids
    "Vanilloideae": "#d62728",  # red
    "Apostasioideae": "#9467bd",  # purple — basal, tiny
}

# A long qualitative palette for tribes (~22 of them). Plotly Dark24 + Light24,
# cycled if a run somehow exceeds 48 distinct tribes.
TRIBE_PALETTE = [
    "#2E91E5",
    "#E15F99",
    "#1CA71C",
    "#FB0D0D",
    "#DA16FF",
    "#222A2A",
    "#B68100",
    "#750D86",
    "#EB663B",
    "#511CFB",
    "#00A08B",
    "#FB00D1",
    "#FC0080",
    "#B2828D",
    "#6C7C32",
    "#778AAE",
    "#862A16",
    "#A777F1",
    "#620042",
    "#1616A7",
    "#DA60CA",
    "#6C4516",
    "#0D2A63",
    "#AF0038",
    "#FD3216",
    "#00FE35",
    "#6A76FC",
    "#FED4C4",
    "#FE00CE",
    "#0DF9FF",
    "#F6F926",
    "#FF9616",
    "#479B55",
    "#EEA6FB",
    "#DC587D",
    "#D626FF",
    "#6E899C",
    "#00B5F7",
    "#B68E00",
    "#C9FBE5",
    "#FF0092",
    "#22FFA7",
    "#E3EE9E",
    "#86CE00",
    "#BC7196",
    "#7E7DCD",
    "#FC6955",
    "#E48F72",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prototypes", default="data/osr/prototypes_full.npz")
    p.add_argument("--taxonomy", default="data/label_audit/taxonomy.json")
    p.add_argument(
        "--out-html",
        default=None,
        help="Destination .html (interactive Plotly, CDN plotly.js).",
    )
    p.add_argument(
        "--out-png",
        default=None,
        help="Destination .png (static matplotlib render, cover style).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.3)
    args = p.parse_args()
    if not args.out_html and not args.out_png:
        p.error("at least one of --out-html / --out-png is required")
    return args


def _level_traces(df, level, order, colors, labels):
    """One Scattergl marker trace per category at a taxonomic ``level``.

    customdata carries [genus, subfamily, tribe, images] so the hover is the same
    rich readout regardless of which level colors the points.
    """
    import plotly.graph_objects as go

    traces = []
    for cat in order:
        m = df[level].to_numpy() == cat
        sub = df[m]
        cd = np.column_stack(
            [
                sub["genus"].to_numpy(),
                sub["subfamily"].to_numpy(),
                sub["tribe"].to_numpy(),
                sub["images"].to_numpy().astype(str),
            ]
        )
        traces.append(
            go.Scattergl(
                x=sub["UMAP-1"],
                y=sub["UMAP-2"],
                mode="markers",
                name=labels[cat],
                legendgroup=level,
                hovertext=sub["species"],
                customdata=cd,
                hovertemplate=(
                    "<b>%{hovertext}</b><br>genus %{customdata[0]}"
                    "<br>%{customdata[1]} · tribe %{customdata[2]}"
                    "<br>%{customdata[3]} images<extra></extra>"
                ),
                marker=dict(
                    size=4, opacity=0.75, color=colors[cat], line=dict(width=0)
                ),
                visible=(level == "subfamily"),
            )
        )
    return traces


def write_html(
    df, sub_order, sub_labels, tribe_order, tribe_labels, title_base, out_html: str
) -> None:
    import plotly.graph_objects as go

    sub_colors = {s: SUBFAMILY_COLORS[s] for s in sub_order}
    tribe_colors = {
        t: TRIBE_PALETTE[i % len(TRIBE_PALETTE)] for i, t in enumerate(tribe_order)
    }

    sub_traces = _level_traces(df, "subfamily", sub_order, sub_colors, sub_labels)
    tribe_traces = _level_traces(df, "tribe", tribe_order, tribe_colors, tribe_labels)
    fig = go.Figure(data=sub_traces + tribe_traces)

    n_sub, n_tribe = len(sub_traces), len(tribe_traces)
    vis_sub = [True] * n_sub + [False] * n_tribe
    vis_tribe = [False] * n_sub + [True] * n_tribe
    title_sub = f"{title_base} — colored by WCVP subfamily"
    title_tribe = f"{title_base} — colored by WCVP tribe"

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.10,
                yanchor="top",
                bgcolor="white",
                bordercolor="#cccccc",
                buttons=[
                    dict(
                        label="color: subfamily",
                        method="update",
                        args=[
                            {"visible": vis_sub},
                            {"title.text": title_sub, "legend.title.text": "subfamily"},
                        ],
                    ),
                    dict(
                        label="color: tribe",
                        method="update",
                        args=[
                            {"visible": vis_tribe},
                            {"title.text": title_tribe, "legend.title.text": "tribe"},
                        ],
                    ),
                ],
            )
        ],
        title=dict(text=title_sub, x=0.0, font=dict(size=14)),
        legend_title_text="subfamily",
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        autosize=True,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displaylogo": False},
    )
    print(
        f"wrote {out_html} ({os.path.getsize(out_html) / 1024:.0f} KB; "
        f"{n_sub} subfamily + {n_tribe} tribe traces)"
    )


def write_png(df, order, legend_labels, title, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df["subfamily"].to_numpy()
    x = df["UMAP-1"].to_numpy()
    y = df["UMAP-2"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw most-populous subfamily first so the tiny clades sit on top, visible.
    for s in order:
        m = sub == s
        ax.scatter(
            x[m],
            y[m],
            s=7,
            color=SUBFAMILY_COLORS[s],
            edgecolors="none",
            alpha=0.6,
            label=legend_labels[s],
            rasterized=True,
        )
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=False,
        markerscale=2.5,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png} ({os.path.getsize(out_png) / 1024:.0f} KB)")


def main() -> None:
    import pandas as pd
    import umap  # heavy import — defer

    args = parse_args()

    print(f"loading prototypes from {args.prototypes}")
    d = np.load(args.prototypes, allow_pickle=True)
    binomials = np.asarray([str(b) for b in d["binomials"]])
    centroids = d["centroids"].astype(np.float32)
    n_rows = (
        d["n_rows_used"].astype(int)
        if "n_rows_used" in d.files
        else np.full(len(binomials), -1)
    )
    print(f"  {len(binomials)} binomials, dim={centroids.shape[1]}")

    print(f"loading taxonomy from {args.taxonomy}")
    with open(args.taxonomy) as f:
        tax = json.load(f)

    def field(b: str, key: str) -> str:
        rec = tax.get(b, {})
        if isinstance(rec, dict):
            return rec.get(key, "") or ""
        # tolerate the legacy [genus, tribe, subfamily] list form
        idx = {"genus": 0, "tribe": 1, "subfamily": 2}[key]
        return (rec[idx] if len(rec) > idx else "") or ""

    genus = np.asarray([field(b, "genus") for b in binomials])
    tribe = np.asarray([field(b, "tribe") or "(unassigned)" for b in binomials])
    subfamily = np.asarray([field(b, "subfamily") for b in binomials])

    keep = np.asarray([s in SUBFAMILY_COLORS for s in subfamily])
    dropped = int((~keep).sum())
    print(
        f"  subfamily known & orchid: {int(keep.sum())} "
        f"(dropped {dropped} empty/non-orchid)"
    )

    centroids = centroids[keep]
    binomials = binomials[keep]
    genus = genus[keep]
    tribe = tribe[keep]
    subfamily = subfamily[keep]
    n_rows = n_rows[keep]

    print(
        f"fitting UMAP (n_neighbors={args.umap_neighbors}, "
        f"min_dist={args.umap_min_dist}, cosine, seed={args.seed})"
    )
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
        n_components=2,
        n_jobs=1,
    )
    coords = np.asarray(reducer.fit_transform(centroids))
    print(f"  UMAP done: shape={coords.shape}")

    df = pd.DataFrame(
        {
            "UMAP-1": coords[:, 0],
            "UMAP-2": coords[:, 1],
            "species": binomials,
            "genus": genus,
            "tribe": tribe,
            "subfamily": subfamily,
            "images": n_rows,
        }
    )

    # Subfamily: fixed phylogenetic order, most-populous drawn first.
    sub_order = [s for s in SUBFAMILY_COLORS if (subfamily == s).any()]
    sub_counts = {s: int((subfamily == s).sum()) for s in sub_order}
    sub_labels = {s: f"{s} (n={sub_counts[s]})" for s in sub_order}
    df["subfamily_label"] = df["subfamily"].map(sub_labels)

    # Tribe: order by count desc (most-populous first → small tribes draw on top).
    uniq_tribes = sorted(set(tribe.tolist()))
    tribe_counts = {t: int((tribe == t).sum()) for t in uniq_tribes}
    tribe_order = sorted(uniq_tribes, key=lambda t: -tribe_counts[t])
    tribe_labels = {t: f"{t} (n={tribe_counts[t]})" for t in tribe_order}
    print(f"  {len(tribe_order)} tribes")

    title_base = f"orchid-clip-v8 prototype UMAP — {len(df):,} species"

    if args.out_html:
        write_html(
            df,
            sub_order,
            sub_labels,
            tribe_order,
            tribe_labels,
            title_base,
            args.out_html,
        )
    if args.out_png:
        png_title = f"{title_base} colored by WCVP subfamily"
        write_png(df, sub_order, sub_labels, png_title, args.out_png)


if __name__ == "__main__":
    main()
