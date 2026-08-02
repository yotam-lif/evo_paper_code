r"""Plot raw deleterious DFE tails for Limdi ancestors and evolved strains.

This is deliberately descriptive: no model fit, smoothing, weighting, or
normalization is applied.  The reported per-gene effect is the mean of the
Green/Red technical libraries, as in the rest of the Limdi analysis.  Missing
values are removed by ``limdi_gene_series``.  Equal-width raw histogram counts
are shown on a logarithmic y axis over -0.5 <= s <= 0.01.

Outputs
-------
    data/limdi_raw_dfe_tails_histogram.csv
    figs_paper/limdi_raw_dfe_tails_semilog.png
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from cmn.cmn_exper import limdi_gene_series  # noqa: E402


LOWER = -0.5
UPPER = 0.01
BIN_WIDTH = 0.005

# Put the two ancestors first and preserve the ten-background main-analysis set.
PANEL_ORDER = (
    "REL606",
    "REL607",
    "Ara-1",
    "Ara-3",
    "Ara-4",
    "Ara-5",
    "Ara-6",
    "Ara+1",
    "Ara+2",
    "Ara+3",
    "Ara+5",
    "Ara+6",
)

CSV_PATH = REPO_DIR / "data" / "limdi_raw_dfe_tails_histogram.csv"
FIG_PATH = REPO_DIR / "figs_paper" / "limdi_raw_dfe_tails_semilog.png"


def load_histograms():
    edges = np.linspace(
        LOWER,
        UPPER,
        int(round((UPPER - LOWER) / BIN_WIDTH)) + 1,
    )
    histograms = {}
    rows = []

    for background in PANEL_ORDER:
        values = limdi_gene_series(background).to_numpy(float)
        values = values[np.isfinite(values)]
        window = values[(values >= LOWER) & (values <= UPPER)]
        counts, _ = np.histogram(window, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        histograms[background] = {
            "counts": counts,
            "n_measured": values.size,
            "n_window": window.size,
        }
        rows.extend(
            {
                "background": background,
                "bin_left": left,
                "bin_right": right,
                "bin_center": center,
                "count": int(count),
                "n_measured": int(values.size),
                "n_window": int(window.size),
            }
            for left, right, center, count in zip(
                edges[:-1], edges[1:], centers, counts
            )
        )

    return edges, histograms, pd.DataFrame(rows)


def plot(edges, histograms):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(10.2, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    positive_counts = np.concatenate(
        [
            spec["counts"][spec["counts"] > 0]
            for spec in histograms.values()
        ]
    )
    y_max = 10 ** np.ceil(np.log10(positive_counts.max()))

    for ax, background in zip(axes.ravel(), PANEL_ORDER):
        spec = histograms[background]
        counts = spec["counts"].astype(float)
        counts[counts == 0] = np.nan
        ax.stairs(counts, edges, color="#355f8d", linewidth=1.25)
        ax.axvline(0.0, color="0.55", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlim(LOWER, UPPER)
        ax.set_ylim(0.8, y_max)
        ax.grid(axis="y", which="major", color="0.88", linewidth=0.6)
        ax.set_title(background, loc="left", fontweight="bold")
        ax.text(
            0.98,
            0.94,
            f"{spec['n_window']:,} / {spec['n_measured']:,} genes",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.35",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Selection coefficient, s")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Raw genes per {BIN_WIDTH:g}-wide bin")

    fig.suptitle(
        "Raw Limdi DFE tails: equal-width counts, no normalization",
        fontsize=13,
    )
    fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_inline_html(path, edges, histograms):
    """Write a theme-aware in-conversation small-multiple rendering."""
    payload = [
        {
            "name": background,
            "counts": histograms[background]["counts"].tolist(),
            "nMeasured": int(histograms[background]["n_measured"]),
            "nWindow": int(histograms[background]["n_window"]),
        }
        for background in PANEL_ORDER
    ]
    fragment = f"""
<div id="limdi-raw-tail-grid">
  <div class="tail-grid" aria-label="Raw Limdi DFE tail histograms"></div>
</div>
<style>
  #limdi-raw-tail-grid .tail-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px 14px;
  }}
  #limdi-raw-tail-grid .tail-panel {{
    min-width: 0;
  }}
  #limdi-raw-tail-grid .tail-heading {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 4px;
  }}
  #limdi-raw-tail-grid .tail-heading h3 {{
    margin: 0;
  }}
  #limdi-raw-tail-grid canvas {{
    display: block;
    width: 100%;
    height: 178px;
  }}
  @media (max-width: 620px) {{
    #limdi-raw-tail-grid .tail-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
  }}
  @media (max-width: 410px) {{
    #limdi-raw-tail-grid .tail-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById("limdi-raw-tail-grid");
  const grid = root.querySelector(".tail-grid");
  const series = {json.dumps(payload, separators=(",", ":"))};
  const edges = {json.dumps(edges.tolist(), separators=(",", ":"))};
  const lower = {LOWER};
  const upper = {UPPER};
  const yMax = 1000;

  series.forEach((item, index) => {{
    const panel = document.createElement("section");
    panel.className = "tail-panel";
    const heading = document.createElement("div");
    heading.className = "tail-heading";
    const label = document.createElement("h3");
    label.textContent = item.name;
    const n = document.createElement("span");
    n.className = "text-small text-muted";
    n.textContent = `${{item.nWindow.toLocaleString()}} / ${{item.nMeasured.toLocaleString()}} genes`;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute(
      "aria-label",
      `${{item.name}} raw gene counts in 0.005-wide selection-coefficient bins from -0.5 to 0.01, with a logarithmic count axis`
    );
    canvas.dataset.index = String(index);
    heading.append(label, n);
    panel.append(heading, canvas);
    grid.append(panel);
  }});

  function token(name) {{
    return getComputedStyle(root).getPropertyValue(name).trim();
  }}

  function draw(canvas, item, index) {{
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const box = canvas.getBoundingClientRect();
    const width = Math.max(220, box.width);
    const height = 178;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);

    const foreground = token("--foreground");
    const muted = token("--muted-foreground");
    const border = token("--border");
    const seriesColor = token("--viz-series-1");
    const margin = {{left: 38, right: 8, top: 6, bottom: 28}};
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const x = value => margin.left + (value - lower) / (upper - lower) * w;
    const y = value => margin.top + (1 - Math.log10(value) / Math.log10(yMax)) * h;

    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.strokeStyle = border;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    [1, 10, 100, 1000].forEach(tick => {{
      const py = y(tick);
      ctx.beginPath();
      ctx.moveTo(margin.left, py);
      ctx.lineTo(margin.left + w, py);
      ctx.stroke();
      ctx.fillText(tick === 1000 ? "10³" : String(tick), margin.left - 5, py);
    }});

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    [-0.5, -0.3, -0.1, 0].forEach(tick => {{
      const px = x(tick);
      ctx.beginPath();
      ctx.moveTo(px, margin.top + h);
      ctx.lineTo(px, margin.top + h + 4);
      ctx.stroke();
      ctx.fillText(tick.toFixed(tick === 0 ? 1 : 1), px, margin.top + h + 7);
    }});

    ctx.strokeStyle = muted;
    ctx.beginPath();
    ctx.moveTo(x(0), margin.top);
    ctx.lineTo(x(0), margin.top + h);
    ctx.stroke();

    ctx.strokeStyle = seriesColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let drawing = false;
    item.counts.forEach((count, i) => {{
      if (count <= 0) {{
        drawing = false;
        return;
      }}
      const left = x(edges[i]);
      const right = x(edges[i + 1]);
      const py = y(count);
      if (!drawing) {{
        ctx.moveTo(left, py);
        drawing = true;
      }} else {{
        ctx.lineTo(left, py);
      }}
      ctx.lineTo(right, py);
    }});
    ctx.stroke();

    ctx.fillStyle = foreground;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("s", margin.left + w / 2, height);
    if (index % 3 === 0) {{
      ctx.save();
      ctx.translate(10, margin.top + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("raw count", 0, 0);
      ctx.restore();
    }}
  }}

  function render() {{
    root.querySelectorAll("canvas").forEach(canvas => {{
      const index = Number(canvas.dataset.index);
      draw(canvas, series[index], index);
    }});
  }}

  requestAnimationFrame(render);
  new ResizeObserver(render).observe(root);
}})();
</script>
""".strip()
    path.write_text(fragment + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inline-html",
        type=Path,
        help="Optional path for a theme-aware inline visualization fragment.",
    )
    args = parser.parse_args()
    edges, histograms, table = load_histograms()
    table.to_csv(CSV_PATH, index=False)
    plot(edges, histograms)
    if args.inline_html is not None:
        write_inline_html(args.inline_html, edges, histograms)
    for background in PANEL_ORDER:
        spec = histograms[background]
        print(
            f"{background:6s}  "
            f"N measured={spec['n_measured']:4d}  "
            f"N in window={spec['n_window']:4d}"
        )
    print(CSV_PATH)
    print(FIG_PATH)
    if args.inline_html is not None:
        print(args.inline_html)


if __name__ == "__main__":
    main()
