r"""Fit deleterious Limdi DFE tails by log-log CCDF regression.

For x = -s > 0 and p(s) proportional to |s|^(-(1 + mu)), the deleterious
survival count obeys

    N(S <= -x) = B x^(-mu).

The fit is performed on 80 equally spaced log thresholds over
0.01 <= x <= 0.04.  Each DFE gets a free intercept.  Both independent slopes
and a common slope with background-specific intercepts are reported.  This is
a descriptive cumulative fit, not a power-law MLE: ECDF points are correlated.

Outputs
-------
    data/limdi_dfe_ccdf_powerlaw_001_004.csv
    data/limdi_dfe_ccdf_powerlaw_001_004_curves.csv
    figs_paper/limdi_dfe_ccdf_powerlaw_001_004.png
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


X_MIN = 0.01
X_MAX = 0.04
N_THRESHOLDS = 80
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 260726

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

RESULT_PATH = REPO_DIR / "data" / "limdi_dfe_ccdf_powerlaw_001_004.csv"
CURVE_PATH = (
    REPO_DIR / "data" / "limdi_dfe_ccdf_powerlaw_001_004_curves.csv"
)
FIG_PATH = (
    REPO_DIR / "figs_paper" / "limdi_dfe_ccdf_powerlaw_001_004.png"
)


def survival_counts(magnitudes, thresholds):
    ordered = np.sort(np.asarray(magnitudes, float))
    return ordered.size - np.searchsorted(ordered, thresholds, side="left")


def regress_log_ccdf(thresholds, counts):
    """OLS log count = intercept - mu log threshold."""
    log_x = np.log(np.asarray(thresholds, float))
    log_y = np.log(np.asarray(counts, float))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = intercept + slope * log_x
    residual = log_y - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((log_y - log_y.mean()) ** 2))
    return {
        "mu": float(-slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - sse / sst),
        "fitted": np.exp(fitted),
        "sse": sse,
        "sst": sst,
    }


def bootstrap_mu(magnitudes, thresholds, rng):
    magnitudes = np.asarray(magnitudes, float)
    estimates = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.choice(magnitudes, size=magnitudes.size, replace=True)
        counts = survival_counts(sample, thresholds)
        estimates[index] = regress_log_ccdf(thresholds, counts)["mu"]
    return estimates


def common_slope_summary(results, backgrounds):
    """Equal-background-weight common slope with a free intercept per DFE."""
    backgrounds = tuple(backgrounds)
    common_mu = float(
        np.mean([results[name]["mu"] for name in backgrounds])
    )
    common_bootstrap = np.mean(
        np.vstack(
            [results[name]["bootstrap_mu"] for name in backgrounds]
        ),
        axis=0,
    )
    common_ci = np.quantile(common_bootstrap, [0.025, 0.975])
    total_sse = 0.0
    total_sst = 0.0
    log_x = np.log(results[backgrounds[0]]["thresholds"])
    fitted = {}
    for background in backgrounds:
        log_y = np.log(results[background]["counts"])
        intercept = float(np.mean(log_y + common_mu * log_x))
        fitted[background] = np.exp(intercept - common_mu * log_x)
        total_sse += float(
            np.sum((log_y - np.log(fitted[background])) ** 2)
        )
        total_sst += float(np.sum((log_y - log_y.mean()) ** 2))
    return {
        "backgrounds": backgrounds,
        "mu": common_mu,
        "mu_ci_low": float(common_ci[0]),
        "mu_ci_high": float(common_ci[1]),
        "within_r_squared": float(1.0 - total_sse / total_sst),
        "fitted": fitted,
    }


def load_and_fit():
    thresholds = np.geomspace(X_MIN, X_MAX, N_THRESHOLDS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = {}
    curve_rows = []

    for background in PANEL_ORDER:
        effects = limdi_gene_series(background).to_numpy(float)
        effects = effects[np.isfinite(effects)]
        magnitudes = -effects[effects < 0.0]
        counts = survival_counts(magnitudes, thresholds)
        fit = regress_log_ccdf(thresholds, counts)
        bootstrap = bootstrap_mu(magnitudes, thresholds, rng)
        fit.update(
            {
                "n_measured": int(effects.size),
                "n_negative": int(magnitudes.size),
                "count_at_001": int(counts[0]),
                "count_at_004": int(counts[-1]),
                "mu_ci_low": float(np.quantile(bootstrap, 0.025)),
                "mu_ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_mu": bootstrap,
                "counts": counts,
                "thresholds": thresholds,
            }
        )
        results[background] = fit
        curve_rows.extend(
            {
                "background": background,
                "x": float(x),
                "s": float(-x),
                "tail_count": int(count),
                "individual_fit_count": float(fitted),
            }
            for x, count, fitted in zip(
                thresholds, counts, fit["fitted"]
            )
        )

    # Because every background uses the same log-x grid and receives equal
    # weight, each fixed-intercept common slope is the mean individual slope.
    summaries = {
        "ALL_COMMON_FIXED_INTERCEPTS": common_slope_summary(
            results, PANEL_ORDER
        ),
        "EVOLVED_COMMON_FIXED_INTERCEPTS": common_slope_summary(
            results, PANEL_ORDER[2:]
        ),
        "REL_COMMON_FIXED_INTERCEPTS": common_slope_summary(
            results, PANEL_ORDER[:2]
        ),
    }
    curve_table = pd.DataFrame(curve_rows)
    curve_table["common_fit_count"] = np.concatenate(
        [
            summaries["ALL_COMMON_FIXED_INTERCEPTS"]["fitted"][name]
            for name in PANEL_ORDER
        ]
    )
    return thresholds, results, summaries, curve_table


def write_results(results, summaries):
    rows = [
        {
            "background": background,
            "mu": results[background]["mu"],
            "mu_ci_low": results[background]["mu_ci_low"],
            "mu_ci_high": results[background]["mu_ci_high"],
            "loglog_r_squared": results[background]["r_squared"],
            "n_measured": results[background]["n_measured"],
            "n_negative": results[background]["n_negative"],
            "tail_count_at_x_0.01": results[background]["count_at_001"],
            "tail_count_at_x_0.04": results[background]["count_at_004"],
        }
        for background in PANEL_ORDER
    ]
    for label, summary in summaries.items():
        backgrounds = summary["backgrounds"]
        rows.append(
            {
                "background": label,
                "mu": summary["mu"],
                "mu_ci_low": summary["mu_ci_low"],
                "mu_ci_high": summary["mu_ci_high"],
                "loglog_r_squared": summary["within_r_squared"],
                "n_measured": sum(
                    results[name]["n_measured"] for name in backgrounds
                ),
                "n_negative": sum(
                    results[name]["n_negative"] for name in backgrounds
                ),
                "tail_count_at_x_0.01": sum(
                    results[name]["count_at_001"]
                    for name in backgrounds
                ),
                "tail_count_at_x_0.04": sum(
                    results[name]["count_at_004"]
                    for name in backgrounds
                ),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)


def plot(results, common):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(10.2, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, background in zip(axes.ravel(), PANEL_ORDER):
        fit = results[background]
        ax.plot(
            fit["thresholds"],
            fit["counts"],
            color="#31688e",
            linewidth=1.5,
            label="empirical cumulative count",
        )
        ax.plot(
            fit["thresholds"],
            fit["fitted"],
            color="#b12a90",
            linewidth=1.3,
            linestyle="--",
            label="individual power-law fit",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", color="0.88", linewidth=0.6)
        ax.set_title(
            f"{background}   "
            rf"$\mu={fit['mu']:.3f}$, $R^2={fit['r_squared']:.4f}$",
            loc="left",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Deleterious magnitude $x=-s$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Raw count with $S\leq -x$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        rf"CCDF power-law slopes over $0.01\leq -s\leq0.04$: "
        rf"common $\mu={common['mu']:.3f}$",
        fontsize=13,
    )
    fig.savefig(FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_inline_html(path, results):
    series = [
        {
            "name": background,
            "mu": round(results[background]["mu"], 4),
            "r2": round(results[background]["r_squared"], 5),
            "x": np.round(results[background]["thresholds"], 7).tolist(),
            "observed": results[background]["counts"].astype(int).tolist(),
            "fitted": np.round(
                results[background]["fitted"], 3
            ).tolist(),
        }
        for background in PANEL_ORDER
    ]
    fragment = f"""
<div id="limdi-ccdf-tail-fits">
  <div class="ccdf-grid" aria-label="Limdi cumulative DFE tail fits"></div>
</div>
<style>
  #limdi-ccdf-tail-fits .ccdf-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px 14px;
  }}
  #limdi-ccdf-tail-fits .ccdf-panel {{
    min-width: 0;
  }}
  #limdi-ccdf-tail-fits .ccdf-heading {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 4px;
  }}
  #limdi-ccdf-tail-fits .ccdf-heading h3 {{
    margin: 0;
  }}
  #limdi-ccdf-tail-fits canvas {{
    display: block;
    width: 100%;
    height: 174px;
  }}
  @media (max-width: 620px) {{
    #limdi-ccdf-tail-fits .ccdf-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
  }}
  @media (max-width: 410px) {{
    #limdi-ccdf-tail-fits .ccdf-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById("limdi-ccdf-tail-fits");
  const grid = root.querySelector(".ccdf-grid");
  const series = {json.dumps(series, separators=(",", ":"))};

  series.forEach((item, index) => {{
    const panel = document.createElement("section");
    panel.className = "ccdf-panel";
    const heading = document.createElement("div");
    heading.className = "ccdf-heading";
    const label = document.createElement("h3");
    label.textContent = item.name;
    const fit = document.createElement("span");
    fit.className = "text-small text-muted";
    fit.textContent = `μ=${{item.mu.toFixed(3)}} · R²=${{item.r2.toFixed(4)}}`;
    const canvas = document.createElement("canvas");
    canvas.dataset.index = String(index);
    canvas.setAttribute("role", "img");
    canvas.setAttribute(
      "aria-label",
      `${{item.name}} log-log cumulative deleterious-tail count and power-law fit from 0.01 to 0.04; fitted mu ${{item.mu.toFixed(3)}}`
    );
    heading.append(label, fit);
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
    const height = 174;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);

    const foreground = token("--foreground");
    const muted = token("--muted-foreground");
    const border = token("--border");
    const empirical = token("--viz-series-1");
    const fitted = token("--viz-series-2");
    const margin = {{left: 42, right: 8, top: 6, bottom: 30}};
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const logX0 = Math.log10(0.01);
    const logX1 = Math.log10(0.04);
    const allY = item.observed.concat(item.fitted);
    const logY0 = Math.floor(Math.log10(Math.min(...allY)));
    const logY1 = Math.ceil(Math.log10(Math.max(...allY)));
    const xScale = value => margin.left + (Math.log10(value) - logX0) / (logX1 - logX0) * w;
    const yScale = value => margin.top + (1 - (Math.log10(value) - logY0) / (logY1 - logY0)) * h;

    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.strokeStyle = border;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let power = logY0; power <= logY1; power += 1) {{
      const value = 10 ** power;
      const py = yScale(value);
      ctx.beginPath();
      ctx.moveTo(margin.left, py);
      ctx.lineTo(margin.left + w, py);
      ctx.stroke();
      ctx.fillText(power === 3 ? "10³" : String(value), margin.left - 5, py);
    }}

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    [0.01, 0.02, 0.04].forEach(value => {{
      const px = xScale(value);
      ctx.beginPath();
      ctx.moveTo(px, margin.top + h);
      ctx.lineTo(px, margin.top + h + 4);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), px, margin.top + h + 7);
    }});

    function line(values, color, dashed) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.beginPath();
      values.forEach((value, i) => {{
        const px = xScale(item.x[i]);
        const py = yScale(value);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }});
      ctx.stroke();
      ctx.setLineDash([]);
    }}
    line(item.observed, empirical, false);
    line(item.fitted, fitted, true);

    ctx.fillStyle = foreground;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("−s", margin.left + w / 2, height);
    if (index % 3 === 0) {{
      ctx.save();
      ctx.translate(10, margin.top + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("tail count", 0, 0);
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

    _, results, summaries, curves = load_and_fit()
    common = summaries["ALL_COMMON_FIXED_INTERCEPTS"]
    write_results(results, summaries)
    curves.to_csv(CURVE_PATH, index=False)
    plot(results, common)
    if args.inline_html is not None:
        write_inline_html(args.inline_html, results)

    for label, summary in summaries.items():
        print(
            f"{label}: "
            f"mu={summary['mu']:.6f} "
            f"[{summary['mu_ci_low']:.6f}, "
            f"{summary['mu_ci_high']:.6f}], "
            f"within-R2={summary['within_r_squared']:.6f}"
        )
    for background in PANEL_ORDER:
        fit = results[background]
        print(
            f"{background:6s} mu={fit['mu']:.6f} "
            f"[{fit['mu_ci_low']:.6f}, {fit['mu_ci_high']:.6f}] "
            f"R2={fit['r_squared']:.6f} "
            f"counts={fit['count_at_001']}->{fit['count_at_004']}"
        )
    print(RESULT_PATH)
    print(CURVE_PATH)
    print(FIG_PATH)
    if args.inline_html is not None:
        print(args.inline_html)


if __name__ == "__main__":
    main()
