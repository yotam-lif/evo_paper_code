r"""Diagnose a common Limdi DFE-tail exponent and its far-tail extrapolation.

Part 1 fixes mu to the equal-background estimate from the twelve DFE CCDFs over
0.01 <= -s <= 0.04.  Only a separate intercept is fitted for each DFE, and the
loss relative to the independent-slope fit is reported.

Part 2 extends REL606, REL607, Ara-1, Ara-3, Ara+2, and Ara+6 to -s = 0.5.
It compares the short-window common-mu extrapolation with a new best log-CCDF
slope over the full 0.01 <= -s <= 0.5 interval.

These are descriptive regressions on correlated empirical cumulative counts,
not power-law maximum-likelihood fits.

Outputs
-------
    data/limdi_common_mu_individual_diagnostics.csv
    data/limdi_common_mu_individual_curves.csv
    data/limdi_extended_ccdf_001_050_diagnostics.csv
    data/limdi_extended_ccdf_001_050_curves.csv
    figs_paper/limdi_common_mu_individual_diagnostics.png
    figs_paper/limdi_extended_ccdf_001_050.png
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
from code_tmp.fit_limdi_dfe_ccdf_powerlaw import (  # noqa: E402
    PANEL_ORDER,
    regress_log_ccdf,
    survival_counts,
)


SHORT_MIN = 0.01
SHORT_MAX = 0.04
EXTENDED_MAX = 0.5
SHORT_POINTS = 80
EXTENDED_POINTS = 180

EXTENDED_ORDER = (
    "REL606",
    "REL607",
    "Ara-1",
    "Ara-3",
    "Ara+2",
    "Ara+6",
)

BASE_RESULT_PATH = (
    REPO_DIR / "data" / "limdi_dfe_ccdf_powerlaw_001_004.csv"
)
COMMON_RESULT_PATH = (
    REPO_DIR / "data" / "limdi_common_mu_individual_diagnostics.csv"
)
COMMON_CURVE_PATH = (
    REPO_DIR / "data" / "limdi_common_mu_individual_curves.csv"
)
EXTENDED_RESULT_PATH = (
    REPO_DIR / "data" / "limdi_extended_ccdf_001_050_diagnostics.csv"
)
EXTENDED_CURVE_PATH = (
    REPO_DIR / "data" / "limdi_extended_ccdf_001_050_curves.csv"
)
COMMON_FIG_PATH = (
    REPO_DIR / "figs_paper" / "limdi_common_mu_individual_diagnostics.png"
)
EXTENDED_FIG_PATH = (
    REPO_DIR / "figs_paper" / "limdi_extended_ccdf_001_050.png"
)


def load_common_mu():
    table = pd.read_csv(BASE_RESULT_PATH)
    row = table.loc[
        table["background"] == "ALL_COMMON_FIXED_INTERCEPTS"
    ]
    if len(row) != 1:
        raise ValueError(
            "Expected one ALL_COMMON_FIXED_INTERCEPTS row in "
            f"{BASE_RESULT_PATH}"
        )
    return float(row.iloc[0]["mu"])


def magnitudes_for(background):
    effects = limdi_gene_series(background).to_numpy(float)
    effects = effects[np.isfinite(effects)]
    return -effects[effects < 0.0]


def fixed_slope_fit(thresholds, counts, mu):
    log_x = np.log(np.asarray(thresholds, float))
    log_y = np.log(np.asarray(counts, float))
    intercept = float(np.mean(log_y + mu * log_x))
    fitted = np.exp(intercept - mu * log_x)
    residual = log_y - np.log(fitted)
    sse = float(np.sum(residual**2))
    sst = float(np.sum((log_y - log_y.mean()) ** 2))
    return {
        "mu": float(mu),
        "intercept": intercept,
        "fitted": fitted,
        "sse": sse,
        "r_squared": float(1.0 - sse / sst),
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_relative_residual": float(
            np.max(np.abs(fitted / counts - 1.0))
        ),
    }


def diagnose_short_window(common_mu):
    thresholds = np.geomspace(SHORT_MIN, SHORT_MAX, SHORT_POINTS)
    diagnostics = {}
    curve_rows = []
    result_rows = []

    for background in PANEL_ORDER:
        magnitudes = magnitudes_for(background)
        counts = survival_counts(magnitudes, thresholds)
        individual = regress_log_ccdf(thresholds, counts)
        common = fixed_slope_fit(
            thresholds, counts, common_mu
        )
        sse_ratio = common["sse"] / individual["sse"]
        diagnostics[background] = {
            "thresholds": thresholds,
            "counts": counts,
            "individual": individual,
            "common": common,
            "sse_ratio": float(sse_ratio),
        }
        result_rows.append(
            {
                "background": background,
                "overall_mu_fixed": common_mu,
                "individual_mu": individual["mu"],
                "individual_r_squared": individual["r_squared"],
                "common_mu_r_squared": common["r_squared"],
                "common_mu_log_rmse": common["log_rmse"],
                "common_mu_max_abs_relative_residual": common[
                    "max_abs_relative_residual"
                ],
                "common_to_individual_sse_ratio": sse_ratio,
            }
        )
        curve_rows.extend(
            {
                "background": background,
                "x": float(x),
                "s": float(-x),
                "tail_count": int(observed),
                "common_mu_fit_count": float(common_count),
                "individual_mu_fit_count": float(individual_count),
            }
            for x, observed, common_count, individual_count in zip(
                thresholds,
                counts,
                common["fitted"],
                individual["fitted"],
            )
        )

    pd.DataFrame(result_rows).to_csv(COMMON_RESULT_PATH, index=False)
    pd.DataFrame(curve_rows).to_csv(COMMON_CURVE_PATH, index=False)
    return diagnostics


def diagnose_extended(common_mu):
    short_thresholds = np.geomspace(
        SHORT_MIN, SHORT_MAX, SHORT_POINTS
    )
    thresholds = np.geomspace(
        SHORT_MIN, EXTENDED_MAX, EXTENDED_POINTS
    )
    diagnostics = {}
    curve_rows = []
    result_rows = []

    for background in EXTENDED_ORDER:
        magnitudes = magnitudes_for(background)
        short_counts = survival_counts(magnitudes, short_thresholds)
        short_common = fixed_slope_fit(
            short_thresholds, short_counts, common_mu
        )
        counts = survival_counts(magnitudes, thresholds)
        if np.any(counts <= 0):
            raise ValueError(
                f"{background} has zero cumulative counts before x=0.5"
            )
        common_extrapolation = np.exp(
            short_common["intercept"]
            - common_mu * np.log(thresholds)
        )
        full_fit = regress_log_ccdf(thresholds, counts)
        predicted_to_observed_at_05 = float(
            common_extrapolation[-1] / counts[-1]
        )
        common_log_residual = (
            np.log(counts) - np.log(common_extrapolation)
        )
        diagnostics[background] = {
            "thresholds": thresholds,
            "counts": counts,
            "common_extrapolation": common_extrapolation,
            "full_fit": full_fit,
            "count_at_05": int(counts[-1]),
            "common_predicted_at_05": float(
                common_extrapolation[-1]
            ),
            "predicted_to_observed_at_05": (
                predicted_to_observed_at_05
            ),
            "common_extrapolation_log_rmse": float(
                np.sqrt(np.mean(common_log_residual**2))
            ),
        }
        result_rows.append(
            {
                "background": background,
                "short_window_common_mu": common_mu,
                "full_range_best_mu": full_fit["mu"],
                "full_range_best_r_squared": full_fit["r_squared"],
                "observed_tail_count_at_x_0.5": int(counts[-1]),
                "common_mu_predicted_count_at_x_0.5": float(
                    common_extrapolation[-1]
                ),
                "common_mu_predicted_to_observed_at_x_0.5": (
                    predicted_to_observed_at_05
                ),
                "common_mu_extrapolation_log_rmse": diagnostics[
                    background
                ]["common_extrapolation_log_rmse"],
            }
        )
        curve_rows.extend(
            {
                "background": background,
                "x": float(x),
                "s": float(-x),
                "tail_count": int(observed),
                "short_common_mu_extrapolated_count": float(
                    common_count
                ),
                "full_range_fit_count": float(full_count),
            }
            for x, observed, common_count, full_count in zip(
                thresholds,
                counts,
                common_extrapolation,
                full_fit["fitted"],
            )
        )

    pd.DataFrame(result_rows).to_csv(
        EXTENDED_RESULT_PATH, index=False
    )
    pd.DataFrame(curve_rows).to_csv(
        EXTENDED_CURVE_PATH, index=False
    )
    return diagnostics


def configure_plotting():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_short(diagnostics, common_mu):
    configure_plotting()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(10.2, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, background in zip(axes.ravel(), PANEL_ORDER):
        diagnostic = diagnostics[background]
        thresholds = diagnostic["thresholds"]
        ax.plot(
            thresholds,
            diagnostic["counts"],
            color="#31688e",
            linewidth=1.5,
            label="empirical cumulative count",
        )
        ax.plot(
            thresholds,
            diagnostic["common"]["fitted"],
            color="#e36a1d",
            linewidth=1.4,
            linestyle="--",
            label=rf"fixed common $\mu={common_mu:.3f}$",
        )
        ax.plot(
            thresholds,
            diagnostic["individual"]["fitted"],
            color="#8e3b9c",
            linewidth=1.1,
            linestyle=":",
            label="individual slope",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", color="0.88", linewidth=0.6)
        ax.set_title(
            f"{background}  "
            rf"$\mu_i={diagnostic['individual']['mu']:.3f}$, "
            rf"$R^2_{{common}}={diagnostic['common']['r_squared']:.3f}$",
            loc="left",
        )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Deleterious magnitude $x=-s$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Raw count with $S\leq -x$")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(
        rf"Each DFE with the same fixed slope $\mu={common_mu:.3f}$",
        fontsize=13,
    )
    fig.savefig(COMMON_FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_extended(diagnostics, common_mu):
    configure_plotting()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(9.0, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, background in zip(axes.ravel(), EXTENDED_ORDER):
        diagnostic = diagnostics[background]
        thresholds = diagnostic["thresholds"]
        ax.plot(
            thresholds,
            diagnostic["counts"],
            color="#31688e",
            linewidth=1.6,
            label="empirical cumulative count",
        )
        ax.plot(
            thresholds,
            diagnostic["common_extrapolation"],
            color="#e36a1d",
            linewidth=1.4,
            linestyle="--",
            label=rf"short-range common $\mu={common_mu:.3f}$",
        )
        ax.plot(
            thresholds,
            diagnostic["full_fit"]["fitted"],
            color="#8e3b9c",
            linewidth=1.2,
            linestyle=":",
            label="best full-range slope",
        )
        ax.axvline(
            SHORT_MAX,
            color="0.55",
            linewidth=0.8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", color="0.88", linewidth=0.6)
        ax.set_title(
            f"{background}  "
            rf"$\mu_{{0.01-0.5}}="
            rf"{diagnostic['full_fit']['mu']:.3f}$, "
            rf"$R^2={diagnostic['full_fit']['r_squared']:.3f}$",
            loc="left",
        )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Deleterious magnitude $x=-s$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Raw count with $S\leq -x$")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(
        r"Selected DFE cumulative tails extended to $s=-0.5$",
        fontsize=13,
    )
    fig.savefig(EXTENDED_FIG_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_inline_html(path, root_id, order, diagnostics, extended):
    series = []
    for background in order:
        diagnostic = diagnostics[background]
        if extended:
            free_fit = diagnostic["full_fit"]
            common_values = diagnostic["common_extrapolation"]
            subtitle = (
                f"mu_full={free_fit['mu']:.3f} | "
                f"R2={free_fit['r_squared']:.3f}"
            )
            aria_detail = (
                f"best full-range mu {free_fit['mu']:.3f}"
            )
        else:
            free_fit = diagnostic["individual"]
            common_values = diagnostic["common"]["fitted"]
            subtitle = (
                f"mu_i={free_fit['mu']:.3f} | "
                f"R2_common={diagnostic['common']['r_squared']:.3f}"
            )
            aria_detail = (
                f"individual mu {free_fit['mu']:.3f} and "
                f"common-fit R squared "
                f"{diagnostic['common']['r_squared']:.3f}"
            )
        series.append(
            {
                "name": background,
                "subtitle": subtitle,
                "aria": aria_detail,
                "x": np.round(
                    diagnostic["thresholds"], 7
                ).tolist(),
                "observed": diagnostic["counts"].astype(int).tolist(),
                "common": np.round(common_values, 3).tolist(),
                "free": np.round(free_fit["fitted"], 3).tolist(),
            }
        )

    columns = 2 if extended else 3
    vertical = SHORT_MAX if extended else None
    common_label = (
        "common μ=.527 extrapolation" if extended else "fixed common μ=.527"
    )
    free_label = (
        "best 0.01–0.5 slope" if extended else "individual slope"
    )
    fragment = f"""
<div id="{root_id}">
  <div class="ccdf-legend text-small" aria-label="Series legend">
    <span><i class="empirical-line" aria-hidden="true"></i>empirical</span>
    <span><i class="common-line" aria-hidden="true"></i>{common_label}</span>
    <span><i class="free-line" aria-hidden="true"></i>{free_label}</span>
  </div>
  <div class="ccdf-grid"></div>
</div>
<style>
  #{root_id} .ccdf-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    margin-bottom: 12px;
  }}
  #{root_id} .ccdf-legend span {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  #{root_id} .ccdf-legend i {{
    display: inline-block;
    width: 24px;
    border-top-width: 2px;
    border-top-style: solid;
  }}
  #{root_id} .empirical-line {{
    border-color: var(--viz-series-1);
  }}
  #{root_id} .common-line {{
    border-color: var(--viz-series-2);
    border-top-style: dashed !important;
  }}
  #{root_id} .free-line {{
    border-color: var(--viz-series-3);
    border-top-style: dotted !important;
  }}
  #{root_id} .ccdf-grid {{
    display: grid;
    grid-template-columns: repeat({columns}, minmax(0, 1fr));
    gap: 18px 14px;
  }}
  #{root_id} .ccdf-panel {{
    min-width: 0;
  }}
  #{root_id} .ccdf-heading {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 4px;
  }}
  #{root_id} .ccdf-heading h3 {{
    margin: 0;
  }}
  #{root_id} canvas {{
    display: block;
    width: 100%;
    height: 186px;
  }}
  @media (max-width: 620px) {{
    #{root_id} .ccdf-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
  }}
  @media (max-width: 410px) {{
    #{root_id} .ccdf-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById("{root_id}");
  const grid = root.querySelector(".ccdf-grid");
  const series = {json.dumps(series, separators=(",", ":"))};
  const vertical = {json.dumps(vertical)};

  series.forEach((item, index) => {{
    const panel = document.createElement("section");
    panel.className = "ccdf-panel";
    const heading = document.createElement("div");
    heading.className = "ccdf-heading";
    const label = document.createElement("h3");
    label.textContent = item.name;
    const subtitle = document.createElement("span");
    subtitle.className = "text-small text-muted";
    subtitle.textContent = item.subtitle
      .replace("mu_i", "μᵢ")
      .replace("mu_full", "μfull")
      .replace("R2_common", "R²common")
      .replace("R2", "R²");
    const canvas = document.createElement("canvas");
    canvas.dataset.index = String(index);
    canvas.setAttribute("role", "img");
    canvas.setAttribute(
      "aria-label",
      `${{item.name}} empirical cumulative tail, common-mu curve, and free-slope curve; ${{item.aria}}`
    );
    heading.append(label, subtitle);
    panel.append(heading, canvas);
    grid.append(panel);
  }});

  function token(name) {{
    return getComputedStyle(root).getPropertyValue(name).trim();
  }}

  function draw(canvas, item, index) {{
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const box = canvas.getBoundingClientRect();
    const width = Math.max(240, box.width);
    const height = 186;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);

    const foreground = token("--foreground");
    const muted = token("--muted-foreground");
    const border = token("--border");
    const empirical = token("--viz-series-1");
    const common = token("--viz-series-2");
    const free = token("--viz-series-3");
    const margin = {{left: 42, right: 8, top: 6, bottom: 30}};
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const logX0 = Math.log10(item.x[0]);
    const logX1 = Math.log10(item.x[item.x.length - 1]);
    const allY = item.observed.concat(item.common, item.free);
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
      const tick = power === 3 ? "10³" : power === 2 ? "10²" : String(value);
      ctx.fillText(tick, margin.left - 5, py);
    }}

    const xTicks = vertical === null
      ? [0.01, 0.02, 0.04]
      : [0.01, 0.04, 0.1, 0.5];
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xTicks.forEach(value => {{
      const px = xScale(value);
      ctx.beginPath();
      ctx.moveTo(px, margin.top + h);
      ctx.lineTo(px, margin.top + h + 4);
      ctx.stroke();
      ctx.fillText(String(value), px, margin.top + h + 7);
    }});

    if (vertical !== null) {{
      ctx.strokeStyle = muted;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xScale(vertical), margin.top);
      ctx.lineTo(xScale(vertical), margin.top + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    function line(values, color, dash) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.setLineDash(dash);
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
    line(item.observed, empirical, []);
    line(item.common, common, [5, 4]);
    line(item.free, free, [2, 3]);

    ctx.fillStyle = foreground;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("−s", margin.left + w / 2, height);
    if (index % {columns} === 0) {{
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
    parser.add_argument("--common-inline-html", type=Path)
    parser.add_argument("--extended-inline-html", type=Path)
    args = parser.parse_args()

    common_mu = load_common_mu()
    short = diagnose_short_window(common_mu)
    extended = diagnose_extended(common_mu)
    plot_short(short, common_mu)
    plot_extended(extended, common_mu)
    if args.common_inline_html is not None:
        write_inline_html(
            args.common_inline_html,
            "limdi-common-mu-diagnostics",
            PANEL_ORDER,
            short,
            extended=False,
        )
    if args.extended_inline_html is not None:
        write_inline_html(
            args.extended_inline_html,
            "limdi-extended-ccdf-diagnostics",
            EXTENDED_ORDER,
            extended,
            extended=True,
        )

    print(f"Fixed common mu = {common_mu:.9f}")
    for background in PANEL_ORDER:
        diagnostic = short[background]
        print(
            f"{background:6s} "
            f"mu_i={diagnostic['individual']['mu']:.6f} "
            f"R2_common={diagnostic['common']['r_squared']:.6f} "
            f"max_rel={diagnostic['common']['max_abs_relative_residual']:.4f}"
        )
    print("Extended:")
    for background in EXTENDED_ORDER:
        diagnostic = extended[background]
        print(
            f"{background:6s} "
            f"mu_full={diagnostic['full_fit']['mu']:.6f} "
            f"R2_full={diagnostic['full_fit']['r_squared']:.6f} "
            f"pred/obs@0.5={diagnostic['predicted_to_observed_at_05']:.3f}"
        )
    for path in (
        COMMON_RESULT_PATH,
        COMMON_CURVE_PATH,
        EXTENDED_RESULT_PATH,
        EXTENDED_CURVE_PATH,
        COMMON_FIG_PATH,
        EXTENDED_FIG_PATH,
    ):
        print(path)
    if args.common_inline_html is not None:
        print(args.common_inline_html)
    if args.extended_inline_html is not None:
        print(args.extended_inline_html)


if __name__ == "__main__":
    main()
