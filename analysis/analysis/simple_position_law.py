"""Build the oversimplified Section V / Fig. D5 explanation.

This deliberately throws away the detailed rho_p form and fits the cartoon
law

    tau(d) = tau_max * (1 - exp(-d / d_star))

to the existing position-only half-time curves.
"""

import base64
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
OUT_HTML = os.path.join(ROOT, "transient_law_derivation_full.html")
OUT_FIG = os.path.join(FIG, "d5_simple_position_law.png")
OUT_JSON = os.path.join(RES, "simple_position_law.json")


def tag_info(tag):
    m = re.match(r"N(\d+)_P(\d+)", tag)
    if not m:
        raise ValueError(f"Cannot parse tag {tag!r}")
    return int(m.group(1)), int(m.group(2))


def fit_shared_law(data):
    """Fit tau/N = a * (1 - exp(-(d/N) / b)) across all D5 curves."""
    xs, ys = [], []
    for tag, result in data.items():
        N, _ = tag_info(tag)
        xs.extend(result["grid"]["dN"])
        ys.extend(np.asarray(result["grid"]["t_half_vmeas"], float) / N)
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)

    best = None
    for b in np.geomspace(0.02, 3.0, 12000):
        f = 1 - np.exp(-x / b)
        a = float(np.dot(y, f) / np.dot(f, f))
        pred = a * f
        rmse_abs = float(np.sqrt(np.mean((pred - y) ** 2)))
        rmse_rel = float(np.sqrt(np.mean(((pred - y) / y) ** 2)))
        if best is None or rmse_abs < best["rmse_abs"]:
            best = {"a": a, "b": float(b), "rmse_abs": rmse_abs,
                    "rmse_rel": rmse_rel}
    return best


def summarize(data, fit):
    summary = {"shared_fit": fit, "models": {}}
    a, b = fit["a"], fit["b"]
    for tag, result in data.items():
        N, p = tag_info(tag)
        x = np.asarray(result["grid"]["dN"], float)
        y = np.asarray(result["grid"]["t_half_vmeas"], float) / N
        pred = a * (1 - np.exp(-x / b))
        rel = (pred - y) / y

        measured = []
        for point in result["points"]:
            yhat = a * N * (1 - np.exp(-point["d"] / (b * N)))
            measured.append({
                "dN": point["d"] / N,
                "measured_t_half": point["t_half"],
                "simple_t_half": float(yhat),
                "measured_over_simple": point["t_half"] / float(yhat),
            })
        ratios = [m["measured_over_simple"] for m in measured]
        lam = 2 * (p - 1) + 4
        v0 = result["points"][0]["v_rem"]
        alpha = deep_half_alpha(v0)
        candidate_specs = [
            ("free midrange fit", a * N, b * N),
            ("rho-memory tangent", 2 * N / lam, N / lam),
            ("endpoint half-time", N / 4, N / (4 * alpha)),
        ]
        d_raw = np.asarray(result["grid"]["dN"], float) * N
        y_raw = np.asarray(result["grid"]["t_half_vmeas"], float)
        curve_tests = []
        for name, tau_max, d_star in candidate_specs:
            pred_raw = tau_max * (1 - np.exp(-d_raw / d_star))
            rel_raw = (pred_raw - y_raw) / y_raw
            point_ratios = []
            for point in result["points"]:
                point_pred = tau_max * (1 - np.exp(-point["d"] / d_star))
                point_ratios.append(point["t_half"] / point_pred)
            curve_tests.append({
                "name": name,
                "tau_max": tau_max,
                "d_star": d_star,
                "d_star_over_N": d_star / N,
                "rms_relative_error_vs_detailed_curve": float(np.sqrt(np.mean(rel_raw ** 2))),
                "max_relative_error_vs_detailed_curve": float(np.max(np.abs(rel_raw))),
                "measured_ratio_min": float(min(point_ratios)),
                "measured_ratio_max": float(max(point_ratios)),
            })
        summary["models"][tag] = {
            "N": N,
            "p": p,
            "tau_max": a * N,
            "d_star": b * N,
            "lambda": lam,
            "rho_memory_length": N / lam,
            "tangent_tau_max": 2 * N / lam,
            "near_basin_v_rem": v0,
            "near_basin_half_slope": alpha,
            "endpoint_matched_half_tau_max": N / 4,
            "endpoint_matched_half_d_star": N / (4 * alpha),
            "curve_tests": curve_tests,
            "rms_relative_error_vs_detailed_curve": float(np.sqrt(np.mean(rel ** 2))),
            "max_relative_error_vs_detailed_curve": float(np.max(np.abs(rel))),
            "measured_ratio_min": float(min(ratios)),
            "measured_ratio_max": float(max(ratios)),
            "measured_points": measured,
        }
    return summary


def deep_half_alpha(v):
    """Deep-basin half-time coefficient t_half = alpha(v) d for c=1."""
    a = (1 + v) ** 2
    b = -(4 + 3 * v)
    c = 3
    return float((-b - np.sqrt(b * b - 4 * a * c)) / (2 * a))


def make_figure(data, summary):
    a = summary["shared_fit"]["a"]
    b = summary["shared_fit"]["b"]

    model_items = sorted(summary["models"].items(), key=lambda kv: kv[1]["p"])
    colors = {2: "#1f77b4", 3: "#d62728"}

    W, H = 1800, 700
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(24)
    small = load_font(19)
    tiny = load_font(16)
    title_font = load_font(31)

    draw.text((W // 2, 22), "Fig. D5, simplified: current depth sets a saturating clock",
              fill="#222222", font=title_font, anchor="ma")

    panels = [(70, 105, 540, 505), (650, 105, 1120, 505), (1230, 105, 1700, 505)]
    for panel, (tag, model) in zip(panels[:2], model_items):
        N, p = model["N"], model["p"]
        grid = data[tag]["grid"]
        x = np.asarray(grid["dN"], float)
        detailed = np.asarray(grid["t_half_vmeas"], float)
        simple = a * N * (1 - np.exp(-x / b))
        rho_tangent = model["tangent_tau_max"] * (
            1 - np.exp(-(x * N) / model["rho_memory_length"])
        )
        half_endpoint = model["endpoint_matched_half_tau_max"] * (
            1 - np.exp(-(x * N) / model["endpoint_matched_half_d_star"])
        )
        pts = data[tag]["points"]

        ymax = max(float(detailed.max()), float(simple.max()),
                   float(rho_tangent.max()), float(half_endpoint.max()),
                   max(pt["t_half"] for pt in pts)) * 1.12
        plot_axes(draw, panel, xlim=(0, 0.46), ylim=(0, ymax),
                  xlabel="depth d/N", ylabel="half-time", font=small, tiny=tiny)
        plot_line(draw, panel, x, detailed, (0, 0.46), (0, ymax), "#555555", width=5)
        plot_line(draw, panel, x, simple, (0, 0.46), (0, ymax), "#d62728", width=6)
        plot_line(draw, panel, x, rho_tangent, (0, 0.46), (0, ymax), "#1f77b4", width=4)
        plot_line(draw, panel, x, half_endpoint, (0, 0.46), (0, ymax), "#2ca02c", width=4)
        plot_points(draw, panel, [pt["d"] / N for pt in pts],
                    [pt["t_half"] for pt in pts], (0, 0.46), (0, ymax),
                    "#6a3fb5", r=8)

        x0, y0, x1, _ = panel
        draw.text(((x0 + x1) // 2, y0 - 42), f"p={p}, N={N}",
                  fill="#222222", font=font, anchor="ma")
        draw.text((x0 + 16, y0 + 42),
                  f"rms error {100 * model['rms_relative_error_vs_detailed_curve']:.1f}%",
                  fill="#333333", font=tiny)
        legend(draw, x0 + 70, panel[3] + 18, [
            ("full d-only law", "#555555"),
            ("free fit", "#d62728"),
            ("rho tangent", "#1f77b4"),
            ("half endpoint", "#2ca02c"),
            ("measured", "#6a3fb5"),
        ], tiny)

    panel = panels[2]
    xlim = (0.018, 0.46)
    ylim = (0.007, 0.65)
    plot_axes(draw, panel, xlim=xlim, ylim=ylim, xlabel="depth d/N",
              ylabel="beneficial fraction n+/N", font=small, tiny=tiny, logx=True, logy=True)
    x0, y0, x1, _ = panel
    draw.text(((x0 + x1) // 2, y0 - 42), "observable proxy for d",
              fill="#222222", font=font, anchor="ma")
    for tag, model in model_items:
        p = model["p"]
        cal = data[tag]["n_pos_calibration"]
        plot_line(draw, panel, cal["dN"], cal["nposN"], xlim, ylim,
                  colors[p], width=4, logx=True, logy=True)
        plot_points(draw, panel, cal["dN"], cal["nposN"], xlim, ylim,
                    colors[p], r=5, logx=True, logy=True)
    legend(draw, x1 - 120, y0 + 20, [("p=2", colors[2]), ("p=3", colors[3])], tiny)

    img.save(OUT_FIG)


def load_font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def transform(x, y, panel, xlim, ylim, logx=False, logy=False):
    x0, y0, x1, y1 = panel
    px0, py0, px1, py1 = x0 + 52, y0 + 20, x1 - 15, y1 - 48

    def scale(v, lo, hi, log):
        if log:
            v, lo, hi = np.log10(v), np.log10(lo), np.log10(hi)
        return (v - lo) / (hi - lo)

    sx = scale(np.asarray(x, float), xlim[0], xlim[1], logx)
    sy = scale(np.asarray(y, float), ylim[0], ylim[1], logy)
    return px0 + sx * (px1 - px0), py1 - sy * (py1 - py0)


def plot_axes(draw, panel, xlim, ylim, xlabel, ylabel, font, tiny, logx=False, logy=False):
    x0, y0, x1, y1 = panel
    px0, py0, px1, py1 = x0 + 52, y0 + 20, x1 - 15, y1 - 48
    draw.rectangle((px0, py0, px1, py1), outline="#888888", width=2)

    if logx:
        xticks = [0.02, 0.05, 0.1, 0.2, 0.4]
    else:
        xticks = [0, 0.1, 0.2, 0.3, 0.4]
    if logy:
        yticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    else:
        top = ylim[1]
        step = nice_step(top / 4)
        yticks = list(np.arange(0, top + 0.5 * step, step))

    for xt in xticks:
        if xlim[0] <= xt <= xlim[1]:
            px, _ = transform([xt], [ylim[0]], panel, xlim, ylim, logx, logy)
            px = float(px[0])
            draw.line((px, py0, px, py1), fill="#eeeeee", width=1)
            draw.text((px, py1 + 8), format_tick(xt), fill="#333333", font=tiny, anchor="ma")
    for yt in yticks:
        if ylim[0] <= yt <= ylim[1]:
            _, py = transform([xlim[0]], [yt], panel, xlim, ylim, logx, logy)
            py = float(py[0])
            draw.line((px0, py, px1, py), fill="#eeeeee", width=1)
            draw.text((px0 - 8, py), format_tick(yt), fill="#333333", font=tiny, anchor="rm")

    draw.text(((px0 + px1) // 2, y1 - 14), xlabel, fill="#222222", font=font, anchor="ma")
    draw.text((x0 + 2, py0 - 8), ylabel, fill="#222222", font=font)


def nice_step(value):
    if value <= 0:
        return 1
    exp = np.floor(np.log10(value))
    frac = value / (10 ** exp)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def format_tick(value):
    if value == 0:
        return "0"
    if value < 1:
        return f"{value:g}"
    return f"{value:.0f}"


def plot_line(draw, panel, xs, ys, xlim, ylim, color, width=3, logx=False, logy=False):
    px, py = transform(xs, ys, panel, xlim, ylim, logx, logy)
    pts = [(float(x), float(y)) for x, y in zip(px, py)
           if np.isfinite(x) and np.isfinite(y)]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def plot_points(draw, panel, xs, ys, xlim, ylim, color, r=5, logx=False, logy=False):
    px, py = transform(xs, ys, panel, xlim, ylim, logx, logy)
    for x, y in zip(px, py):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r),
                     fill=color, outline="white", width=2)


def legend(draw, x, y, items, font):
    for i, (label, color) in enumerate(items):
        yy = y + 24 * i
        draw.line((x, yy + 9, x + 28, yy + 9), fill=color, width=5)
        draw.text((x + 36, yy), label, fill="#222222", font=font)


def img_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def build_html(summary):
    a = summary["shared_fit"]["a"]
    b = summary["shared_fit"]["b"]
    rows = []
    analytic_rows = []
    fit_rows = []
    for tag, model in sorted(summary["models"].items(), key=lambda kv: kv[1]["p"]):
        rows.append(
            "<tr>"
            f"<td>{model['p']}</td>"
            f"<td>{model['N']}</td>"
            f"<td>{model['tau_max']:.1f}</td>"
            f"<td>{model['d_star']:.1f}</td>"
            f"<td>{100 * model['rms_relative_error_vs_detailed_curve']:.1f}%</td>"
            f"<td>{100 * model['max_relative_error_vs_detailed_curve']:.1f}%</td>"
            f"<td>{model['measured_ratio_min']:.2f}-{model['measured_ratio_max']:.2f}</td>"
            "</tr>"
        )
        for test in model["curve_tests"]:
            fit_rows.append(
                "<tr>"
                f"<td>{model['p']}</td>"
                f"<td>{model['N']}</td>"
                f"<td>{test['name']}</td>"
                f"<td>{test['tau_max']:.1f}</td>"
                f"<td>{test['d_star']:.1f}</td>"
                f"<td>{test['d_star_over_N']:.3f}</td>"
                f"<td>{100 * test['rms_relative_error_vs_detailed_curve']:.1f}%</td>"
                f"<td>{100 * test['max_relative_error_vs_detailed_curve']:.1f}%</td>"
                f"<td>{test['measured_ratio_min']:.2f}-{test['measured_ratio_max']:.2f}</td>"
                "</tr>"
            )
        analytic_rows.append(
            "<tr>"
            f"<td>{model['p']}</td>"
            f"<td>{model['N']}</td>"
            f"<td>{model['lambda']:.0f}</td>"
            f"<td>{model['rho_memory_length']:.1f}</td>"
            f"<td>{model['rho_memory_length'] / model['N']:.3f}</td>"
            f"<td>{model['endpoint_matched_half_d_star']:.1f}</td>"
            f"<td>{model['endpoint_matched_half_d_star'] / model['N']:.3f}</td>"
            f"<td>{model['near_basin_half_slope']:.3f}</td>"
            "</tr>"
        )

    html = HTML_TEMPLATE
    html = html.replace("__FIG_B64__", img_b64(OUT_FIG))
    html = html.replace("__A__", f"{a:.3f}")
    html = html.replace("__B__", f"{b:.3f}")
    html = html.replace("__ROWS__", "\n".join(rows))
    html = html.replace("__ANALYTIC_ROWS__", "\n".join(analytic_rows))
    html = html.replace("__FIT_ROWS__", "\n".join(fit_rows))
    return html


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Section V, simplified: the position-only transient clock</title>
<script>
window.MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']],
                        displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script id="MathJax-script" async
 src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
 body {font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; line-height:1.65;
       max-width:880px; margin:0 auto; padding:34px 20px; color:#202124; background:#fbfbfb;}
 h1 {font-size:1.75em; border-bottom:3px solid #6a3fb5; padding-bottom:.35em; line-height:1.25;}
 h2 {font-size:1.22em; margin-top:2em; border-bottom:1px solid #c8c8c8; padding-bottom:.2em;}
 p {margin-bottom:1em;}
 figure {margin:1.45em 0; text-align:center;}
 figure img {max-width:100%; border:1px solid #ddd; border-radius:4px; background:#fff;}
 figcaption {font-size:.9em; color:#444; text-align:left; margin-top:.55em; padding:0 8px;}
 .box {margin:1.3em 0; padding:.9em 1.1em; background:#f3eefc; border-left:4px solid #6a3fb5;}
 .step {margin:1.3em 0; padding:.9em 1.1em; background:#eef6ff; border-left:4px solid #0066cc;}
 .warn {margin:1.3em 0; padding:.9em 1.1em; background:#fff8e6; border-left:4px solid #d99a00;}
 table {border-collapse:collapse; margin:1.1em auto; font-size:.92em;}
 th,td {border:1px solid #ccc; padding:.38em .7em; text-align:center;}
 th {background:#eef2f7;}
 code {background:#f0f0f0; padding:.05em .3em; border-radius:3px; font-size:.92em;}
 .dim {color:#777; font-size:.9em;}
</style>
</head>
<body>

<h1>Section V in one page: a position-only clock</h1>

<p class="dim">This is the intentionally simplified replacement for the long derivation.
The full version was preserved as <code>transient_law_derivation_full_preserved.html</code>.
Assumption for this page: $c=1$.</p>

<div class="box">
<p><b>The cartoon claim.</b> If the walk is currently a distance $d$ from the maximum, the
transient-correlation half-time is approximately</p>
$$\boxed{\tau(d)=\tau_{\max}\left(1-e^{-d/d_*}\right)}$$
<p>For the two Fig. D5 examples, a single normalized fit is already good enough:</p>
$$\boxed{\tau(d)\approx __A__\,N\left(1-e^{-d/(__B__\,N)}\right)}.$$
</div>

<h2>1. What survives from the full law</h2>

<p>The exact-looking law had three distances and a complicated pool-memory factor
$\rho_p$. For prediction, Section V plugs in the typical future path:
$$u_{12}(\Delta t)\approx \Delta t,\qquad d_2(\Delta t)\approx d-v_{\rm rem}\Delta t.$$
When you differentiate the resulting curve at $\Delta t=0$, the two terms involving
$v_{\rm rem}$ cancel. That is the useful part. It says:</p>

<div class="step">
$$\text{clock at depth }d \;\propto\; 1-\rho_p(d).$$
</div>

<p>In words: the remaining transient variance sets the time scale. Near the maximum there is
little variance left, so the clock is short. Far away, the pool has already forgotten most of
the maximum, so the clock cannot grow forever.</p>

<h2>2. How to simplify $\rho_p$ consistently</h2>

<p>The detailed $\rho_p(d)$ knows about Krawtchouk factors, flipped-spin signs, clipping, and
adaptivity. To explain Fig. D5 rather than re-derive it, keep only the shape it must have:</p>
<ul>
<li>$\rho_p(0)=1$: at zero distance, the spectrum is perfectly correlated with itself.</li>
<li>$\rho_p(d)$ decreases as $d$ grows.</li>
<li>It has one effective memory length.</li>
</ul>

<p>The simplest function with those properties is a single exponential,
$$\rho_p(d)\approx e^{-d/d_*}.$$
Substituting this into "clock $\propto 1-\rho_p(d)$" gives the saturation law above.</p>

<div class="warn">
<p><b>What got swept under the rug?</b> The difference between a tangent time and a half-time,
the exact small-distance slope, the $p$-dependence, and the clipping of the original
$\rho_p$. All of that is now absorbed into the two fitted constants
$\tau_{\max}$ and $d_*$. This is less exact and much easier to remember.</p>
</div>

<h2>3. Can $d_*$ be derived?</h2>

<p>Yes, but only after deciding what the simplified exponential is trying to match.</p>

<p><b>If the exponential is replacing $\rho_p$ itself</b>, the small-distance slope gives a
clean derivation. The detailed law has
$$\rho_p'(0)=-\Lambda/N,\qquad \Lambda=2(p-1)+4.$$
Matching $e^{-d/d_*}$ to that slope gives
$$d_*^{(\rho)}=\frac{N}{\Lambda}.$$
This is the microscopic memory length of the pool law. It belongs to the tangent-time formula
$$\tau_{\rm tan}(d)=\frac{2N}{\Lambda}\left(1-e^{-d/(N/\Lambda)}\right).$$</p>

<p><b>If the exponential is replacing the Fig. D5 half-time curve</b>, two extra effects enter:
the half-time is shorter than the tangent time, and the far-field half-time saturates near
$N/4$ because the clipped flip factor in $\rho_p(u)$ hits zero at $u=N/4$. In the deep basin,
the counting formula gives
$$\rho(\Delta t)\approx
\frac{2d-(1+v_{\rm rem})\Delta t}{2\sqrt{d(d-v_{\rm rem}\Delta t)}}.$$
Setting this to $1/2$ gives $t_{1/2}\approx \alpha(v_{\rm rem})d$, where
$$\alpha(v)=
\frac{(4+3v)-\sqrt{(4+3v)^2-12(1+v)^2}}{2(1+v)^2}.$$
Matching that slope to a saturation curve with plateau $N/4$ gives
$$d_*^{(1/2)}\approx \frac{N}{4\alpha(v_{\rm rem})}\approx 0.33N.$$
That is an analytic, endpoint-matched estimate for the half-time version.</p>

<table>
<tr><th>$p$</th><th>$N$</th><th>$\Lambda$</th><th>$d_*^{(\rho)}$</th>
<th>$d_*^{(\rho)}/N$</th><th>$d_*^{(1/2)}$</th>
<th>$d_*^{(1/2)}/N$</th><th>$\alpha(v_{\rm rem})$</th></tr>
__ANALYTIC_ROWS__
</table>

<p>The fitted $d_*=0.587N$ above is therefore not a fundamental length. It is an effective
midrange parameter: it is what you get when both $\tau_{\max}$ and $d_*$ are allowed to float
so that one red curve shadows the full d-only law over the plotted depths. For derivation,
$N/\Lambda$ is the clean answer for $\rho_p$; roughly $N/3$ is the endpoint-matched answer for
the half-time cartoon.</p>

<h2>4. Reading Fig. D5</h2>

<figure>
<img src="data:image/png;base64,__FIG_B64__" alt="Simplified Fig. D5">
<figcaption><b>Fig. D5, simplified.</b> The grey curve is the full d-only law from the original
Section V calculation: plug $u_{12}=\Delta t$ and $d_2=d-v_{\rm rem}\Delta t$ into the full
kernel-difference formula, then numerically read off its half-time. The purple points are the
measured transient half-times. Red is the best free two-parameter saturation fit. Blue uses
the analytic $d_*^{(\rho)}=N/\Lambda$ tangent curve. Green uses the endpoint-matched half-time
estimate $d_*^{(1/2)}\approx N/3$. The right panel keeps the operational punchline: measure
$n_+$, infer $d$, then plug that $d$ into the chosen clock.</figcaption>
</figure>

<table>
<tr><th>$p$</th><th>$N$</th><th>curve</th><th>$\tau_{\max}$</th><th>$d_*$</th>
<th>$d_*/N$</th><th>rms error vs full d-only law</th><th>max error</th>
<th>measured/curve range</th></tr>
__FIT_ROWS__
</table>

<h2>5. The shortest usable recipe</h2>

<ol>
<li>Count the currently beneficial moves, $n_+$.</li>
<li>Use the right panel calibration to translate $n_+$ into an estimated depth $\hat d$.</li>
<li>Forecast the transient half-time with
$$\tau(\hat d)\approx 0.5N\left(1-e^{-\hat d/(0.59N)}\right).$$</li>
</ol>

<p>That is the whole simplified Section V: depth is the clock, the clock grows roughly
linearly near the maximum, and it saturates at about half the system size in accepted flips.</p>

<p class="dim">Reproduce: <code>analysis/simple_position_law.py</code> rebuilds this page,
<code>analysis/figures/d5_simple_position_law.png</code>, and
<code>analysis/results/simple_position_law.json</code>. The detailed source curves come from
<code>analysis/results/position_only.json</code>.</p>

</body>
</html>
"""


def main():
    with open(os.path.join(RES, "position_only.json")) as fh:
        data = json.load(fh)
    fit = fit_shared_law(data)
    summary = summarize(data, fit)
    make_figure(data, summary)
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(OUT_HTML, "w") as fh:
        fh.write(build_html(summary))
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
