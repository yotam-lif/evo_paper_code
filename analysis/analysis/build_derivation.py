"""Build the standalone step-by-step derivation document
(transient_law_derivation.html) with its two pedagogical figures."""

import base64
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis_lib as al  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def rho_p(u, N, p):
    u = np.asarray(u, float)
    return al.rho_unflip_theory(u, N, [p]) * np.maximum(1 - 4 * u / N, 0)


# ---------------------------------------------------------------------------
# Fig D1: the configuration triangle
# ---------------------------------------------------------------------------

def fig_triangle():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    P = {"t1": (0.08, 0.25), "t2": (0.62, 0.72), "f": (0.95, 0.18)}
    lbl = {"t1": r"$\sigma(t_1)$" + "\n(reference)", "t2": r"$\sigma(t_2)$" + "\n(later time)",
           "f": r"$\sigma_f = \sigma(T)$" + "\n(terminal maximum)"}
    for k, (x, y) in P.items():
        ax.plot(x, y, "o", ms=14, color="#6a3fb5" if k != "f" else "#d62728", zorder=5)
        dy = -0.14 if k != "t2" else 0.09
        ax.text(x, y + dy, lbl[k], ha="center", fontsize=11,
                va="top" if k != "t2" else "bottom")
    def edge(a, b, text, off, color="#444"):
        (x1, y1), (x2, y2) = P[a], P[b]
        ax.plot([x1, x2], [y1, y2], "-", color=color, lw=1.6)
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text,
                fontsize=12, color=color, ha="center")
    edge("t1", "t2", r"$u_{12}$", (-0.06, 0.02))
    edge("t2", "f", r"$u(t_2,T)=d_2$", (0.14, 0.03))
    edge("t1", "f", r"$u(t_1,T)=d_1$", (0.0, -0.055))
    ax.text(0.5, 1.0, "three configurations, three mutual Hamming distances;\n"
                      r"the pool covariance of the spectra depends only on the distance: $C(u)$",
            ha="center", fontsize=10.5, va="top", transform=ax.transAxes)
    ax.set_xlim(-0.12, 1.22)
    ax.set_ylim(-0.05, 1.02)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d1_triangle.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig D2: anatomy of the formula on a real bin + worked example numbers
# ---------------------------------------------------------------------------

def fig_anatomy():
    tag, p, N = "N1000_P2", 2, 1000
    z = np.load(os.path.join(RES, f"meas_dense_{tag}.npz"))
    Ts = z["T"]
    lo, hi = 0.10, 0.14
    rows = {"tr": [], "u12": [], "d2": [], "d1": []}
    for r in range(len(Ts)):
        T = int(Ts[r])
        d_f_r = z["scal_d_f"][r]
        for j in range(z["ref_t"].shape[1]):
            tr_, d1 = int(z["ref_t"][r, j]), float(z["ref_d_f"][r, j])
            if tr_ < 0 or tr_ >= T - 4 or not (lo <= d1 / N < hi):
                continue
            rows["tr"].append(z["pair_rho_trans"][r, j, tr_:T + 1])
            rows["u12"].append(z["pair_u"][r, j, tr_:T + 1].astype(float))
            rows["d2"].append(d_f_r[tr_:T + 1].astype(float))
            rows["d1"].append(d1)
    Lm = min(len(a) for a in rows["tr"])
    tr = np.nanmean([a[:Lm] for a in rows["tr"]], axis=0)
    u12 = np.nanmean([a[:Lm] for a in rows["u12"]], axis=0)
    d2 = np.nanmean([a[:Lm] for a in rows["d2"]], axis=0)
    d1 = float(np.mean(rows["d1"]))
    dt = np.arange(Lm)

    r1 = rho_p(np.array([d1]), N, p)[0]
    r2 = rho_p(d2, N, p)
    ru = rho_p(u12, N, p)
    num = ru + 1 - r1 - r2
    den = 2 * np.sqrt(np.maximum((1 - r1) * (1 - r2), 1e-12))
    pred = num / den

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(dt, ru, color="#1f77b4", label=r"$\rho_p(u_{12}(\Delta t))$  (pair $t_1,t_2$)")
    ax.plot(dt, r2, color="#2ca02c", label=r"$\rho_p(d_2(\Delta t))$  (pair $t_2,T$)")
    ax.axhline(r1, color="#d62728", ls="--", label=fr"$\rho_p(d_1)={r1:.3f}$  (pair $t_1,T$)")
    ax.set_xlabel(r"$\Delta t$ (accepted moves)")
    ax.set_ylabel("pairwise pool correlation")
    ax.set_title(f"ingredients, measured walk: p={p}, N={N}, "
                 fr"$d_1\approx{d1:.0f}$", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.plot(dt, num, color="#9467bd", label=r"numerator  $\rho_p(u_{12})+1-\rho_p(d_1)-\rho_p(d_2)$")
    ax.plot(dt, den, color="#8c564b", label=r"denominator  $2\sqrt{(1-\rho_p(d_1))(1-\rho_p(d_2))}$")
    ax.plot(dt, pred, color="#d62728", lw=2.2, label="ratio = the law")
    ax.plot(dt, tr, color="#6a3fb5", lw=2.2, alpha=0.85, label=r"measured $\rho_{\rm trans}$")
    ax.set_xlabel(r"$\Delta t$ (accepted moves)")
    ax.set_title("assembly: numerator / denominator vs data", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d2_anatomy.png"), dpi=160)
    plt.close(fig)

    # worked example at a specific lag
    k = 40
    ex = dict(N=N, p=p, d1=d1, dt=int(dt[k]), u12=float(u12[k]), d2=float(d2[k]),
              r1=float(r1), r2=float(r2[k]), ru=float(ru[k]),
              num=float(num[k]), den=float(den[k]), pred=float(pred[k]),
              meas=float(tr[k]))
    with open(os.path.join(RES, "derivation_example.json"), "w") as fh:
        json.dump(ex, fh, indent=1)
    return ex


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def img(name, caption):
    with open(os.path.join(FIG, name), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{name}">'
            f'<figcaption>{caption}</figcaption></figure>')


def build_html(ex):
    krow2 = " ".join(f"<td>{(1000-1-2*u)/(1000-1):.3f}</td>" for u in (0, 10, 50, 100, 200))
    HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Full derivation: the kernel-difference law</title>
<script>
window.MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']],
                        displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script id="MathJax-script" async
 src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
 body {font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; line-height:1.7;
       max-width:920px; margin:0 auto; padding:40px 20px; color:#222; background:#fbfbfb;}
 h1 {font-size:1.6em; border-bottom:3px solid #6a3fb5; padding-bottom:.4em;}
 h2 {font-size:1.28em; margin-top:2.2em; border-bottom:1px solid #bbb; padding-bottom:.2em;}
 h3 {font-size:1.05em; margin-top:1.6em;}
 p {text-align:justify; margin-bottom:1.05em;}
 figure {margin:1.6em 0; text-align:center;}
 figure img {max-width:100%; border:1px solid #ddd; border-radius:4px; background:#fff;}
 figcaption {font-size:.88em; color:#444; text-align:justify; margin-top:.5em; padding:0 8px;}
 .box {margin:1.4em 0; padding:.9em 1.2em; background:#f3eefc; border-left:4px solid #6a3fb5;}
 .step {margin:1.4em 0; padding:.9em 1.2em; background:#eef6ff; border-left:4px solid #0066cc;}
 .warn {background:#fff8e6; border-left:4px solid #d99a00; margin:1.4em 0; padding:.9em 1.2em;}
 table {border-collapse:collapse; margin:1.2em auto; font-size:.93em;}
 th,td {border:1px solid #ccc; padding:.4em .8em; text-align:center;}
 th {background:#eef2f7;}
 code {background:#f0f0f0; padding:.05em .3em; border-radius:3px; font-size:.92em;}
 .dim {color:#777; font-size:.9em;}
 li {margin-bottom:.45em;}
</style>
</head>
<body>

<h1>Full derivation of the all-times transient law
$$\rho_{\rm trans}(t_1,t_2) = \frac{\rho_p(u_{12}) + 1 - \rho_p(d_1) - \rho_p(d_2)}
{2\sqrt{\big(1-\rho_p(d_1)\big)\big(1-\rho_p(d_2)\big)}}$$</h1>

<p class="dim">Self-contained companion to <code>scrambling_report.html</code>, Sec. 8.2.
Every step is either an explicit computation shown here, an exact identity, or a clearly
flagged approximation with its measured accuracy. Conventions follow the data:
$\operatorname{Var}(J)=p!/N^{p-1}$; time $=$ accepted flips.</p>

<h2>0. What we are computing, and the geometry of the statement</h2>

<p>Along a greedy walk that ends at the local maximum $\sigma_f=\sigma(T)$, define for every
move $i$ its <b>transient</b>: the deviation of its energy change from the terminal value,
$$a_i(t) \;\equiv\; \Delta_i(t) - \Delta_i(T).$$
The object of interest is the Pearson correlation of the transients across the pool of $N$
moves, between two times $t_1<t_2$:
$$\rho_{\rm trans}(t_1,t_2)
= \frac{\operatorname{Cov}_i\big(a_i(t_1),\,a_i(t_2)\big)}
{\sqrt{\operatorname{Var}_i\,a_i(t_1)\;\operatorname{Var}_i\,a_i(t_2)}}.$$
Three configurations appear: $\sigma(t_1)$, $\sigma(t_2)$, and $\sigma_f$. They form a triangle
whose side lengths are mutual Hamming distances (Fig. D1): $u_{12}$ between the two times, and
&mdash; this is a definition, not an assumption &mdash;
$$u(t_1,T) = d_H\big(\sigma(t_1),\sigma_f\big) \equiv d_1,\qquad u(t_2,T) \equiv d_2 .$$
The whole derivation consists of (I) one physical input &mdash; the pool covariance of the
spectra at two configurations depends only on their mutual distance, $C(u)$, with a formula we
derive from the couplings &mdash; and (II) two lines of covariance algebra applied to the
triangle.</p>

__D1__

<h2>I. The input: the pairwise pool covariance $C(u)$</h2>

<p>We need the covariance, across the pool of moves, of the spectrum at two configurations
$\sigma$, $\sigma'$ at mutual Hamming distance $u$ along the walk. It is built from four
pieces, each derived below: (a) stationarity, (b) the unflipped-spin correlation
(a counting computation), (c) the flipped-spin correlation (an exact sign identity), and
(d) the composition of the pool, which involves how the walk selects flips.</p>

<h3>I.a Stationarity: only the mutual distance matters</h3>
<p>For any $\varepsilon\in\{\pm1\}^N$, replacing
$J_{i_1\dots i_p}\to J_{i_1\dots i_p}\,\varepsilon_{i_1}\cdots\varepsilon_{i_p}$ leaves the
coupling distribution invariant (each coupling is a symmetric Gaussian) and maps
$E(\sigma)\to E(\varepsilon\circ\sigma)$. Choosing $\varepsilon=\sigma$ translates $\sigma$ to
the all-$+$ configuration. Hence the joint disorder statistics of the two spectra depend only
on the <em>relative</em> configuration $\sigma\circ\sigma'$ &mdash; by permutation symmetry,
only on the number $u$ of disagreeing sites. So a single function of one integer,
$C(u)$, is the complete pairwise input. (This is also why no coordinate tied to $\sigma_f$ can
enter $C$: the field has no special point; $\sigma_f$ will enter only through the triangle
geometry.)</p>

<h3>I.b Unflipped spins: the Krawtchouk correlation</h3>
<p>Write $\Delta_i = -2\sigma_i h_i$ with the local field
$h_i = \sum_{S} J_{i,S}\prod_{j\in S}\sigma_j$, the sum running over the
$\binom{N-1}{p-1}$ subsets $S$ of the other $N-1$ sites of size $p-1$. Take a spin $i$ with
$\sigma_i'=\sigma_i$ (not flipped between the two configurations). Because distinct couplings
are independent with variance $\operatorname{Var}(J)$,
$$\mathbb E\big[h_i h_i'\big]
= \operatorname{Var}(J)\sum_{|S|=p-1}\ \prod_{j\in S}\sigma_j\sigma_j' .$$
Each product is $+1$ if $S$ contains an even number of the $u$ disagreeing sites and $-1$ if
odd. Counting subsets by how many disagreeing sites $j$ they contain:
$$\sum_{|S|=p-1}\prod_{j\in S}\sigma_j\sigma_j'
= \sum_{j=0}^{p-1}(-1)^j\binom{u}{j}\binom{N-1-u}{p-1-j}
\;=\; K_{p-1}(u;\,N-1),$$
a Krawtchouk polynomial. Since $\operatorname{Var}(h_i) = \binom{N-1}{p-1}\operatorname{Var}(J)$,
$$\rho_{\rm unflipped}(u) = \frac{K_{p-1}(u;N-1)}{\binom{N-1}{p-1}}
\;\xrightarrow{\;N\gg1\;}\; q^{\,p-1},\qquad q = 1-\tfrac{2u}{N}.$$
Worked case $p=2$ (one disagreeing site kills one term each):
$K_1(u;N-1)/(N-1) = \frac{(N-1-u)-u}{N-1} = 1-\frac{2u}{N-1}$ &mdash; exactly linear in $u$.
Sample values at $N=1000$, $u = 0,10,50,100,200$:</p>
<table><tr><th>$u$</th><th>0</th><th>10</th><th>50</th><th>100</th><th>200</th></tr>
<tr><td>$\rho_{\rm unflipped}$ ($p{=}2$)</td>__KROW2__</tr></table>
<p>For $p=3$: $K_2(u;M)/\binom{M}{2} = 1 - \frac{4u(M-u)}{M(M-1)}$, $M=N-1$. This branch is
verified directly in the data (Fig. D3, blue).</p>

<h3>I.c Flipped spins: exact sign reversal</h3>
<p>Flipping spin $i$ maps $\Delta_i\to-\Delta_i$ <em>exactly</em>: flipping back must restore
the energy, so $\Delta_i(\sigma^{(i)}) = E(\sigma)-E(\sigma^{(i)}) = -\Delta_i(\sigma)$. More
generally, if $i$ is flipped an odd number of times between the two configurations,
$\Delta_i' = -2\sigma_i'h_i' = +2\sigma_i h_i'$ (the field $h_i$ does not contain
$\sigma_i$), so its correlation is the <em>negative</em> Krawtchouk branch:
$$\rho_{\rm flipped}(u) = -\,\frac{K_{p-1}(u-1;N-1)}{\binom{N-1}{p-1}} \simeq -\,q^{\,p-1}.$$
Also verified directly (Fig. D3, orange).</p>

<h3>I.d Composition of the pool: who got flipped, and how they were chosen</h3>
<p>Between two walk configurations at distance $u$, exactly $u$ spins are flipped an odd number
of times; at leading order (fresh sites, verified $\dot u = 1$ early) these are $u$ spins
flipped once, a fraction $f = u/N$ of the pool. The pool covariance is the mixture of the two
branches &mdash; but the flipped spins are not a random sample: the walk chose them. Two exact
Gaussian computations quantify this.</p>
<p><b>Size bias.</b> SSWM accepts move $i$ with probability $\propto\Delta_i^+$. For a large
pool with $\Delta\sim\mathcal N(0,s^2)$, the accepted value is distributed as
$x\,p(x)\mathbf 1_{x>0}/\mathbb E[X^+]$, so its second moment is
$$\mathbb E[\Delta^2\,|\,{\rm sel}]
= \frac{\int_0^\infty x^3\,\varphi_s(x)\,dx}{\int_0^\infty x\,\varphi_s(x)\,dx}
= \frac{2s^3/\sqrt{2\pi}}{s/\sqrt{2\pi}} = 2s^2
\qquad\Rightarrow\qquad \beta \equiv \frac{\mathbb E[\Delta^2|{\rm sel}]}{s^2} = 2,$$
using $\int_0^\infty x\varphi = s/\sqrt{2\pi}$ and $\int_0^\infty x^3\varphi = 2s^3/\sqrt{2\pi}$
(integrate by parts once).</p>
<p><b>Cross moment of a flipped spin.</b> If $i$ flips at time $s$ between $t_1$ and $t_2$:
its value at $s$ is size-biased, the flip reverses the sign, and the kernel factors compose
multiplicatively along fresh sites ($q_{13}=q_{12}q_{23}+O(N^{-2})$):
$$\mathbb E\big[\Delta_i(t_1)\Delta_i(t_2)\big]
= -\,\beta\,s^2\,\rho_K\big(u(t_1,s)\big)\rho_K\big(u(s,t_2)\big)
\;\approx\; -\,\beta\,s^2\rho_K(u_{12}).$$
<b>Depletion of the unflipped.</b> The reference second moment is conserved exactly:
$(1-f)\,\mathbb E[\Delta^2|{\rm unflipped}] + f\,\beta s^2 = s^2$, so the unflipped sub-pool
carries $\mathbb E[\Delta^2|{\rm unflip}] = s^2(1-\beta f)/(1-f)$.</p>
<p><b>Assembly.</b> Adding the two sub-pools,
$$\frac{C(u)}{s^2} = (1-f)\,\rho_K(u)\frac{1-\beta f}{1-f} \;+\; f\big(-\beta\,\rho_K(u)\big)
= \rho_K(u)\,(1-2\beta f)
\;=\; \underbrace{\frac{K_{p-1}(u;N-1)}{\binom{N-1}{p-1}}\Big(1-\frac{4u}{N}\Big)_{\!+}}_{\textstyle \equiv\ \rho_p(u)}$$
for SSWM ($\beta=2$). This is the input law. It was verified as a whole at the start of the
walk (Fig. D3, cyan curve on its dashed prediction), and its pieces were verified separately:
the Krawtchouk branches (Fig. D3), the value $\beta=2$ (controls: uniform acceptance and
neutral dynamics shift the pool constant exactly as $\beta\to1$ predicts), and $\dot u = 1$.</p>

<div class="box">
<p><b>$\rho_p$ in words.</b> $\rho_p(u)$ is the fraction of the spectrum's pool covariance
that survives $u$ accepted flips: take the whole spectrum at one configuration and at another
configuration $u$ flips away (any two &mdash; past, future, start, end; only their mutual
distance matters), and $\rho_p(u)$ is their correlation across the pool of moves. Its first
factor is the field decorrelation of the not-flipped spins (Krawtchouk $\to q^{p-1}$); its
second factor, $(1-4u/N)$, is the bookkeeping of the $u$ flipped spins: each carries its
exactly sign-reversed $\Delta$, and SSWM preferentially flipped large ones ($\beta=2$). It
contains no reference to $\sigma_f$ or to the future &mdash; those enter later, purely through
which distances the triangle hands to it.</p>
</div>

__D3__

<div class="step">
<p><b>The step people miss: $C(u)$ applies to the pair $(t, T)$ as well.</b> The terminal
configuration is just another configuration of the walk, at mutual distance $u(t,T)=d_H(t)$
from $\sigma(t)$. The spins disagreeing between $\sigma(t)$ and $\sigma_f$ &mdash; the set
$A(t)$, $|A(t)|=d$ &mdash; are precisely the spins that get flipped an odd number of times in
$(t,T]$, and those flips are SSWM-accepted flips like any others, so the same mixture
bookkeeping with $f=d/N$ applies:
$$\operatorname{Cov}\big(\Delta(t),\Delta(T)\big) = s^2\rho_p(d_H(t)).$$
No new physics is introduced at this step; the same three verified ingredients are evaluated
at a different distance. (Approximations inherited: "flipped odd $=$ flipped once", and the
annealed treatment of the walk's adaptivity; both are quantified in Sec. IV.)</p>
</div>

<h2>II. The algebra: from $C(u)$ to the boxed law</h2>

<p>Now the two lines. Covariance is bilinear, so for the differences
$a_i(t)=\Delta_i(t)-\Delta_i(T)$:</p>
<div class="step">
<p><b>Line 1 (expand).</b>
$$\operatorname{Cov}\big(a(t_1),a(t_2)\big)
= \underbrace{\operatorname{Cov}\big(\Delta(t_1),\Delta(t_2)\big)}_{C(u_{12})}
- \underbrace{\operatorname{Cov}\big(\Delta(t_1),\Delta(T)\big)}_{C(d_1)}
- \underbrace{\operatorname{Cov}\big(\Delta(T),\Delta(t_2)\big)}_{C(d_2)}
+ \underbrace{\operatorname{Var}\big(\Delta(T)\big)}_{C(0)=s^2}$$
Each of the four terms is the pairwise input evaluated on one side of the triangle of Fig. D1
(the last one on the degenerate side of length 0). Dividing by $s^2$:
$$\operatorname{Cov}\big(a(t_1),a(t_2)\big) = s^2\Big[\rho_p(u_{12}) + 1 - \rho_p(d_1)
- \rho_p(d_2)\Big].$$</p>
<p><b>Line 2 (normalize).</b> Setting $t_2=t_1$ in Line 1 (so $u_{12}=0$, $d_2=d_1$):
$$\operatorname{Var}\big(a(t)\big) = s^2\big[1 + 1 - 2\rho_p(d)\big] = 2s^2\big(1-\rho_p(d)\big).$$
Therefore
$$\rho_{\rm trans}(t_1,t_2)
= \frac{s^2\big[\rho_p(u_{12}) + 1 - \rho_p(d_1) - \rho_p(d_2)\big]}
{\sqrt{2s^2\big(1-\rho_p(d_1)\big)\cdot 2s^2\big(1-\rho_p(d_2)\big)}}
= \boxed{\;\frac{\rho_p(u_{12}) + 1 - \rho_p(d_1) - \rho_p(d_2)}
{2\sqrt{\big(1-\rho_p(d_1)\big)\big(1-\rho_p(d_2)\big)}}\;}$$
The $s^2$ cancels; nothing else was used. All three arguments &mdash; $u_{12}(\Delta t)$,
$d_1$, $d_2(\Delta t)$ &mdash; are read off the walk, so the law has no adjustable
content.</p>
</div>

<h2>III. Worked numeric example and the anatomy of the formula</h2>

<p>Take the $p=2$, $N=1000$ data, references at depth $d_1\approx__EXD1__$ (bin average of
__EXN__ references), at lag $\Delta t=__EXDT__$. The measured walk gives
$u_{12}=__EXU12__$ and $d_2=__EXD2__$. Then, evaluating
$\rho_p(u) = \big(1-\frac{2u}{N-1}\big)\big(1-\frac{4u}{N}\big)$:</p>
<table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>$\rho_p(u_{12})$</td><td>__EXRU__</td></tr>
<tr><td>$\rho_p(d_1)$</td><td>__EXR1__</td></tr>
<tr><td>$\rho_p(d_2)$</td><td>__EXR2__</td></tr>
<tr><td>numerator $= \rho_p(u_{12})+1-\rho_p(d_1)-\rho_p(d_2)$</td><td>__EXNUM__</td></tr>
<tr><td>denominator $= 2\sqrt{(1-\rho_p(d_1))(1-\rho_p(d_2))}$</td><td>__EXDEN__</td></tr>
<tr><td><b>law: ratio</b></td><td><b>__EXPRED__</b></td></tr>
<tr><td><b>measured $\rho_{\rm trans}$</b></td><td><b>__EXMEAS__</b></td></tr>
</table>

__D2__

<h2>IV. Limits, and what is approximate</h2>

<h3>IV.a Deep basin: the constants cancel and geometry emerges</h3>
<p>For $d_1,d_2,u_{12}\ll N$, linearize $\rho_p(u)\approx1-\Lambda u/N$ with
$\Lambda = 2(p-1)+4$ (kernel slope $+$ flip term). Then
$$\text{numerator} \approx \frac{\Lambda}{N}\big(d_1+d_2-u_{12}\big),\qquad
\text{denominator} \approx 2\sqrt{\frac{\Lambda d_1}{N}\cdot\frac{\Lambda d_2}{N}}
= \frac{2\Lambda}{N}\sqrt{d_1d_2},$$
and $\Lambda/N$ cancels:
$$\rho_{\rm trans} \to \frac{d_1+d_2-u_{12}}{2\sqrt{d_1d_2}} = \frac{|A_1\cap A_2|}{\sqrt{d_1d_2}},$$
pure counting (the disagreement-set overlap; the last equality is
$u_{12}=|A_1\triangle A_2| = d_1+d_2-2|A_1\cap A_2|$). With toward-flips removing sites from
$A$ at rate $(1+v)/2$ and away-flips adding fresh ones at rate $(1-v)/2$:
$|A_1\cap A_2|\approx d\,e^{-(1+v)\Delta t/2d}$ while $\sqrt{d_1d_2}\approx d\,e^{-v\Delta t/2d}$
&mdash; the drift cancels in the ratio and $\rho_{\rm trans}\approx e^{-\Delta t/2d}$: the
$\tau = R^2/2$ shell law. Note what cancelled: <em>every</em> per-flip constant
($p$, $\beta$, $\pi$, &hellip;). This is why the basin dynamics looks purely geometric and needed
no flip factor.</p>

<h3>IV.b Far field</h3>
<p>At the equator $\rho_p(d_1),\rho_p(d_2)$ are small and slowly varying; the lag dependence is
carried by $\rho_p(u_{12})$, and the transient decorrelates on the pool-kernel scale. (The
transient is not the EMD; from an equatorial reference the EMD itself is governed directly by
the pool law, as established in the main report.)</p>

<h3>IV.c The approximations, listed, with measured consequences</h3>
<ul>
<li><b>"Flipped odd $=$ flipped once"</b> and the multiplicative kernel chain: exact to
$O(u^2/N^2)$; degrade near the walk's end where back-flips occur ($\dot u = 0.92$&ndash;$1.0$
measured; the formula uses the <em>measured</em> $u_{12}$, absorbing most of this).</li>
<li><b>Common $s^2$ for all four terms:</b> the spectrum width drifts along the walk
($2.82\to2.08$ for $p{=}2$); the ratio structure cancels the common scale but not the drift
between $t_1$, $t_2$, $T$. Part of the residual at large depth.</li>
<li><b>Annealed treatment of adaptivity:</b> the walk's selection builds correlations with the
disorder beyond the size-bias bookkeeping (the "conditioning sag" of the main report,
$\lesssim0.1$ in the pairwise correlation at deep $u$).</li>
<li><b>Clipping $(1-4u/N)_+$:</b> the linear flip factor is the leading term; it crosses zero
at $u=N/4$, and pairs with larger separations (equatorial references at long lags) are where
the law is weakest.</li>
</ul>
<p>Net measured accuracy of the boxed law (pointwise, $\rho_{\rm trans}\in[0.25,0.97]$):
$\le0.013$&ndash;$0.04$ for $d/N\le0.34$ at both $p$; $0.09$ in the $p{=}2$ equator bin
(concentrated beyond the half-time); the smallest-$d$ $p{=}3$ bin is sampling-noise limited.
Fig. D4 shows the law against the data at five depths.</p>

__D4__

<h2>V. A predictive, position-only form: the timescale from where you are now</h2>

<p>As written, the boxed law takes the <em>measured</em> trajectory
($u_{12}(\Delta t)$, $d_2(\Delta t)$) as input &mdash; it is a consistency statement, not a
forecast. To forecast from the current position, substitute the typical trajectory, whose two
rates are dynamical constants of the walk, not future data:
$$u_{12}(\Delta t) = c\,\Delta t\quad (c = 0.95\pm0.03\ \text{measured},\ \approx1),\qquad
d_2(\Delta t) = d - v_{\rm rem}\,\Delta t .$$
Now compute the initial decay rate of the resulting curve. Write
$\rho_p'(0) = -\Lambda/N$ with $\Lambda = 2(p-1)+4$ (kernel slope plus flip slope). The
numerator and denominator of the law contribute:
$$-\frac{d}{d\Delta t}\ln(\text{num})\Big|_0
= \frac{c\,\Lambda/N \;-\; v_{\rm rem}\,\rho_p'(d)}{2\big(1-\rho_p(d)\big)},\qquad
-\frac{d}{d\Delta t}\ln(\text{den})\Big|_0
= \frac{-\,v_{\rm rem}\,\rho_p'(d)}{2\big(1-\rho_p(d)\big)},$$
and the two $v_{\rm rem}$ terms <b>cancel exactly</b> in the difference (the same cancellation
that produced the drift-free $R^2/2$ law in the basin, now shown to hold at every depth):</p>

<div class="step">
<p>$$\text{rate}(d) = \frac{c\,\Lambda}{2N\big(1-\rho_p(d)\big)}
\qquad\Longleftrightarrow\qquad
\tau_{\rm tangent}(d) = \frac{2N\big(1-\rho_p(d)\big)}{c\,\Lambda},\qquad \Lambda = 2(p-1)+4 .$$
<b>The timescale depends only on the current depth</b> $d$, the size $N$, and the interaction
order &mdash; not on the drift, and not on any future information beyond the dynamical constant
$c\approx1$. Reading: $2N(1-\rho_p(d))/N\Lambda$ is (remaining transient variance at depth
$d$) $\div$ (the universal per-move covariance loss $\Lambda/2N$). Limits: deep basin,
$1-\rho_p(d)\approx\Lambda d/N$, so $\tau\to 2d/c$ &mdash; the $R^2/2$ shell law; equator,
$\rho_p\to0$, so $\tau\to 2N/c\Lambda$.</p>
</div>

<p>Because the transient-correlation <em>curve</em> is convex, its half-time is shorter than
the tangent time; the half-time of the $d$-only curve is computed once and for all
(numerically, no data input). Validation against the measured half-times at every depth
(Fig. D5; the two red curves &mdash; measured $v_{\rm rem}(d)$ versus a constant global $v$
&mdash; nearly coincide, the numerical face of the cancellation):</p>
<table>
<tr><th>$p$</th><th>$d/N$</th><th>measured $t_{1/2}$</th><th>$d$-only law</th><th>ratio</th></tr>
<tr><td>2</td><td>0.062</td><td>48.6</td><td>48.7</td><td>1.00</td></tr>
<tr><td>2</td><td>0.120</td><td>95.2</td><td>92.6</td><td>1.03</td></tr>
<tr><td>2</td><td>0.195</td><td>161.5</td><td>148.2</td><td>1.09</td></tr>
<tr><td>2</td><td>0.310</td><td>217.8</td><td>212.0</td><td>1.03</td></tr>
<tr><td>2</td><td>0.391</td><td>207.7</td><td>247.6</td><td>0.84</td></tr>
<tr><td>3</td><td>0.062</td><td>13.5</td><td>14.2</td><td>0.95</td></tr>
<tr><td>3</td><td>0.120</td><td>27.7</td><td>26.7</td><td>1.04</td></tr>
<tr><td>3</td><td>0.195</td><td>41.3</td><td>42.0</td><td>0.98</td></tr>
<tr><td>3</td><td>0.300</td><td>55.9</td><td>58.7</td><td>0.95</td></tr>
</table>

__D5__

<p><b>Knowing $d$ without the future.</b> One honest objection remains: $d$ itself is the
distance to a maximum you have not reached. Operationally, the walk's position can be read off
the <em>current spectrum</em>: the number of beneficial moves $n_+$ is a monotone function of
depth (Fig. D5, right panel gives the measured calibration $n_+(d)$ for both models). Measure
$n_+$ now, read off $\hat d$, evaluate $\tau(\hat d)$ &mdash; a forecast that uses no future
information at all. (A first-principles theory of the $n_+\!\leftrightarrow\!d$ relation near
the maximum is part of the pseudogap physics and is not attempted here; the calibration is
empirical but static &mdash; one curve per model, measured once.)</p>

<p class="dim">Scope reminder: this section predicts the <em>transient-correlation</em>
timescale. The subset-EMD timescale coincides with it deep in the basin (up to the
$(1+v_{\rm rem})$ amplitude factor) and crosses over to the pool-law timescale
$N/(2p-2+\pi)$ in the far field; the full position-only EMD curve inherits the open
selection-split problem of the main report, Sec. 8.2.</p>

<div class="warn">
<p><b>Why not the "exact" eigen-identity instead?</b> For $p=2$ one is tempted by
$E=\frac12 r^\top\Lambda r$, $\Delta_i = r^\top\Lambda\delta^i + \frac12\delta^{i\top}\Lambda\delta^i$,
which gives $a_i(t) = (r(t)-r_f)^\top\Lambda\delta^i$ with <em>fixed</em> move vectors
$\delta^i$ &mdash; apparently exact, no approximations. But $\delta^i = -2v_i\sigma_i$ carries
the <em>current</em> spin: for the $d/N$ spins disagreeing with $\sigma_f$ the vector has the
opposite sign, and the identity with fixed $\delta^i$ silently drops that. Tested directly
(diagonalizing $J$ run by run): deviations up to $0.29$ at the equator &mdash; worse than the
kernel-difference law, whose $\rho_p$ carries the parity bookkeeping. The lesson: the
flip-parity accounting is not a refinement; it is load-bearing.</p>
</div>

<p class="dim">Reproduce: <code>analysis/transient_law_test.py</code> (law variants and the
eigen-identity test), <code>analysis/alltimes_figs.py</code> (Fig. D4),
<code>analysis/build_derivation.py</code> (this document and Figs. D1&ndash;D2).</p>

</body>
</html>
"""
    for k, v in [("__EXD1__", f"{ex['d1']:.0f}"), ("__EXN__", "20"),
                 ("__EXDT__", str(ex["dt"])), ("__EXU12__", f"{ex['u12']:.1f}"),
                 ("__EXD2__", f"{ex['d2']:.1f}"), ("__EXRU__", f"{ex['ru']:.4f}"),
                 ("__EXR1__", f"{ex['r1']:.4f}"), ("__EXR2__", f"{ex['r2']:.4f}"),
                 ("__EXNUM__", f"{ex['num']:.4f}"), ("__EXDEN__", f"{ex['den']:.4f}"),
                 ("__EXPRED__", f"{ex['pred']:.3f}"), ("__EXMEAS__", f"{ex['meas']:.3f}"),
                 ("__KROW2__", krow2)]:
        HTML = HTML.replace(k, v)

    FIGS = {
        "__D1__": ("d1_triangle.png",
                   "Fig. D1 &mdash; The three configurations and their mutual distances. The "
                   "pairwise input C(u) is evaluated on each side of this triangle; the "
                   "distances to the terminal maximum are, by definition, the radial "
                   "coordinates d<sub>1</sub>, d<sub>2</sub>."),
        "__D3__": ("a1_ingredients.png",
                   "Fig. D3 &mdash; The input, verified piece by piece (reference t=0): "
                   "unflipped spins on the Krawtchouk branch +q<sup>p-1</sup> (blue), flipped "
                   "spins on the sign-reversed branch (orange), and the assembled pool law "
                   "&rho;<sub>p</sub>(u) = q<sup>p-1</sup>(1-4u/N) (cyan data on dashed "
                   "prediction)."),
        "__D2__": ("d2_anatomy.png",
                   "Fig. D2 &mdash; Anatomy of the formula on the worked bin (p=2, N=1000, "
                   "d<sub>1</sub>&asymp;120). Left: the three ingredients &mdash; the pairwise "
                   "correlation at the measured mutual distance u<sub>12</sub>(&Delta;t) "
                   "(blue), at the shrinking distance d<sub>2</sub>(&Delta;t) to the terminal "
                   "maximum (green), and the constant &rho;<sub>p</sub>(d<sub>1</sub>) (red "
                   "dashed). Right: numerator, denominator, their ratio (red), and the "
                   "measured transient correlation (violet)."),
        "__D5__": ("d5_position_only.png",
                   "Fig. D5 &mdash; The position-only timescale. Left/middle: the half-time of "
                   "the d-only law (red; solid with the measured v<sub>rem</sub>(d), dotted "
                   "with a constant global v &mdash; they nearly coincide: the drift cancels) "
                   "and its tangent form 2N(1-&rho;<sub>p</sub>(d))/c&Lambda; (grey dashed), "
                   "against the measured transient half-times (points). Right: the empirical "
                   "calibration n<sub>+</sub>(d), letting the current spectrum stand in for "
                   "the unknown depth."),
        "__D4__": ("u3_alltimes_geometry.png",
                   "Fig. D4 &mdash; The boxed law (dashed) against the measured transient "
                   "correlation (solid) at five depths, both p; lag rescaled by each curve's "
                   "half-time. Agreement &le; 0.04 pointwise for d/N &le; 0.34; the p=2 "
                   "equator bin deviates up to 0.09 beyond its half-time (Sec. IV.c)."),
    }
    for key, (fname, caption) in FIGS.items():
        HTML = HTML.replace(key, img(fname, caption))

    out = os.path.join(ROOT, "transient_law_derivation.html")
    with open(out, "w") as fh:
        fh.write(HTML)
    print(f"written: {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    fig_triangle()
    ex = fig_anatomy()
    print(json.dumps(ex, indent=1))
    build_html(ex)
