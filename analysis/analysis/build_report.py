"""Assemble the EMD-centric HTML report (MathJax + base64-embedded figures)."""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

with open(os.path.join(RES, "stats_emd.json")) as fh:
    SE = json.load(fh)
with open(os.path.join(RES, "kappa.json")) as fh:
    KJ = json.load(fh)
with open(os.path.join(RES, "intermediate.json")) as fh:
    IJ = json.load(fh)


def img(name, caption):
    with open(os.path.join(FIG, name), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{name}">'
            f'<figcaption>{caption}</figcaption></figure>')


def kappa_table():
    rows = []
    for r in sorted(SE["kappa_vs_N"], key=lambda r: (r["p"], r["N"])):
        rows.append(f"<tr><td>{r['p']}</td><td>{r['N']}</td>"
                    f"<td>{r['kappa']:.2f} &plusmn; {r['err']:.2f}</td></tr>")
    return "\n".join(rows)


def inter_table():
    rows = []
    for tag, p in [("N1000_P2", 2), ("N300_P3", 3)]:
        for r in IJ[tag]:
            rows.append(
                f"<tr><td>{p}</td><td>{r['dN']:.2f}</td><td>{r['n']}</td>"
                f"<td>{r['t_data']:.0f} &plusmn; {r['t_data_err']:.0f}</td>"
                f"<td>{r['t_K']:.0f}</td><td>{r['t_B']:.0f}</td>"
                f"<td>{r['t_harm']:.0f}</td><td>{r['t_prod']:.0f}</td>"
                f"<td><b>{r['t_data']/r['t_prod']:.2f}</b></td>"
                f"<td>{r['t_data']/r['t_harm']:.2f}</td></tr>")
    return "\n".join(rows)


def late_table():
    rows = []
    for p in (2, 3):
        fl = dict((round(d), f) for d, f in SE["floors"][f"p{p}"])
        for c in SE["late"][f"p{p}"]:
            if c["tau_emd"] != c["tau_emd"]:
                continue
            d = c["d"]
            floor = fl.get(round(d))
            floor_s = f"{floor:.2f}" if floor is not None else "&mdash;"
            rows.append(
                f"<tr><td>{p}</td><td>{d:.0f}</td><td>{c['v_rem']:.2f}</td>"
                f"<td>{c['tau_emd']:.0f}</td><td>{c['tau_ampl_angle']:.0f}</td>"
                f"<td>{2*d/(1+c['v_rem']):.0f}</td><td>{2*d:.0f}</td>"
                f"<td>{floor_s}</td></tr>")
    return "\n".join(rows)


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The EMD scrambling timescale in the p-spin model</title>
<script>
window.MathJax = {tex: {inlineMath: [['$','$'],['\\(','\\)']],
                        displayMath: [['$$','$$'],['\\[','\\]']]}};
</script>
<script id="MathJax-script" async
 src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
 body {font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; line-height:1.65;
       max-width:960px; margin:0 auto; padding:40px 20px; color:#222; background:#fbfbfb;}
 h1 {font-size:1.7em; border-bottom:3px solid #6a3fb5; padding-bottom:.4em;}
 h2 {font-size:1.3em; margin-top:2.2em; border-bottom:1px solid #bbb; padding-bottom:.2em;}
 h3 {font-size:1.05em; margin-top:1.5em;}
 p {text-align:justify; margin-bottom:1.05em;}
 figure {margin:1.6em 0; text-align:center;}
 figure img {max-width:100%; border:1px solid #ddd; border-radius:4px; background:#fff;}
 figcaption {font-size:.88em; color:#444; text-align:justify; margin-top:.5em; padding:0 8px;}
 .box {margin:1.4em 0; padding:.9em 1.2em; background:#f3eefc; border-left:4px solid #6a3fb5;}
 .result {background:#eefaf0; border-left:4px solid #2e9e4f;}
 .note {background:#fff8e6; border-left:4px solid #d99a00;}
 table {border-collapse:collapse; margin:1.2em auto; font-size:.93em;}
 th,td {border:1px solid #ccc; padding:.4em .8em; text-align:center;}
 th {background:#eef2f7;}
 td.l {text-align:left;}
 .dim {color:#777; font-size:.9em;}
 code {background:#f0f0f0; padding:.05em .3em; border-radius:3px; font-size:.92em;}
 .toc {background:#f4f4f4; padding:1em 1.5em; border-radius:6px; font-size:.95em;}
 .toc a {text-decoration:none; color:#4a2f85;}
 li {margin-bottom:.4em;}
</style>
</head>
<body>

<h1>The EMD scrambling timescale in the $p$-spin model: what sets it in each regime</h1>

<p class="dim">Analysis of the stored SSWM greedy-ascent walks on pure $p$-spin landscapes
($p=2$: $N=100$&ndash;$2000$; $p=3$: $N=100$&ndash;$500$; 10 disorder realizations each), with
control dynamics and a synthetic surrogate process. Code and reproduction instructions:
<code>analysis/</code>. Supporting correlation-level measurements are in Appendix B; the main
text uses only the EMD.</p>

<div class="toc">
<b>Contents</b><br>
<a href="#read">How to read this report</a><br>
<a href="#obs">1. The observable</a><br>
<a href="#walk">2. The walk in numbers: two speeds, one parabola</a><br>
<a href="#early">3. Result: the early-time EMD law, and every curve in Fig. 3 explained</a><br>
<a href="#derive">4. Where the law comes from, in three steps</a><br>
<a href="#checks">5. Checks: system sizes, and the correct coordinate</a><br>
<a href="#notv">6. Why the drift $v$ does not set the timescale</a><br>
<a href="#late">7. Late times: the basin regime</a><br>
<a href="#regimes">8. The whole walk: EMD timescale regime by regime</a><br>
<a href="#alltimes">&emsp;8.2 An all-times theory? What is exact, and what is ruled out</a><br>
<a href="#sym">9. Two symmetries of the EMD</a><br>
<a href="#appA">Appendix A. Derivations</a><br>
<a href="#appB">Appendix B. Supporting correlation measurements</a><br>
<a href="#appC">Appendix C. Methods</a><br>
<a href="#appD">Appendix D. What does not fit perfectly</a>
</div>

<h2 id="read">How to read this report</h2>

<p>The question, from the context document: what governs early-time scrambling of the
flip spectrum along a greedy $p$-spin ascent, and why does the Model-X expectation
&mdash; scrambling at the radial drift rate $v\approx0.62$ &mdash; fail? Here we answer it for
the subset-to-full earth-mover distance (EMD), the primary scrambling measure. The answer has
three parts, one per regime of the walk:</p>

<div class="box result">
<p style="margin-bottom:.4em"><b>The EMD timescale, in one box.</b> Time $t$ counts accepted
flips. $u$ is the number of spins on which the two compared configurations differ
(the mutual Hamming distance), and $q=1-2u/N$ their mutual overlap.</p>
<ul>
<li><b>Kernel regime</b> (roughly the first 60% of the walk):
$\ \widetilde W(t) = q_{0t}^{\,p-1}\big(1-\pi u/N\big)$ with $u\simeq t$, so
$$\tau_{\rm EMD} \;=\; \frac{N}{2(p-1)+\pi}\qquad
(\approx 194 \text{ for } p{=}2,\,N{=}1000;\ \approx 42 \text{ for } p{=}3,\,N{=}300).$$
The timescale depends only on $N$, the interaction order $p$, and the acceptance rule
(the $\pi$). It does not depend on the drift $v$, the distance to the terminal maximum, or
the shell radius $R$.</li>
<li><b>Basin regime</b> (last ~40%, once $d_H(t_{\rm ref},\sigma_f) \lesssim d^* \approx 0.15N$):
$$\tau_{\rm EMD} \;\approx\; \frac{2d}{1+v_{\rm rem}} \;=\; \frac{R^2}{2(1+v_{\rm rem})}
\;\approx\; d,$$
where $d = d_H(t_{\rm ref},\sigma_f)$ and $v_{\rm rem}\approx0.8$&ndash;$1$ is the remaining
drift. This is the only place the drift enters, and it enters through the geometry of the
shrinking basin, not through any radial decorrelation of the landscape.</li>
<li><b>Floor</b>: for references inside the basin the EMD does not decay to zero but to a
measured floor of $0.28$&ndash;$0.56$; the frozen terminal spectrum permanently remembers the
subset.</li>
<li><b>Intermediate regime</b> ($d^*/2\lesssim d\lesssim 2d^*$): both mechanisms act at once
&mdash; the measured half-time is about half of either alone &mdash; and adding their rates
summarizes it empirically to $10$&ndash;$25\%$ (Sec. 8.1). At a deeper level (Sec. 8.2): the
<em>transient</em> correlation obeys one derived law at every depth (the kernel-difference
formula, built from the verified pool kernel; accurate to $\le0.04$ for $d/N\le0.34$), with no
interpolation; the two natural candidates for a closed all-times EMD formula
(endpoint-conditioned Gaussian; the mean-field master equation) are ruled out by direct test,
which pins down exactly what a full theory must contain.</li>
</ul>
</div>

<p>Section 3 presents the early-time result and explains, curve by curve, the figure that
contains it. Section 4 derives the law in three steps. Sections 5&ndash;6 give the checks and
the reason $v$ is irrelevant. Sections 7&ndash;8 cover the basin regime and assemble the whole
picture. Derivations are spelled out in Appendix A; the correlation-level measurements that
back the derivation are in Appendix B.</p>

<div class="box">
<p style="margin-bottom:.3em"><b>Notation</b> (used throughout):</p>
<ul style="margin-bottom:0">
<li>$\Delta_i(t)$: energy change of flipping spin $i$ at time $t$; the "spectrum" is
$\{\Delta_i(t)\}_{i=1}^N$. Time $=$ accepted flips.</li>
<li>$M = \{i:\Delta_i(t_{\rm ref})>0\}$: the raisers at the reference time.
$\widetilde W(t)$: the $W_1$ (earth-mover) distance between $\{\Delta_i(t)\}_{i\in M}$ and the
full spectrum, divided by its value at $t_{\rm ref}$.</li>
<li>$u(t_1,t_2)$: mutual Hamming distance between the configurations at $t_1$ and $t_2$;
$q = 1-2u/N$ their mutual overlap.</li>
<li>$d_H(t) = d_H(\sigma(t),\sigma_f)$: Hamming distance to the terminal local maximum
("radial coordinate"); $R^2 = 4d_H(N-d_H)/N$ the shell radius; $v$: the drift rate,
$d_H(t)\approx d_H(0)-vt$.</li>
<li>$\xi(q)$: landscape kernel; for the pure model $\mathbb E[E(\sigma)E(\sigma')] \propto
q^p$, so $\xi'(q)/\xi'(1) = q^{p-1}$.</li>
</ul>
</div>

<h2 id="obs">1. The observable</h2>

<p>At a reference configuration, mark the moves that would raise the energy (about half of
them). Let the walk run. At each later time, re-evaluate those same marked moves and ask: does
their distribution of energy changes still look special, or has it relaxed into the full
spectrum? The normalized EMD $\widetilde W(t)$ is that question made quantitative: it starts at
$1$ (at the reference the marked set is exactly the positive part of the spectrum) and decays
toward $0$ as the marked set forgets it was special. Figure 1 shows the raw picture.</p>

__F1__

<p>Two exact facts about this observable are worth fixing at the start.</p>
<p><b>Raisers and lowerers give the same curve.</b> The full spectrum is a weighted mixture of
the marked set and its complement: $F_{\rm full}=wF_M+(1-w)F_{M^c}$, with $w=|M|/N$ fixed at the
reference. Therefore $W_1({\rm full},M)=(1-w)\int|F_M-F_{M^c}|$ and
$W_1({\rm full},M^c)=w\int|F_M-F_{M^c}|$: after normalization both equal
$$\widetilde W(t) \;=\; \frac{\int|F_M(t)-F_{M^c}(t)|}
{\int|F_M(t_{\rm ref})-F_{M^c}(t_{\rm ref})|}.$$
The raiser-EMD and lowerer-EMD are the <em>same number</em>, configuration by configuration
(verified to $10^{-7}$; Fig. 9). This explains the document's observation that either subset
works, and it gives the cleanest way to think about the observable: <b>$\widetilde W$ measures
how separated the two halves of the reference spectrum still are, relative to their initial
complete separation.</b></p>
<p><b>The subset membership is frozen; only the values move.</b> Everything below is about how
the values $\Delta_i(t)$ of a fixed set of spins wander.</p>

<h2 id="walk">2. The walk in numbers: two speeds, one parabola</h2>

__F2__

<p>Three facts about the walk frame the whole problem (Fig. 2).</p>
<p><b>The walk has two speeds.</b> The distance to the terminal maximum falls linearly,
$d_H(t)\approx d_H(0)-vt$, with $v=0.615\pm0.013$ for $p{=}2$ (the $v\approx0.62$ of the
context document) and $v=0.745\pm0.031$ for $p{=}3$. But the distance from any fixed reference
grows at speed $1$: $du/dt=0.995$&ndash;$1.000$ over the first third of the walk. Every accepted
flip moves the system one full Hamming unit; only the net fraction $v$ of that motion points
toward $\sigma_f$. Equivalently, a fraction $(1+v)/2\approx0.8$ of flips approach the terminal
maximum and $(1-v)/2\approx0.2$ retreat from it. Which of the two speeds sets the EMD clock is
exactly what the naive radial transfer gets wrong.</p>
<p><b>The radius is a parabola.</b> Because $\cos\theta = 1-2d_H/N$ is what drifts linearly, the
shell radius $R^2 = 4d_H(N-d_H)/N$ traces the parabola implied by linear $d_H(t)$ &mdash; flat
near the equator, square-root collapse near the maximum. Nothing dynamical stalls at the
equator; it is a change of coordinates.</p>
<p><b>Nothing rescales globally.</b> The spectrum width shrinks only from $2.8$ to $2.1$
($p{=}2$) over the entire walk, while the number of raisers goes to zero. There is no analog of
Model X's overall $R\,E(R)$ prefactor; scrambling here cannot be geometric rescaling.</p>

<h2 id="early">3. Result: the early-time EMD law, and every curve in Fig. 3 explained</h2>

<div class="box result">
<p><b>Early-time law.</b> From a reference far from the terminal maximum,
$$\widetilde W(t)\;=\;\underbrace{q_{0t}^{\,p-1}}_{\text{field decorrelation}}\;
\underbrace{\big(1-\pi\,u/N\big)}_{\text{move reversal}},\qquad q_{0t}=1-\tfrac{2u}{N},\quad
u\simeq t,$$
so the initial decay rate is $\big[2(p-1)+\pi\big]/N$ and
$$\boxed{\ \tau_{\rm EMD} = \frac{N}{2(p-1)+\pi}\ }\qquad
\tau=194\ (p{=}2,N{=}1000),\qquad \tau=42\ (p{=}3,N{=}300).$$
The two factors are the two things that happen to the marked values: (i) the landscape's
field decorrelates under displacement, one kernel factor per flip; (ii) each accepted move has
its own value exactly sign-reversed, $\Delta\to-\Delta$, and SSWM preferentially accepts large
values ($\pi = 2\times$ the mean selected gain over the mean positive gain; Sec. 4).</p>
</div>

__F3__

<p>Figure 3 is the central figure. Its six curves, in the order of the legend:</p>
<ol>
<li><b>EMD data.</b> $\widetilde W(t)$ from the reference $t_{\rm ref}=0$, averaged over the 10
runs (band: standard error). Nothing is fitted anywhere in this figure.</li>
<li><b>The early-time law</b> $q_{0t}^{p-1}(1-\pi u/N)$, evaluated with the <em>measured</em>
mean $u(t)$ (which is $\simeq t$ early on) and no free parameters. It lies on the data down to
$\widetilde W\approx0.3$. Below that it undershoots: $(1-\pi u/N)$ is the first term of an
expansion in $u/N$ and cannot hold once $\pi u/N$ is order one. The full theory at large $u$ is
curve (4).</li>
<li><b>The kernel factor alone</b>, $q_{0t}^{p-1}$. This is what the EMD would do if accepted
moves did not reverse their own $\Delta$ &mdash; pure field decorrelation. It is exact in that
hypothetical (Sec. 4, step 2). The gap between (3) and the data is therefore the entire
contribution of move reversal: about a factor $1.6$ in rate for $p{=}2$. For $p{=}3$ the kernel
factor is relatively more important ($2(p-1)=4$ vs $\pi$).</li>
<li><b>The landscape-free surrogate.</b> A simulation of the law's ingredients and nothing
else: $N$ i.i.d. Gaussian numbers; each step, pick one positive number with probability
proportional to its value (the SSWM rule), flip its sign, and multiply all other numbers by the
one-flip kernel factor $c=1-2(p-1)/(N-1)$, refreshing the lost variance with independent noise.
There is no landscape, no configuration, no $\sigma_f$, no drift, no geometry. Its EMD
(computed identically) lies on the p-spin data to $\approx0.01$ ($p{=}2$) and $\approx0.03$
($p{=}3$) everywhere, including the deep tail where the linearized law (2) fails.
<b>Early-time EMD scrambling contains nothing but the kernel and the flip bookkeeping.</b></li>
<li><b>Naive radial transfer, version 1:</b> same law, but with the radial clock &mdash;
displacement replaced by radial progress, $u\to vt$. This is the quantitative version of
"the scrambling slope is the drift rate $v$". It is slower than the data by exactly the factor
$1/v$ ($1.6\times$ for $p{=}2$, $1.35\times$ for $p{=}3$), because the walk decorrelates with
every flip, not only with the net-inward part of its motion.</li>
<li><b>Naive radial transfer, version 2:</b> $e^{-vt/d_0}$, the "time-to-maximum" clock
(Model&nbsp;X's radial scrambling time transplanted: correlations die when the walk has covered
its initial radius). Predicts $\tau=d_0/v = 659$ ($p{=}2$) and $126$ ($p{=}3$) &mdash; a factor
$3$&ndash;$3.5$ too slow, and with the wrong dependences: the measured $\tau$ does not grow
with $d_0$ at all (Sec. 8).</li>
</ol>

<h2 id="derive">4. Where the law comes from, in three steps</h2>

<p><b>Step 1 &mdash; the field decorrelates only through displacement.</b> The couplings'
distribution is invariant under the gauge transformation $J\to J\,\varepsilon_{i_1}\!\cdots
\varepsilon_{i_p}$, $\varepsilon\in\{\pm1\}^N$, which translates the landscape by
$\varepsilon$. So the joint statistics of two spectra depend only on the disagreement set of
the two configurations &mdash; by symmetry, only on $u$. There is no special point in the field:
$\sigma_f$ is made by the dynamics, not by the disorder. A direct average (Appendix A.1) gives,
for a spin not flipped between the two times,
$$\operatorname{Corr}\big(\Delta_i(t_1),\Delta_i(t_2)\big)
= \frac{K_{p-1}(u;N-1)}{\binom{N-1}{p-1}} \;\xrightarrow{N\gg1}\; q^{\,p-1},$$
a Krawtchouk polynomial of the mutual distance &mdash; and for a spin flipped an odd number of
times, the same with a minus sign ($\Delta_i\to-\Delta_i$ exactly under its own flip). Both
branches are verified in Appendix B (Fig. B1). One kernel factor per flip; radial and
tangential flips count identically.</p>

<p><b>Step 2 &mdash; with no move reversal, the EMD <em>is</em> the kernel correlation.</b>
Suppose for a moment no marked move were ever taken. Per spin, $(\Delta_i(0),\Delta_i(t))$ is
bivariate Gaussian with correlation $\rho=q^{p-1}$, so the raiser values at time $t$ are
distributed as $\rho\Delta_0^+ + \text{noise}$ and the lowerer values as its mirror image. The
right-shifted law stochastically dominates the left-shifted one, and the $W_1$ distance between
a distribution and its dominated mirror is exactly the difference of means, $2\rho\,m_+$
(with $m_+ = \mathbb E[\Delta|\Delta>0]$). Normalizing by the reference value $2m_+$:
$$\widetilde W(t) = \rho(t) = q_{0t}^{\,p-1}\qquad\text{exactly, for Gaussian spectra without
flips.}$$
This is why curve (3) of Fig. 3 is a meaningful baseline and not just a plotting choice.</p>

<p><b>Step 3 &mdash; move reversal, with SSWM's size bias, gives the $(1-\pi u/N)$ factor.</b>
Each accepted flip takes one member of the raiser cloud and reinserts it, sign-reversed, on the
lowerer side. Because SSWM accepts with probability $\propto\Delta$, the moved member is not
typical: its mean value is the size-biased mean
$\mathbb E[\Delta\cdot\Delta^+]/\mathbb E[\Delta^+]=\sqrt{\pi/2}\,s = (\pi/2)\,m_+$. To first
order in the flipped fraction, each flip therefore reduces the raiser&ndash;lowerer separation
by $2\times(\pi/2)m_+$ out of $2m_+$ (once for the removal from the right, once for the mirror
insertion on the left; Appendix A.2 does this properly at the level of CDFs). After $u$ flips:
$$\widetilde W = q^{\,p-1}\Big(1-\kappa\,\frac{u}{N}\Big),\qquad
\kappa \;=\; \frac{2\,\mathbb E[\Delta_{\rm selected}]}{\mathbb E[\Delta\,|\,\Delta>0]}
\;=\;\pi\quad\text{for SSWM}.$$
The same formula gives $\kappa=2$ for acceptance rules with no size bias (uniform among
raisers, or fully random flips). The constant is a property of the <em>acceptance rule</em>,
not of the landscape &mdash; a sharp, testable statement.</p>

<p>Putting $u\simeq t$ (unit displacement speed; Fig. 2) converts the law into a timescale in
accepted moves: rate $=[2(p-1)+\pi]/N$. Measured values of $\kappa$ (curve-level fits, no
window protocol): p-spin data $3.32\pm0.19$ ($p{=}2$, $N{=}1000$) and $2.88\pm0.34$
($p{=}3$, $N{=}300$); high-precision surrogate $3.12$ and $3.15$, against $\pi=3.14$. With the
uniform rule instead: data $1.90$/$2.06$, surrogate $1.94$/$1.94$, against $2$.</p>

<h2 id="checks">5. Checks: system sizes, and the correct coordinate</h2>

__F6__

<p>Fig. 4 collects the fitted $\kappa$ for every dataset: all sizes of both $p$ scatter around
$\pi$ with no trend in $N$ &mdash; the $1/N$ scaling of the rate and the value of the constant
are both as derived. (Fitting $\kappa$ at the curve level avoids the window-protocol
sensitivity that plagues log-slope fits of non-exponential laws; Appendix C.)</p>

__F4__

<p>Fig. 5 shows the same law without touching time at all. Top row: $\widetilde W$ from all
references in the first 60% of every walk, plotted against the <em>mutual</em> overlap
$q(t_{\rm ref},t)$, falls on one curve &mdash; the law evaluated at that overlap (red). Bottom
row: the same data plotted against the <em>radial</em> coordinate $q_f(t)$ (overlap with the
terminal maximum) is not a function at all: $\widetilde W=1$ occurs at every $q_f$, and
references at different points of the walk trace disjoint fans. A radial description of the
spectrum has nothing to attach to; the mutual overlap is the coordinate that works.</p>

<h2 id="notv">6. Why the drift $v$ does not set the timescale</h2>

<p>The stationarity of Step 1 already says why no radial quantity can appear in the early-time
law: the field statistics do not know where $\sigma_f$ is, and the walk decorrelates the
spectrum with every unit of displacement, of which the drift is only the net-inward component.
The direct test is to change the drift while keeping the landscapes, and watch the EMD
constant.</p>

__F5__

<p>Fig. 6: on the same disorder realizations we run (i) the original SSWM walks ($v=0.62$ /
$0.74$), (ii) greedy walks that accept uniformly among raisers ($v=0.47$ / $0.54$), and (iii)
neutral walks that flip uniformly random spins, ignoring the spectrum entirely ($v=0$; there
is not even a terminal maximum). The fitted $\kappa$ moves from $\pi$ to $2$ &mdash; exactly
tracking the acceptance rule's size bias, as predicted by the boxed formula of Step 3 &mdash;
and does not track $v$ at all. A walk with <em>zero</em> drift scrambles the subset at the
full kernel-plus-flips rate. The open squares are the surrogate run with each rule, which has
no drift by construction and reproduces both constants.</p>

<p>Three further observations close the case against $v$:</p>
<ul>
<li>$v$ is not universal: it varies with $p$ ($0.615\to0.745$) and with $N$ ($0.77$ at
$N{=}200$ down to $0.61$ at $N{=}2000$ for $p{=}2$), while the measured EMD constants stay
pinned at the kernel values (Fig. 4).</li>
<li>The $p$-dependence of the measured rate is the kernel's, $2(p-1)+\pi$: going
$p=2\to3$ adds $2$ to the rate constant. A drift-based rate $\propto v$ would change by only
20%.</li>
<li>Even the most honest transfer of Model X's closed-form correlation reduces, near the
equator, to the <em>kernel-linear</em> law $\widetilde W\approx q_{0t}$. That happens to be the
right field factor for $p{=}2$ (the SK model is the kernel-linear case) &mdash; which is why the
transfer looks nearly reasonable there &mdash; but it misses the factor $p-1$ for $p{=}3$ and
misses the move-reversal term for every $p$: in Model&nbsp;X a taken move's energy change does
not reverse; on the hypercube it reverses exactly.</li>
</ul>

<h2 id="late">7. Late times: the basin regime</h2>

<p>Near the terminal maximum the annealed picture must fail, precisely because the dynamics
conditioned $\sigma_f$ to be a maximum. Write the spectrum there as
$$\Delta_i(t) = \underbrace{\Delta_i^{f}}_{\text{frozen, }\le0}
+ \underbrace{a_i(t)}_{\text{transient}},$$
where $\Delta_i^f$ is the terminal spectrum and the transient $a_i$ vanishes as the walk ends.
The transient lives on the shell of radius $R(t)$ around $\sigma_f$: its leading term is linear
in the tangential direction $\hat u(t)$, with amplitude $\propto R(t)$. Two facts, both derived
in Appendix A.3 and both testable, control the late-time EMD:</p>
<ul>
<li><b>Angle:</b> the tangential directions at two times decorrelate through the renewal of the
disagreement set $A$ (the sites where $\sigma(t)$ still differs from $\sigma_f$):
$\hat u_1\!\cdot\!\hat u_2 = |A_1\cap A_2|/\sqrt{d_1d_2}$ exactly (up to $O(d/N)$). Toward-flips
remove sites from $A$, away-flips add fresh ones; in the correlation the drift <em>cancels</em>
between removal and dilution and the initial rate is $c/2d\simeq 2/R^2$ &mdash; the known
$\tau=R^2/2$ shell-mixing law, with $c=\dot u\approx0.95$ the local displacement speed. This
part is verified pointwise, parameter-free, in Appendix B (Fig. B3).</li>
<li><b>Amplitude:</b> the subset's specialness is carried by the transient, whose scale
$R(t)\propto\sqrt{d(t)}$ collapses deterministically as $d(t)=d-v_{\rm rem}\Delta t$. The EMD,
unlike an amplitude-normalized correlation, feels this factor $\sqrt{d_2/d_1}$ directly.</li>
</ul>
<p>Multiplying the two,
$$\widetilde W(\Delta t)\;\approx\;\sqrt{\tfrac{d_2}{d_1}}\;\big(\hat u_1\!\cdot\!\hat u_2\big)
\qquad\Longrightarrow\qquad
\tau_{\rm EMD}\;\approx\;\frac{2d}{c+v_{\rm rem}}\;\simeq\;\frac{R^2}{2(1+v_{\rm rem})}
\;\approx\; d .$$
<b>This is where the drift genuinely enters the EMD</b> &mdash; through the deterministic
shrinkage of the basin transient, i.e. through the radial <em>position</em>, not through any
radial decorrelation of the field. And this is also the correct discrete analog of Model-X
"radial scrambling": a deterministic amplitude collapse confined to the conditioned
neighborhood of the maximum, with the time-to-maximum $d/v \simeq R^2/4v$ <em>quadratic</em> in
$R$ (the parabola of Fig. 2), rather than a far-field decorrelation mechanism with rate $v$.</p>

__F7__

<p>Fig. 7, top: late-time EMD curves with the lag rescaled by $2d/(1+v_{\rm rem})$, against the
full amplitude$\times$angle prediction (dashed, computed from each run's measured $d_1$, $d_2$,
$u_{12}$ &mdash; no parameters). Bottom: the fitted timescales against $d_{\rm ref}$, bracketed
by $2d$ (pure shell mixing, too slow) and lying on $2d/(1+v_{\rm rem})$. Two caveats are visible
and expected. At the largest $d_{\rm ref}$ the measured decay is <em>faster</em> than the basin
prediction because the kernel mechanism still contributes there (next section: the two rates
add). And the curves level off at a nonzero <b>floor</b> (arrowheads): measured
$\widetilde W_\infty = 0.43$ and $0.28$ ($p{=}2$, $d_{\rm ref}\approx62,125$), $0.56$ and $0.28$
($p{=}3$, $d_{\rm ref}\approx20,37$). The floor is the frozen spectrum's permanent memory of the
subset: members of $M$ are biased toward small $|\Delta_i^f|$, and that bias never relaxes. So
"the subset becomes indistinguishable from the full spectrum" is exact in the kernel regime and
only approximate for references inside the basin. (At the smallest $d_{\rm ref}$ the subset has
too few members to measure an EMD at all &mdash; fewer than a dozen raisers survive.)</p>

<h2 id="regimes">8. The whole walk: EMD timescale regime by regime</h2>

__F8__

<p>Fig. 8 puts the local EMD timescale $\tau_{\rm EMD}(t_{\rm ref})$ &mdash; measured by the
identical fit at every reference along the walk &mdash; against the reference's radial position
$d_H(t_{\rm ref})$. The data ride the flat kernel value $N/(2p-2+\pi)$ while the reference is
far out, then bend onto the basin line $2d/(1+v_{\rm rem})$; the crossing of the two branches
(adding the rates, pink band, describes the data through the bend) defines
$$d^* \;=\; \frac{N\,(1+v_{\rm rem})}{2\,[2(p-1)+\pi]}\qquad
d^*/N \approx 0.18\ (p{=}2),\qquad 0.13\ (p{=}3),$$
which the walk reaches at $t^*/T\approx0.6$ for both $p$ &mdash; the same point where the
linear-drift window of $d_H(t)$ ends. In summary:</p>

<table>
<tr><th>regime</th><th>where</th><th>EMD law</th><th>timescale</th><th>what sets it</th></tr>
<tr>
<td class="l"><b>Kernel</b> (far field)</td>
<td class="l">$d_H(t_{\rm ref})\gtrsim d^*$; first $\sim$60% of the walk</td>
<td class="l">$q_{0t}^{\,p-1}\,(1-\pi u/N)$, $u\simeq \Delta t$</td>
<td class="l">$\tau=\dfrac{N}{2(p-1)+\pi}$<br>$=194$ ($p{=}2$, $N{=}10^3$); $42$ ($p{=}3$, $N{=}300$)</td>
<td class="l">interaction order (kernel curvature $2(p-1)$) $+$ move reversal with SSWM size
bias ($\pi$). Independent of position, drift, $d_0$, $R$.</td>
</tr>
<tr>
<td class="l"><b>Basin</b></td>
<td class="l">$d_H(t_{\rm ref})\lesssim d^*$; last $\sim$40%</td>
<td class="l">$\sqrt{d_2/d_1}\,\big(\hat u_1\!\cdot\!\hat u_2\big)$</td>
<td class="l">$\tau=\dfrac{2d}{1+v_{\rm rem}}=\dfrac{R^2}{2(1+v_{\rm rem})}\approx d$</td>
<td class="l">shell mixing (disagreement-set renewal, rate $c/2d$) $\times$ deterministic
amplitude collapse (rate $v_{\rm rem}/2d$) &mdash; the only entry point of the drift.</td>
</tr>
<tr>
<td class="l"><b>Floor</b></td>
<td class="l">references inside the basin, $\Delta t\to$ end</td>
<td class="l">$\widetilde W \to \widetilde W_\infty$</td>
<td class="l">$\widetilde W_\infty = 0.28$&ndash;$0.56$ measured</td>
<td class="l">frozen terminal spectrum: the subset's small-$|\Delta^f|$ bias never relaxes.</td>
</tr>
<tr>
<td class="l"><b>Crossover</b></td>
<td class="l">$d^* = \dfrac{N(1+v_{\rm rem})}{2[2(p-1)+\pi]}$</td>
<td class="l" colspan="2">rates add: $\tau^{-1}\approx\dfrac{2(p-1)+\pi}{N}
+\dfrac{1+v_{\rm rem}}{2d}$ &mdash; an interpolation, tested quantitatively in
Sec. 8.1: accurate to $10$&ndash;$25\%$ over $d\in[d^*/2,\,2d^*]$.</td>
<td class="l">$d^*/N\approx0.18$ ($p{=}2$), $0.13$ ($p{=}3$); reached at
$t^*/T\approx0.6$.</td>
</tr>
</table>

<p><b>Worked example</b> ($p=2$, $N=1000$, $d_0=405$, $T\approx600$). A subset marked at
$t_{\rm ref}=0$ relaxes with $\tau=194$ accepted moves; so does a subset marked at
$t_{\rm ref}=200$, even though $d_H$ has meanwhile dropped from 405 to about 280 &mdash; the
kernel clock does not care. A subset marked at $t_{\rm ref}\approx450$ ($d_H\approx125$) sits
near the crossover and relaxes with $\tau\approx60$&ndash;$90$. A subset marked at
$d_H\approx60$ relaxes with $\tau\approx50\approx d_H$, and retains a floor
$\widetilde W_\infty\approx0.4$ forever. Under the naive radial picture the first number would
have been $d_0/v\approx660$ and position-dependent throughout &mdash; more than three times too
slow at the start and wrong in shape thereafter.</p>

<h3 id="inter">8.1 Numerics through the crossover: how good is "rates added"?</h3>

<p>The added-rates line of the table is an interpolation, not a derived law. To test it
properly we re-measured the walks with references pinned <em>densely</em> in depth
($14$ target depths, $d/N=0.05$&ndash;$0.41$, all 10 runs, binned with $\ge4$ references per
bin) and switched to a protocol-free timescale: $t_{1/2}$, the first crossing of
$\widetilde W = 1/2$ of the bin-averaged curve (bootstrap errors over runs). Against it we put
four parameter-free predictions, built per bin from measured quantities only:</p>
<ul>
<li>$t_K$ &mdash; <b>kernel only</b>: half-time of the closed-form law
$q_u^{p-1}(1-\pi u/N)$, converted to moves with the bin's measured mutual speed
$c=0.90$&ndash;$0.97$;</li>
<li>$t_B$ &mdash; <b>basin only</b>: half-time of the amplitude$\times$angle curve
$\sqrt{d_2/d_1}\,\hat u_1\!\cdot\!\hat u_2$ built from each run's measured
$(d_1,d_2(\Delta t),u_{12}(\Delta t))$;</li>
<li><b>rates added</b>, in two equivalent flavors: the harmonic combination
$(1/t_K+1/t_B)^{-1}$ and the curve-level product $K\times B$ (its half-time
$t_{\rm prod}$);</li>
<li>$\min(t_K,t_B)$ &mdash; a <b>sharp crossover</b> (each regime keeps its own law, no
cooperation).</li>
</ul>

__I1__

<p>Three quantitative conclusions (Fig. 8b, Table 3):</p>
<ol>
<li><b>Both mechanisms genuinely act at once in the intermediate regime.</b> At $d\approx d^*$
the measured half-time is about <em>half</em> of either single mechanism's: $p{=}2$,
$d/N=0.13$: data $40\pm4$ vs kernel-only $123$, basin-only $79$; $p{=}3$, $d/N=0.15$: data
$15.9\pm0.4$ vs $27$ and $26$. Consequently the sharp-crossover description fails by up to a
factor $2.0$ ($p{=}2$) / $1.65$ ($p{=}3$) exactly where the crossover matters. The subset
forgets roughly twice as fast as either mechanism alone.</li>
<li><b>Adding the rates describes the crossover window to $10$&ndash;$25\%$ with no free
parameters.</b> Over $d\in[\sim d^*/2,\ \sim 1.7d^*]$ (i.e. $d/N=0.09$&ndash;$0.29$ for $p{=}2$,
$0.11$&ndash;$0.21$ for $p{=}3$) the product form gives
$t_{1/2}^{\rm data}/t_{1/2}^{\rm prod}=0.76$&ndash;$1.11$ ($p{=}2$) and $0.97$&ndash;$1.14$
($p{=}3$); the harmonic form is comparable, systematically $\sim8\%$ faster. So: yes &mdash;
in the intermediate, not-entirely-near-field regime the EMD timescale <em>is</em> the
crossover timescale of the table, at the $10$&ndash;$25\%$ level.</li>
<li><b>The residuals are systematic and instructive</b> (Fig. 8c). Just below $d^*$ the $p{=}2$
data is up to $\sim25\%$ <em>faster</em> than the interpolation ($d/N\approx0.13$) &mdash; the
conditioning sag of Appendix D, where the annealed kernel is slightly too optimistic mid-walk.
Toward the far field the interpolation becomes too <em>fast</em>
(data/product $\to1.27$&ndash;$1.29$ at the far anchors): at the equator the basin curve's
angle factor double-counts displacement decorrelation that the kernel already contains, so the
basin term should simply be dropped there &mdash; kernel-only reproduces the far anchors to
$5$&ndash;$6\%$ (data/kernel $=0.95$ and $0.94$).</li>
</ol>

__I2__

<p><b>Practical prescription.</b> Use kernel-only for $d\gtrsim2d^*$, rates-added (product
form) within $d^*/2\lesssim d\lesssim2d^*$, basin-only below $d^*/2$ (where it matches to a few
percent; Sec. 7). This composite reproduces every measured $t_{1/2}$ in both models to
$\lesssim15\%$.</p>

<p><b>Table 3</b> &mdash; EMD half-times through the crossover (accepted moves; predictions
parameter-free; ratios in bold are the rates-added product form):</p>
<table>
<tr><th>$p$</th><th>$d/N$</th><th>$n_{\rm refs}$</th><th>$t_{1/2}$ data</th>
<th>$t_K$</th><th>$t_B$</th><th>harm.</th><th>prod.</th>
<th>data/prod</th><th>data/harm</th></tr>
__INTERTABLE__
</table>

<h3 id="alltimes">8.2 An all-times theory? What is exact, and what is ruled out</h3>

<p>Adding rates is a summary, not a theory. This section reports what a first-principles,
interpolation-free description can and cannot be, based on direct tests.</p>

<p><b>What is derived and holds at all times: the kernel-difference law for the live part.</b>
Define the transient of every move as its deviation from the terminal spectrum,
$a_i(t) = \Delta_i(t) - \Delta_i(T)$. Its two-time correlation follows in three lines from one
verified input and one identity, with nothing assumed (a fully self-contained, step-by-step
version of this derivation &mdash; every integral, the counting argument, a worked numeric
example, and the anatomy of the formula on real data &mdash; is in the companion document
<code>transient_law_derivation.html</code>):</p>
<ol>
<li><em>Input</em> (Secs. 3&ndash;4, Fig. B1): the pool-level two-point function of the
spectrum between any two configurations at mutual distance $u$ is
$C(u) = s^2\rho_p(u)$ with
$$\rho_p(u) = \frac{K_{p-1}(u;N-1)}{\binom{N-1}{p-1}}\Big(1-\frac{4u}{N}\Big)_{\!+},$$
the Krawtchouk kernel for unflipped spins combined with the exact $-q^{p-1}$ branch of flipped
spins and the SSWM size bias ($2\beta = 4$). Every piece of this was derived and verified
independently, at the start of the walk.</li>
<li><em>Algebra</em>: covariances of differences expand,
$$\operatorname{Cov}\big(a(t_1),a(t_2)\big) = C(u_{12}) - C\big(u(t_1,T)\big)
- C\big(u(t_2,T)\big) + C(0),$$
and the mutual distance from any configuration to the end <em>is</em> the radial coordinate:
$u(t,T) = d_H(t) \equiv d$. This is where the walk's geometry enters &mdash; through an
identity, not a model.</li>
<li><em>Normalize</em>:
$$\boxed{\;\rho_{\rm trans}(t_1,t_2) \;=\;
\frac{\rho_p(u_{12}) + 1 - \rho_p(d_1) - \rho_p(d_2)}
{2\sqrt{\big(1-\rho_p(d_1)\big)\big(1-\rho_p(d_2)\big)}}\;}$$</li>
</ol>
<p>In the deep basin all the linear coefficients of $\rho_p$ cancel between numerator and
denominator, leaving the pure counting form $(d_1+d_2-u_{12})/2\sqrt{d_1d_2} =
|A_1\cap A_2|/\sqrt{d_1d_2}$ &mdash; the disagreement-set law with its drift cancellation and
$\tau = R^2/2$; in the far field it decays on the pool-kernel scale. Measured accuracy
(pointwise maximum deviation over $\rho_{\rm trans}\in[0.25,0.97]$, dense references):</p>
<table>
<tr><th></th><th>$d/N\approx0.06$</th><th>$0.12$</th><th>$0.20$</th><th>$0.31$</th><th>$0.39$</th></tr>
<tr><td>$p=2$</td><td>$0.013$</td><td>$0.039$</td><td>$0.011$</td><td>$0.026$</td><td>$0.093$</td></tr>
<tr><td>$p=3$</td><td>$0.11^{\,*}$</td><td>$0.014$</td><td>$0.030$</td><td>$0.034$</td><td>&mdash;</td></tr>
</table>
<p class="dim">($^*$few references and a tiny transient variance in that bin; the deviation is
dominated by sampling noise. The $p{=}2$ equator bin's $0.09$ is real and concentrated beyond
the half-time; see Appendix D.)</p>
<p>Two remarks. First, the naive geometric route one might try instead &mdash; the $p{=}2$
eigen-identity $\Delta_i(t) = \Delta_i^f + w(t)^\top\Lambda\delta^i$ with $w = r - r_f$,
evaluated with fixed move vectors &mdash; is exact only for spins that agree with $\sigma_f$;
the move vectors of the $d/N$ disagreeing spins carry the opposite sign. Tested directly
(diagonalizing $J$ run by run), that version deviates by up to $0.29$ at the equator, i.e.
<em>worse</em> than the kernel-difference law: the flip-parity bookkeeping, which the pool
kernel $\rho_p$ carries and the naive identity drops, is essential. Second, this law governs
the live part, not the EMD itself: the EMD mixes the transient with the frozen component
through the raiser selection, which is the remaining open piece below.</p>

__U3__

<p><b>Ruled out (i): endpoint-conditioned Gaussian ("pinned kernel").</b> The canonical
interpolation-free construction: take the verified pairwise law
$\hat\rho(u) = \rho_K(u)\,e^{-\pi u/N}$ and condition the Gaussian pool on the terminal
spectrum; the separation then transfers by partial regression,
$$\widetilde W(\Delta t) = \frac{\hat\rho(u_{12})-\hat\rho(d_1)\hat\rho(d_2)}{1-\hat\rho(d_1)^2}
+ \Phi\,\frac{\hat\rho(d_2)-\hat\rho(u_{12})\hat\rho(d_1)}{1-\hat\rho(d_1)^2},$$
with $\Phi$ the measured floor. By construction this reduces to the far-field law at the
equator and to amplitude$\times$angle deep in the basin (all bookkeeping constants cancel in
the ratio &mdash; a satisfying explanation of why the basin needed no $\pi$-factor). But
through the crossover it fails badly: measured $t_{1/2}$ over predicted $= 0.43$&ndash;$0.60$
($p{=}2$) and $0.59$&ndash;$0.70$ ($p{=}3$) at $d/N\approx0.1$&ndash;$0.2$ (Figs. 8e&ndash;8f).
The true pinning to the terminal state is roughly twice as strong as Gaussian conditioning on
the endpoint values. The diagnosis is quantitative: outside the basin the frozen and transient
components are strongly <em>anti-correlated</em> (exact variance bookkeeping at the equator,
$p{=}2$: $V_f + V_w + 2\,\mathrm{Cov} \approx 1.1 + 3.2 - 2.3 = 2.0 = $ the spectrum variance),
and the pool is heterogeneous in flip parity; a homogeneous Gaussian reduction destroys
precisely the structure that drives the crossover.</p>

__U1__

__U2__

<p><b>Ruled out (ii): the mean-field master equation.</b> The derived $O(1/N)$ single-spin
process (the Eastham&ndash;Blythe&ndash;Bray&ndash;Moore description generalized to $p$): per
accepted move, $\lambda_i \to \lambda_i - \frac{2(p-1)}{N}(\lambda_i+\lambda_k) + \eta_i$ with
$\operatorname{Var}\eta = 4p(p-1)/N$, the flipped spin reversing sign, SSWM selection. This
contains the kernel contraction, the flip bookkeeping, and the convection that depletes the
raisers &mdash; and nothing geometric. It reproduces the far field ($t_{1/2}=120$ at
$t_{\rm ref}=0$ vs data $109$, $p{=}2$), but it <em>cannot</em> describe the basin, for a sharp
reason: its convection is a common shift, invisible to the shift-invariant EMD, and its kicks
are fresh; all its correlation rates are $O(1/N)$, never $O(1/d)$. Numerically it does not even
terminate: $0/40$ runs ($N{=}1000$, $p{=}2$) and $0/100$ runs ($N{=}300$, $p{=}3$) reach a
local maximum within $1.6N$ moves &mdash; the raiser count stalls at $n_+/N\approx0.07$
(median), well above the depths where basin physics lives. The existence of the terminal
maximum, the floors, and the $\tau\approx d$ basin scale are all configuration-geometric
effects, provably outside this model class.</p>

<div class="box">
<p><b>What an all-times theory must contain.</b> The structure is a two-component pool:
a static, <em>one-sided, pseudogapped</em> frozen part $\Delta_i^f$ and a transient whose
two-time correlation is the derived kernel-difference law above, with a depth-dependent
frozen&ndash;transient anti-correlation (at the equator, $p{=}2$:
$V_f + V_a + 2\,\mathrm{Cov} \approx 1.1 + 3.2 - 2.3 = 2.0 = $ the spectrum variance). The EMD
at all times is the raiser&ndash;lowerer separation transfer of this mixture; its crossover is
controlled by how the selection at the reference splits between the two components as their
weights, cross-correlation, and the selection threshold drift along the walk. Every ingredient
is now measured or derived except one: the joint selection-split functional, which is
non-Gaussian because the frozen part is one-sided. Closing it (a one-dimensional integral over
the terminal spectrum's distribution, with the kernel-difference covariances as input) is the
sharply posed open problem; simple homogeneous closures &mdash; Gaussian conditioning,
mean-field kicks, or rate addition &mdash; each fail in a now-documented way.</p>
</div>

<h2 id="sym">9. Two symmetries of the EMD</h2>

__F9__

<p>Reading the same subset backward along the walk gives the same decay as forward at short and
moderate lags (Fig. 9) &mdash; as it must, since the law depends only on the mutual displacement
$u$, which grows in both directions. The forward curve eventually saturates at the basin floor
while the backward curve keeps falling: the slow drift toward the frozen maximum is the only
arrow of time in the observable. And the raiser/lowerer identity of Section 1 holds to machine
precision (annotation).</p>

<h2 id="appA">Appendix A. Derivations</h2>

<h3>A.1 The kernel two-point function (Step 1)</h3>
<p>With $\Delta_i=-2\sigma_ih_i$ and $h_i=\sum_{|S|=p-1}J_{i,S}\prod_{j\in S}\sigma_j$ ($S$ over
subsets of the other $N-1$ sites; $\operatorname{Var}J = p!/N^{p-1}$ in the data's convention),
two configurations differing on $u$ sites, with spin $i$ unflipped, give
$$\mathbb E[h_ih_i'] = \operatorname{Var}(J)\sum_{|S|=p-1}\prod_{j\in S}\sigma_j\sigma_j'
= \operatorname{Var}(J)\,K_{p-1}(u;N-1),$$
the Krawtchouk polynomial $K_k(u;M)=\sum_j(-1)^j\binom{u}{j}\binom{M-u}{k-j}$ (the hypercube
analog of Gegenbauer polynomials). Normalizing, $\rho_{\rm unflipped}=K_{p-1}(u;N-1)/
\binom{N-1}{p-1}\to q^{p-1}$; exactly $1-2u/(N-1)$ for $p=2$. If spin $i$ is flipped an odd
number of times, $\Delta_i' = +2\sigma_ih_i'$ and the correlation is $-K_{p-1}(u-1;N-1)/
\binom{N-1}{p-1}\simeq -q^{p-1}$. For a mixed kernel $\xi(q)=\sum_p c_pq^p$ the same computation
weighted by sectors gives $\rho=\xi'(q)/\xi'(1)$ (verified on the mixed dataset, Fig. B4, with
its additive floor $\xi'(0)/\xi'(1)$). Along a fresh-site walk the kernel factors compose
multiplicatively to $O(u^2/N^2)$, which is what the surrogate implements.</p>

<h3>A.2 The EMD law (Steps 2&ndash;3)</h3>
<p><b>No flips.</b> Per spin $(\Delta_0,\Delta_t)$ is bivariate Gaussian with correlation
$\rho$. The conditional laws $A=\mathcal L(\Delta_t|\Delta_0>0)$ and
$\tilde A=\mathcal L(\Delta_t|\Delta_0<0)$ are mirror images with means $\pm\rho m_+$,
$m_+=\sqrt{2/\pi}\,s$, and $A$ stochastically dominates $\tilde A$; hence
$W_1(A,\tilde A)=\int(F_{\tilde A}-F_A)=2\rho m_+$ exactly, and $\widetilde W=\rho$.</p>
<p><b>Flips.</b> After $u$ flips (all drawn from raisers at early times; fraction $2f$ of $M$,
$f=u/N$), the raiser side is $F_M = A + 2f\,(G^- - S)$ to $O(f)$: a mass $2f$ of would-be
members with law $S$ (the size-biased selected values, evolved; mean $(\pi/2)m_+\rho$) is
replaced by its sign-reversed image $G^-$ (mean $-(\pi/2)m_+\rho$). The lowerer side is
unperturbed at this order. Since $G^-$ lies entirely to the left of $S$,
$F_{G^-}-F_S\ge0$ everywhere, and while the base gap keeps one sign,
$$\int|F_M-F_{M^c}| = 2\rho m_+ - 2f\big[\text{mean}(S)-\text{mean}(G^-)\big]
= 2\rho m_+\Big(1-2f\,\frac{\mathbb E[\Delta_{\rm sel}]}{m_+}\Big),$$
i.e. $\widetilde W=\rho\,(1-\kappa u/N)$ with $\kappa = 2\,\mathbb E[\Delta_{\rm sel}]/m_+$.
SSWM selects with probability $\propto\Delta^+$:
$\mathbb E[\Delta_{\rm sel}]=\mathbb E[X^2 1_{X>0}]/\mathbb E[X1_{X>0}]=\sqrt{\pi/2}\,s
=(\pi/2)m_+$, so $\kappa=\pi$. Uniform-among-raisers: $\mathbb E[\Delta_{\rm sel}]=m_+$,
$\kappa=2$; unconditioned flips likewise give $2$. Higher orders in $f$, flips originating from
$M^c$, and selection-depletion corrections are all contained in the surrogate, which fixes
$\kappa_{\rm eff}=3.12$&ndash;$3.15$ &mdash; consistent with $\pi$ at the percent level.
(Contrast with the pool Pearson correlation, whose flip term carries the <em>second</em>
moment of the selected values, $2\beta$ with $\beta=\mathbb E[\Delta^2_{\rm sel}]/s^2=2$:
each observable picks up the moment of the selected flips that it measures; Appendix B.)</p>

<h3>A.3 The basin regime</h3>
<p>Exact overlap decomposition on the hypercube: $q_{12}=q_{1f}q_{2f}
+\sqrt{(1-q_{1f}^2)(1-q_{2f}^2)}\,\hat u_1\!\cdot\!\hat u_2$. With $q=1-2d/N$ and
$u_{12}=d_1+d_2-2|A_1\cap A_2|$ this reads, for $d\ll N$,
$$\hat u_1\!\cdot\!\hat u_2 = \frac{d_1+d_2-u_{12}}{2\sqrt{d_1d_2}}
= \frac{|A_1\cap A_2|}{\sqrt{d_1d_2}}.$$
Renewal estimate: toward-flips (rate $(1+v)/2$ per move) each remove one site from $A$;
away-flips (rate $(1-v)/2$) add fresh sites. So $|A_1\cap A_2|\approx d\,e^{-(1+v)\Delta t/2d}$
while $\sqrt{d_1d_2}\approx d\,e^{-v\Delta t/2d}$, and in the ratio the drift cancels:
$\hat u_1\!\cdot\!\hat u_2\approx e^{-c\Delta t/2d}$ with $c$ the mutual speed &mdash; the
$\tau=R^2/2$ law, now mechanistic. This angle factor is verified parameter-free against the
transient part of the spectrum (Fig. B3: fitted times agree to 2&ndash;6% at every depth and
both $p$). The transient's amplitude is $\propto R(t)\propto\sqrt{d(t)}$ (the leading,
linear-in-$\hat u$ term of the restriction of the energy to the shell), so the EMD &mdash;
which measures the raiser&ndash;lowerer separation in absolute units set by the frozen width
&mdash; carries the extra factor $\sqrt{d_2/d_1}\approx e^{-v_{\rm rem}\Delta t/2d}$.
Multiplying: rate $(c+v_{\rm rem})/2d$.</p>

<h2 id="appB">Appendix B. Supporting correlation measurements</h2>

<p>These are the field-level facts the EMD theory is built on, measured directly.</p>

__A1__

<p>Fig. B1: the two branches of the two-point function at work &mdash; unflipped spins on
$+q^{p-1}$ (Krawtchouk), flipped spins near $-q^{p-1}$, and the pool correlation on
$q^{p-1}(1-4u/N)$: the pool's flip term is $2\beta=4$ (second moment of the selected values),
against the EMD's $\kappa=\pi$ (first moment). The controls move the pool constant to
$2\beta=2$ under the uniform and neutral rules, in step with the EMD's $\pi\to2$.</p>

__A2__

<p>Fig. B2: the unflipped correlation from all references collapses on the annealed kernel
$\xi'(q)/\xi'(1)$ as a function of mutual overlap. The systematic sag below the annealed curve
at deep $q$ (up to $\sim0.1$) is absent in the neutral control (max deviation $0.010$ at
$N{=}1000$), which pins it on the adaptive dynamics: it is the smooth onset of the conditioned
basin, not a property of the field.</p>

__A3__

<p>Fig. B3: the parameter-free test of the basin angle factor: Pearson correlation of the
transient part $\Delta_i(t)-\Delta_i^f$ (solid) against the disagreement-set formula evaluated
on each run's measured $(d_1,d_2,u_{12})$ (dashed).</p>

__A4__

<p>Fig. B4: kernel generality on the mixed $1{+}2{+}3$-spin dataset: the correlation follows
$\xi'(q)/\xi'(1)=(1+2q+3q^2)/6$ and heads to the additive floor $1/6$ &mdash; the $p{=}1$ part
of every $\Delta_i$ is constant along the walk and never scrambles. The pure 3-spin data on the
same axes decay toward zero. This is the "additive limit has no scrambling" statement of the
context document, visible inside a single spectrum.</p>

<h2 id="appC">Appendix C. Methods</h2>
<ul>
<li><b>Data.</b> Stored SSWM walks (<code>N*_P*_pure_repeats10.pkl</code>): couplings
($\operatorname{Var}J=p!/N^{p-1}$, twice the context document's convention &mdash; irrelevant
after normalization), initial configuration, accepted-flip sequence. Replay via incremental
updates (<code>analysis_lib.py</code> on top of <code>helper.py</code>), validated per walk:
every accepted flip had $\Delta>0$, the terminal configuration is a strict local maximum,
recomputation error $<10^{-5}$.</li>
<li><b>EMD.</b> <code>scipy.stats.wasserstein_distance</code> between the subset's values and
the full spectrum's values, normalized at the reference. References: fixed fractions of $T$
plus references pinned at $d_H\approx N/8, N/16, N/32$; both time directions.</li>
<li><b>Fits.</b> $\kappa$: least squares of $\widetilde W / \rho_{\rm kernel}(u)$ against
$1-\kappa u/N$ over $\widetilde W\ge0.82$ (curve-level, protocol-free). Timescales
$\tau$: log-slope over $\widetilde W\in[0.55,0.97]$, identical protocol for data and theory
curves (none of the laws is exponential). Errors: bootstrap over the 10 realizations.</li>
<li><b>Crossover numerics (Sec. 8.1).</b> Dense references pinned at
$d/N=0.05,\dots,0.41$ (<code>run_intermediate.py</code>); half-time $t_{1/2}$ by linear
interpolation of the bin-averaged forward curve through $1/2$. $t_K$ from the closed-form
kernel law with the bin's measured mutual speed (the curve-based version, where defined,
agrees to a few percent but is truncated by the walk end at small $d$); $t_B$ from the
amplitude$\times$angle curve on the measured geometry
(<code>intermediate_analysis.py</code>).</li>
<li><b>Controls.</b> Uniform greedy: <code>weighted=False</code> relaxations from the same
initial conditions. Neutral: uniformly random flip sequences on the same landscapes. Surrogate:
i.i.d. Gaussians, kernel factor $1-2(p-1)/(N-1)$ per accepted move, sign reversal of the
selected value; $3000$ realizations for the $\kappa$ fits.</li>
<li><b>Reproduce.</b> <code>run_measurements.py main|sweep</code> &rarr;
<code>run_controls.py</code> &rarr; <code>run_precision.py</code> &rarr;
<code>make_figures_emd.py</code> &rarr; <code>build_report.py</code>.</li>
</ul>

<p><b>Table C1</b> &mdash; fitted $\kappa$ (SSWM data, all sizes; theory $\pi\approx3.14$):</p>
<table>
<tr><th>$p$</th><th>$N$</th><th>$\kappa$</th></tr>
__KAPPATABLE__
</table>

<p><b>Table C2</b> &mdash; basin-regime EMD timescales (accepted moves): measured vs the
amplitude$\times$angle prediction fitted identically, the leading-order tangent
$2d/(1+v_{\rm rem})$, pure shell mixing $2d$, and the measured floor:</p>
<table>
<tr><th>$p$</th><th>$d_{\rm ref}$</th><th>$v_{\rm rem}$</th><th>$\tau_{\rm EMD}$</th>
<th>ampl$\times$angle</th><th>$2d/(1{+}v_{\rm rem})$</th><th>$2d$</th>
<th>floor $\widetilde W_\infty$</th></tr>
__LATETABLE__
</table>

<h2 id="appD">Appendix D. What does not fit perfectly</h2>
<ul>
<li><b>Mid-$q$ sag.</b> Beyond $u/N\sim0.15$ the measured EMD (and the underlying correlation)
sits below the annealed law by up to $\sim0.1$ before the basin regime takes over (Figs. 5,
B2). The neutral control shows no sag, so it is a conditioning effect of the adaptive walk.
Computing it is open. The surrogate, interestingly, tracks the data (not the annealed law)
through most of this &mdash; part of the sag is selection bookkeeping beyond $O(f)$, already
contained there.</li>
<li><b>$p=3$ basin EMD.</b> With $N=300$, the surviving raiser pools at basin references are
small ($\lesssim30$ spins) and the fitted late-time $\tau$ scatters by tens of percent around
the prediction (Table C2, e.g. measured $28$ vs tangent $20$ at $d\approx20$, $20$ vs $32$ at
$d\approx37$). The $p{=}2$ mid-depth point matches to 1% ($50.5$ vs $50.7$). Larger $N$ at
$p{=}3$ would sharpen this.</li>
<li><b>Rates-added is an interpolation, not a law.</b> Quantified in Sec. 8.1: it holds to
$10$&ndash;$25\%$ through $d\in[d^*/2,2d^*]$, but its residuals are systematic &mdash; the
$p{=}2$ data runs up to $\sim25\%$ faster than it just below $d^*$ (the conditioning sag), and
in the far field it over-adds (the basin angle factor at the equator double-counts kernel
decorrelation; drop the basin term there, where kernel-only is accurate to $5$&ndash;$6\%$).</li>
<li><b>Two principled all-times closures are ruled out by direct test</b> (Sec. 8.2):
endpoint-conditioned Gaussian regression (up to $2.3\times$ too slow through the crossover;
the real pinning is stronger than Gaussian conditioning on the terminal values) and the
mean-field master-equation process (correct far field; provably no basin: common-shift
convection plus fresh kicks; $0/140$ runs terminate). The surviving derived structure is the
kernel-difference law for the transient ($\le0.04$ pointwise for $d/N\le0.34$; $0.09$ in the
$p{=}2$ equator bin, concentrated beyond the half-time, where higher-order conditioning and
the clipped flip factor enter) plus the one-sided frozen component; the open piece is the
non-Gaussian selection split between them.</li>
<li><b>What sets $v$</b> (and its drift with $N$, $p$, and along the walk, $0.55\to0.88$)
remains the one place radial information lives; it is a self-consistency property of the walk,
untouched here.</li>
</ul>

<p class="dim">Couplings convention, walk statistics, and all correlation-level laws
(including the pool Pearson's $2\beta$ flip term) are documented in the analysis scripts.
Earlier draft of this report (correlation-centric): superseded by this version; its constant
"$2p+\pi/2$" for the EMD rate came from an incomplete bookkeeping of the flip term and is
corrected here to $2(p-1)+\pi$, now surrogate-verified at the percent level.</p>

</body>
</html>
"""

HTML = HTML.replace("__KAPPATABLE__", kappa_table())
HTML = HTML.replace("__LATETABLE__", late_table())
HTML = HTML.replace("__INTERTABLE__", inter_table())

FIGS = {
    "__F1__": ("f1_observable.png",
               "Fig. 1 &mdash; What the EMD sees (one run each; top p=2, bottom p=3). Scatter: "
               "the spectrum at time t against the spectrum at the reference; violet points are "
               "the marked raisers, grey the rest. At t=0 the raisers are exactly the right "
               "half. As the walk proceeds their values spread and drift down with the whole "
               "spectrum until the violet marginal matches the grey one. Right: the normalized "
               "EMD tracks this relaxation; dots mark the three snapshots. What to look at: by "
               "one timescale (dotted line) the separation is mostly gone."),
    "__F2__": ("f2_walk.png",
               "Fig. 2 &mdash; The walk in numbers (10 runs; top p=2, N=1000; bottom p=3, "
               "N=300). Left: distance to the terminal maximum falls linearly at v (dashed), "
               "while distance from the starting point grows at speed 1 (orange): each accepted "
               "flip is one Hamming unit of displacement, of which only the net fraction v "
               "points inward. Middle: the shell radius follows the parabola implied by linear "
               "d<sub>H</sub>(t); the equator 'stall' is pure coordinates. Right: the spectrum's "
               "width barely changes while the raisers deplete &mdash; nothing rescales "
               "globally. What to look at: the two different slopes on the left; they are the "
               "candidate clocks, and Fig. 3 decides between them."),
    "__F3__": ("f3_money.png",
               "Fig. 3 &mdash; The early-time EMD and its explanation (reference at t=0; band: "
               "&plusmn;1 s.e.m. over 10 runs). Curves as numbered in the legend and explained "
               "in the text: (1) data; (2) the parameter-free early-time law "
               "q<sup>p-1</sup>(1-&pi;u/N) using the measured u(t); (3) the kernel factor alone "
               "&mdash; the EMD if accepted moves did not sign-reverse; (4) the landscape-free "
               "surrogate, which contains only the kernel factor and the flip bookkeeping and "
               "lies on the data everywhere; (5) the same law run on the radial clock u&rarr;vt "
               "('slope = drift'), too slow by 1/v; (6) the time-to-maximum clock "
               "e<sup>-vt/d<sub>0</sub></sup>, too slow by a factor 3&ndash;3.5. What to look "
               "at: data on (2) and (4); both naive curves wrong in slope and shape; the gap "
               "between (3) and (1) is the move-reversal effect."),
    "__F4__": ("f4_collapse.png",
               "Fig. 5 &mdash; The correct coordinate is the mutual overlap. Top: EMD from all "
               "references in the first 60% of every walk (color: reference position) against "
               "the mutual overlap q(t<sub>ref</sub>,t); the red curve is the early-time law. "
               "One coordinate, one curve (spread grows at deep q where the O(u/N) law and the "
               "onset of conditioning bite &mdash; Appendix D). Bottom: the same data against "
               "the radial coordinate q<sub>f</sub>(t): no functional relation &mdash; "
               "&#87;&#771;=1 occurs at every q<sub>f</sub>. A radial theory has nothing to "
               "attach to."),
    "__F5__": ("f5_controls.png",
               "Fig. 6 &mdash; The flip constant follows the acceptance rule, not the drift. "
               "Same landscapes, three dynamics: SSWM (v=0.62/0.74), uniform-acceptance greedy "
               "(v=0.47/0.54), neutral random flipping (v=0). Filled points: fitted &kappa; "
               "with bootstrap errors; open squares: the surrogate run with the corresponding "
               "rule (no drift by construction). What to look at: &kappa; sits at &pi; for SSWM "
               "and at 2 for both no-size-bias rules, irrespective of v &mdash; including v=0."),
    "__F6__": ("f6_scaling.png",
               "Fig. 4 &mdash; The fitted flip constant &kappa; for every dataset (p=2: "
               "N=100&ndash;2000; p=3: N=100&ndash;500; bootstrap errors). Consistent with "
               "&kappa;=&pi; throughout, with no trend in N: the EMD rate is "
               "[2(p-1)+&pi;]/N across a factor 20 in system size."),
    "__F7__": ("f7_late.png",
               "Fig. 7 &mdash; Basin regime (references pinned at d<sub>ref</sub>&asymp;N/32, "
               "N/16, N/8). Top: EMD vs lag rescaled by 2d/(1+v<sub>rem</sub>); dashed: the "
               "amplitude&times;angle prediction computed from each run's measured geometry (no "
               "parameters); arrowheads: the floors. Bottom: fitted timescales vs "
               "d<sub>ref</sub>, between 2d (pure shell mixing; too slow) and on "
               "2d/(1+v<sub>rem</sub>). What to look at: &tau;&asymp;d<sub>ref</sub>; the "
               "largest-d point decays faster than the basin prediction because the kernel "
               "mechanism still contributes there (Fig. 8); the smallest-d bin has too few "
               "raisers for an EMD."),
    "__F8__": ("f8_crossover.png",
               "Fig. 8 &mdash; The EMD timescale along the whole walk: fitted &tau; at every "
               "reference vs the reference's distance to the terminal maximum. Flat kernel "
               "branch N/(2p-2+&pi;) far out; basin branch 2d/(1+v<sub>rem</sub>) near the "
               "maximum; pink: the two rates added. The crossover d* (vertical line) is reached "
               "at about 60% of the walk for both p &mdash; the same point where the linear "
               "drift window of d<sub>H</sub>(t) ends."),
    "__F9__": ("f9_symmetry.png",
               "Fig. 9 &mdash; Symmetries. The subset marked at 0.4T relaxes at the same rate "
               "read forward or backward (the law depends only on the mutual displacement); the "
               "forward curve eventually feels the basin floor. The raiser-subset and "
               "lowerer-subset EMDs are identical to machine precision (annotation) &mdash; the "
               "exact mixture identity of Sec. 1."),
    "__I1__": ("i1_intermediate.png",
               "Fig. 8b &mdash; Numerics through the crossover. Left and middle: bin-averaged "
               "EMD at two intermediate depths (d/N = 0.21 and 0.11) against the parameter-free "
               "curves: kernel only (blue), basin only (green), and their product (red, = rates "
               "added at curve level). The data falls between the single-mechanism curves and "
               "near their product until the late floor. Right: half-time t<sub>1/2</sub> vs "
               "depth: data (points) against kernel-only, basin-only, rates-added (red), and "
               "the sharp min(t<sub>K</sub>,t<sub>B</sub>) (dotted). What to look at: at "
               "d&asymp;d* the data sits a factor ~2 below both single-mechanism curves and on "
               "the rates-added curve; toward the far field it leaves the rates-added curve and "
               "climbs to kernel-only."),
    "__I2__": ("i2_ratios.png",
               "Fig. 8c &mdash; Accuracy of each description: ratio of measured to predicted "
               "half-time vs depth (band: &plusmn;15%). Rates-added (red circles / brown "
               "squares for the harmonic / product forms) stays within 10&ndash;25% through the "
               "crossover window; kernel-only (up-triangles) and basin-only (down-triangles) "
               "fail by up to a factor 2&ndash;3 outside their own regimes; the sharp min "
               "(crosses) fails by up to 2 at the center. The far-field drift of the red/brown "
               "points above 1 is the double-counting of the basin term at the equator "
               "(Appendix D): kernel-only is the correct far-field description."),
    "__U3__": ("u3_alltimes_geometry.png",
               "Fig. 8d &mdash; The kernel-difference law for the transient correlation at "
               "every depth. Solid: measured correlation of the transient part from references "
               "at d/N = 0.06 to 0.39 (colors), lag rescaled by each law curve's half-time. "
               "Dashed: the boxed formula evaluated on each run's measured (d<sub>1</sub>, "
               "d<sub>2</sub>(&Delta;t), u<sub>12</sub>(&Delta;t)) &mdash; derived, not "
               "fitted; no parameters, no interpolation. Pointwise agreement &le; 0.04 for "
               "d/N &le; 0.34 (both p); the p=2 equator bin deviates by up to 0.09 beyond its "
               "half-time (Appendix D)."),
    "__U1__": ("u1_unified.png",
               "Fig. 8e &mdash; The endpoint-conditioned ('pinned-kernel') candidate against "
               "the data at three depths (columns; rows: p=2, p=3). Dashed red: the "
               "partial-regression law with the measured floor &Phi;; dotted: its live part. "
               "It works at the far anchor and (by construction) deep in the basin, but is far "
               "too slow through the crossover: the true pinning to the terminal state is "
               "about twice the Gaussian-endpoint value."),
    "__U2__": ("u2_unified_ratio.png",
               "Fig. 8f &mdash; Accuracy of the pinned-kernel law: measured over predicted "
               "half-time. It fails by up to 2.3&times; (p=2) at d/N &asymp; 0.13 and "
               "approaches 1 only at the far anchor &mdash; the quantitative demonstration "
               "that single-endpoint Gaussian conditioning understates the conditioning of "
               "the walk's neighborhood of &sigma;<sub>f</sub>."),
    "__A1__": ("a1_ingredients.png",
               "Fig. B1 &mdash; The two branches of the spin-level correlation (reference t=0): "
               "unflipped spins on the Krawtchouk kernel +q<sup>p-1</sup>, flipped spins near "
               "-q<sup>p-1</sup> (each accepted move reverses its own &Delta; exactly), and the "
               "pool Pearson on q<sup>p-1</sup>(1-4u/N) &mdash; its flip term is the second "
               "moment of the selected values (2&beta;=4), where the EMD's is the first "
               "(&kappa;=&pi;)."),
    "__A2__": ("a2_rho_collapse.png",
               "Fig. B2 &mdash; The unflipped correlation from all references against the "
               "mutual overlap: collapse onto the annealed kernel &xi;'(q)/&xi;'(1), with the "
               "conditioning sag at deep q (absent in the neutral control)."),
    "__A3__": ("a3_geometry.png",
               "Fig. B3 &mdash; Parameter-free test of the basin angle factor: the transient "
               "part's correlation (solid) against the disagreement-set formula "
               "(d<sub>1</sub>+d<sub>2</sub>-u<sub>12</sub>)/(2&radic;(d<sub>1</sub>d<sub>2"
               "</sub>)) evaluated on the measured walk (dashed). Fitted times agree to "
               "2&ndash;6% at every depth and both p; the drift cancels here, which is why "
               "&tau;=R&sup2;/2 was the right shell-mixing law."),
    "__A4__": ("a4_mixed.png",
               "Fig. B4 &mdash; Kernel generality: the mixed 1+2+3-spin model follows the "
               "general law &xi;'(q)/&xi;'(1) and saturates toward the additive floor 1/6 (its "
               "p=1 component never scrambles), while the pure 3-spin correlation decays to "
               "zero."),
}
for key, (fname, caption) in FIGS.items():
    HTML = HTML.replace(key, img(fname, caption))

out = os.path.join(ROOT, "scrambling_report.html")
with open(out, "w") as fh:
    fh.write(HTML)
print(f"report written: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
