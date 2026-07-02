# Fraction of beneficial spins under greedy dynamics in the pure $p$-spin model

**Question.** During a greedy / SSWM adaptive walk, is there an $N$-independent
function of time giving the fraction of *beneficial* spins still available?
**Answer.** Yes. With time measured as flips-per-spin $\tau = t/N$,

$$\boxed{\,b(\tau)\;\approx\;\tfrac12\,(1-p\,\tau)_+^{2}\,}
\qquad\Longleftrightarrow\qquad
\frac{db}{d\tau} = -\,p\,\sqrt{2b},\quad b(0)=\tfrac12 .$$

Two features of this are **exact and derived** ($b(0)=\tfrac12$ and the initial
slope $-p$); the full quadratic is an **effective law**, excellent in the bulk and
confirmed numerically, with a small correction in the tail. This note explains
every piece — in particular *why the rate is $p$* and *where the $\sqrt{2b}$ comes
from*, which are the two things that were not obvious.

Figures produced alongside this note:
`beneficial_spins_result.png` (the collapse), `explain_sqrt2b.png` (the mechanism),
`ode_test.png` (the ODE check).

---

## 1. Setup and definitions

Pure $p$-spin Hamiltonian, couplings with variance $p!/N^{p-1}$:

$$H(\sigma)=\sum_{i_1<\dots<i_p} J_{i_1\dots i_p}\,\sigma_{i_1}\cdots\sigma_{i_p}.$$

The **flip spectrum** is the energy change on flipping spin $i$:

$$\Delta_i \;=\; \Delta H_i \;=\; -2\,\sigma_i h_i,
\qquad h_i=\!\!\sum_{I\ni i}\! J_I\!\!\prod_{k\in I,\,k\neq i}\!\!\sigma_k .$$

In your code's convention the walk *increases* $H$ (it maximises fitness), so a spin
is **beneficial** when $\Delta_i>0$, and a local maximum is reached when no spin is
beneficial. Define

$$b(\tau)=\frac{\#\{i:\Delta_i>0\}}{N},\qquad \tau=\frac{t}{N}\ \ (\text{flips per spin}).$$

Two standing facts we will use repeatedly:

* **Variance of the spectrum.** Each $\Delta_i$ is a sum over the
  $\binom{N-1}{p-1}$ interactions through $i$, so
  $$\operatorname{Var}(\Delta_i)=4\binom{N-1}{p-1}\frac{p!}{N^{p-1}}\;\xrightarrow{N\to\infty}\;4p\equiv\sigma_0^2 .$$
  ($\sigma_0^2=8$ for the SK model $p=2$, $12$ for $p=3$.)

* **Random start.** At $\tau=0$ the configuration is random, so the $\Delta_i$ are
  symmetric about $0$ $\Rightarrow$ **$b(0)=\tfrac12$** exactly.

---

## 2. Why the initial rate is exactly $-p$

This is the crucial and clean piece. Work per single flip; recall $d\tau = 1/N$, so
one unit of $\tau$ is $N$ flips. Let $B=Nb$ be the number of beneficial spins and
track $\Delta B$ per flip.

### 2a. The flipped spin itself: $-1$

We flip a beneficial spin $k$ (so $\Delta_k>0$). Flipping $k$ reverses every
interaction that contains $k$, so its own field flips sign:
$\Delta_k \to -\Delta_k < 0$. It leaves the beneficial set.

$$\Delta B_{\text{self}} = -1 \quad\text{per flip}.$$

If this were the whole story, $db/d\tau=-1$ and $b$ would hit $0$ at $\tau=\tfrac12$
for **every** $p$. It is *not* the whole story — flipping $k$ also disturbs everyone
else.

### 2b. Every other spin gets a kick — and the kick is biased *downward*

Flipping $k$ changes each other spin's stability by

$$\delta_j \;=\; 4\!\!\sum_{I\ni j,k}\! J_I\!\prod_{m\in I}\sigma_m ,$$

i.e. only the interactions containing **both** $j$ and $k$ contribute. Each is
$O(1/\sqrt N)$, so a single kick is tiny — but there are $\sim N$ of them per flip,
and, crucially, **they are not zero-mean**.

Here is the mechanism. The spin $k$ was selected *because* it is beneficial:

$$\Delta_k=-2S_k>0 \;\Longrightarrow\; S_k\equiv\sum_{I\ni k}J_I\!\prod_{m\in I}\sigma_m<0 .$$

The kick $\delta_j$ is $4\times$ the *piece* of that same sum $S_k$ that also involves
$j$. Because the whole sum is conditioned negative, that piece is biased negative
too. The fraction of $k$'s interactions that also contain $j$ is

$$\frac{\binom{N-2}{p-2}}{\binom{N-1}{p-1}}=\frac{p-1}{N-1},$$

and for a sum of equal-variance terms the conditional mean of a sub-sum is that
fraction of the total, giving

$$\mathbb E[\delta_j\mid k]
=4\cdot\frac{p-1}{N-1}\,S_k
=-\,\frac{2(p-1)}{N}\,\Delta_k \;+\;O(N^{-2}).$$

> **Intuition.** *Flipping a beneficial spin nudges the stability of **every** other
> spin downward.* The spin you chose was the one whose field pointed "the wrong way";
> since it shares couplings with everyone, undoing its frustration slightly
> re-frustrates all its neighbours. This is a coherent, one-directional push, not
> random noise — that is why it survives averaging.

### 2c. How many of the others cross zero: $-(p-1)$

A spin $j$ leaves the beneficial set if the downward push carries it across
$\Delta=0$. The expected number that cross is (density at the threshold) $\times$
(mean downward shift) $\times$ ($N$ spins):

$$\Delta B_{\text{kick}}
=-\,N\,\rho(0)\,\big|\mathbb E[\delta_j]\big|
=-\,2(p-1)\,\rho(0)\,\mathbb E[\Delta_k],$$

where $\rho(0)$ is the density of stabilities at $0$ (normalised to $1$) and
$\mathbb E[\Delta_k]$ is the mean stability of the *selected* spin.

Evaluate at $\tau=0$ (Gaussian of width $\sigma_0$), with SSWM selection
(flip probability $\propto \Delta_k$, i.e. proportional to the fitness gain):

$$\rho(0)=\frac{1}{\sigma_0\sqrt{2\pi}},\qquad
\mathbb E[\Delta_k]=\frac{\langle\Delta^2\rangle_+}{\langle\Delta\rangle_+}
=\frac{\sigma_0^2}{\sigma_0\sqrt{2/\pi}}=\sigma_0\sqrt{\tfrac\pi2}.$$

The product is where the magic happens — **$\sigma_0$ cancels**:

$$\rho(0)\,\mathbb E[\Delta_k]
=\frac{1}{\sigma_0\sqrt{2\pi}}\cdot\sigma_0\sqrt{\tfrac\pi2}
=\frac{1}{2}.$$

Hence $\Delta B_{\text{kick}} = -2(p-1)\cdot\tfrac12 = -(p-1)$.

### 2d. Add them up

$$\frac{db}{d\tau}\Big|_{0}
=\underbrace{-1}_{\text{flipped spin}}\;\underbrace{-\,(p-1)}_{\text{push on the rest}}
=\;\boxed{-p}.$$

So the "$p$" is literally **1 (the spin you flip) plus $p-1$ (the spins your flip
knocks out)**. The extra $p-1$ is a pure interaction effect: at $p=1$ (independent
sites) it vanishes and the rate is just $-1$; each extra body in the interaction adds
one more spin knocked out per flip. The width $\sigma_0$ — and with it all the messy
$\sqrt{\pi/2}$ factors — drops out, which is why the answer is a clean integer $-p$.

*Numerics:* measured initial slope is $-1.9\to-2.0$ for $p=2$ (all $N$) and climbs
toward $-3$ for $p=3$ ($-2.76,-2.95$ at $N=100,200$). ✓

---

## 3. The full curve and where $\sqrt{2b}$ comes from

### 3a. The ODE is just "a length that shrinks at constant speed"

The closed form $b=\tfrac12(1-p\tau)^2$ is *identical* to the statement

$$\frac{db}{d\tau}=-p\sqrt{2b}.$$

The cleanest way to read this: **change variables to $u\equiv\sqrt{2b}$.** Then

$$\frac{du}{d\tau}=\frac{1}{\sqrt{2b}}\frac{db}{d\tau}=-p
\quad\Longrightarrow\quad u(\tau)=1-p\,\tau,
\quad\Longrightarrow\quad b=\tfrac12 u^2=\tfrac12(1-p\tau)^2 .$$

So $\sqrt{2b}$ is a quantity that starts at $1$ and **decreases in a perfectly
straight line, at constant rate $p$**, hitting zero at $\tau=1/p$. The fraction $b$
is quadratic only because it is (half) the *square* of that linear quantity.
That is the entire content of the "$\sqrt{2b}$": it is the thing that is actually
linear in time.

The left panel of `explain_sqrt2b.png` shows exactly this — $\sqrt{2b}$ versus $\tau$
is a straight line of slope $-p$ for both $p=2$ and $p=3$.

### 3b. What is $\sqrt{2b}$, physically? The width of the beneficial band

$\sqrt{2b}$ is not just an algebraic trick — it is (proportional to) the **linear
size of the beneficial part of the spectrum.** Measure the mean gap of the beneficial
spins, $\langle\Delta\rangle_+$ ("how far above threshold the beneficial spins
typically sit"). Along the walk we find

$$\langle\Delta\rangle_+ \;=\; \sigma_0\sqrt{\tfrac{2}{\pi}}\;\sqrt{2b}
\qquad(\text{constant of proportionality holds to a few \% for all }\tau),$$

see the middle panel of `explain_sqrt2b.png` (measured ratio $\approx 2.2$, predicted
$\sigma_0\sqrt{2/\pi}=2.26$ for $p=2$). So:

* $\sqrt{2b}\;\propto\;$ **width** of the beneficial band.
* $b$ (the fraction) $\;\propto\;$ **area** of that band $\;\approx\;$ width $\times$ height.

Picture the beneficial spins as a little triangular sliver sitting above threshold.
Its **width** shrinks linearly in time (the band edge marches toward zero at constant
speed, set by the downward drift of §2), and because area $\sim\text{width}^2$, the
fraction closes **quadratically**. The right panel confirms the differential form
directly: $-db/d\tau$ plotted against $\sqrt{2b}$ is a straight line through the
origin with slope $p$.

> **One-line summary of the mechanism.** *Selection + the conditioning drift close the
> beneficial band from the top at a constant rate; the fraction of beneficial spins is
> the band's area, i.e. the square of its (linearly shrinking) width — hence
> $b\propto(1-p\tau)^2$ and $db/d\tau\propto\sqrt{b}$.*

### 3c. Honest status of the quadratic

What is **exact**: $b(0)=\tfrac12$ and $db/d\tau|_0=-p$ (§2).

What is **effective / empirical**: that the rate stays exactly $-p\sqrt{2b}$ for *all*
$\tau$. It is the simplest law consistent with both exact anchors that *also* has the
right qualitative endgame (rate $\to0$ as $b\to0$, because near a local maximum almost
nothing is beneficial and each flip is nearly self-cancelling). The $-db/d\tau$ vs
$\sqrt{2b}$ collapse (slope $p$ through the origin) is strong evidence that this is the
correct effective law in the bulk.

It is **not** exact in the tail. A drift-only estimate would predict the band width to
collapse *faster* than linearly (giving an exponential-type decay with the walk ending
at $\tau=\ln 2/[2(p-1)]$, e.g. $0.35$ for $p=2$). The real dynamics decays *slower*
late on, because diffusion of the stabilities keeps refilling the band just above
threshold (we see the band height $\rho(0)/\sqrt{2b}$ creep up from $0.14$ to $0.23$
over the walk — a pile-up of marginal spins). That refilling is exactly what makes the
rate fall like $\sqrt{2b}$ rather than staying near $-1$, and it is why the **actual
walk runs slightly past $\tau=1/p$** (§6).

---

## 4. The underlying $N$-independent equation (why a collapse exists at all)

Everything above is a consequence of one $N$-independent evolution equation for the
spectrum density $\rho(\Delta,\tau)$ (normalised to $1$; $b=\int_0^\infty\rho$):

$$\frac{\partial\rho}{\partial\tau}
=\underbrace{8p(p-1)\,\partial_\Delta^2\rho}_{\text{kick diffusion}}
+\underbrace{2(p-1)E_{\rm sel}\,\partial_\Delta\rho}_{\text{conditioning drift}}
-\underbrace{w(\Delta)\mathbf 1_{\Delta>0}}_{\text{flip a beneficial spin}}
+\underbrace{w(-\Delta)\mathbf 1_{\Delta<0}}_{\text{it reappears at }-\Delta},$$

with the SSWM selection density $w(\Delta)=\Delta\,\rho(\Delta)/Z$ on $\Delta>0$,
$Z=\int_0^\infty\!\Delta\rho$, and $E_{\rm sel}=\langle\Delta^2\rangle_+/\langle\Delta\rangle_+$.
The three transport coefficients follow from the couplings:

| quantity | value | meaning |
|---|---|---|
| $\sigma_0^2=\operatorname{Var}(\Delta_i)$ | $4p$ | initial spectrum width$^2$ |
| diffusion coeff. | $8p(p-1)$ | variance injected by kicks per unit $\tau$ |
| drift velocity | $-2(p-1)E_{\rm sel}$ | coherent downward push (§2b) |

Because every coefficient is a pure number in the variable $\tau=t/N$, **the whole
curve $b(\tau)$ is $N$-independent** — that is the theoretical reason the data
collapse. Integrating this PDE with $\rho(\cdot,0)=\mathcal N(0,\sigma_0^2)$ reproduces
$b(0)=\tfrac12$, the slope $-p$, and the sigmoidal decay (parameter-free). It agrees
with the simulation through the bulk and mildly over-holds the tail (its diffusion is
a touch too strong), which is the same tail caveat as §3c.

---

## 5. Numerical validation

Replaying the recorded walks (10 repeats each) and sampling $b$ vs $t/N$:

**Collapse & closed form (SK model, $p=2$).** Curves for
$N=100,200,300,400,500,1000,1500,2000$ fall on one master curve in $\tau=t/N$. The
zero-parameter form $b=\tfrac12(1-2\tau)^2$ matches with RMSE $\approx0.01$ over the
bulk:

| $\tau$ | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|
| simulation ($N=2000$) | 0.400 | 0.310 | 0.228 | 0.164 | 0.072 | 0.028 |
| $\tfrac12(1-2\tau)^2$ | 0.405 | 0.320 | 0.245 | 0.180 | 0.080 | 0.020 |

**Exponent.** Fixing the zero at $\tau=1/p$ and fitting $b=\tfrac12(1-p\tau)^\beta$
returns $\beta\approx2.1$ for $p=2$ (all $N$) and $\beta\to2.01$ for $p=3$ as $N$ grows
— i.e. $\beta=2$.

**The ODE directly.** $-db/d\tau$ vs $\sqrt{2b}$ is linear through the origin with
slope $\approx p$ (best fit $2.07$ for $p=2$; $2.81$ for $p=3$ at $N=300$, still rising
toward $3$).

**Band-width scaling.** $\langle\Delta\rangle_+/\sqrt{2b}=$ const $\approx2.2$
throughout (predicted $2.26$), confirming §3b.

---

## 6. $p$-dependence and the walk length (the tail)

The initial slope $-p$ makes the decay **steeper for larger $p$**: $p=3$ falls twice
as fast at the start as it hits zero, reaching the bulk floor near $\tau=1/p$
($0.5$ for $p=2$, $0.333$ for $p=3$).

The *actual* walk (first time $b$ hits exactly $0$) runs a little past $1/p$ because of
the marginal-spin tail:

| $N$ | 100 | 200 | 300 | 400 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|---|
| $L/N$, $p=2$ | 0.51 | 0.48 | 0.51 | 0.55 | 0.56 | 0.60 | 0.62 | 0.62 |
| $L/N$, $p=3$ | 0.32 | 0.39 | 0.40 | | | | | |

For $p=2$ this grows with $N$ toward $\approx0.62$ — intriguingly close to the
scrambling drift $v\approx0.62$ measured elsewhere in this project (worth a follow-up:
is the greedy walk length per spin equal to the displacement rate $v$?). The quadratic
captures the bulk down to $b\sim0.03$; the last few percent live on the tail that the
effective law does not resolve.

---

## 7. Caveats

1. **Selection rule sets the constant.** The clean $-p$ holds for the
   *fitness-gain–weighted* SSWM rule your data use (flip $\propto\Delta$). Uniform
   choice among beneficial spins gives instead
   $db/d\tau|_0=-1-\tfrac{2(p-1)}{\pi}\approx-1.64$ for $p=2$; steepest descent
   (always the largest $\Delta$) is steeper than $-p$. The $b(0)=\tfrac12$ and the
   *existence* of an $N$-independent curve are rule-independent; the numbers are not.

2. **Bulk vs tail.** $b=\tfrac12(1-p\tau)^2$ is the leading (hydrodynamic) law; it is
   exact at $\tau=0$ (value and slope) and accurate to RMSE $\sim0.01$ in the bulk, but
   the true walk length exceeds $1/p$ by a fluctuation tail (§6).

3. **Finite size.** $p=3$ is only measured to $N=300$; slopes and $L/N$ are still
   drifting toward their $N\to\infty$ values ($-3$, and a limit above $1/3$).

---

## 8. Reproduce

Run inside the project venv (`.venv/bin/python`):

| script | produces |
|---|---|
| `_measure_beneficial.py N*_P2_pure_repeats10.pkl …` | $b(t/N)$ for each file → `beneficial_results.pkl` |
| `_fit_form.py` | closed-form / exponent fits (§5) |
| `_test_ode.py` | $-db/d\tau$ vs $\sqrt{2b}$ (`ode_test.png`) |
| `_spectrum_shape.py` | band width / height scaling (§3b) |
| `_mf_theory.py` | integrates the PDE of §4 (`mf_curves.pkl`) |
| `_final_fig.py`, `_explain_fig.py` | `beneficial_spins_result.png`, `explain_sqrt2b.png` |

**Bottom line.** The fraction of beneficial spins is an $N$-independent function of
$\tau=t/N$: it starts at $\tfrac12$, decays with the exact initial rate $-p$ (one spin
you flip, plus $p-1$ you knock out), and follows $b\approx\tfrac12(1-p\tau)^2$ — the
square of a band width that closes at constant speed $p$ — down to a small fluctuation
tail near $\tau\approx1/p$.
