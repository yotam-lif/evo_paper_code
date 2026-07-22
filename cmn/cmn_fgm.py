import numpy as np
from scipy.stats import ncx2, norm

class Fisher:
    """
    Fisher Geometric Model with Gaussian mutation steps and SSWM relaxation.

    In this version, the selection matrix S is represented only by its eigenvalues
    (i.e., working in the diagonal basis). Both isotropic and anisotropic cases
    are supported through the eigenvalue spectrum.

    Attributes
    ----------
    n : int
        Dimensionality of phenotype space.
    sigma : float
        Standard deviation for Gaussian mutation steps.
    deltas : numpy.ndarray
        Array of pre-sampled mutation steps of shape (m, n).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(self, n, sigma=0.05, m=10**3, random_state=None):
        """
        Initialize the model in the diagonal basis.

        Parameters
        ----------
        n : int
            Number of phenotypic traits (dimensions).
        sigma : float
            Scale parameter for mutations.
        m : int
            Number of mutation vectors to pre-sample.
        random_state : int or numpy.random.Generator, optional
            Seed or RNG for reproducibility.
        """
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        # RNG setup
        if isinstance(random_state, (int, np.integer)):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng()

        # Pre-sample Gaussian mutation steps
        self.deltas = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.m, self.n))
        # Initialize r_0 = (sqrt(n), 0, ..., 0), isotropic so only initial radius of n matters
        # Instead of starting at random r and taking iid gaussian entries of sigma_0 so that r_0 ~ n * sigma_0 ^2,
        # We set scale by sigma_0 = 1 and sigma becomes in units of sigma_0.
        # initial radius scales with n so that effective initial position doesn't become smaller as n increases (mutation size scales with n as well).
        self.r = np.zeros(n)
        self.r[0] = np.sqrt(n)

    def _sample_semicircle(self, n, sigma):
        """
        Sample n values from the semicircle distribution on [-2*sigma, 2*sigma]:
        density f(x) ∝ sqrt(4*sigma^2 - x^2) using rejection sampling.
        """
        radius = 2.0 * sigma
        samples = []
        while len(samples) < n:
            x = self.rng.uniform(-radius, radius)
            accept_prob = np.sqrt(radius**2 - x**2) / radius
            if self.rng.uniform(0.0, 1.0) < accept_prob:
                samples.append(x)
        return np.array(samples)

    def compute_log_fitness(self, r):
        r = np.asarray(r, dtype=float)
        return - float(np.dot(r, r))

    def compute_fitness(self, r):
        """
        Compute fitness: w(r) = exp(log_fitness(r)).
        """
        return float(np.exp(self.compute_log_fitness(r)))

    def compute_dfe(self, r):
        """
        Compute distribution of fitness effects at phenotype r.
        Returns array of w(r + delta_i) - w(r) for each pre-sampled delta.
        """
        r = np.asarray(r, dtype=float)
        w0 = self.compute_fitness(r)
        return np.array([self.compute_fitness(r + delta) - w0 for delta in self.deltas])

    def compute_bdfe(self, dfe):
        """
        Extract beneficial fitness effects and their indices from dfe array.
        """
        dfe = np.asarray(dfe, dtype=float)
        mask = dfe > 0
        return dfe[mask], np.nonzero(mask)[0]

    def sswm_choice(self, bdfe, b_ind):
        """
        Choose a substitution under SSWM: probability ∝ fitness effect.
        """
        bdfe = np.asarray(bdfe, dtype=float)
        total = bdfe.sum()
        if total > 0:
            probs = bdfe / bdfe.sum()
        else:
            probs = bdfe / len(bdfe)
        return int(self.rng.choice(b_ind, p=probs))

    def relax(self, max_steps=10 ** 4):
        """
        Perform an adaptive walk using SSWM.
        Returns list of chosen mutation indices, r history and dfe history.
        """
        traj = [self.r.copy()]
        flips = []
        dfes = []
        for _ in range(max_steps):
            dfe = self.compute_dfe(self.r)
            dfes.append(dfe.copy())
            bdfe, b_ind = self.compute_bdfe(dfe)
            if len(b_ind) == 0:
                break
            choice = self.sswm_choice(bdfe, b_ind)
            flips.append(choice)
            self.r += self.deltas[choice]
            # self.deltas[choice] = -1 * self.deltas[choice]
            traj.append(self.r.copy())
        return flips, traj, dfes


# ──────────────────────────────────────────────────────────────────────────────
# Analytic isotropic FGM distribution of fitness effects (DFE), in LOG-FITNESS.
#
# Derivation: Theory/fgm_dfe.tex and Tenaillon (2014, Theory/utility_of_fgm.pdf). A
# genotype sits at radius r in an n-dimensional phenotype space with Gaussian fitness
# w(x) = exp(-|x|^2 / 2). A mutation adds an isotropic step delta ~ N(0, sigma^2 I_n).
# The SELECTION COEFFICIENT is the LOG-fitness ratio -- the additive coefficient that
# competition assays actually measure, NOT the raw fitness difference exp(-U/2)-exp(-r^2/2):
#
#       s = log(w(r+delta)/w(r)) = (r^2 - U)/2,   U = |r + delta|^2,
#
# with U/sigma^2 ~ noncentral chi^2 (df = n, noncentrality lambda = r^2/sigma^2). So
#
#       s = s_max - (sigma^2/2) X,   X ~ ncx2(n, r^2/sigma^2),   s_max = r^2/2,
#
# where s_max = -log w(r) = r^2/2 is the ancestor maladaptation = the UPPER support of s
# (the best possible mutation reaches the optimum, gaining s = s_max). Support is
# s in (-inf, r^2/2]: ONE-SIDED -- the deleterious tail is unbounded, so only the
# beneficial reach (the data max) bounds r (from below at r_lo = sqrt(2*s_max)).
# Three parameters: n (phenotypic dimension), sigma (mutation step s.d.), r (radius).
# (Earlier versions of this file fitted the fitness DIFFERENCE Delta; the data are
# selection coefficients, so we fit log-fitness s. See Tenaillon eq for p(s).)
# ──────────────────────────────────────────────────────────────────────────────
def fgm_support(r):
    """Support (s_min, s_max) of the log-fitness FGM DFE at radius ``r``: (-inf, r^2/2]."""
    return -np.inf, 0.5 * np.square(r)


def fgm_dfe_logpdf(s, n, sigma, r):
    """Log density of the log-fitness FGM DFE P(s | n, sigma, r), vectorized over ``s``.

    ``n``, ``sigma``, ``r`` are scalars. Returns -inf above the support (s > r^2/2).
    """
    s = np.asarray(s, float)
    s2 = sigma * sigma
    smax = 0.5 * r * r
    out = np.full(s.shape, -np.inf)
    m = s < smax
    if np.any(m):
        X = (r * r - 2.0 * s[m]) / s2          # = 2 (smax - s)/sigma^2 > 0
        out[m] = ncx2.logpdf(X, df=n, nc=r * r / s2) + np.log(2.0 / s2)
    return out


def fgm_dfe_pdf(s, n, sigma, r):
    """Density of the log-fitness FGM DFE (exp of :func:`fgm_dfe_logpdf`)."""
    return np.exp(fgm_dfe_logpdf(s, n, sigma, r))


# scipy's ncx2.cdf sums ~nc/2 series terms, so it is ~20x slower per element once
# the noncentrality nc = r^2/sigma^2 is large (small sigma in the grid). Sankaran's
# (1959) cube-root-normal approximation is O(1) and grows MORE accurate with nc
# (abs error ~1e-4 at nc~80, ~1e-9 by nc~600), so it is the fast path there.
_NCX2_EXACT_MAX = 300.0   # use exact ncx2 while df + 2*nc <= this, Sankaran above


def _sankaran_cdf(y, k, l):
    """Sankaran approximation to the noncentral chi^2 CDF, P(X <= y; df=k, nc=l)."""
    s = k + 2.0 * l
    h = 1.0 - (2.0 / 3.0) * (k + l) * (k + 3.0 * l) / s ** 2
    p = s / (k + l) ** 2
    m = (h - 1.0) * (1.0 - 3.0 * h)
    base = np.maximum(y / (k + l), 1e-300)
    num = base ** h - (1.0 + h * p * (h - 1.0 - 0.5 * (2.0 - h) * m * p))
    den = h * np.sqrt(2.0 * p) * (1.0 + 0.5 * m * p)
    return norm.cdf(num / den)


def _ncx2_cdf(y, df, nc):
    """ncx2 CDF with the Sankaran fast path for large noncentrality. ``df`` and
    ``nc`` broadcast against ``y``."""
    df_b = np.broadcast_to(df, y.shape)
    nc_b = np.broadcast_to(nc, y.shape)
    out = np.empty(y.shape, float)
    big = (df_b + 2.0 * nc_b) > _NCX2_EXACT_MAX
    sm = ~big
    if sm.any():
        out[sm] = ncx2.cdf(y[sm], df=df_b[sm], nc=nc_b[sm])
    if big.any():
        out[big] = _sankaran_cdf(y[big], df_b[big], nc_b[big])
    return out


def _fgm_cdf_s_at(edges, n, sigma, r):
    """F_s(edge) = P(S <= edge) for each bin edge, broadcast over a grid.

    ``n``, ``sigma``, ``r`` are 1-D arrays of length G (grid points); ``edges`` is a
    1-D array of length E. Returns shape (G, E), INCREASING in the edge. Edges at or
    above s_max = r^2/2 map to 1; the deleterious side is unbounded (-> 0).
    """
    n = np.asarray(n, float)[:, None]
    sigma = np.asarray(sigma, float)[:, None]
    r = np.asarray(r, float)[:, None]
    s2 = sigma * sigma
    X = (r * r - 2.0 * edges[None, :]) / s2        # (G, E) = 2(s_max - edge)/sigma^2
    above = X <= 0.0                               # edge >= s_max -> all mass below
    # F_s(edge) = P(S <= edge) = P(X >= X(edge)) = sf(X) = 1 - cdf(X)
    F = 1.0 - _ncx2_cdf(np.where(above, 0.0, X), n, r * r / s2)
    return np.where(above, 1.0, F)


def fgm_bin_probs(edges, n, sigma, r):
    """Per-bin model probabilities P(bin_j) on a parameter grid, shape (G, B)."""
    F = _fgm_cdf_s_at(edges, n, sigma, r)          # (G, B+1), increasing
    return np.clip(F[:, 1:] - F[:, :-1], 0.0, None)


def fgm_bin_loglik(counts, edges, n, sigma, r):
    """Fine-binned multinomial log-likelihood of the log-fitness FGM DFE over a grid.

    Parameters
    ----------
    counts : (B,) array        histogram counts of the observed effects
    edges  : (B+1,) array      monotone bin edges the counts were built on
    n, sigma, r : (G,) arrays  grid of parameter triples to evaluate

    Returns
    -------
    (G,) array of log-likelihoods. A triple is -inf when a populated bin lies above
    s_max = r^2/2 (the beneficial reach is what bounds r from below). The deleterious
    side is unbounded, so r is no longer capped from above by the support; the bins
    only cover the observed range, so mass leaking above s_max / below the data
    discourages an over-large r through the (unnormalised) bin probabilities.
    """
    counts = np.asarray(counts, float)
    edges = np.asarray(edges, float)
    P = fgm_bin_probs(edges, n, sigma, r)          # (G, B)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0.0, np.log(P), -np.inf)
        contrib = np.where(counts[None, :] > 0.0, counts[None, :] * logP, 0.0)
    return contrib.sum(axis=1)
