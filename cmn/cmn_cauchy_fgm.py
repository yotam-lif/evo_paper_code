r"""Distribution and likelihood helpers for the shared-buffer Student FGM.

The phenotype-to-fitness map is the same log-fitness Fisher model used elsewhere:

    s = log(w(r + delta) / w(r)) = -r . delta - |delta|^2 / 2,
    w(x) = exp(-|x|^2 / 2).

The mutation vector is changed from Gaussian to the isotropic shared-buffer family

    delta = sigma * Z / sqrt(2 G),
    Z ~ N_n(0, I),  G ~ Gamma(mu, 1),

so ``|delta|^2 / sigma^2 ~ BetaPrime(n/2, mu)``.  Its magnitude density has
tail exponent ``1 + mu`` and survival exponent ``mu``.  The original multivariate
Cauchy is recovered at ``mu=1/2``, because then ``2 G`` has a chi-squared
distribution with one degree of freedom.

The functions below evaluate the exact DFE density obtained by integrating the
translated isotropic density over spheres centred on the fitness optimum.

All three parameters (effective dimension n, radius r, and Cauchy component scale sigma)
may be non-integer/continuous during inference.  The analytic continuation is normalized
for n > 0 and is useful as an effective-dimensional model.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import (
    betainc,
    betaln,
    gammaln,
    hyp2f1,
    kve,
    logsumexp,
    ndtr,
    roots_jacobi,
)


_HYPERGEOM_QUAD_N = 128
_SURVIVAL_QUAD_N = 192
_ERROR_QUAD_N = 16
_SURVIVAL_NODES, _SURVIVAL_WEIGHTS = leggauss(_SURVIVAL_QUAD_N)
_ERROR_NODES, _ERROR_WEIGHTS = hermgauss(_ERROR_QUAD_N)


def _reduced_hypergeom_log(n, mu, z):
    """Stable Euler-reduced hypergeometric term for the generalized model.

    SciPy's direct hypergeometric evaluator becomes unreliable for large parameters.
    We first extract the ``(1-z)^(-mu-1/2)`` singularity with Euler's
    transformation.  When its second reduced parameter is positive, the remaining
    bounded function is evaluated as an expectation under a beta distribution
    using Gauss-Jacobi quadrature.  Otherwise direct evaluation of the reduced
    function is stable.
    """
    a = 0.25 * n - 0.5 * mu
    b = 0.25 * n - 0.5 * mu - 0.5
    c = 0.5 * n
    if b <= 1e-10:
        reduced = hyp2f1(a, b, c, z)
        return np.log(reduced)
    d = mu + 0.5
    nodes, weights = roots_jacobi(_HYPERGEOM_QUAD_N, d - 1.0, b - 1.0)
    t = 0.5 * (nodes + 1.0)
    log_weights = np.log(weights) - np.log(weights.sum())
    log_prefactor = (
        gammaln(d)
        + gammaln(c)
        - gammaln(b + d)
        - gammaln(a + d)
    )
    log_ratio = (
        np.log1p(-t)[:, None]
        - np.log1p(-t[:, None] * z[None, :])
    )
    return log_prefactor + logsumexp(
        log_weights[:, None]
        + a * log_ratio,
        axis=0,
    )


def cauchy_fgm_support(r):
    """Support of the log-fitness effect: ``(-inf, r^2 / 2]``."""
    return -np.inf, 0.5 * np.square(r)


def cauchy_fgm_dfe_logpdf(s, n, sigma, r, mu=0.5):
    r"""Exact log density of the shared-buffer Student-FGM DFE.

    Writing ``u = |r + delta|^2 = r^2 - 2s``, the radial density of the translated
    isotropic Student family gives

    .. math::

        p(s)=\frac{2\Gamma(\lambda)\sigma^{2\mu}}
        {\Gamma(\mu)\Gamma(n/2)}
        u^{n/2-1} A^{-\lambda}
        {}_2F_1(\lambda/2,(\lambda+1)/2;n/2;z),

    where ``A = sigma^2 + r^2 + u``,
    ``z = 4 r^2 u / A^2``, and ``lambda=n/2+mu``.
    """
    s = np.asarray(s, dtype=float)
    n = float(n)
    sigma = float(sigma)
    r = float(r)
    mu = float(mu)
    out = np.full(s.shape, -np.inf, dtype=float)
    if not (np.isfinite(n) and np.isfinite(sigma) and np.isfinite(r)
            and np.isfinite(mu) and n > 0.0 and sigma > 0.0
            and r >= 0.0 and mu > 0.0):
        return out

    u = r * r - 2.0 * s
    keep = u > 0.0
    if not np.any(keep):
        return out

    uk = u[keep]
    A = sigma * sigma + r * r + uk
    z = np.clip(4.0 * r * r * uk / (A * A), 0.0, 1.0 - 1e-15)
    # Compute log(1-z) from its factorization rather than by subtracting nearly
    # equal numbers.  This remains accurate on the large-n likelihood ridge.
    sqrt_u = np.sqrt(uk)
    log_one_minus_z = (
        np.log(sigma * sigma + np.square(r - sqrt_u))
        + np.log(sigma * sigma + np.square(r + sqrt_u))
        - 2.0 * np.log(A)
    )
    z_reduced = np.clip(-np.expm1(log_one_minus_z), 0.0, 1.0 - 1e-15)
    log_h = (
        -(mu + 0.5) * log_one_minus_z
        + _reduced_hypergeom_log(n, mu, z_reduced)
    )

    lam = 0.5 * n + mu
    log_const = (
        np.log(2.0)
        + gammaln(lam)
        - gammaln(mu)
        - gammaln(0.5 * n)
        + 2.0 * mu * np.log(sigma)
    )
    vals = (
        log_const
        + (0.5 * n - 1.0) * np.log(uk)
        - lam * np.log(A)
        + log_h
    )
    out[keep] = np.where(np.isfinite(vals), vals, -np.inf)
    return out


def cauchy_fgm_dfe_pdf(s, n, sigma, r, mu=0.5):
    """Exact DFE density, the exponential of :func:`cauchy_fgm_dfe_logpdf`."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(cauchy_fgm_dfe_logpdf(s, n, sigma, r, mu=mu))


def cauchy_fgm_fitness_dfe_logpdf(s, n, sigma, r, mu=0.5):
    r"""Exact density of the absolute fitness effect ``s = w_mut - w_0``.

    The shared-buffer closed form describes the log-fitness effect ``x``.
    With ``w_0=exp(-r^2/2)``, the measured absolute effect is
    ``s=w_0(exp(x)-1)``.  The density below applies the inverse transform
    ``x=log(1+s/w_0)`` and the Jacobian ``1/(w_0+s)``.
    """
    s = np.asarray(s, dtype=float)
    r = float(r)
    out = np.full(s.shape, -np.inf, dtype=float)
    if not np.isfinite(r) or r < 0.0:
        return out
    w0 = np.exp(-0.5 * r * r)
    keep = (s > -w0) & (s < 1.0 - w0)
    if np.any(keep):
        shifted = w0 + s[keep]
        x = np.log(shifted / w0)
        out[keep] = (
            cauchy_fgm_dfe_logpdf(
                x,
                n=n,
                sigma=sigma,
                r=r,
                mu=mu,
            )
            - np.log(shifted)
        )
    return out


def cauchy_fgm_fitness_dfe_pdf(s, n, sigma, r, mu=0.5):
    """Density corresponding to :func:`cauchy_fgm_fitness_dfe_logpdf`."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(
            cauchy_fgm_fitness_dfe_logpdf(
                s,
                n=n,
                sigma=sigma,
                r=r,
                mu=mu,
            )
        )


def cauchy_fgm_survival_many_eps(cut, n, sigma, r, eps, mu=0.5):
    r"""Survival probabilities for one or more Gaussian error standard deviations.

    The exact mutation magnitude obeys
    ``q/(q+sigma^2) ~ Beta(n/2, mu)``.  Conditional on q, the direction cosine
    has the spherical marginal distribution, so the survival probability reduces
    to a one-dimensional expectation over log q.  Gaussian measurement error is
    integrated by Gauss-Hermite quadrature and vectorized over ``eps``.
    """
    n = float(n)
    sigma = float(sigma)
    r = float(r)
    mu = float(mu)
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    if not (
        n > 1.0
        and sigma > 0.0
        and r >= 0.0
        and mu > 0.0
        and np.isfinite(mu)
        and np.all(np.isfinite(eps))
        and np.all(eps >= 0.0)
    ):
        return np.full(eps.shape, np.nan, dtype=float)

    # Integrate over y=log(q/sigma^2).  Its log-beta-prime density is smooth
    # even at large n, unlike direct Jacobi quadrature in t=q/(q+sigma^2),
    # whose mass collapses into a boundary layer at t=1.
    a_q = 0.5 * n
    b_q = mu
    z_lo, z_hi = -20.0, 30.0
    z = (
        0.5 * (z_hi - z_lo) * _SURVIVAL_NODES
        + 0.5 * (z_hi + z_lo)
    )
    y = np.log(n) + z
    log_density = (
        a_q * y
        - (a_q + b_q) * np.logaddexp(0.0, y)
        - betaln(a_q, b_q)
    )
    weights = (
        0.5
        * (z_hi - z_lo)
        * _SURVIVAL_WEIGHTS
        * np.exp(log_density)
    )
    weights = weights / weights.sum()
    q = sigma * sigma * np.exp(y)

    errors = np.sqrt(2.0) * eps[:, None] * _ERROR_NODES[None, :]
    err_weights = _ERROR_WEIGHTS / np.sqrt(np.pi)
    shifted_cut = float(cut) - errors[:, :, None]
    if r <= 1e-14:
        conditional = (-0.5 * q[None, None, :]) >= shifted_cut
    else:
        z0 = (
            -shifted_cut - 0.5 * q[None, None, :]
        ) / (r * np.sqrt(q)[None, None, :])
        interior = np.abs(z0) < 1.0
        conditional = np.where(z0 >= 1.0, 1.0, 0.0)
        zi = np.clip(z0, -1.0, 1.0)
        sphere_cdf = (
            0.5
            + 0.5 * np.sign(zi)
            * betainc(0.5, 0.5 * (n - 1.0), np.square(zi))
        )
        conditional = np.where(interior, sphere_cdf, conditional)
    values = np.einsum(
        "h,q,ehq->e",
        err_weights,
        weights,
        conditional,
        optimize=True,
    )
    return np.clip(values, 0.0, 1.0)


def cauchy_fgm_survival(cut, n, sigma, r, eps=0.0, mu=0.5):
    r"""Probability that an observed effect is at least ``cut``.

    This scalar wrapper delegates to :func:`cauchy_fgm_survival_many_eps`.
    """
    return float(cauchy_fgm_survival_many_eps(
        cut,
        n=n,
        sigma=sigma,
        r=r,
        eps=np.array([eps], dtype=float),
        mu=mu,
    )[0])


def cauchy_fgm_fitness_survival_many_eps(
    cut,
    n,
    sigma,
    r,
    eps,
    mu=0.5,
):
    r"""``P(s_obs >= cut)`` for the exact absolute-fitness effect.

    This is the raw-fitness counterpart of
    :func:`cauchy_fgm_survival_many_eps`.  Measurement noise is integrated in
    raw-effect coordinates; every resulting threshold is then mapped to the
    corresponding log-fitness threshold before the angular integral.
    """
    n = float(n)
    sigma = float(sigma)
    r = float(r)
    mu = float(mu)
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    if not (
        n > 1.0
        and sigma > 0.0
        and r >= 0.0
        and mu > 0.0
        and np.isfinite(mu)
        and np.all(np.isfinite(eps))
        and np.all(eps >= 0.0)
    ):
        return np.full(eps.shape, np.nan, dtype=float)

    a_q = 0.5 * n
    b_q = mu
    z_lo, z_hi = -20.0, 30.0
    z = (
        0.5 * (z_hi - z_lo) * _SURVIVAL_NODES
        + 0.5 * (z_hi + z_lo)
    )
    y = np.log(n) + z
    log_density = (
        a_q * y
        - (a_q + b_q) * np.logaddexp(0.0, y)
        - betaln(a_q, b_q)
    )
    weights = (
        0.5
        * (z_hi - z_lo)
        * _SURVIVAL_WEIGHTS
        * np.exp(log_density)
    )
    weights = weights / weights.sum()
    q = sigma * sigma * np.exp(y)

    w0 = np.exp(-0.5 * r * r)
    if not np.isfinite(w0) or w0 <= 0.0:
        return np.full(eps.shape, np.nan, dtype=float)
    errors = np.sqrt(2.0) * eps[:, None] * _ERROR_NODES[None, :]
    raw_cut = float(cut) - errors
    below = raw_cut <= -w0
    above = raw_cut >= 1.0 - w0
    interior_cut = ~(below | above)
    log_cut = np.zeros_like(raw_cut)
    log_cut[interior_cut] = np.log1p(raw_cut[interior_cut] / w0)

    shifted_cut = log_cut[:, :, None]
    if r <= 1e-14:
        conditional = (-0.5 * q[None, None, :]) >= shifted_cut
    else:
        z0 = (
            -shifted_cut - 0.5 * q[None, None, :]
        ) / (r * np.sqrt(q)[None, None, :])
        interior_angle = np.abs(z0) < 1.0
        conditional = np.where(z0 >= 1.0, 1.0, 0.0)
        zi = np.clip(z0, -1.0, 1.0)
        sphere_cdf = (
            0.5
            + 0.5 * np.sign(zi)
            * betainc(0.5, 0.5 * (n - 1.0), np.square(zi))
        )
        conditional = np.where(
            interior_angle,
            sphere_cdf,
            conditional,
        )
    conditional = np.where(below[:, :, None], 1.0, conditional)
    conditional = np.where(above[:, :, None], 0.0, conditional)
    values = np.einsum(
        "h,q,ehq->e",
        _ERROR_WEIGHTS / np.sqrt(np.pi),
        weights,
        conditional,
        optimize=True,
    )
    return np.clip(values, 0.0, 1.0)


def cauchy_fgm_fitness_bin_probs(edges, n, sigma, r, mu=0.5):
    r"""Exact bin probabilities for the absolute-fitness-effect DFE.

    Integrating the angular CDF over the beta-prime mutation magnitude avoids
    point-sampling the integrable singularity at the finite deleterious
    endpoint.
    """
    edges = np.asarray(edges, dtype=float)
    n = float(n)
    sigma = float(sigma)
    r = float(r)
    mu = float(mu)
    if not (
        edges.ndim == 1
        and edges.size >= 2
        and np.all(np.diff(edges) > 0.0)
        and n > 1.0
        and sigma > 0.0
        and r >= 0.0
        and mu > 0.0
    ):
        return np.full(max(edges.size - 1, 0), np.nan, dtype=float)

    a_q = 0.5 * n
    y = np.log(n) + (
        0.5 * (30.0 - (-20.0)) * _SURVIVAL_NODES
        + 0.5 * (30.0 + (-20.0))
    )
    log_density = (
        a_q * y
        - (a_q + mu) * np.logaddexp(0.0, y)
        - betaln(a_q, mu)
    )
    weights = (
        0.5
        * (30.0 - (-20.0))
        * _SURVIVAL_WEIGHTS
        * np.exp(log_density)
    )
    weights = weights / weights.sum()
    q = sigma * sigma * np.exp(y)

    w0 = np.exp(-0.5 * r * r)
    if not np.isfinite(w0) or w0 <= 0.0:
        return np.full(edges.size - 1, np.nan, dtype=float)
    below = edges <= -w0
    above = edges >= 1.0 - w0
    interior = ~(below | above)
    log_edges = np.zeros_like(edges)
    log_edges[interior] = np.log1p(edges[interior] / w0)

    if r <= 1.0e-14:
        survival = (
            (-0.5 * q[None, :]) >= log_edges[:, None]
        ) @ weights
    else:
        z0 = (
            -log_edges[:, None] - 0.5 * q[None, :]
        ) / (r * np.sqrt(q)[None, :])
        interior_angle = np.abs(z0) < 1.0
        conditional = np.where(z0 >= 1.0, 1.0, 0.0)
        zi = np.clip(z0, -1.0, 1.0)
        sphere_cdf = (
            0.5
            + 0.5 * np.sign(zi)
            * betainc(0.5, 0.5 * (n - 1.0), np.square(zi))
        )
        conditional = np.where(
            interior_angle,
            sphere_cdf,
            conditional,
        )
        survival = conditional @ weights
    cdf = 1.0 - survival
    cdf[below] = 0.0
    cdf[above] = 1.0
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    return np.clip(np.diff(cdf), 0.0, 1.0)


def cauchy_fgm_large_n_logpdf(s, C, A):
    r"""Large-n limit at fixed ``C=n*sigma^2`` and ``A=r*sigma``.

    In this limit

    .. math::

        S = -A Z/B - C/(2B^2),

    for independent ``Z ~ N(0,1)`` and ``B=|N(0,1)|``.  Integrating over ``B`` gives
    a closed density involving the modified Bessel function ``K_1``.  ``kve`` is used
    below so the negative-tail exponential cancellation remains numerically stable.
    """
    s = np.asarray(s, dtype=float)
    C = float(C)
    A = float(A)
    out = np.full(s.shape, -np.inf, dtype=float)
    if not (np.isfinite(C) and np.isfinite(A) and C > 0.0 and A > 0.0):
        return out
    radius = np.hypot(A, s)
    argument = C * radius / (2.0 * A * A)
    scaled_bessel = kve(1.0, argument)
    vals = (
        np.log(C)
        - np.log(2.0 * np.pi * A)
        - np.log(radius)
        + np.log(scaled_bessel)
        - C * (s + radius) / (2.0 * A * A)
    )
    return np.where(np.isfinite(vals), vals, -np.inf)


def cauchy_fgm_large_n_pdf(s, C, A):
    """Density corresponding to :func:`cauchy_fgm_large_n_logpdf`."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(cauchy_fgm_large_n_logpdf(s, C, A))


def cauchy_fgm_large_n_survival(cut, C, A, eps=0.0):
    """Survival probability for the exact large-n limiting DFE."""
    C = float(C)
    A = float(A)
    eps = float(eps)
    if not (C > 0.0 and A > 0.0 and eps >= 0.0):
        return float("nan")
    # Gauss-Hermite undersamples the narrow b~0 boundary layer that generates the
    # Cauchy tail.  A high-order Legendre rule on [0,8] resolves it directly; the
    # omitted half-normal mass above 8 is negligible.
    b = 4.0 * (_SURVIVAL_NODES + 1.0)
    weights = (
        4.0
        * _SURVIVAL_WEIGHTS
        * np.sqrt(2.0 / np.pi)
        * np.exp(-0.5 * b * b)
    )
    mean = -C / (2.0 * b * b)
    sd = np.sqrt(A * A / (b * b) + eps * eps)
    probability = ndtr((mean - float(cut)) / sd)
    return float(np.clip(np.sum(weights * probability), 0.0, 1.0))
