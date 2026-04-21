import math
from itertools import combinations

import numpy as np


FLOAT_DTYPE = np.float32
SPIN_DTYPE = np.int8


def _site_index_dtype(N):
    """Pick the smallest unsigned integer dtype that can store spin-site indices."""
    if N - 1 <= np.iinfo(np.uint16).max:
        return np.uint16
    if N - 1 <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _interaction_index_dtype(num_interactions):
    """Pick an integer dtype that can index into the interaction table."""
    if num_interactions <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _as_spin_array(sigma, copy=False):
    """Cast to +/-1 int8 spin array."""
    if copy:
        return np.array(sigma, dtype=SPIN_DTYPE, copy=True)
    return np.asarray(sigma, dtype=SPIN_DTYPE)


# ---------------------------------------------------------------------------
# Building the interaction table
# ---------------------------------------------------------------------------

def _build_spin_indices(N, p):
    """
    Enumerate all C(N, p) distinct p-body interactions and store them
    column-wise: a tuple of p arrays, where array k holds the k-th spin
    index of every interaction (i_1 < i_2 < ... < i_p).
    """
    site_dtype = _site_index_dtype(N)
    tuples = np.array(list(combinations(range(N), p)), dtype=site_dtype)
    return tuple(np.ascontiguousarray(tuples[:, k]) for k in range(p))


def _build_site_interaction_map(N, spin_indices):
    """
    For each spin site i, list the interactions that contain it.

    Returns a length-N list; element i is an array of interaction indices
    (rows into the spin_indices table) where site i appears.
    """
    num_interactions = spin_indices[0].shape[0]
    idx_dtype = _interaction_index_dtype(num_interactions)
    site_map = [[] for _ in range(N)]

    for column in spin_indices:
        for row, site in enumerate(column):
            site_map[int(site)].append(row)

    return [np.array(rows, dtype=idx_dtype) for rows in site_map]


# ---------------------------------------------------------------------------
# Core per-interaction computations
# ---------------------------------------------------------------------------

def _compute_spin_products(sigma, sector, interaction_idx=None):
    """
    Compute the spin product  σ_{i_1} * σ_{i_2} * ... * σ_{i_p}  for each
    interaction in one sector.

    If *interaction_idx* is given, only that subset of interactions is evaluated.
    """
    spin_indices = sector["spin_indices"]

    if interaction_idx is None:
        cols = [sigma[c] for c in spin_indices]
    else:
        cols = [sigma[c[interaction_idx]] for c in spin_indices]

    product = cols[0].copy()
    for c in cols[1:]:
        product *= c
    return product.astype(SPIN_DTYPE, copy=False)


def _scatter_to_sites(per_site, sector, contributions, interaction_idx=None):
    """
    Distribute a per-interaction quantity to every spin site participating
    in that interaction, accumulating into *per_site*.

    For example, if interaction (i, j, k) has contribution c, then
    per_site[i], per_site[j], and per_site[k] each receive +c.
    """
    N = per_site.shape[0]
    for sites in sector["spin_indices"]:
        if interaction_idx is not None:
            sites = sites[interaction_idx]
        per_site += _sum_by_site(sites, contributions, N)


def _sum_by_site(sites, weights, N):
    """Aggregate weights by site index (histogram-based scatter-add)."""
    summed = np.bincount(sites, weights=weights, minlength=N)
    return summed.astype(FLOAT_DTYPE, copy=False)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def init_p_tensor(N, p, random_state=None):
    """
    Initialize one p-body interaction sector.

    Draws C(N, p) Gaussian couplings J_{i_1...i_p} with variance
    p! / N^{p-1}, stored sparsely alongside the spin-index tuples.

    Parameters
    ----------
    N : int
        Number of spins.
    p : int
        Interaction order (body number).
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    dict
        A sector with keys:
        - "order": p
        - "spin_indices": tuple of p arrays (the interaction table)
        - "couplings": the random coupling J_{i_1...i_p} for each interaction
        - "site_to_interactions": for each site, which interactions contain it
    """
    if int(p) != p or p < 1:
        raise ValueError("p must be a positive integer.")
    if p > N:
        raise ValueError("p must satisfy p <= N.")

    rng = np.random.default_rng(random_state)
    p = int(p)

    spin_indices = _build_spin_indices(N, p)
    variance = math.factorial(p) / (N ** (p - 1))
    couplings = rng.normal(
        loc=FLOAT_DTYPE(0.0),
        scale=FLOAT_DTYPE(np.sqrt(variance)),
        size=spin_indices[0].shape[0],
    ).astype(FLOAT_DTYPE)

    return {
        "order": p,
        "spin_indices": spin_indices,
        "couplings": couplings,
        "site_to_interactions": _build_site_interaction_map(N, spin_indices),
    }


def init_tensor(N, p, random_state=None):
    """Alias for ``init_p_tensor``."""
    return init_p_tensor(N, p, random_state=random_state)


def init_J(N, P, random_state=None, pure=False):
    """
    Initialize a mixed or pure p-spin model.

    The Hamiltonian is:

        H(σ) = Σ_p  Σ_{i_1<...<i_p}  J_{i_1...i_p}  σ_{i_1} ... σ_{i_p}

    where the sum over p runs from 1 to P (mixed) or includes only p = P (pure).

    Parameters
    ----------
    N : int
        Number of spins.
    P : int
        Maximum interaction order.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.
    pure : bool, optional
        If True, keep only the P-body sector (pure P-spin model).
        Default is False (mixed model with orders 1 through P).

    Returns
    -------
    dict
        Model with keys "N", "P", "pure", and "sectors".
    """
    if int(P) != P or P < 1:
        raise ValueError("P must be a positive integer.")
    if P > N:
        raise ValueError("P must satisfy P <= N.")

    rng = np.random.default_rng(random_state)
    P = int(P)
    orders = [P] if pure else list(range(1, P + 1))
    sectors = [init_p_tensor(N, p, random_state=rng) for p in orders]
    return {"N": N, "P": P, "pure": bool(pure), "sectors": sectors}


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def compute_lfs(sigma, J):
    """
    Compute the local field at every site.

    The local field h_i = ∂H/∂σ_i, i.e. the sum of all interaction terms
    that contain site i, with σ_i factored out (using σ_i^2 = 1).
    """
    sigma = _as_spin_array(sigma)
    local_fields = np.zeros(J["N"], dtype=FLOAT_DTYPE)

    for sector in J["sectors"]:
        spin_products = _compute_spin_products(sigma, sector)
        weighted_products = sector["couplings"] * spin_products  # J * σ_{i1}...σ_{ip}

        for sites in sector["spin_indices"]:
            # Multiply by σ_site to cancel it from the product (σ^2 = 1),
            # leaving J times the product of the *other* spins.
            field_contributions = weighted_products * sigma[sites]
            local_fields += _sum_by_site(sites, field_contributions, J["N"])

    return local_fields


def compute_fit_slow(sigma, J, f_off=0.0):
    """
    Compute the fitness (Hamiltonian value) of configuration σ.

        fitness = Σ_interactions  J_{i1...ip} σ_{i1} ... σ_{ip}  -  f_off
    """
    sigma = _as_spin_array(sigma)
    fitness = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        spin_products = _compute_spin_products(sigma, sector)
        fitness += np.dot(sector["couplings"], spin_products)

    return float(fitness - FLOAT_DTYPE(f_off))


def compute_fit_off(sigma_init, J):
    """Compute the fitness offset so that fitness(sigma_init) = 1."""
    return compute_fit_slow(sigma_init, J) - 1


def compute_fitness_delta_mutant(sigma, J, k):
    """
    Compute the fitness change ΔF when spin k is flipped.

    Only interactions containing site k are affected, so we restrict
    the sum to those terms for efficiency.
    """
    sigma = _as_spin_array(sigma)
    delta_f = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        affected = sector["site_to_interactions"][k]
        if affected.size == 0:
            continue

        spin_products = _compute_spin_products(sigma, sector, interaction_idx=affected)
        delta_f += np.sum(
            -FLOAT_DTYPE(2.0) * sector["couplings"][affected] * spin_products,
            dtype=FLOAT_DTYPE,
        )

    return float(delta_f)


# ---------------------------------------------------------------------------
# Distribution of fitness effects (DFE)
# ---------------------------------------------------------------------------

def compute_dfe(sigma, J):
    """Compute the DFE: the fitness change ΔF_i for flipping each spin i."""
    sigma = _as_spin_array(sigma)
    return (-FLOAT_DTYPE(2.0) * sigma * compute_lfs(sigma, J)).astype(FLOAT_DTYPE, copy=False)


def compute_bdfe(sigma, J):
    """Return (beneficial DFE values, their site indices)."""
    dfe = compute_dfe(sigma, J)
    return _extract_beneficial(dfe)


def _extract_beneficial(dfe):
    """Extract beneficial (positive) entries and their indices from a DFE."""
    mask = dfe > 0
    return dfe[mask], np.flatnonzero(mask)


def compute_normalized_bdfe(sigma, J):
    """Return the beneficial DFE normalized to a probability distribution."""
    bdfe, b_ind = compute_bdfe(sigma, J)
    norm = np.sum(bdfe, dtype=FLOAT_DTYPE)
    if norm > 0:
        bdfe = bdfe / norm
    return bdfe.astype(FLOAT_DTYPE, copy=False), b_ind


def compute_rank(sigma, J):
    """Count the number of beneficial mutations (rank of the configuration)."""
    dfe = compute_dfe(sigma, J)
    return int(np.count_nonzero(dfe > 0))


# ---------------------------------------------------------------------------
# Spin-flip selection
# ---------------------------------------------------------------------------

def _choose_beneficial_flip(dfe, sswm=True):
    """
    Choose a spin to flip from the beneficial mutations in the DFE.

    If sswm=True, flip probability is proportional to ΔF (strong-selection
    weak-mutation). Otherwise, pick uniformly among beneficial mutations.
    """
    bdfe, beneficial_sites = _extract_beneficial(dfe)
    if sswm:
        probs = bdfe / np.sum(bdfe, dtype=FLOAT_DTYPE)
        return np.random.choice(beneficial_sites, p=probs)
    return np.random.choice(beneficial_sites)


def sswm_flip(sigma, J):
    """Choose a spin to flip using SSWM (strong-selection weak-mutation)."""
    return _choose_beneficial_flip(compute_dfe(sigma, J), sswm=True)


# ---------------------------------------------------------------------------
# Adaptive walk (relaxation)
# ---------------------------------------------------------------------------

def _initialize_relaxation_state(sigma0, J):
    """
    Set up the cached state for an adaptive walk: the current spin
    configuration, its fitness, the full DFE, and per-sector cached
    spin products (to allow incremental updates on each flip).
    """
    sigma = _as_spin_array(sigma0, copy=True)
    dfe = np.zeros(J["N"], dtype=FLOAT_DTYPE)
    fitness = FLOAT_DTYPE(0.0)
    sector_caches = []

    for sector in J["sectors"]:
        spin_products = _compute_spin_products(sigma, sector)
        fitness += np.dot(sector["couplings"], spin_products)
        dfe_contributions = -FLOAT_DTYPE(2.0) * sector["couplings"] * spin_products
        _scatter_to_sites(dfe, sector, dfe_contributions)
        sector_caches.append({"spin_products": spin_products})

    return {"sigma": sigma, "dfe": dfe, "fitness": fitness, "sector_caches": sector_caches}


def _apply_flip(state, J, flip_site):
    """
    Flip spin at *flip_site* and incrementally update the fitness, DFE,
    and cached spin products.

    Only interactions containing flip_site need recomputation.
    """
    delta_f = state["dfe"][flip_site]

    for sector, cache in zip(J["sectors"], state["sector_caches"]):
        affected = sector["site_to_interactions"][flip_site]
        if affected.size == 0:
            continue

        old_spin_products = cache["spin_products"][affected]
        # Flipping one spin negates all spin products containing it,
        # which shifts the DFE by 4 * J * (old product) at each site.
        dfe_updates = FLOAT_DTYPE(4.0) * sector["couplings"][affected] * old_spin_products

        _scatter_to_sites(state["dfe"], sector, dfe_updates, interaction_idx=affected)
        cache["spin_products"][affected] = -old_spin_products

    state["sigma"][flip_site] = -state["sigma"][flip_site]
    state["fitness"] += delta_f


def relax_pspin(sigma0, J, sswm=True):
    """
    Run an adaptive walk until no beneficial mutations remain.

    Returns the sequence of flipped sites.
    """
    flip_sequence = []
    state = _initialize_relaxation_state(sigma0, J)

    while np.any(state["dfe"] > 0):
        flip_site = _choose_beneficial_flip(state["dfe"], sswm=sswm)
        flip_sequence.append(int(flip_site))
        _apply_flip(state, J, int(flip_site))

    return flip_sequence
