import math
from itertools import combinations

import numpy as np


FLOAT_DTYPE = np.float32
SPIN_DTYPE = np.int8


def _site_index_dtype(N):
    """
    Pick the smallest unsigned integer dtype that can store site indices.
    """
    if N - 1 <= np.iinfo(np.uint16).max:
        return np.uint16
    if N - 1 <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _term_row_dtype(num_terms):
    """
    Pick a term-row index dtype for lookup tables.
    """
    if num_terms <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _as_spin_array(sigma, copy=False):
    """
    Store spins as +/-1 int8 values.
    """
    if copy:
        return np.array(sigma, dtype=SPIN_DTYPE, copy=True)
    return np.asarray(sigma, dtype=SPIN_DTYPE)


def _build_site_columns(N, p):
    """
    Build the distinct tuples i_1 < ... < i_p and store them column-wise.
    """
    site_dtype = _site_index_dtype(N)
    index_tuples = np.array(list(combinations(range(N), p)), dtype=site_dtype)
    return tuple(np.ascontiguousarray(index_tuples[:, position]) for position in range(p))


def _build_site_to_term_rows(N, site_columns):
    """
    For each site, list the stored interaction terms that contain it.
    """
    num_terms = site_columns[0].shape[0]
    row_dtype = _term_row_dtype(num_terms)
    site_to_term_rows = [[] for _ in range(N)]

    for site_indices in site_columns:
        for term_idx, site in enumerate(site_indices):
            site_to_term_rows[int(site)].append(term_idx)

    return [np.array(term_rows, dtype=row_dtype) for term_rows in site_to_term_rows]


def _sum_by_site(site_indices, weights, N):
    """
    Aggregate weights by site index using a histogram instead of scatter-add.
    """
    summed = np.bincount(site_indices, weights=weights, minlength=N)
    return summed.astype(FLOAT_DTYPE, copy=False)


def _compute_term_products(sigma, sector, term_rows=None):
    """
    Compute sigma_{i_1} ... sigma_{i_p} for one sector.
    """
    site_columns = sector["site_columns"]
    order = sector["order"]

    if term_rows is None:
        col0 = site_columns[0]
        if order == 1:
            return sigma[col0]
        col1 = site_columns[1]
        if order == 2:
            return (sigma[col0] * sigma[col1]).astype(SPIN_DTYPE, copy=False)
        col2 = site_columns[2]
        if order == 3:
            return (sigma[col0] * sigma[col1] * sigma[col2]).astype(SPIN_DTYPE, copy=False)
        col3 = site_columns[3]
        if order == 4:
            return (
                sigma[col0] * sigma[col1] * sigma[col2] * sigma[col3]
            ).astype(SPIN_DTYPE, copy=False)
    else:
        col0 = site_columns[0][term_rows]
        if order == 1:
            return sigma[col0]
        col1 = site_columns[1][term_rows]
        if order == 2:
            return (sigma[col0] * sigma[col1]).astype(SPIN_DTYPE, copy=False)
        col2 = site_columns[2][term_rows]
        if order == 3:
            return (sigma[col0] * sigma[col1] * sigma[col2]).astype(SPIN_DTYPE, copy=False)
        col3 = site_columns[3][term_rows]
        if order == 4:
            return (
                sigma[col0] * sigma[col1] * sigma[col2] * sigma[col3]
            ).astype(SPIN_DTYPE, copy=False)

    term_products = np.ones(
        site_columns[0].shape[0] if term_rows is None else term_rows.shape[0],
        dtype=SPIN_DTYPE,
    )
    for site_indices in site_columns:
        if term_rows is None:
            term_products *= sigma[site_indices]
        else:
            term_products *= sigma[site_indices[term_rows]]
    return term_products


def _accumulate_term_values(site_values, sector, term_values, term_rows=None):
    """
    Add one scalar value per stored interaction term to every site in that term.
    """
    N = site_values.shape[0]
    for site_indices in sector["site_columns"]:
        if term_rows is not None:
            site_indices = site_indices[term_rows]
        site_values += _sum_by_site(site_indices, term_values, N)


def init_p_tensor(N, p, random_state=None):
    """
    Initialize one p-spin sector in sparse form.

    Instead of storing a full symmetric rank-p tensor, we store only the
    distinct tuples i_1 < ... < i_p and their couplings. This is equivalent to
    the model definition and is much easier to work with numerically.

    Parameters
    ----------
    N : int
        The number of spins.
    p : int
        The interaction order.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    dict
        A sector with keys:
        - "order": the interaction order p
        - "site_columns": tuple of p arrays, one array per position in i_1 < ... < i_p
        - "couplings": Gaussian couplings for those tuples
        - "site_to_term_rows": for each site, the rows of the sector that contain it
    """
    if int(p) != p or p < 1:
        raise ValueError("p must be a positive integer.")
    if p > N:
        raise ValueError("p must satisfy p <= N.")

    rng = np.random.default_rng(random_state)
    p = int(p)

    site_columns = _build_site_columns(N, p)
    variance = math.factorial(p) / (N ** (p - 1))
    couplings = rng.normal(
        loc=FLOAT_DTYPE(0.0),
        scale=FLOAT_DTYPE(np.sqrt(variance)),
        size=site_columns[0].shape[0],
    ).astype(FLOAT_DTYPE)

    return {
        "order": p,
        "site_columns": site_columns,
        "couplings": couplings,
        "site_to_term_rows": _build_site_to_term_rows(N, site_columns),
    }


def init_tensor(N, p, random_state=None):
    """
    Alias for ``init_p_tensor``.
    """
    return init_p_tensor(N, p, random_state=random_state)


def init_J(N, P, random_state=None):
    """
    Initialize the mixed p-spin model for orders p = 1, ..., P.

    Parameters
    ----------
    N : int
        The number of spins.
    P : int
        The maximum interaction order.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    dict
        Model data containing all p-spin sectors.
    """
    if int(P) != P or P < 1:
        raise ValueError("P must be a positive integer.")
    if P > N:
        raise ValueError("P must satisfy P <= N.")

    rng = np.random.default_rng(random_state)
    P = int(P)
    sectors = [init_p_tensor(N, p, random_state=rng) for p in range(1, P + 1)]
    return {"N": N, "P": P, "sectors": sectors}


def compute_lfs(sigma, J):
    """
    Calculate the local fields for the mixed p-spin model.

    The local field at site i is the sum of all interaction terms containing i,
    with sigma_i removed from the product.
    """
    sigma = _as_spin_array(sigma)
    local_fields = np.zeros(J["N"], dtype=FLOAT_DTYPE)

    for sector in J["sectors"]:
        term_products = _compute_term_products(sigma, sector)
        base_term_values = sector["couplings"] * term_products

        for site_indices in sector["site_columns"]:
            other_spin_products = base_term_values * sigma[site_indices]
            local_fields += _sum_by_site(site_indices, other_spin_products, J["N"])

    return local_fields


def compute_dfe(sigma, J):
    """
    Calculate the distribution of fitness effects.
    """
    sigma = _as_spin_array(sigma)
    return (-FLOAT_DTYPE(2.0) * sigma * compute_lfs(sigma, J)).astype(FLOAT_DTYPE, copy=False)


def compute_bdfe(sigma, J):
    """
    Calculate the beneficial distribution of fitness effects.
    """
    dfe = compute_dfe(sigma, J)
    return _compute_bdfe_from_dfe(dfe)


def _compute_bdfe_from_dfe(dfe):
    """
    Extract the beneficial fitness effects and their indices from a DFE.
    """
    bdfe = dfe[dfe > 0]
    b_ind = np.flatnonzero(dfe > 0)
    return bdfe, b_ind


def _choose_flip_from_dfe(dfe, sswm=True):
    """
    Choose a beneficial flip directly from a precomputed DFE.
    """
    bdfe, beneficial_idx = _compute_bdfe_from_dfe(dfe)
    if sswm:
        flip_probabilities = bdfe / np.sum(bdfe, dtype=FLOAT_DTYPE)
        return np.random.choice(beneficial_idx, p=flip_probabilities)
    return np.random.choice(beneficial_idx)


def compute_normalized_bdfe(sigma, J):
    """
    Calculate the normalized beneficial distribution of fitness effects.
    """
    bdfe, b_ind = compute_bdfe(sigma, J)
    norm = np.sum(bdfe, dtype=FLOAT_DTYPE)
    if norm > 0:
        bdfe = bdfe / norm
    return bdfe.astype(FLOAT_DTYPE, copy=False), b_ind


def compute_rank(sigma, J):
    """
    Calculate the rank of the spin configuration.
    """
    dfe = compute_dfe(sigma, J)
    return int(np.count_nonzero(dfe > 0))


def sswm_flip(sigma, J):
    """
    Choose a spin to flip using SSWM probabilities.
    """
    return _choose_flip_from_dfe(compute_dfe(sigma, J), sswm=True)


def compute_fit_off(sigma_init, J):
    """
    Calculate the fitness offset for the given configuration.
    """
    return compute_fit_slow(sigma_init, J) - 1


def compute_fit_slow(sigma, J, f_off=0.0):
    """
    Compute the fitness of the configuration sigma.
    """
    sigma = _as_spin_array(sigma)
    fitness = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        term_products = _compute_term_products(sigma, sector)
        fitness += np.dot(sector["couplings"], term_products)

    return float(fitness - FLOAT_DTYPE(f_off))


def compute_fitness_delta_mutant(sigma, J, k):
    """
    Compute the fitness change for a mutant at site k.
    """
    sigma = _as_spin_array(sigma)
    delta_f = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        affected_term_rows = sector["site_to_term_rows"][k]
        if affected_term_rows.size == 0:
            continue

        term_products = _compute_term_products(sigma, sector, term_rows=affected_term_rows)
        delta_f += np.sum(
            -FLOAT_DTYPE(2.0) * sector["couplings"][affected_term_rows] * term_products,
            dtype=FLOAT_DTYPE,
        )

    return float(delta_f)


def _initialize_relaxation_state(sigma0, J):
    """
    Build the DFE, cached fitness, and cached term products for relaxation.
    """
    sigma = _as_spin_array(sigma0, copy=True)
    dfe = np.zeros(J["N"], dtype=FLOAT_DTYPE)
    fitness = FLOAT_DTYPE(0.0)
    sector_states = []

    for sector in J["sectors"]:
        term_products = _compute_term_products(sigma, sector)
        fitness += np.dot(sector["couplings"], term_products)
        term_dfe_contributions = -FLOAT_DTYPE(2.0) * sector["couplings"] * term_products
        _accumulate_term_values(dfe, sector, term_dfe_contributions)
        sector_states.append({"term_products": term_products})

    return {"sigma": sigma, "dfe": dfe, "fitness": fitness, "sector_states": sector_states}


def _apply_flip_to_relaxation_state(state, J, flip_idx):
    """
    Update sigma, cached fitness, cached term products, and the DFE after one flip.
    """
    delta_f = state["dfe"][flip_idx]

    for sector, sector_state in zip(J["sectors"], state["sector_states"]):
        affected_term_rows = sector["site_to_term_rows"][flip_idx]
        if affected_term_rows.size == 0:
            continue

        old_term_products = sector_state["term_products"][affected_term_rows]
        dfe_updates = FLOAT_DTYPE(4.0) * sector["couplings"][affected_term_rows] * old_term_products

        _accumulate_term_values(
            state["dfe"],
            sector,
            dfe_updates,
            term_rows=affected_term_rows,
        )
        sector_state["term_products"][affected_term_rows] = -old_term_products

    state["sigma"][flip_idx] = -state["sigma"][flip_idx]
    state["fitness"] += delta_f


def relax_pspin(sigma0, J, sswm=True):
    """
    Relax the mixed p-spin model with the given interactions.
    """
    flip_sequence = []
    state = _initialize_relaxation_state(sigma0, J)

    while np.any(state["dfe"] > 0):
        flip_idx = _choose_flip_from_dfe(state["dfe"], sswm=sswm)
        flip_sequence.append(int(flip_idx))
        _apply_flip_to_relaxation_state(state, J, int(flip_idx))

    return flip_sequence
