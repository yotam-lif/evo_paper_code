import math
from itertools import combinations

import numpy as np


FLOAT_DTYPE = np.float32
SPIN_DTYPE = np.int8


# ---------------------------------------------------------------------------
# Spin configurations and flip histories
# ---------------------------------------------------------------------------

def init_sigma(N):
    """
    Initialize a random spin configuration for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.

    Returns
    -------
    numpy.ndarray
        The spin configuration.
    """
    return np.random.choice([-1, 1], N)


def compute_sigma_from_hist(sigma_0, hist, t=None):
    """
    Compute sigma from the initial sigma and the flip history up to flip number 't'.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history - the indices of spins flipped, in order.
    t : int
        The flip number up to which to compute the spin configuration. The default is None.

    Returns
    -------
    numpy.ndarray
        The spin configuration after t flips.
    """
    sigma = np.copy(sigma_0)
    if t is None:
        rel_hist = hist
    else:
        rel_hist = hist[:t]
    for flip in rel_hist:
        sigma[flip] *= -1
    return sigma


def curate_sigma_list(sigma_0, hist, ts):
    """
    Curate the sigma list to have num_points elements.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history - the indices of spins flipped, in order.
    ts : list of int
        The indices of flips in 'hist' to recreate sigma at.

    Returns
    -------
    list
        The curated list of spin configurations.
    """
    sigma_list = []
    for t in ts:
        sigma_t = compute_sigma_from_hist(sigma_0, hist, t)
        sigma_list.append(sigma_t)
    return sigma_list


# ---------------------------------------------------------------------------
# Small dtype / casting helpers
# ---------------------------------------------------------------------------

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


def compute_energy(sigma, J, h_off=0.0):
    """
    Compute the energy (Hamiltonian value) of configuration σ.

        energy = Σ_interactions  J_{i1...ip} σ_{i1} ... σ_{ip}  -  h_off
    """
    sigma = _as_spin_array(sigma)
    energy = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        spin_products = _compute_spin_products(sigma, sector)
        energy += np.dot(sector["couplings"], spin_products)

    return float(energy - FLOAT_DTYPE(h_off))


def compute_energy_off(sigma_init, J):
    """Compute the energy offset so that energy(sigma_init) = 1."""
    return compute_energy(sigma_init, J) - 1


def compute_energy_delta_flip(sigma, J, k):
    """
    Compute the energy change ΔH when spin k is flipped.

    Only interactions containing site k are affected, so we restrict
    the sum to those terms for efficiency.
    """
    sigma = _as_spin_array(sigma)
    delta = FLOAT_DTYPE(0.0)

    for sector in J["sectors"]:
        affected = sector["site_to_interactions"][k]
        if affected.size == 0:
            continue

        spin_products = _compute_spin_products(sigma, sector, interaction_idx=affected)
        delta += np.sum(
            -FLOAT_DTYPE(2.0) * sector["couplings"][affected] * spin_products,
            dtype=FLOAT_DTYPE,
        )

    return float(delta)


def compute_flip_spectrum(sigma, J):
    """Compute the single-flip spectrum: the energy change ΔH_i for flipping each spin i."""
    sigma = _as_spin_array(sigma)
    return (-FLOAT_DTYPE(2.0) * sigma * compute_lfs(sigma, J)).astype(FLOAT_DTYPE, copy=False)


def compute_positive_spectrum(sigma, J):
    """Return (positive single-flip effects, their site indices)."""
    spectrum = compute_flip_spectrum(sigma, J)
    return _extract_positive(spectrum)


def _extract_positive(spectrum):
    """Extract positive entries and their indices from a flip spectrum."""
    mask = spectrum > 0
    return spectrum[mask], np.flatnonzero(mask)


def compute_normalized_positive_spectrum(sigma, J):
    """Return the positive single-flip spectrum normalized to a probability distribution."""
    positive, p_ind = compute_positive_spectrum(sigma, J)
    norm = np.sum(positive, dtype=FLOAT_DTYPE)
    if norm > 0:
        positive = positive / norm
    return positive.astype(FLOAT_DTYPE, copy=False), p_ind


def count_positive_flips(sigma, J):
    """Count the number of flips that increase the energy."""
    spectrum = compute_flip_spectrum(sigma, J)
    return int(np.count_nonzero(spectrum > 0))


# ---------------------------------------------------------------------------
# Spin-flip selection
# ---------------------------------------------------------------------------

def _choose_positive_flip(spectrum, weighted=True):
    """
    Choose a spin to flip from the energy-increasing flips in the spectrum.

    If weighted=True, flip probability is proportional to ΔH. Otherwise,
    pick uniformly among the energy-increasing flips.
    """
    positive, positive_sites = _extract_positive(spectrum)
    if weighted:
        probs = positive / np.sum(positive, dtype=FLOAT_DTYPE)
        return np.random.choice(positive_sites, p=probs)
    return np.random.choice(positive_sites)


def weighted_flip(sigma, J):
    """Choose a spin to flip with probability proportional to its energy gain."""
    return _choose_positive_flip(compute_flip_spectrum(sigma, J), weighted=True)


# ---------------------------------------------------------------------------
# Greedy walk (relaxation)
# ---------------------------------------------------------------------------

def _initialize_relaxation_state(sigma0, J):
    """
    Set up the cached state for a greedy walk: the current spin
    configuration, its energy, the full flip spectrum, and per-sector cached
    spin products (to allow incremental updates on each flip).
    """
    sigma = _as_spin_array(sigma0, copy=True)
    spectrum = np.zeros(J["N"], dtype=FLOAT_DTYPE)
    energy = FLOAT_DTYPE(0.0)
    sector_caches = []

    for sector in J["sectors"]:
        spin_products = _compute_spin_products(sigma, sector)
        energy += np.dot(sector["couplings"], spin_products)
        spectrum_contributions = -FLOAT_DTYPE(2.0) * sector["couplings"] * spin_products
        _scatter_to_sites(spectrum, sector, spectrum_contributions)
        sector_caches.append({"spin_products": spin_products})

    return {"sigma": sigma, "spectrum": spectrum, "energy": energy, "sector_caches": sector_caches}


def _apply_flip(state, J, flip_site):
    """
    Flip spin at *flip_site* and incrementally update the energy, flip
    spectrum, and cached spin products.

    Only interactions containing flip_site need recomputation.
    """
    delta = state["spectrum"][flip_site]

    for sector, cache in zip(J["sectors"], state["sector_caches"]):
        affected = sector["site_to_interactions"][flip_site]
        if affected.size == 0:
            continue

        old_spin_products = cache["spin_products"][affected]
        # Flipping one spin negates all spin products containing it,
        # which shifts the spectrum by 4 * J * (old product) at each site.
        spectrum_updates = FLOAT_DTYPE(4.0) * sector["couplings"][affected] * old_spin_products

        _scatter_to_sites(state["spectrum"], sector, spectrum_updates, interaction_idx=affected)
        cache["spin_products"][affected] = -old_spin_products

    state["sigma"][flip_site] = -state["sigma"][flip_site]
    state["energy"] += delta


def relax_pspin(sigma0, J, weighted=True):
    """
    Run a greedy walk until no energy-increasing flips remain.

    Returns the sequence of flipped sites.
    """
    flip_sequence = []
    state = _initialize_relaxation_state(sigma0, J)

    while np.any(state["spectrum"] > 0):
        flip_site = _choose_positive_flip(state["spectrum"], weighted=weighted)
        flip_sequence.append(int(flip_site))
        _apply_flip(state, J, int(flip_site))

    return flip_sequence


# ---------------------------------------------------------------------------
# Object-oriented interface
# ---------------------------------------------------------------------------

class PSpin:
    """
    Mixed / pure p-spin model as an object.

    Bundles the landscape (the interaction sectors ``J``), the current
    spin configuration (state) and an energy offset ``h_off`` into one object.
    The offset is stored on the model and applied on every energy computation,
    so that -- once :meth:`set_offset` has pinned it -- the initial
    configuration has energy 1.

    The landscape is kept in the same dict layout produced by :func:`init_J`
    (exposed as the read-only :attr:`J` view), so every module-level function in
    this file remains usable with ``model.J``.

    Attributes
    ----------
    N, P, pure : landscape dimensions / flags (see :func:`init_J`).
    sectors : list of interaction sectors (the couplings J).
    sigma : current spin configuration (state).
    h_off : energy offset subtracted on every energy computation.
    """

    def __init__(self, N, P, sigma_init=None, random_state=None, pure=False):
        model = init_J(N, P, random_state=random_state, pure=pure)
        self.N = model["N"]
        self.P = model["P"]
        self.pure = model["pure"]
        self.sectors = model["sectors"]
        self.h_off = FLOAT_DTYPE(0.0)
        if sigma_init is not None:
            self.sigma = _as_spin_array(sigma_init, copy=True)
            self.set_offset(self.sigma)
        else:
            self.sigma = None

    @property
    def J(self):
        """Read-only dict view of the landscape, as returned by :func:`init_J`."""
        return {"N": self.N, "P": self.P, "pure": self.pure, "sectors": self.sectors}

    def set_offset(self, sigma_init):
        """Store the offset so that ``compute_energy(sigma_init) == 1``."""
        self.h_off = FLOAT_DTYPE(compute_energy_off(sigma_init, self.J))
        return self.h_off

    def compute_energy(self, sigma):
        """Energy of ``sigma``, with the stored offset applied."""
        return compute_energy(sigma, self.J, self.h_off)

    def compute_flip_spectrum(self, sigma):
        """Single-flip spectrum at ``sigma``."""
        return compute_flip_spectrum(sigma, self.J)

    def compute_positive_spectrum(self, sigma):
        """Positive single-flip spectrum (values, indices) at ``sigma``."""
        return compute_positive_spectrum(sigma, self.J)

    def count_positive_flips(self, sigma):
        """Number of energy-increasing flips at ``sigma``."""
        return count_positive_flips(sigma, self.J)

    def relax(self, sigma0=None, weighted=True):
        """Run a greedy walk, updating :attr:`sigma`. Returns flip sequence."""
        if sigma0 is None:
            if self.sigma is None:
                raise ValueError("No configuration to relax: pass sigma0 or set self.sigma.")
            sigma0 = self.sigma
        flip_sequence = relax_pspin(sigma0, self.J, weighted=weighted)
        sigma = _as_spin_array(sigma0, copy=True)
        for site in flip_sequence:
            sigma[site] = -sigma[site]
        self.sigma = sigma
        return flip_sequence