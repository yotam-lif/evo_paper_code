import numpy as np
from collections import defaultdict

class FitnessSampler:
    """Callable class to sample fitness values with given mean and std."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self):
        return np.float32(np.random.normal(self.mean, self.std))

class NK:
    """
    The NK model.
    N: Number of loci.
    K: Each locus's fitness depends on itself and its K neighbors.
    """

    def __init__(self, N, K, mean=0.0, std=1.0, seed=None):
        """
        Initialize the NK model.

        Parameters
        ----------
        N : int
            Number of loci.
        K : int
            Number of neighbors per locus.
        mean : float
            Mean of the Gaussian distribution used for fitness draws.
        std : float
            Standard deviation of the Gaussian distribution used for fitness draws.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.N = N
        self.K = K
        self.mean = mean
        self.std = std

        if seed is not None:
            np.random.seed(seed)

        # fis[i] is a dictionary where:
        # key: tuple of states (S_i, S_j1, ..., S_jK)
        # value: fitness value drawn from Gaussian
        # defaultdict takes care of sampling a new RV if the key is not in dictionary i
        draw_fitness = FitnessSampler(mean, std)
        self.fis = [defaultdict(draw_fitness) for _ in range(N)]
        # Precompute neighbor indices (circular)
        self.neighbor_indices = [
            [(i + offset) % self.N for offset in range(self.K + 1)] for i in range(self.N)
        ]
        # Precompute dependents for each locus, where dependents are the indices which fitness is affected by locus i
        self.dependents = [[] for _ in range(self.N)]
        for j in range(self.N):
            for i in self.neighbor_indices[j]:
                self.dependents[i].append(j)

    def compute_fitness(self, sigma):
        """
        Compute the total fitness of a given configuration.

        The fitness is the SUM of f_i over all loci i, i.e. an EXTENSIVE quantity.
        This matches the p-spin convention (:func:`cmn_pspin.compute_fit_slow` sums
        over all interactions without dividing), so fitness effects from the two
        models live on the same scale and can be compared directly.

        NOTE: this previously returned the *mean* (fit_sum / N), an intensive
        quantity ~N times smaller. Any dataset pickled under that older convention
        (e.g. the existing files in data/NK/) stores intensive effects and must be
        multiplied by N before being compared with the p-spin.

        Parameters
        ----------
        sigma : numpy.ndarray
            Array of locus states, typically -1 or +1.

        Returns
        -------
        numpy.float32
            The total fitness of the configuration
        """
        fit_sum = np.float32(0.0)
        for i in range(self.N):
            # Identify the pattern: locus i and its K neighbors (circular)
            # We'll consider the next K loci in a circular fashion.
            indices = self.neighbor_indices[i]
            kclique_i = tuple(int(sigma[idx]) for idx in indices)
            # defaultdict takes care of the case where kclique_i is not in contributions[i]
            fit_sum += self.fis[i][kclique_i]
        # The total fitness is the extensive sum of all f_i (no 1/N normalization)
        return np.float32(fit_sum)

    def get_fis(self):
        """
        Get the fitness values for all loci and patterns.

        Returns
        -------
        list:
            List of dictionaries where each dictionary contains the fitness values for a locus.
        """
        return self.fis

def compute_dfe(sigma, nk):
    """
    Compute the distribution of fitness effects (DFE) for a given configuration.

    Parameters
    ----------
    sigma : numpy.ndarray
        Array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.

    Returns
    -------
    numpy.ndarray(numpy.float32)
        The DFE for the given configuration.
    """
    dfe = np.zeros(nk.N, dtype=np.float32)
    curr_fit = nk.compute_fitness(sigma)
    for i in range(nk.N):
        # avoid excess copying of sigma prime by switching back each flip
        sigma[i] = -sigma[i]
        dfe[i] = nk.compute_fitness(sigma) - curr_fit
        sigma[i] = -sigma[i]
    return dfe


def compute_bdfe(dfe):
    """
    Compute the beneficial distribution of fitness effects (bDFE) for a given configuration.

    Parameters
    ----------
    dfe: numpy.ndarray

    Returns
    -------
    numpy.ndarray
        The bdfe for the given configuration.
    numpy.ndarray
        Indices of beneficial mutations.
    """
    bdfe = dfe[dfe > 0]
    b_ind = np.where(dfe > 0)[0]
    return bdfe, b_ind


def compute_rank(dfe):
    """
    Compute the rank of a given configuration.
    The rank is the number of beneficial mutations.

    Parameters
    ----------
    dfe: numpy.ndarray

    Returns
    -------
    int
        The rank of the configuration.
    """
    return np.sum(dfe > 0)


def sswm_choice(bdfe, b_ind):
    """
    Select a site for mutation using the Strong Selection Weak Mutation (SSWM) model.

    Parameters
    ----------
    bdfe: numpy.ndarray
        The beneficial distribution of fitness effects.
    b_ind: numpy.ndarray
        Indices of beneficial mutations.

    Returns
    -------
    int
        The index of the selected site for mutation.
    """
    sum_bdfe = np.sum(bdfe)
    if sum_bdfe != 0:
        bdfe = bdfe / sum_bdfe
    ind = np.random.choice(b_ind, p=bdfe)
    return ind

def relax_nk(sigma_init, nk):
    """
    Relax a given configuration using the NK model.

    Parameters
    ----------
    sigma_init : numpy.ndarray
        Initial array of locus states, typically -1 or +1.
    nk : NK
        An instance of the NK model.

    Returns
    -------
    numpy.ndarray
        The relaxed configuration.
    """
    sigma = np.copy(sigma_init)
    dfe = compute_dfe(sigma, nk)
    rank = compute_rank(dfe)
    flip_hist = []
    dfes = [dfe]
    while rank > 0:
        bdfe, b_ind = compute_bdfe(dfe)
        i = sswm_choice(bdfe, b_ind)
        flip_hist.append(i)
        sigma[i] = -sigma[i]
        dfe = compute_dfe(sigma, nk)
        dfes.append(dfe)
        rank = compute_rank(dfe)
    return flip_hist, dfes

def propagate_forward(dfe1, dfe2):
    # Find out the bdfe and it's indexes
    bdfe_anc, b_ind_anc = compute_bdfe(dfe1)
    prop_bdfe = dfe2[b_ind_anc]
    return bdfe_anc, prop_bdfe

def propagate_backward(dfe1, dfe2):
    # Find out the bdfe and it's indexes
    bdfe_evo, b_ind_evo = compute_bdfe(dfe2)
    prop_bdfe = dfe1[b_ind_evo]
    return bdfe_evo, prop_bdfe