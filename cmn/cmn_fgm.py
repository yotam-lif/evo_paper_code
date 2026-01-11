import numpy as np

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

    def relax(self, max_steps=1000):
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
            self.deltas[choice] = -1 * self.deltas[choice]
            traj.append(self.r.copy())
        return flips, traj, dfes
