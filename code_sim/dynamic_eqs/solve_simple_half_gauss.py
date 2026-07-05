import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from cmn.cmn_eqs import theta, rhs, normalize

# Parameters
s_max = 100.0  # Maximum s value
s_min = -s_max  # Minimum s value
N_s = 600  # Number of spatial grid points
t_min = 0.0  # Start time
t_max = 5.0  # End time
c = 1.0  # Speed of the drift term
D = 4.0
sig = 2.0  # Standard deviation
T_num = 200  # Number of time points
eps = 1e-5  # Small number to avoid division by zero

# Create directory if it doesn't exist
output_dir = '../../plots/solve_transport_rw'
os.makedirs(output_dir, exist_ok=True)

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]

x_max = 40
x_min = -x_max
N_x = int((x_max - x_min) / ds)
x = np.linspace(x_min, x_max, N_x)

# Time points where the solution is computed
t_eval = np.linspace(t_min, t_max, T_num)

# Initial condition: Gaussian centered at s = s0
s0 = 0.0  # Center of the Gaussian
p0_gauss = np.exp(-((s - s0) ** 2) / (2 * sig ** 2))
p0_gauss /= np.sum(p0_gauss) * ds  # Normalize
p0_neg = p0_gauss * theta(-s)

# Solve the PDE using solve_ivp with normalization after each step
solution_neg = solve_ivp(
    rhs, [t_min, t_max], p0_neg, t_eval=t_eval, method='RK45',
    vectorized=False, events=None
)
p_all_n = np.apply_along_axis(normalize, 0, solution_neg.y)

# -----------------------------
# Plotting the solution at selected times
# -----------------------------


plt.figure(figsize=(10, 6))
time_indices = [0, len(t_eval) // 4, len(t_eval) // 2, 3 * len(t_eval) // 4, -1]
for idx in time_indices:
    if idx < p_all_n.shape[1]:  # Ensure the index is within bounds
        sol = p_all_n[:, idx]
        sol_on_x = np.interp(x, s, sol)
        plt.plot(x, sol_on_x, label=f't = {t_eval[idx]:.3f}')
plt.title('Solution of the PDE at Different Times')
plt.xlabel('Position s')
plt.ylabel('Concentration p(s, t)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'solution_plot_n.png'))
plt.close()
