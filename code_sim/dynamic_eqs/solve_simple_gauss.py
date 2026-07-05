import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

from cmn.cmn_eqs import normalize, rhs

# Parameters
s_max, s_min = 100.0, -100.0
N_s = 600
t_min, t_max = 0.0, 5.0
c, D = 1.0, 4.0       # drift speed and diffusion coefficient
sig = 2.0             # initial Gaussian width
T_num = 200

# Spatial grid
s = np.linspace(s_min, s_max, N_s)
ds = s[1] - s[0]

# Initial condition: normalized Gaussian on s
p0 = np.exp(-((s - 0.0)**2) / (2 * sig**2))
p0 /= np.sum(p0) * ds

# Time points
t_eval = np.linspace(t_min, t_max, T_num)

# Solve ∂p/∂t = flip_term + drift_term + diff_term
sol = solve_ivp(
    fun=rhs,
    t_span=(t_min, t_max),
    y0=p0,
    t_eval=t_eval,
    args=(s, ds, c, D),
    method='RK45'
)

# Renormalize each snapshot so ∑p(s)ds = 1
p_all = np.apply_along_axis(lambda col: normalize(col, ds), 0, sol.y)

# Prepare plotting grid
x_min, x_max = -40, 40
N_x = int((x_max - x_min)/ds) + 1
x = np.linspace(x_min, x_max, N_x)

# Make output directory
output_dir = '../../plots/solve_transport_rw'
os.makedirs(output_dir, exist_ok=True)

# Plot at a few select times
plt.figure(figsize=(10, 6))
for idx in [0, len(t_eval)//4, len(t_eval)//2, 3*len(t_eval)//4, -1]:
    u = p_all[:, idx]
    plt.plot(x, np.interp(x, s, u), label=f't = {t_eval[idx]:.2f}')
plt.title('Solution of the PDE at Different Times')
plt.xlabel('Position $s$')
plt.ylabel('$p(s, t)$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'solution_plot_g.png'))
plt.close()
