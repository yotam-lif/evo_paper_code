import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from cmn.cmn_fgm import Fisher

def run_simulation(r, n, sig, m, max_steps):
    """Worker function for a single simulation repeat."""
    model = Fisher(n=n, sigma=sig, m=m, random_state=r)
    flips, traj, dfes = model.relax(max_steps=max_steps)
    return {
        'flips': flips,
        'traj': traj,
        'dfes': dfes
    }

if __name__ == "__main__":
    n = 32

    sig = 0.05
    max_steps = 1000
    repeats = 10 ** 3
    m = n * 10 ** 3

    # Use all available CPUs
    with ProcessPoolExecutor() as executor:
        # Map the worker function over the range of repeats
        results = list(executor.map(
            run_simulation,
            range(repeats),
            [n]*repeats,
            [sig]*repeats,
            [m]*repeats,
            [max_steps]*repeats
        ))

    # Save the results to a pickle file
    output_file = f'fgm_rps{repeats}_n{n}_sig{sig}.pkl'
    output_dir = '../FGM'
    output_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Data saved to {output_path}")