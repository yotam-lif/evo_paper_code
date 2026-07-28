import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
from concurrent.futures import ProcessPoolExecutor
from cmn.cmn_fgm import Fisher

def run_simulation(
    r,
    n,
    sig,
    m,
    max_steps,
    mutation_distribution="gaussian",
    mu=0.5,
    initial_radius=None,
):
    """Worker function for a single simulation repeat."""
    model = Fisher(
        n=n,
        sigma=sig,
        m=m,
        random_state=r,
        mutation_distribution=mutation_distribution,
        mu=mu,
        initial_radius=initial_radius,
    )
    flips, traj, dfes = model.relax(max_steps=max_steps)
    return {
        'flips': flips,
        'traj': traj,
        'dfes': dfes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate FGM model data.')
    parser.add_argument('--n', type=int, required=True, help='Number of dimensions')
    parser.add_argument('--sig', type=float, default=0.05, help='Mutation size sigma')
    parser.add_argument('--m', type=int, required=True, help='Number of loci')
    parser.add_argument('--repeats', type=int, required=True, help='Number of repeats')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max relaxation steps')
    parser.add_argument(
        '--mutation-distribution',
        choices=('gaussian', 'heavy_tailed'),
        default='gaussian',
    )
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument(
        '--initial-radius',
        type=float,
        default=None,
        help='Default: sqrt(n), as in the old simulations',
    )
    parser.add_argument('--output-dir', default='../FGM')
    args = parser.parse_args()

    n = args.n
    sig = args.sig
    max_steps = args.max_steps
    repeats = args.repeats
    m = args.m

    # Use all available CPUs
    with ProcessPoolExecutor() as executor:
        # Map the worker function over the range of repeats
        results = list(executor.map(
            run_simulation,
            range(repeats),
            [n]*repeats,
            [sig]*repeats,
            [m]*repeats,
            [max_steps]*repeats,
            [args.mutation_distribution]*repeats,
            [args.mu]*repeats,
            [args.initial_radius]*repeats,
        ))

    # Save the results to a pickle file
    if args.mutation_distribution == 'gaussian' and args.initial_radius is None:
        output_file = f'fgm_rps{repeats}_n{n}_sig{sig}.pkl'
    else:
        output_file = (
            f'fgm_rps{repeats}_n{n}_m{m}_sig{sig}_'
            f'{args.mutation_distribution}_mu{args.mu}_r{args.initial_radius}.pkl'
        )
    output_dir = args.output_dir
    output_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Data saved to {output_path}")
