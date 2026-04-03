import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmn import cmn
from cmn import cmn_pspin


def generate_single_data_pspin(N: int, P: int, seed: int | None = None) -> dict:
    """
    Generate a single mixed p-spin simulation dataset.

    Parameters
    ----------
    N : int
        Number of spins.
    P : int
        Maximum interaction order in the mixed p-spin model.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
          - 'init_sigma': Initial spin configuration.
          - 'J': Mixed p-spin interaction data.
          - 'flip_seq': List of spin indices flipped during the SSWM walk.
    """
    if seed is not None:
        np.random.seed(seed)

    init_sigma = cmn.init_sigma(N).astype(np.int8, copy=False)
    J = cmn_pspin.init_J(N, P, random_state=seed)
    flip_seq = cmn_pspin.relax_pspin(init_sigma, J, sswm=True)

    return {
        "init_sigma": init_sigma,
        "J": J,
        "flip_seq": flip_seq,
    }


def generate_data_pspin(
    N: int,
    P: int,
    n_repeats: int,
    output_dir: str,
    seed: int | None = None,
    max_workers: int | None = None,
) -> None:
    """
    Generate multiple mixed p-spin datasets in parallel and save them to a pickle file.

    Parameters
    ----------
    N : int
        Number of spins.
    P : int
        Maximum interaction order in the mixed p-spin model.
    n_repeats : int
        Number of independent repeats to simulate.
    output_dir : str
        Directory where the output pickle will be saved.
    seed : int, optional
        Seed used to generate independent worker seeds.
    max_workers : int, optional
        Number of worker processes for parallel generation.
    """
    parent_rng = np.random.default_rng(seed)
    repeat_seeds = parent_rng.integers(0, 2**32, size=n_repeats, dtype=np.uint32)

    if max_workers == 1:
        data = [
            generate_single_data_pspin(N, P, int(repeat_seed))
            for repeat_seed in repeat_seeds
        ]
    else:
        try:
            data = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(generate_single_data_pspin, N, P, int(repeat_seed))
                    for repeat_seed in repeat_seeds
                ]
                for future in futures:
                    data.append(future.result())
        except PermissionError:
            data = [
                generate_single_data_pspin(N, P, int(repeat_seed))
                for repeat_seed in repeat_seeds
            ]

    os.makedirs(output_dir, exist_ok=True)

    filename = f"N{N}_P{P}_repeats{n_repeats}.pkl"
    output_file = os.path.join(output_dir, filename)

    with open(output_file, "wb") as handle:
        pickle.dump(data, handle)

    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mixed p-spin model simulation data (SSWM adaptive walks)."
    )
    parser.add_argument("--N", type=int, required=True, help="Number of spins")
    parser.add_argument(
        "--P",
        "--p",
        dest="P",
        type=int,
        required=True,
        help="Maximum interaction order in the mixed p-spin model",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        required=True,
        help="Number of independent simulations to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output pickle file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible data generation",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Optional number of worker processes",
    )

    args = parser.parse_args()
    generate_data_pspin(
        args.N,
        args.P,
        args.n_repeats,
        args.output_dir,
        seed=args.seed,
        max_workers=args.max_workers,
    )
