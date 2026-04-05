import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "PSPIN"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmn import cmn
from cmn import cmn_pspin


def generate_single_data_pspin(
    N: int,
    P: int,
    seed: int | None = None,
    pure: bool = False,
) -> dict:
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
    pure : bool, optional
        If True, generate a pure P-spin model. Default is False, which
        generates the mixed model with orders 1 through P.

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
    J = cmn_pspin.init_J(N, P, random_state=seed, pure=pure)
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
    seed: int | None = 1,
    max_workers: int | None = None,
    pure: bool = False,
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
    pure : bool, optional
        If True, generate a pure P-spin model. Default is False, which
        generates the mixed model with orders 1 through P.
    """
    if max_workers is None:
        max_workers = n_repeats

    parent_rng = np.random.default_rng(seed)
    repeat_seeds = parent_rng.integers(0, 2**32, size=n_repeats, dtype=np.uint32)

    if max_workers == 1:
        data = [
            generate_single_data_pspin(N, P, int(repeat_seed), pure=pure)
            for repeat_seed in repeat_seeds
        ]
    else:
        try:
            data = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(generate_single_data_pspin, N, P, int(repeat_seed), pure)
                    for repeat_seed in repeat_seeds
                ]
                for future in futures:
                    data.append(future.result())
        except PermissionError:
            data = [
                generate_single_data_pspin(N, P, int(repeat_seed), pure=pure)
                for repeat_seed in repeat_seeds
            ]

    os.makedirs(output_dir, exist_ok=True)

    model_label = "pure" if pure else "mixed"
    filename = f"N{N}_P{P}_{model_label}_repeats{n_repeats}.pkl"
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
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Default is the sibling PSPIN directory next to this script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for reproducible data generation. Default is 1.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of worker processes. Default is n_repeats.",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="If set, generate a pure P-spin model instead of the default mixed model.",
    )

    args = parser.parse_args()
    generate_data_pspin(
        args.N,
        args.P,
        args.n_repeats,
        args.output_dir,
        seed=args.seed,
        max_workers=args.max_workers,
        pure=args.pure,
    )
