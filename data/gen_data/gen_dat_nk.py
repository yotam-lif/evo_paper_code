import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from cmn import cmn, cmn_nk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gen_dat_nk.log"),
        logging.StreamHandler()
    ]
)

# Create the output directory if it doesn't exist
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../NK'))
os.makedirs(output_dir, exist_ok=True)


def process(N, k):
    """Simulates the NK model and returns the flip sequence and the model state."""
    try:
        init_sigma = cmn.init_sigma(N)
        NK_init = cmn_nk.NK(N, k)
        flip_seq, dfes = cmn_nk.relax_nk(init_sigma, NK_init)
        return {'init_sigma': init_sigma, 'flip_seq': flip_seq, 'dfes': dfes}
    except Exception as e:
        logging.error(f"Error in process(N={N}, k={k}): {e}")
        return None


def main(N, k, num_repeats, num_workers):
    """Main function to run the simulation in parallel using ProcessPoolExecutor."""
    logging.info(f"Starting simulation with N={N}, k={k}, num_repeats={num_repeats}, num_workers={num_workers}")
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process, N, k) for _ in range(num_repeats)]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Save the results to a pickle file
    output_path = os.path.join(output_dir, f'N_{N}_K_{k}_repeats_{num_repeats}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"Simulation completed. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NK model gen_data.')
    parser.add_argument('--N', type=int, required=True, help='Number of loci')
    parser.add_argument('--K', type=int, required=True, help='Number of neighbors per locus')
    parser.add_argument('--num_repeats', type=int, required=True, help='Number of repeats')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of workers for parallel processing')

    args = parser.parse_args()

    # Ensure num_workers does not exceed available cores
    num_workers = min(args.num_workers, multiprocessing.cpu_count())
    logging.info(f"Using {num_workers} workers.")

    main(args.N, args.K, args.num_repeats, num_workers)
