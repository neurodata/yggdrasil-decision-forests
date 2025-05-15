import subprocess
import os
import re
import statistics
import csv
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


# Argument parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csv", "synthetic"], required=True,
                        help="Experiment mode: 'csv' to load data via train_forest, 'rng' to generate via train_forest synthetic")
    parser.add_argument("--threads", type=int, default=-1,
                        help="Number of threads to use. Use -1 for all logical CPUs.")
    parser.add_argument("--threads_list", type=int, nargs="+", default=None,
                        help="List of number of threads to test, e.g. --threads_list 1 2 4 8 16 32 64")
    return parser.parse_args()


# Grid definitions
global n_values, d_values
n_values = [128, 256, 512, 1024, 2048, 4096]
d_values = [128, 256, 512, 1024, 2048, 4096]


# Save helper
def save_matrix(matrix, filepath, title_row=None):
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        if title_row:
            writer.writerow(title_row)
        writer.writerow(["n"] + d_values)
        for n in n_values:
            writer.writerow([n] + [matrix[n][d] for d in d_values])


def main():
    args = get_args()

    # Build the list of thread configurations to run
    if args.threads_list is not None:
        threads_to_test = args.threads_list
    else:
        threads_to_test = [args.threads]

    base_results_dir = "ariel_results"
    os.makedirs(base_results_dir, exist_ok=True)
    binary = "./bazel-bin/examples/train_oblique_forest"

    for t in threads_to_test:
        # resolve -1 to all logical CPUs
        if t == -1:
            thread_count = os.cpu_count()
            logging.info(f"\n\nUnlimited threads requested. Found {thread_count} logical CPUs. Calling train_oblique_forest with --threads={thread_count}\n\n")
        else:
            thread_count = t

        avg_csv = os.path.join(base_results_dir, f"matrix_avg_results_{thread_count}.csv")
        std_csv = os.path.join(base_results_dir, f"matrix_std_results_{thread_count}.csv")

        # Initialize matrices
        avg_matrix = {n: {d: "" for d in d_values} for n in n_values}
        std_matrix = {n: {d: "" for d in d_values} for n in n_values}

        # Pre-create result files with headers
        with open(avg_csv, "w", newline='') as f:
            csv.writer(f).writerow(["n"] + d_values)
        with open(std_csv, "w", newline='') as f:
            csv.writer(f).writerow(["n"] + d_values)

        if args.mode == "csv":
            # CSV mode static args
            static_args = [
                "--label_col=Target",
                "--projection_density_factor=3.0",
                f"--num_threads={thread_count}"
            ]
            time_rx = re.compile(r"Training time: ([\d.]+) seconds")
            header = ["YDF Built from Source", "d=1000", "nnz=3", "trees=50", "3 repeats", "-1 depth", "<CPU>"]

            data_dir = "ariel_test_data/random_csvs"
            for n in n_values:
                for d in d_values:
                    filename = f"random_n={n}_d={d}.csv"
                    path = os.path.join(data_dir, filename)
                    if not os.path.exists(path):
                        logging.warning(f"Skipping missing file: {filename}")
                        continue
                    print(f"\nRunning on: {filename}")
                    times = []
                    for i in range(3):
                        cmd = [binary, "--input_mode=csv", f"--train_csv={path}"] + static_args
                        try:
                            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                            m = time_rx.search(out)
                            if m:
                                t_measured = float(m.group(1))
                                times.append(t_measured)
                                print(f"  Run {i+1}: {t_measured:.4f} s")
                            else:
                                logging.error(f"  Run {i+1}: parse fail")
                        except subprocess.CalledProcessError as e:
                            logging.error(f"  Run {i+1}: error\n{e.output}")
                    if times:
                        avg = statistics.mean(times)
                        std = statistics.stdev(times) if len(times) > 1 else 0.0
                        avg_matrix[n][d] = f"{avg:.4f}"
                        std_matrix[n][d] = f"{std:.4f}"
                        print(f"  ➤ Avg: {avg:.4f}s | Std: {std:.4f}s")
                    else:
                        logging.critical("  ➤ All runs failed.")

                    # Save after each cell
                    save_matrix(avg_matrix, avg_csv, header)
                    save_matrix(std_matrix, std_csv, header)

        else:  # rng mode
            static_args = [
                "--label_mod=2",
                "--projection_density_factor=3.0",
                "--num_trees=50",
                "--tree_depth=-1",
                f"--num_threads={thread_count}",
                "--max_num_projections=1000",
                "--num_projections_exponent=1"
            ]
            time_rx = re.compile(r"Training wall-time: ([\d.]+)s")
            header = ["YDF RNG-generated", "d variable", "nnz=3", "trees=50", "7 repeats", "-1 depth", "<CPU>"]

            for n in n_values:
                for d in d_values:
                    print(f"\nRunning: rows={n}, cols={d}")
                    times = []
                    for i in range(7):
                        cmd = [binary, "--input_mode=synthetic", f"--rows={n}", f"--cols={d}"] + static_args
                        try:
                            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                            m = time_rx.search(out)
                            if m:
                                t_measured = float(m.group(1))
                                times.append(t_measured)
                                print(f"  Run {i+1}: {t_measured:.4f} s")
                            else:
                                logging.error(f"  Run {i+1}: parse fail")
                        except subprocess.CalledProcessError as e:
                            logging.error(f"  Run {i+1}: error\n{e.output}")
                    if times:
                        avg = statistics.mean(times)
                        std = statistics.stdev(times) if len(times) > 1 else 0.0
                        avg_matrix[n][d] = f"{avg:.4f}"
                        std_matrix[n][d] = f"{std:.4f}"
                        print(f"  ➤ Avg: {avg:.4f}s | Std: {std:.4f}s")
                    else:
                        logging.critical("  ➤ All runs failed.")

                    # Save after each cell
                    save_matrix(avg_matrix, avg_csv, header)
                    save_matrix(std_matrix, std_csv, header)


if __name__ == "__main__":
    main()
