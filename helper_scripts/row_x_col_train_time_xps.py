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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", choices=["csv", "synthetic"], default="synthetic",
                        help="Experiment mode: 'csv' to load data via train_forest, 'rng' to generate via train_forest synthetic")
    parser.add_argument("--sort_method", choices=["SortFeature", "SortIndex"], default="SortFeature",
                        help="Use SortIndex to save the results to sort_index dir instead of regular 'SortFeature'. Has no effect on C++ binary")
    # Runtime params.
    parser.add_argument("--threads", type=int, default=-1,
                        help="Number of threads to use. Use -1 for all logical CPUs.")
    parser.add_argument("--threads_list", type=int, nargs="+", default=None,
                        help="List of number of threads to test, e.g. --threads_list 1 2 4 8 16 32 64")
    parser.add_argument("--rows_list", type=int, nargs="+", default=[128, 256, 512,1024],
                        help="List of number of rows of the input matrix to test, e.g. --rows_list 128 256 512. Default: [128, 256, 512,1024]")
    parser.add_argument("--cols_list", type=int, nargs="+", default=[128,256,512,1024,2048,4096],
                        help="List of number of cols of the input matrix to test, e.g. --cols_list 128 256 512. Default: [128,256,512,1024,2048,4096]")
    parser.add_argument("--repeats", type=int, default=7,
                        help="Number of times to repeat & avg. experiments. Default: 7")
    
    # Model params
    parser.add_argument("--num_trees", type=int, default=50,
                        help="Number of trees in the Random Forest. Default: 50")
    parser.add_argument("--tree_depth", type=int, default=-1,
                        help="Limit depth of trees in Random Forest. Default: -1 (Unlimited)")
    parser.add_argument("--projection_density_factor", type=int, default=3,
                    help="Number of nonzeros per projection. Default: 3")
    parser.add_argument("--max_num_projections", type=int, default=1000,
                    help="Maximum number of projections. WARNING: YDF doesn't always obey this! Default: 1000")

    return parser.parse_args()


def save_matrix(matrix, filepath, title_row=None):
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        if title_row:
            writer.writerow(title_row)
        writer.writerow(["n"] + d_values)
        for n in n_values:
            writer.writerow([n] + [matrix[n][d] for d in d_values])


def get_cpu_model_proc():
    """
    Reads /proc/cpuinfo and returns the first 'model name' value.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    # split only on the first ':' → [key, value]
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return "Could not access /proc/cpuinfo to get CPU model name"
    

def main():
    args = get_args()
    
    global n_values, d_values
    n_values = args.rows_list
    d_values = args.cols_list

    if args.threads_list is not None:
        threads_to_test = args.threads_list
    else:
        threads_to_test = [args.threads]

    base_results_dir = os.path.join("ariel_results", "runtime_heatmap", get_cpu_model_proc(), args.sort_method)
    os.makedirs(base_results_dir, exist_ok=True)
    binary = "./bazel-bin/examples/train_oblique_forest"

    for t in threads_to_test:
        print("Running w/ Threads=", t)
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

        header = ["YDF Fisher-Yates", f"per-proj. nnz={args.projection_density_factor}", f"trees={args.num_trees}", f"{args.repeats} repeats", f"{args.tree_depth} depth", get_cpu_model_proc(), f"{str(t)} thread(s)"]

        if args.input_mode == "csv":
            # CSV mode static args
            static_args = [
                "--label_col=Target",
                f"--projection_density_factor={args.projection_density_factor}.0",
                f"--num_threads={thread_count}"
            ]
            time_rx = re.compile(r"Training time: ([\d.]+) seconds")
            

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
                    for i in range(args.repeats):
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
                f"--projection_density_factor={args.projection_density_factor}.0",
                f"--num_trees={args.num_trees}",
                f"--tree_depth={args.tree_depth}",
                f"--num_threads={thread_count}",
                f"--max_num_projections={args.max_num_projections}",
            ]
            time_rx = re.compile(r"Training wall-time: ([\d.]+)s")

            for n in n_values:
                for d in d_values:
                    print(f"\nRunning: rows={n}, cols={d}")
                    times = []
                    for i in range(args.repeats):
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
