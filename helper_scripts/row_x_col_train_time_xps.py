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


def disable_e_cores():
    """Disable E-cores (cores 6 to nproc-1)"""
    try:
        nproc = int(subprocess.run(['nproc'], capture_output=True, text=True).stdout.strip())
        if nproc > 6:
            result = subprocess.run(['sudo', 'chcpu', '-d', f'6-{nproc-1}'], 
                                  capture_output=True, text=True, check=True)
            print(f"Disabled E-cores 6-{nproc-1}")
            if result.stdout: print(result.stdout.strip())
            if result.stderr: print(result.stderr.strip())
    except Exception as e:
        print(f"Warning: Could not disable E-cores: {e}")


def enable_e_cores():
    """Re-enable E-cores (cores 6-15)"""
    try:
        result = subprocess.run(['sudo', 'chcpu', '-e', '6-15'], 
                              capture_output=True, text=True, check=True)
        print("Re-enabled E-cores 6-15")
        if result.stdout: print(result.stdout.strip())
        if result.stderr: print(result.stderr.strip())
    except Exception as e:
        print(f"Warning: Could not re-enable E-cores: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", choices=["csv", "synthetic"], default="synthetic",
                        help="Experiment mode: 'csv' to load data via train_forest, 'rng' to generate via train_forest synthetic")
    parser.add_argument("--experiment_name", type=str, default="untitled_experiment",
                        help="Name for the experiment, used in the output CSV filename")
    # Runtime params.
    parser.add_argument("--numerical_split_type", type=str, choices=["Exact", "Random", "Equal Width"], required=True,
                        help="Whether to use Exact or Histogram splits")
    parser.add_argument("--num_threads", type=int, default=-1,
                        help="Number of threads to use. Use -1 for all logical CPUs.")
    parser.add_argument("--threads_list", type=int, nargs="+", default=None,
                        help="List of number of threads to test, e.g. --threads_list 1 2 4 8 16 32 64")
    parser.add_argument("--rows_list", type=int, nargs="+", default=[128, 256, 512,1024],
                        help="List of number of rows of the input matrix to test, e.g. --rows_list 128 256 512. Default: [128, 256, 512,1024]")
    parser.add_argument("--cols_list", type=int, nargs="+", default=[128,256,512,1024],
                        help="List of number of cols of the input matrix to test, e.g. --cols_list 128 256 512. Default: [128,256,512,1024,2048,4096]")
    parser.add_argument("--repeats", type=int, default=5,
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


def save_combined_matrix(avg_matrix, std_matrix, filepath, title_row=None):
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        if title_row:
            writer.writerow(title_row)
        
        # Create header row with averages and std columns (5 columns apart)
        header = ["n"] + d_values + [""] * 5 + [f"{d}_std" for d in d_values]
        writer.writerow(header)
        
        # Write data rows
        for n in n_values:
            row = [n]
            # Add average values
            row.extend([avg_matrix[n][d] for d in d_values])
            # Add 5 empty columns
            row.extend([""] * 5)
            # Add std values
            row.extend([std_matrix[n][d] for d in d_values])
            writer.writerow(row)


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
        threads_to_test = [args.num_threads]

    base_results_dir = os.path.join("ariel_results", "runtime_heatmap", get_cpu_model_proc())
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

        combined_csv = os.path.join(base_results_dir, f"{args.experiment_name}_{thread_count}.csv")

        # Initialize matrices
        avg_matrix = {n: {d: "" for d in d_values} for n in n_values}
        std_matrix = {n: {d: "" for d in d_values} for n in n_values}

        header = ["YDF Fisher-Yates", f"per-proj. nnz={args.projection_density_factor}", f"trees={args.num_trees}", f"{args.repeats} repeats", f"{args.tree_depth} depth", get_cpu_model_proc(), f"{str(t)} thread(s)"]

        if args.input_mode == "csv":
            # CSV mode static args
            static_args = [
                "--label_col=Target",
                f"--projection_density_factor={args.projection_density_factor}.0",
                f"--num_threads={thread_count}",
                f"--numerical_split_type={args.numerical_split_type}"
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
                            # Disable E-cores before running the binary
                            disable_e_cores()
                            
                            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                            
                            # Re-enable E-cores after the binary is done
                            enable_e_cores()
                            
                            m = time_rx.search(out)
                            if m:
                                t_measured = float(m.group(1))
                                times.append(t_measured)
                                print(f"  Run {i+1}: {t_measured:.4f} s")
                            else:
                                logging.error(f"  Run {i+1}: parse fail")
                        except subprocess.CalledProcessError as e:
                            # Make sure to re-enable E-cores even if binary fails
                            enable_e_cores()
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
                    save_combined_matrix(avg_matrix, std_matrix, combined_csv, header)

        else:  # rng mode
            static_args = [
                "--label_mod=2",
                f"--projection_density_factor={args.projection_density_factor}.0",
                f"--num_trees={args.num_trees}",
                f"--tree_depth={args.tree_depth}",
                f"--num_threads={thread_count}",
                f"--max_num_projections={args.max_num_projections}",
                f"--numerical_split_type={args.numerical_split_type}"
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
                            # Make sure to re-enable E-cores even if binary fails
                            enable_e_cores()
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
                    save_combined_matrix(avg_matrix, std_matrix, combined_csv, header)


if __name__ == "__main__":
    # Disable E-cores before running the binary
    disable_e_cores()
    main()
    # Re-enable E-cores after the binary is done
    enable_e_cores()