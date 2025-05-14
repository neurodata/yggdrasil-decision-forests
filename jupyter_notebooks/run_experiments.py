import subprocess
import os
import re
import statistics
import csv
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["csv", "rng"], required=True, help="Experiment mode: 'csv' to load data from files, 'rng' to generate data")
args = parser.parse_args()

# Common paths and parameters
results_dir = "ariel_results"
avg_csv = os.path.join(results_dir, "matrix_avg_results.csv")
std_csv = os.path.join(results_dir, "matrix_std_results.csv")

# Grid
n_values = [128, 256, 512, 1024, 2048, 4096]
d_values = [128, 256, 512, 1024, 2048, 4096]

# Initialize results
avg_matrix = {n: {d: "" for d in d_values} for n in n_values}
std_matrix = {n: {d: "" for d in d_values} for n in n_values}

def save_matrix(matrix, filepath, title_row):
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title_row)
        writer.writerow(["n"] + d_values)
        for n in n_values:
            row = [n] + [matrix[n][d] for d in d_values]
            writer.writerow(row)

# Execution for CSV mode
if args.mode == "csv":
    data_dir = "ariel_test_data/random_csvs"
    binary_path = "./bazel-bin/examples/ariel_rf_train"
    n_runs = 3
    static_args = ["--label_col=Target", "--projection_density_factor=3.0", "--num_threads=96"]
    time_pattern = re.compile(r"Training time: ([\d.]+) seconds")
    filename_pattern = re.compile(r"random_n=(\d+)_d=(\d+)\.csv")
    header = ["YDF Built from Source", "d=1000", "nnz=3", "trees=50", f"{n_runs} repeats", "-1 depth", "<CPU>"]

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        match = filename_pattern.match(filename)
        if not match:
            print(f"Skipping unrecognized filename: {filename}")
            continue

        n_val, d_val = int(match.group(1)), int(match.group(2))
        if n_val not in n_values or d_val not in d_values:
            print(f"Skipping unexpected n/d: {n_val}/{d_val}")
            continue

        print(f"\nRunning on: {filename}")
        times = []
        csv_path = os.path.join(data_dir, filename)

        for i in range(n_runs):
            cmd = [binary_path, f"--train_csv={csv_path}"] + static_args
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                match = time_pattern.search(output)
                if match:
                    time_sec = float(match.group(1))
                    times.append(time_sec)
                    print(f"  Run {i+1}: {time_sec:.4f} seconds")
                else:
                    print(f"  Run {i+1}: Failed to parse time")
            except subprocess.CalledProcessError as e:
                print(f"  Run {i+1}: Error running command\n{e.output}")

        if times:
            avg = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"  ➤ Avg: {avg:.4f} s | Std: {std:.4f} s")
            avg_matrix[n_val][d_val] = f"{avg:.4f}"
            std_matrix[n_val][d_val] = f"{std:.4f}"
        else:
            print("  ➤ All runs failed.")

# Execution for rng mode
elif args.mode == "rng":
    binary_path = "./bazel-bin/examples/synthetic_matrix_train"
    n_runs = 7
    static_args = [
        "--label_mod=2",
        "--projection_density_factor=3.0",
        "--trees=50",
        "--depth=-1",
        "--threads=96",
        "--max_num_projections=1000",
        "--num_projections_exponent=1"
    ]
    header = ["YDF with in-file RNG-generated Matrices", "d=1000 (HARD-CODED!)", "nnz=3", "trees=50", f"{n_runs} repeats", "-1 depth", "<CPU>"]
    time_pattern = re.compile(r"Ariel Training time: ([\d.]+) s")

    for n in n_values:
        for d in d_values:
            print(f"\nRunning: rows={n}, cols={d}")
            times = []
            for i in range(n_runs):
                cmd = [binary_path, f"--rows={n}", f"--cols={d}"] + static_args
                try:
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    match = time_pattern.search(output)
                    if match:
                        t = float(match.group(1))
                        times.append(t)
                        print(f"  Run {i+1}: {t:.4f} seconds")
                    else:
                        print(f"  Run {i+1}: Failed to parse time")
                except subprocess.CalledProcessError as e:
                    print(f"  Run {i+1}: Error\n{e.output}")

            if times:
                avg = statistics.mean(times)
                std = statistics.stdev(times) if len(times) > 1 else 0.0
                print(f"  ➤ Avg: {avg:.4f} s | Std: {std:.4f} s")
                avg_matrix[n][d] = f"{avg:.4f}"
                std_matrix[n][d] = f"{std:.4f}"
            else:
                print("  ➤ All runs failed.")

# Save final matrices
save_matrix(avg_matrix, avg_csv, header)
save_matrix(std_matrix, std_csv, header)
