import subprocess
import os
import re
import statistics
import csv

# Paths
data_dir = "ariel_test_data/random_csvs"
results_dir = "ariel_results"
binary_path = "./bazel-bin/examples/ariel_rf_train"
avg_csv = os.path.join(results_dir, "matrix_avg_results.csv")
std_csv = os.path.join(results_dir, "matrix_std_results.csv")
n_runs = 3

# Flags and regex
static_args = ["--label_col=Target", "--projection_density_factor=3.0", "--num_threads=96"]
time_pattern = re.compile(r"Training time: ([\d.]+) seconds")
filename_pattern = re.compile(r"random_n=(\d+)_d=(\d+)\.csv")

# Grid values
n_values = [128, 256, 512, 1024, 2048, 4096]
d_values = [128, 256, 512, 1024, 2048, 4096]

# Init empty matrices
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

# Initial save of empty files
header = ["YDF Built from Source", "d=1000", "nnz=3", "trees=50", "3 repeats", "-1 depth", " <CPU> "]
save_matrix(avg_matrix, avg_csv, header)
save_matrix(std_matrix, std_csv, header)

# Process each file
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".csv"):
        continue

    match = filename_pattern.match(filename)
    if not match:
        print(f"Skipping unrecognized filename: {filename}")
        continue

    n_val = int(match.group(1))
    d_val = int(match.group(2))

    if n_val not in n_values or d_val not in d_values:
        print(f"Skipping unexpected n/d: {n_val}/{d_val}")
        continue

    csv_path = os.path.join(data_dir, filename)
    print(f"\nRunning on: {filename}")
    times = []

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

        # Update matrices
        avg_matrix[n_val][d_val] = f"{avg:.4f}"
        std_matrix[n_val][d_val] = f"{std:.4f}"

        # Save immediately
        save_matrix(avg_matrix, avg_csv, header)
        save_matrix(std_matrix, std_csv, header)
    else:
        print("  ➤ All runs failed.")
