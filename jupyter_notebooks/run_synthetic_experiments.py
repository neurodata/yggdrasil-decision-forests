import subprocess
import os
import re
import statistics
import csv

# Paths
results_dir = "ariel_results"
binary_path = "./bazel-bin/examples/synthetic_matrix_train"
avg_csv = os.path.join(results_dir, "matrix_avg_results.csv")
std_csv = os.path.join(results_dir, "matrix_std_results.csv")
n_runs = 7

# Flag arguments for all runs
static_args = [
    "--label_mod=2",
    "--projection_density_factor=3.0",
    "--trees=50",
    "--depth=-1",
    "--threads=96",
    "--max_num_projections=1000",
    "--num_projections_exponent=1"
]

# Grid values
n_values = [128, 256, 512, 1024, 2048, 4096]
d_values = [128, 256, 512, 1024, 2048, 4096]

# Initialize empty results
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

# Initial header
header = ["YDF with Synthetic Matrices", "d=1000 (HARD-CODED!)", str("nnz="+n_runs), "trees=50", "3 repeats", "-1 depth", "<CPU>"]
save_matrix(avg_matrix, avg_csv, header)
save_matrix(std_matrix, std_csv, header)

# Run grid
for n in n_values:
    for d in d_values:
        print(f"\nRunning: rows={n}, cols={d}")
        times = []

        for i in range(n_runs):
            cmd = [binary_path, f"--rows={n}", f"--cols={d}"] + static_args
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                # Parse timing output
                match = re.search(r"Ariel Training time: ([\d.]+) s", output)
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

        # Save after each combo
        save_matrix(avg_matrix, avg_csv, header)
        save_matrix(std_matrix, std_csv, header)
