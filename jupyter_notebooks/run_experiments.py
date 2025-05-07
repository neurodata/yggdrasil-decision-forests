import subprocess
import os
import re
import statistics

# Path to directory with CSV files
data_dir = "ariel_test_data/random_csvs"

# Binary path
binary_path = "./bazel-bin/examples/ariel_rf_train"

# Other static flags
static_args = [
    "--label_col=Target",
    "--projection_density_factor=3.0"
]

# Number of times to run each experiment
n_runs = 3

# Regex to extract training time from output
time_pattern = re.compile(r"Training time: ([\d.]+) seconds")

results = {}

for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".csv"):
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
        results[filename] = (avg, std)
        print(f"  ➤ Avg: {avg:.4f} s | Std: {std:.4f} s")
    else:
        results[filename] = (None, None)
        print("  ➤ All runs failed.")

# Summary
print("\n=== Summary ===")
for fname, (avg, std) in results.items():
    if avg is not None:
        print(f"{fname:40s}  Avg: {avg:.4f} s  | Std: {std:.4f} s")
    else:
        print(f"{fname:40s}  FAILED")
