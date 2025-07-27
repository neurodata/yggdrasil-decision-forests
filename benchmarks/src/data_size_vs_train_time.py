import subprocess
import os
import re
import statistics
import csv
import argparse
import logging
import signal
import sys
import atexit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Global flag to track E-core state
cpu_modified = False

def configure_cpu_for_benchmarks(enable_pcore_only=True):
    """
    Configure CPU for benchmarking.
    
    Args:
        enable_pcore_only: If True, disable HT/E-cores/turbo. If False, restore all.
    """
    global cpu_modified
    
    action = "--disable" if enable_pcore_only else "--enable"
    cmd = ["sudo", "./benchmarks/src/utils/disable_cpu_e_features.sh", action]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Update global flag based on action
        cpu_modified = enable_pcore_only
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to configure CPU: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def cleanup_and_exit(signum=None, frame=None):
    """Cleanup function to restore CPU configuration before exiting"""
    global cpu_modified
    if cpu_modified:
        print("\nCleaning up: Restoring CPU configuration...")
        configure_cpu_for_benchmarks(False)  # This will set cpu_modified = False
    if signum is not None:
        print(f"\nReceived signal {signum}, exiting cleanly.")
        sys.exit(1)


def setup_signal_handlers():
    """Setup signal handlers for graceful cleanup"""
    signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_and_exit)  # Termination signal
    atexit.register(cleanup_and_exit)  # Fallback for other exit scenarios


def build_binary():
    """Build the binary using bazel. Returns True if successful, False otherwise."""
    build_cmd = [
        'bazel', 'build', '-c', 'opt', 
        '--config=fixed_1000_projections', 
        '//examples:train_oblique_forest'
    ]
    
    print("Building binary...")
    print(f"Running: {' '.join(build_cmd)}")
    
    try:
        result = subprocess.run(
            build_cmd, 
            capture_output=False, 
            text=True, 
            check=True,
            env=os.environ.copy(),  # Preserve current environment
            cwd=os.getcwd()         # Explicitly set working directory
        )
        
        print("✅ Build succeeded!")
        if result.stdout:
            logging.info(f"Build stdout:\n{result.stdout}")
        if result.stderr:
            logging.info(f"Build stderr:\n{result.stderr}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Build failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Build stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Build stderr:\n{e.stderr}")
        return False
    
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error during build: {e}")
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", choices=["csv", "synthetic"], default="synthetic",
                        help="Experiment mode: 'csv' to load data via train_forest, 'rng' to generate via train_forest synthetic")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name for the experiment, used in the output CSV filename")
    # Runtime params.
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads to use. Use -1 for all logical CPUs.")
    parser.add_argument("--threads_list", type=int, nargs="+", default=None,
                        help="List of number of threads to test, e.g. --threads_list 1 2 4 8 16 32 64")
    parser.add_argument("--rows_list", type=int, nargs="+", default=[128, 256, 512,1024],
                        help="List of number of rows of the input matrix to test, e.g. --rows_list 128 256 512. Default: [128, 256, 512,1024]")
    parser.add_argument("--cols_list", type=int, nargs="+", default=[128,256,512,1024],
                        help="List of number of cols of the input matrix to test, e.g. --cols_list 128 256 512. Default: [128,256,512,1024]")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of times to repeat & avg. experiments. Use at least 5 for publishable results. Default: 1, for speed")
    
    # Model params
    parser.add_argument("--feature_split_type", type=str, choices=["Axis Aligned", "Oblique"], required=True,
                    help="Whether to use Exact or Histogram splits")
    parser.add_argument("--numerical_split_type", type=str, choices=["Exact", "Random", "Equal Width"], required=True,
                    help="Whether to use Exact or Histogram splits")
    parser.add_argument("--num_trees", type=int, default=50,
                        help="Number of trees in the Random Forest. Default: 50")
    parser.add_argument("--tree_depth", type=int, default=-1,
                        help="Limit depth of trees in Random Forest. Default: -1 (Unlimited)")
    parser.add_argument("--projection_density_factor", type=int, default=3,
                    help="Number of nonzeros per projection. Default: 3")
    parser.add_argument("--max_num_projections", type=int, default=1000,
                    help="Maximum number of projections. WARNING: YDF doesn't always obey this! Default: 1000")

    return parser.parse_args()


def save_combined_matrix(avg_matrix, std_matrix, filepath, params=None):
    """Save results with parameters to the right (after 2 blanks), similar to chrono parsing code."""
    
    # Create the main results data
    results_data = []
    
    # Header row
    header = ["n"] + d_values + [""] * 5 + [f"{d}_std" for d in d_values]
    results_data.append(header)
    
    # Data rows
    for n in n_values:
        row = [n]
        # Add average values
        row.extend([avg_matrix[n][d] for d in d_values])
        # Add 5 empty columns
        row.extend([""] * 5)
        # Add std values
        row.extend([std_matrix[n][d] for d in d_values])
        results_data.append(row)
    
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        
        if params:
            # Convert params dict to list of [Parameter, Value] pairs
            param_rows = [[k, v] for k, v in params.items()]
            # Add header for parameters
            param_rows.insert(0, ["Parameter", "Value"])
            
            # Ensure both tables have the same number of rows
            n_rows = max(len(results_data), len(param_rows))
            
            # Pad with empty rows if needed
            while len(results_data) < n_rows:
                results_data.append([""] * len(results_data[0]))
            while len(param_rows) < n_rows:
                param_rows.append(["", ""])
            
            # Write combined data: [results] [2 gaps] [parameters]
            for i in range(n_rows):
                combined_row = results_data[i] + ["", ""] + param_rows[i]
                writer.writerow(combined_row)
        else:
            # No parameters, just write results
            for row in results_data:
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


def run_binary_with_cleanup(cmd):
    """Run binary command without toggling E-cores (they should stay disabled)"""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        raise e
    except KeyboardInterrupt:
        # Handle Ctrl+C during subprocess execution
        print("\nKeyboard interrupt received during binary execution...")
        raise
    

def main():
    # Setup signal handlers first
    setup_signal_handlers()
    
    args = get_args()
    
    # Build the binary first - exit if build fails
    if not build_binary():
        print("\n❌ Cannot proceed with benchmarks - build failed!")
        sys.exit(1)
    
    global n_values, d_values
    n_values = args.rows_list
    d_values = args.cols_list

    if args.threads_list is not None:
        threads_to_test = args.threads_list
    else:
        threads_to_test = [args.num_threads]

    base_results_dir = os.path.join("benchmarks/results", "runtime_heatmap", get_cpu_model_proc())
    os.makedirs(base_results_dir, exist_ok=True)
    binary = "./bazel-bin/examples/train_oblique_forest"

    # Disable E-cores once at the beginning. Only do it for my CPU with E-cores
    configure_cpu_for_benchmarks(True)

    try:
        for t in threads_to_test:
            print("Running w/ Threads=", t)
            # resolve -1 to all logical CPUs
            if t == -1:
                thread_count = os.cpu_count()
                logging.info(f"\n\nUnlimited threads requested. Found {thread_count} logical CPUs. Calling train_oblique_forest with --threads={thread_count}\n\n")
            else:
                thread_count = t

            combined_csv = os.path.join(base_results_dir, f"{args.feature_split_type} | {args.numerical_split_type} | {args.num_threads} thread(s) | {args.experiment_name}.csv")

            # Initialize matrices
            avg_matrix = {n: {d: "" for d in d_values} for n in n_values}
            std_matrix = {n: {d: "" for d in d_values} for n in n_values}

            params = {
                "Benchmark": args.experiment_name,

                "Projection Density Factor": args.projection_density_factor,
                "Trees": args.num_trees,
                "depth": args.tree_depth,
                "feature_split_type": args.feature_split_type,
                "numerical_split_type": args.numerical_split_type,

                "cpu_model": get_cpu_model_proc(),
                "threads": str(t),
                "repeats": args.repeats,
            }

            if args.input_mode == "csv":
                # Add max_num_projections to params for CSV mode if needed
                params["max_num_projections"] = args.max_num_projections
                
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
                            cmd = [binary, "--input_mode=csv", f"--train_csv={path}", f"--feature_split_type={args.feature_split_type}"] + static_args
                            try:
                                out = run_binary_with_cleanup(cmd)
                                
                                m = time_rx.search(out)
                                if m:
                                    t_measured = float(m.group(1))
                                    times.append(t_measured)
                                    print(f"  Run {i+1}: {t_measured:.4f} s")
                                else:
                                    logging.error(f"  Run {i+1}: parse fail")
                            except subprocess.CalledProcessError as e:
                                logging.error(f"  Run {i+1}: error\n{e.output}")
                            except KeyboardInterrupt:
                                print(f"\nKeyboard interrupt during run {i+1}")
                                raise
                        if times:
                            avg = statistics.mean(times)
                            std = statistics.stdev(times) if len(times) > 1 else 0.0
                            avg_matrix[n][d] = f"{avg:.4f}"
                            std_matrix[n][d] = f"{std:.4f}"
                            print(f"  ➤ Avg: {avg:.4f}s | Std: {std:.4f}s")
                        else:
                            logging.critical("  ➤ All runs failed.")

                        # Save after each cell
                        save_combined_matrix(avg_matrix, std_matrix, combined_csv, params)

            else:  # synthetic mode
                params["max_num_projections"] = args.max_num_projections
                
                static_args = [
                    "--label_mod=2",
                    f"--projection_density_factor={args.projection_density_factor}.0",
                    f"--num_trees={args.num_trees}",
                    f"--tree_depth={args.tree_depth}",
                    f"--num_threads={thread_count}",
                    f"--max_num_projections={args.max_num_projections}",
                    f"--feature_split_type={args.feature_split_type}",
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
                                out = run_binary_with_cleanup(cmd)
                                
                                m = time_rx.search(out)
                                if m:
                                    t_measured = float(m.group(1))
                                    times.append(t_measured)
                                    print(f"  Run {i+1}: {t_measured:.4f} s")
                                else:
                                    logging.error(f"  Run {i+1}: parse fail")
                            except subprocess.CalledProcessError as e:
                                logging.error(f"  Run {i+1}: error\n{e.output}")
                            except KeyboardInterrupt:
                                print(f"\nKeyboard interrupt during run {i+1}")
                                raise
                        if times:
                            avg = statistics.mean(times)
                            std = statistics.stdev(times) if len(times) > 1 else 0.0
                            avg_matrix[n][d] = f"{avg:.4f}"
                            std_matrix[n][d] = f"{std:.4f}"
                            print(f"  ➤ Avg: {avg:.4f}s | Std: {std:.4f}s")
                        else:
                            logging.critical("  ➤ All runs failed.")

                        # Save after each cell
                        save_combined_matrix(avg_matrix, std_matrix, combined_csv, params)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Ensure cleanup happens regardless of how we exit
        cleanup_and_exit()


if __name__ == "__main__":
    main()