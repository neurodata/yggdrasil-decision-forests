import subprocess
import os
import logging
import signal
import sys
import atexit
import argparse


# Global flag to track E-core state
cpu_modified = False
# Remember the *exact* Bazel command we executed, to add to CSV
last_build_cmd = ""


def get_base_parser():
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid duplicate help
    
    # Common arguments
    parser.add_argument("--input_mode", choices=["uniform", "trunk", "csv"], default="csv")
    parser.add_argument("--train_csv", default="benchmarks/data/processed_wise1_data.csv")
    parser.add_argument("--label_col", default="Cancer Status")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--feature_split_type", default="Oblique",
                       choices=["Axis Aligned", "Oblique"])
    parser.add_argument("--numerical_split_type", default="Exact",
                       choices=["Exact", "Random", "Equal Width", "Dynamic Random Histogramming", "Dynamic Equal Width Histogramming"])
    parser.add_argument("--tree_depth", type=int)
    parser.add_argument("--num_threads", type=int, required=True)
    parser.add_argument("--num_trees", type=int)  # Note: different defaults in your files
    parser.add_argument("--projection_density_factor", type=int)
    parser.add_argument("--max_num_projections", type=int)
    parser.add_argument("--sample_projection_mode", choices=["Fast", "Slow"], default="Fast")
    parser.add_argument("--enable_fast_equal_width_binning", action="store_true")
    
    return parser


def configure_cpu_for_benchmarks(enable_pcore_only=True):
    """
    Configure CPU for benchmarking.
    
    Args:
        enable_pcore_only: If True, disable HT/E-cores/turbo. If False, restore all.
    """
    global cpu_modified

    if get_cpu_model_proc() == "Intel(R) Core(TM) Ultra 9 185H":
        action = "--disable" if enable_pcore_only else "--enable"
        cmd = ["sudo", "./benchmarks/src/utils/set_cpu_e_features.sh", action]
        
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
            sys.exit(1)
    else:
        print("Skipping changing CPU E-features. CPU not Intel Core Ultra 9 185H")

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


def build_binary(args, chrono_mode):
    """Build the binary using bazel. Returns True if successful, False otherwise."""
    
    base_cmd = ['bazel', 'build', '--ui_event_filters=-warning',
                '-c', 'opt', '--config=fixed_1000_projections']
    finished_cmd = base_cmd[:] # ← work on a copy

    if args.numerical_split_type == "Dynamic Random Histogramming":
        finished_cmd.append('--config=enable_dynamic_random_histogramming')
    elif args.numerical_split_type == "Dynamic Equal Width Histogramming":
        finished_cmd.append('--config=enable_dynamic_equal_width_histogramming')
    if args.sample_projection_mode == "Slow":
        finished_cmd.append('--config=slow_sample_projections')
    
    if chrono_mode:
        finished_cmd.append('--config=multithreaded_chrono_profile')

    if args.enable_fast_equal_width_binning:
        finished_cmd.append('--config=enable_fast_equal_width_binning')
        
    finished_cmd.append("--ui_event_filters=-warning")
    finished_cmd.append('//examples:train_oblique_forest')

    global last_build_cmd
    last_build_cmd = " ".join(finished_cmd)

    print("Building binary...")
    print(f"Running: {' '.join(finished_cmd)}")
    
    try:
        result = subprocess.run(
            finished_cmd, 
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
