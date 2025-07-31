import subprocess
import os
import logging
import signal
import sys
import atexit


# Global flag to track E-core state
cpu_modified = False

def configure_cpu_for_benchmarks(enable_pcore_only=True):
    """
    Configure CPU for benchmarking.
    
    Args:
        enable_pcore_only: If True, disable HT/E-cores/turbo. If False, restore all.
    """
    global cpu_modified

    if get_cpu_model_proc() == "Intel Core Ultra 9 185H":
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
    
    base_cmd = ['bazel', 'build', '-c', 'opt', '--config=fixed_1000_projections']
    finished_cmd = base_cmd

    if args.numerical_split_type == "Dynamic Histogramming":
        finished_cmd.append('--config=enable_dynamic_histogramming')
    
    if chrono_mode:
        finished_cmd.append('--config=chrono_profile')
    
    finished_cmd.append('//examples:train_oblique_forest')

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
