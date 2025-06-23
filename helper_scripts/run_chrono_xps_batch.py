"""Run YDF synthetic-data benchmarks and parse timing output.

Supports two log formats:
  • default (simple)       – the original, less verbose timing lines
  • --verbose (per-tree)   – the newer, more verbose per-tree & projection summary

The resulting per-tree timing table is written to
    ../ariel_results/per_function_timing/<CPU_MODEL>/<rows>_x_<cols>/<walltime>.xlsx
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from typing import Callable

import pandas as pd
import logging

def get_cpu_model_proc() -> str:
    """
    Reads /proc/cpuinfo and returns the first 'model name' value, sanitized for file paths.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    # sanitize for filesystem (replace spaces/slashes)
                    return model.replace(" ", "_").replace("/", "-")
    except FileNotFoundError:
        return "unknown_cpu"
    return "unknown_cpu"

# ── columns we keep in the XLSX, in the order we want ───────────────────────────
ORDER = [
    "Selecting Bootstrapped Samples",
    "Initialization of FindBestCondOblique",
    "SampleProjection",
    "ApplyProjection",
    "ApplyProjection w/o Sort",
    "Sort inside ApplyProjection",
    "Bucket Allocation & Initialization=0",
    "Filling & Finalizing the Buckets",
    "SortFeature",
    "ScanSplits",
    "Post-processing after Training all Trees",
    "EvaluateProjection",
    "FillExampleBucketSet (next 3 calls)",
]

# log contains slightly different spellings – normalise them here
RENAMES = {
    "Post-processing after Train": "Post-processing after Training all Trees",
    "FillExampleBucketSet (calls 3 above)": "FillExampleBucketSet (next 3 calls)",
}


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_mode", choices=["csv", "synthetic"], default="synthetic",
                        # help="Experiment mode: 'csv' to load data via train_forest, 'rng' to generate via train_forest synthetic")
    parser.add_argument("--sort_method", choices=["SortFeature", "SortIndex"], default="SortFeature",
                        help="Use SortIndex to save the results to sort_index dir instead of regular 'SortFeature'. Has no effect on C++ binary")
    # Runtime params.
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads to use. Use -1 for all logical CPUs.")
    parser.add_argument("--threads_list", type=int, nargs="+", default=None,
                        help="List of number of threads to test, e.g. --threads_list 1 2 4 8 16 32 64")
    parser.add_argument("--rows", type=int, default=524288, help="Rows of the synthetic input matrix")
    parser.add_argument("--cols", type=int, default=1024, help="Columns of the synthetic input matrix")
    parser.add_argument("--repeats", type=int, default=7,
                        help="Number of times to repeat & avg. experiments. Default: 7")
    
    # Model params
    parser.add_argument("--num_trees", type=int, default=1,
                        help="Number of trees in the Random Forest. Default: 1")
    parser.add_argument("--tree_depth", type=int, default=2,
                        help="Limit depth of trees in Random Forest. -1 = Unlimited. Default: 2")
    parser.add_argument("--projection_density_factor", type=int, default=3,
                    help="Number of nonzeros per projection. Default: 3")
    parser.add_argument("--max_num_projections", type=int, default=1,
                    help="Maximum number of projections. WARNING: YDF doesn't always obey this! Default: 1000")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Parse the newer, more verbose per-tree timing output.  "
             "Omit for legacy/simple format.",
    )
    return parser.parse_args()

# ── helpers ─────────────────────────────────────────────────────────────────────

def parse_train_time(log: str) -> str:
    m = re.search(r"Training wall-time:\s*([\d\.]+s)", log)
    if not m:
        raise LookupError("Training time string couldn't be found in log output")
    return m.group(1)


DEPTH_RE      = re.compile(r"^Depth\s+(\d+)")
FUNCTION_RE   = re.compile(
    r"^\s*(?:-\s*)*(?P<name>[^:]+?)\s*(?:took|Took):?\s*(?P<secs>[0-9.eE+-]+)s",
    re.IGNORECASE,
)

def parse_log_with_depth(log: str) -> pd.DataFrame:
    rows, cur_depth = [], None
    for line in log.splitlines():
        m_depth = DEPTH_RE.match(line.strip())
        if m_depth:
            cur_depth = int(m_depth.group(1))          # update section state
            continue

        m_fun = FUNCTION_RE.match(line)
        if m_fun and cur_depth is not None:
            rows.append({
                "depth":    cur_depth,
                "function": m_fun.group("name").strip(),
                "time_s":   float(m_fun.group("secs")),
            })

    if not rows:
        raise ValueError("No depth-annotated timing found.")

    # aggregate identical function calls that happen multiple times
    long  = pd.DataFrame(rows).groupby(["depth", "function"], as_index=False)["time_s"].sum()
    wide  = long.pivot(index="depth", columns="function", values="time_s")       \
                .rename(columns=RENAMES)                                         \
                .reindex(columns=ORDER, fill_value=0.0)                          \
                .sort_index()                                                    \
                .reset_index()

    return wide


PARSER_MAP = {
    False: parse_log_with_depth,    # simple build prints your banners
    True:  parse_log_with_depth,    # verbose build also prints banners
}



def save_tbl_to_xlsx(
    tbl: pd.DataFrame,
    rows: int,
    cols: int,
    file_dir: str,
    training_time: str,
) -> None:
    dir_path = os.path.join(file_dir, f"{rows}_x_{cols}")
    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, f"{training_time}.xlsx")
    print(f"▪ Saving to {full_path}")
    tbl.to_excel(full_path, index=False)

if __name__ == "__main__":
    args = get_args()

    # dynamic base directory based on CPU model
    base_dir = os.path.join(
        "..", "ariel_results", "per_function_timing", get_cpu_model_proc(), args.sort_method
    )
    os.makedirs(base_dir, exist_ok=True)

    for i in range(args.repeats):
        cmd = [
            "../bazel-bin/examples/train_oblique_forest",
            "--input_mode=synthetic",
            f"--max_num_projections={args.max_num_projections}",
            f"--num_trees={args.num_trees}",
            f"--num_threads={args.num_threads}",
            f"--tree_depth={args.tree_depth}",
            f"--rows={args.rows}",
            f"--cols={args.cols}",
        ]
        log_output = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        ).stdout

        training_time = parse_train_time(log_output)
        tbl = PARSER_MAP[args.verbose](log_output)
        save_tbl_to_xlsx(tbl, args.rows, args.cols, base_dir, training_time)
