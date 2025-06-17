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


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YDF synthetic benchmark & parse timing logs")
    p.add_argument("--rows", type=int, required=True, help="Rows of the synthetic input matrix")
    p.add_argument("--cols", type=int, required=True, help="Columns of the synthetic input matrix")
    p.add_argument("--repeats", type=int, default=7, help="How many experiments to run. Default: 7")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Parse the newer, more verbose per-tree timing output.  "
             "Omit for legacy/simple format.",
    )
    return p.parse_args()

# ── helpers ─────────────────────────────────────────────────────────────────────

def parse_train_time(log: str) -> str:
    m = re.search(r"Training wall-time:\s*([\d\.]+s)", log)
    if not m:
        raise LookupError("Training time string couldn't be found in log output")
    return m.group(1)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Group + pivot so every tree becomes a row and columns are functions."""
    long = df.groupby(["tree", "function"], as_index=False)["time_s"].sum()
    wide = (
        long
        .pivot(index="tree", columns="function", values="time_s")
        .rename(columns=RENAMES)
        .fillna(0.0)
    )
    wide = wide.reindex(columns=ORDER, fill_value=0.0)
    return wide.reset_index()

# ── parsers ─────────────────────────────────────────────────────────────────────

def parse_log_simple(log: str) -> pd.DataFrame:
    TIMING_RE = re.compile(
        r"^\s*(?:-\s*)*(?P<name>[^:]+?)\s+(?:Took|took):\s+(?P<secs>[0-9.eE+-]+)s",
        re.IGNORECASE,
    )
    rows, cur_tree = [], -1
    for line in log.splitlines():
        if "Selecting Bootstrapped Samples Took" in line:
            cur_tree += 1
        m = TIMING_RE.match(line)
        if m and cur_tree >= 0:
            rows.append({
                "tree": cur_tree + 1,
                "function": m.group("name").strip(),
                "time_s": float(m.group("secs")),
            })
    return _finalise(pd.DataFrame(rows))


def parse_log_verbose(log: str) -> pd.DataFrame:
    TIMING_RE = re.compile(
        r"^\s*(?:-\s*)*(?P<name>[^:]+?)\s*(?:took|Took)?\s*:?\s*(?P<secs>[0-9.eE+-]+)s",
        re.IGNORECASE,
    )
    rows, cur_tree = [], -1
    for line in log.splitlines():
        stripped = line.strip()
        if stripped.startswith("=== Timing summary"):
            continue
        if "Selecting Bootstrapped Samples" in stripped and "Took" in stripped:
            cur_tree += 1
        m = TIMING_RE.match(line)
        if m and cur_tree >= 0:
            rows.append({
                "tree": cur_tree + 1,
                "function": m.group("name").strip(),
                "time_s": float(m.group("secs")),
            })
    if not rows:
        raise ValueError("No timing information found – did you forget --verbose?")
    return _finalise(pd.DataFrame(rows))

PARSER_MAP: dict[bool, Callable[[str], pd.DataFrame]] = {
    False: parse_log_simple,
    True: parse_log_verbose,
}

# ── XLSX writer ─────────────────────────────────────────────────────────────────

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
        "..", "ariel_results", "per_function_timing", get_cpu_model_proc()
    )
    os.makedirs(base_dir, exist_ok=True)

    for i in range(args.repeats):
        cmd = [
            "../bazel-bin/examples/train_oblique_forest",
            "--input_mode=synthetic",
            "--max_num_projections=1",
            "--num_trees=20",
            "--num_threads=1",
            "--tree_depth=2",
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
