#!/usr/bin/env python3
"""Run YDF with parallel-chrono, write per-tree-depth CSV."""

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from collections import defaultdict

import pandas as pd
import utils.utils as utils               # your local helper module

# -----------------------------------------------------------------------------
# 1. CLI
# -----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_mode", choices=["synthetic", "csv"], default="synthetic")
    p.add_argument("--train_csv", default="benchmarks/data/processed_wise1_data.csv")
    p.add_argument("--label_col", default="Cancer Status")
    p.add_argument("--experiment_name", default="")
    p.add_argument("--feature_split_type", default="Oblique",
                   choices=["Axis Aligned", "Oblique"])
    p.add_argument("--numerical_split_type", default="Exact",
                   choices=["Exact", "Random", "Equal Width", "Dynamic Histogramming"])
    p.add_argument("--tree_depth", type=int, default=-1)
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--num_trees", type=int, default=5)
    p.add_argument("--projection_density_factor", type=int, default=3)
    p.add_argument("--max_num_projections", type=int, default=1000)
    p.add_argument("--save_log", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2.  Parsing parallel-chrono output
# -----------------------------------------------------------------------------
TREELINE_RX = re.compile(
    r"thread\s+(\d+)\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"SampleProj\s+([0-9.eE+-]+)s\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"
    r"EvalProj\s+([0-9.eE+-]+)s"
)
TRAIN_RX = re.compile(r"Training wall-time:\s*([0-9.eE+-]+)s")

def parse_parallel_chrono(log: str) -> pd.DataFrame:
    rows = []
    for line in log.splitlines():
        m = TREELINE_RX.search(line)
        if not m:
            continue
        _, tree, depth, t_sp, t_pe, t_ep = m.groups()
        rows.append((
            int(tree), int(depth),
            float(t_sp), float(t_pe), float(t_ep),
        ))

    if not rows:
        raise ValueError("No parallel-chrono lines found in log")

    df = pd.DataFrame(
        rows,
        columns=[
            "tree", "depth",
            "SampleProjection", "ProjectionEvaluate", "EvaluateProjection"
        ],
    ).sort_values(["tree", "depth"])

    # placeholder columns (kept for backward compatibility)
    df["nodes"] = 0
    df["total_samples"] = 0

    df = df[[
        "tree", "depth", "nodes", "total_samples",
        "SampleProjection", "ProjectionEvaluate", "EvaluateProjection"
    ]]
    return df


# -----------------------------------------------------------------------------
# 3.  Convenience: CSV writer with params block on the right
# -----------------------------------------------------------------------------
def write_csv(table: pd.DataFrame, params: dict[str, object], path: str):
    p_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
    n = max(len(table), len(p_df))
    gap = pd.DataFrame({"": [""] * n, "  ": [""] * n})
    (table.reindex(range(n)).fillna("")
         .pipe(lambda left: pd.concat([left, gap, p_df.reindex(range(n)).fillna("")],
                                      axis=1))
         ).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


# -----------------------------------------------------------------------------
# 4.  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    utils.setup_signal_handlers()
    a = get_args()

    # Build with chrono mode
    if not utils.build_binary(a, chrono_mode=True):
        print("❌ Build failed", file=sys.stderr)
        sys.exit(1)

    exp_name = f"{a.feature_split_type} | {a.numerical_split_type} | {a.num_threads}t | {a.experiment_name}"
    out_dir = os.path.join(
        "benchmarks/results", "per_function_timing",
        utils.get_cpu_model_proc(), exp_name, f"{a.rows}_x_{a.cols}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ----- assemble cmd -------------------------------------------------------
    cmd = ["./bazel-bin/examples/train_oblique_forest",
           f"--num_trees={a.num_trees}",
           f"--tree_depth={a.tree_depth}",
           f"--num_threads={a.num_threads}",
           f"--projection_density_factor={a.projection_density_factor}",
           f"--max_num_projections={a.max_num_projections}",
           f"--feature_split_type={a.feature_split_type}",
           ]

    cmd.append("--numerical_split_type=Exact" if a.numerical_split_type == "Dynamic Histogramming"
               else f"--numerical_split_type={a.numerical_split_type}")

    if a.input_mode == "synthetic":
        cmd += ["--input_mode=synthetic", f"--rows={a.rows}", f"--cols={a.cols}"]
    else:
        cmd += ["--input_mode=csv",
                f"--train_csv={a.train_csv}",
                f"--label_col={a.label_col}"]

    # ----- run ----------------------------------------------------------------
    try:
        utils.configure_cpu_for_benchmarks(True)
        t0 = time.perf_counter()
        log = subprocess.run(cmd, text=True, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        dt = time.perf_counter() - t0
        print(f"\n▶ binary ran in {dt:.3f}s")

        if a.save_log:
            with open(os.path.join(out_dir, f"{exp_name}.log"), "w") as f:
                f.write(log)

        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)   # strip ANSI
        table = parse_parallel_chrono(log_plain)
        wall = TRAIN_RX.search(log_plain).group(1)
        write_csv(table, vars(a), os.path.join(out_dir, f"{wall}.csv"))
        print("CSV written.")

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
    finally:
        utils.cleanup_and_exit()