#!/usr/bin/env python3
"""Run YDF with parallel-chrono, write per-tree-depth CSV (thread-pivoted)."""

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path

import pandas as pd
import utils.utils as utils      # your helper module
# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------
TIMING_RX = re.compile(
    r"thread\s+(\d+)\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"SampleProj\s+([0-9.eE+-]+)s\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"
    r"EvalProj\s+([0-9.eE+-]+)s"
)
TRAIN_RX = re.compile(r"Training wall-time:\s*([0-9.eE+-]+)s")


def parse_parallel_chrono(log: str) -> pd.DataFrame:
    rows = []
    for line in log.splitlines():
        m = TIMING_RX.search(line)
        if not m:
            continue
        tid, tree, depth, nodes, samples, t_sp, t_pe, t_ep = m.groups()
        rows.append((
            int(tree), int(depth), int(tid), int(nodes), int(samples),
            float(t_sp), float(t_pe), float(t_ep)
        ))

    if not rows:
        raise ValueError("No parallel-chrono lines found in log")

    df = pd.DataFrame(
        rows,
        columns=["tree", "depth", "thread",
                 "nodes", "samples",
                 "SampleProjection", "ProjectionEvaluate", "EvaluateProjection"]
    )

    # combine duplicates (shouldn’t happen but safe)
    df = (df.groupby(["tree", "depth", "thread"], as_index=False)
            .sum(numeric_only=True))

    # pivot on thread id
    wide = (
        df.pivot_table(index=["tree", "depth"],
                       columns="thread",
                       values=["SampleProjection",
                               "ProjectionEvaluate",
                               "EvaluateProjection",
                               "samples"],   # per-thread sample count
                       fill_value=0.0)
          .sort_index(axis=1)
    )

    # flatten multilevel columns  (('SampleProjection',1330) → 'SampleProjection_thr1330')
    wide.columns = [f"{func}_thr{tid}" for func, tid in wide.columns]

    # ‘nodes’ column (identical for any thread) – take from first occurrence
    nodes = (df.groupby(["tree", "depth"], as_index=False)
               .first()[["tree", "depth", "nodes"]])

    wide = nodes.merge(wide, on=["tree", "depth"]).sort_values(["tree", "depth"])
    return wide.reset_index(drop=True)

# --------------------------------------------------------------------------
# CSV helper
# --------------------------------------------------------------------------
def write_csv(left: pd.DataFrame, params: dict[str, object], path: str):
    right = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
    n = max(len(left), len(right))
    gap = pd.DataFrame({"": [""] * n, "  ": [""] * n})
    (left.reindex(range(n)).fillna("")
         .pipe(lambda l: pd.concat([l, gap,
                                    right.reindex(range(n)).fillna("")], axis=1))
     ).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    utils.setup_signal_handlers()
    a = get_args()

    if not utils.build_binary(a, chrono_mode=True):
        print("❌ build failed", file=sys.stderr)
        sys.exit(1)

    exp = f"{a.feature_split_type} | {a.numerical_split_type} | {a.num_threads}t | {a.experiment_name}"
    out_dir = Path("benchmarks/results/per_function_timing") / utils.get_cpu_model_proc() / exp / f"{a.rows}_x_{a.cols}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # command ----------------------------------------------------------
    cmd = ["./bazel-bin/examples/train_oblique_forest",
           f"--num_trees={a.num_trees}",
           f"--tree_depth={a.tree_depth}",
           f"--num_threads={a.num_threads}",
           f"--projection_density_factor={a.projection_density_factor}",
           f"--max_num_projections={a.max_num_projections}",
           f"--feature_split_type={a.feature_split_type}"]

    cmd.append("--numerical_split_type=Exact"
               if a.numerical_split_type == "Dynamic Histogramming"
               else f"--numerical_split_type={a.numerical_split_type}")

    if a.input_mode == "synthetic":
        cmd += ["--input_mode=synthetic", f"--rows={a.rows}", f"--cols={a.cols}"]
    else:
        cmd += ["--input_mode=csv",
                f"--train_csv={a.train_csv}",
                f"--label_col={a.label_col}"]

    # run --------------------------------------------------------------
    try:
        utils.configure_cpu_for_benchmarks(True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, text=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        log = proc.stdout
        print(f"\n▶ binary ran in {dt:.2f}s")

        if a.save_log:
            (out_dir / f"{exp}.log").write_text(log)

        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)
        table = parse_parallel_chrono(log_plain)
        wall = TRAIN_RX.search(log_plain).group(1)
        write_csv(table, vars(a), out_dir / f"{wall}.csv")
        print("CSV written to", out_dir)

    except Exception as e:
        print("❌", e, file=sys.stderr)
    finally:
        utils.cleanup_and_exit()