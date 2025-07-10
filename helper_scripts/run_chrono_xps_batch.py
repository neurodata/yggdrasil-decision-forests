"""Run YDF synthetic-data benchmarks and parse timing output.

Produces one CSV per run:

    • columns   A…?   – timing table (tree × depth × functions)
    • columns  +2 gap – two empty spacer columns
    • columns   …Z    – key-value "Run-Parameters" block
"""

from __future__ import annotations
import argparse, csv, os, re, subprocess, time
from collections import defaultdict
from typing import Callable

import pandas as pd

# ─────────────────────────── misc helpers ───────────────────────────
def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for l in f:
                if l.startswith("model name"):
                    return (
                        l.split(":", 1)[1]
                        .strip().replace(" ", "_").replace("/", "-")
                    )
    except FileNotFoundError:
        pass
    return "unknown_cpu"

def disable_e_cores():
    """Disable E-cores (cores 6 to nproc-1)"""
    try:
        nproc = int(subprocess.run(['nproc'], capture_output=True, text=True).stdout.strip())
        if nproc > 6:
            result = subprocess.run(['sudo', 'chcpu', '-d', f'6-{nproc-1}'], 
                                  capture_output=True, text=True, check=True)
            print(f"Disabled E-cores 6-{nproc-1}")
            if result.stdout: print(result.stdout.strip())
            if result.stderr: print(result.stderr.strip())
    except Exception as e:
        print(f"Warning: Could not disable E-cores: {e}")

def enable_e_cores():
    """Re-enable E-cores (cores 6-15)"""
    try:
        result = subprocess.run(['sudo', 'chcpu', '-e', '6-15'], 
                              capture_output=True, text=True, check=True)
        print("Re-enabled E-cores 6-15")
        if result.stdout: print(result.stdout.strip())
        if result.stderr: print(result.stderr.strip())
    except Exception as e:
        print(f"Warning: Could not re-enable E-cores: {e}")


ORDER = [
    "Selecting Bootstrapped Samples",
    "Initialization of FindBestCondOblique",
    "SampleProjection", "ApplyProjection",
    "Bucket Allocation & Initialization=0",
    "Filling & Finalizing the Buckets", "SortFeature", "ScanSplits",
    "Post-processing after Training all Trees",
    "EvaluateProjection",
    "FillExampleBucketSet (next 3 calls)",
]
RENAMES = {
    "Post-processing after Train": "Post-processing after Training all Trees",
    "FillExampleBucketSet (calls 3 above)": "FillExampleBucketSet (next 3 calls)",
}

# ─────────────────────────── CLI ────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_mode", choices=["synthetic", "csv"], default="synthetic")
    p.add_argument("--train_csv",
                   default="ariel_test_data/processed_wise1_data.csv")
    p.add_argument("--label_col", default="Cancer Status")
    p.add_argument("--experiment_name", type=str, default="untitled_experiment",
                   help="Name for the experiment, used in the output directory path")
    p.add_argument("--numerical_split_type", type=str, 
                   choices=["Exact", "Random", "Equal Width"], 
                   default="Exact",
                   help="Numerical split type for the random forest")
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--rows", type=int, default=524288)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--num_trees", type=int, default=1)
    p.add_argument("--tree_depth", type=int, default=2)
    p.add_argument("--projection_density_factor", type=int, default=3)
    p.add_argument("--max_num_projections", type=int, default=1)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ─────────────────────────── regexes ───────────────────────────────
TRAIN_RX = re.compile(r"Training wall-time:\s*([0-9.eE+-]+)s")
BOOT_TAG = "Selecting Bootstrapped Samples"
DEPTH_TAG = "Depth "
TOOK_TAG = " took:"               # exactly one leading space — leave as-is
STRIP_SET = " \t-"

# ─────────────────────────── parsing (fast path only) ──────────────
def _num(tok: str) -> float:
    tok = tok.rstrip()
    if tok.endswith('s'):
        tok = tok[:-1]
    return float(tok)


def fast_parse_tree_depth(log: str) -> pd.DataFrame:
    rows: list[tuple[int, int, str, float]] = []
    node_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    cur_tree = -1
    cur_depth: int | None = None

    for line in log.splitlines():

        # ── new tree (depth 0) ─────────────────────────────────────
        if BOOT_TAG in line:
            cur_tree += 1
            node_counts[(cur_tree, 0)] += 1           # depth-0 node
            rows.append(
                (
                    cur_tree, 0, ORDER[0],
                    _num(line.rsplit(maxsplit=1)[-1]),
                )
            )
            cur_depth = None
            continue

        # ── depth header ──────────────────────────────────────────
        if line.lstrip().startswith(DEPTH_TAG):
            cur_depth = int(line.lstrip()[len(DEPTH_TAG):].split()[0])
            node_counts[(cur_tree, cur_depth)] += 1   # count this node
            continue

        # ── skip lines until at least one tree seen ──────────────
        if cur_tree < 0 or TOOK_TAG not in line:
            continue

        # ── timing line ──────────────────────────────────────────
        name_part, _, rest = line.partition(TOOK_TAG)
        time_s = _num(rest.split()[0])

        clean = name_part.lstrip(STRIP_SET).rstrip()
        clean = RENAMES.get(clean, clean)

        rows.append((cur_tree, cur_depth, clean, time_s))

    if not rows:
        raise ValueError("No timing lines parsed")

    df = pd.DataFrame(rows, columns=["tree", "depth",
                                     "function", "time_s"])

    wide = (
        df.pivot_table(index=["tree", "depth"],
                       columns="function",
                       values="time_s",
                       aggfunc="sum",
                       fill_value=0.0)
          .reindex(columns=ORDER, fill_value=0.0)
          .reset_index()
    )

    # merge the node counts
    counts_df = (
        pd.DataFrame(
            [(t, d, c) for (t, d), c in node_counts.items()],
            columns=["tree", "depth", "nodes"]
        )
    )
    wide = wide.merge(counts_df, on=["tree", "depth"], how="left")
    wide["nodes"] = wide["nodes"].fillna(0).astype(int)

    cols = ["tree", "depth", "nodes"] + ORDER
    return wide[cols]


PARSER: dict[bool, Callable[[str], pd.DataFrame]] = {
    False: fast_parse_tree_depth,
    True:  fast_parse_tree_depth,
}

# ─────────────────────────── CSV writer ────────────────────────────
def write_csv(table: pd.DataFrame, params: dict[str, object], path: str):
    """Write timing table left-aligned, params block to the right (after 2 blanks)."""
    p_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])

    n_rows = max(len(table), len(p_df))
    tbl = table.reindex(range(n_rows)).fillna("")
    p_df = p_df.reindex(range(n_rows)).fillna("")
    gap = pd.DataFrame({"": [""] * n_rows, "  ": [""] * n_rows})

    pd.concat([tbl, gap, p_df], axis=1).to_csv(
        path, index=False, quoting=csv.QUOTE_MINIMAL
    )

# ─────────────────────────── main ──────────────────────────────────
if __name__ == "__main__":
    a = get_args()
    out_dir = os.path.join(
        "ariel_results", "per_function_timing",
        cpu_model(), a.experiment_name, f"{a.rows}_x_{a.cols}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # print(f"Running with input dims {a.rows}x{a.cols}" )
    print(f"Running with args {a}")

    for rep in range(a.repeats):

        cmd = [
            "./bazel-bin/examples/train_oblique_forest",
            f"--num_trees={a.num_trees}",
            f"--tree_depth={a.tree_depth}",
            f"--num_threads={a.num_threads}",
            f"--projection_density_factor={a.projection_density_factor}",
            f"--max_num_projections={a.max_num_projections}",
            f"--numerical_split_type={a.numerical_split_type}",
        ]

        if a.input_mode == "synthetic":
            cmd += [
                "--input_mode=synthetic",
                f"--rows={a.rows}", f"--cols={a.cols}",
            ]
        else:  # csv
            cmd += [
                f"--train_csv={a.train_csv}",
                f"--label_col={a.label_col}",
            ]

        # Disable E-cores before running the binary
        disable_e_cores()
        
        t0 = time.perf_counter()
        log = subprocess.run(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=True
        ).stdout
        t1 = time.perf_counter()
        print(f"\n▶ binary ran in {t1 - t0:.3f}s")

        # Re-enable E-cores after the binary is done
        enable_e_cores()

        log = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)  # strip ANSI
        table = PARSER[a.verbose](log)
        t2 = time.perf_counter()

        wall = TRAIN_RX.search(log).group(1)               # e.g. "0.0084s"
        print(f"parsing output took {t2 - t1:.3f}s")

        csv_path = os.path.join(out_dir, f"{wall}.csv")
        params = dict(
            rows=a.rows, cols=a.cols,
            num_trees=a.num_trees, tree_depth=a.tree_depth,
            proj_density_factor=a.projection_density_factor,
            max_projections=a.max_num_projections,
            num_threads=a.num_threads, experiment_name=a.experiment_name,
            numerical_split_type=a.numerical_split_type,
            cpu_model=cpu_model(), repeat_index=rep + 1,
        )
        write_csv(table, params, csv_path)
        t3 = time.perf_counter()

        print(f"writing csv took {t3 - t2:.3f}s\nCSV written to {csv_path}")

        if a.save_log:
            with open(os.path.join(out_dir, f"{wall}.log"), "w") as f:
                f.write(log)