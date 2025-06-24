"""Run YDF synthetic-data benchmarks and parse timing output.

Produces one CSV per run:

    • columns   A…?   – timing table (tree × depth × functions)
    • columns  +2 gap – two empty spacer columns
    • columns   …Z    – key-value “Run-Parameters” block

Console prints a per-run timing breakdown:
    ▶ run binary: … s | parse: … s | write csv: … s
"""

from __future__ import annotations
import argparse, csv, os, re, subprocess, time
from typing import Callable
import pandas as pd

# ─────────────────────────── misc helpers ───────────────────────────
def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for l in f:
                if l.startswith("model name"):
                    return (l.split(":", 1)[1]
                              .strip().replace(" ", "_").replace("/", "-"))
    except FileNotFoundError:
        pass
    return "unknown_cpu"

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
    p.add_argument("--sort_method", choices=["SortFeature", "SortIndex"],
                   default="SortFeature")
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
DEPTH_RX = re.compile(r"^Depth\s+(\d+)")
BOOT_RX  = re.compile(r"Selecting Bootstrapped Samples")
FUNC_RX  = re.compile(r"^\s*(?:-\s*)*(?P<name>[^:]+?)\s*"
                      r"(?:took|Took):?\s*([0-9.eE+-]+)s",
                      re.IGNORECASE)

# ─────────────────────────── parsing ───────────────────────────────
def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    long = (df.groupby(["tree", "depth", "function"], as_index=False)["time_s"]
              .sum())
    wide = (long.pivot(index=["tree", "depth"],
                       columns="function",
                       values="time_s")
                .rename(columns=RENAMES)
                .reindex(columns=ORDER, fill_value=0.0)
                .reset_index())
    return wide[["tree", "depth"] +
                [c for c in wide.columns if c not in {"tree", "depth"}]]

def parse_tree_depth(log: str) -> pd.DataFrame:
    rows, cur_tree, cur_depth = [], -1, None
    for line in log.splitlines():
        if BOOT_RX.search(line):
            cur_tree += 1
            m = FUNC_RX.match(line)
            if m:
                rows.append(dict(tree=cur_tree, depth=0,
                                 function=m.group("name").strip(),
                                 time_s=float(m.group(2))))
            cur_depth = None
            continue
        m = DEPTH_RX.match(line.strip())
        if m:
            cur_depth = int(m.group(1)); continue
        m = FUNC_RX.match(line)
        if m and cur_depth is not None and cur_tree >= 0:
            rows.append(dict(tree=cur_tree, depth=cur_depth,
                             function=m.group("name").strip(),
                             time_s=float(m.group(2))))
    if not rows:
        raise ValueError("No timing lines parsed.")
    return _finalise(pd.DataFrame(rows))


import pandas as pd

BOOT_TAG   = "Selecting Bootstrapped Samples"
DEPTH_TAG  = "Depth "
TOOK_TAG   = " took:"        # exactly one leading space — leave as-is
STRIP_SET  = " \t-"          # what we strip from the *left* of the name

def fast_parse_tree_depth(log: str) -> pd.DataFrame:
    rows, cur_tree, cur_depth = [], -1, None

    for line in log.splitlines():
        # ── new tree ──────────────────────────────────────────────
        if BOOT_TAG in line:
            cur_tree += 1
            rows.append((cur_tree, 0, ORDER[0],
                         float(line.rsplit(maxsplit=1)[-1][:-1])))
            cur_depth = None
            continue

        # ── depth header (allow leading spaces) ──────────────────
        if line.lstrip().startswith(DEPTH_TAG):
            cur_depth = int(line.lstrip()[len(DEPTH_TAG):].split()[0])
            continue

        # ── skip lines until we’ve seen the first tree ───────────
        if cur_tree < 0 or TOOK_TAG not in line:
            continue

        # ── timing line ─────────────────────────────────────────
        name_part, _, rest = line.partition(TOOK_TAG)
        time_s = float(rest.split()[0][:-1])          # "<num>s" → float

        # kill any leading spaces / tabs / dashes
        clean = name_part.lstrip(STRIP_SET).rstrip()
        clean = RENAMES.get(clean, clean)             # apply 2 renames

        rows.append((cur_tree, cur_depth, clean, time_s))

    if not rows:
        raise ValueError("No timing lines parsed")

    df = pd.DataFrame(rows, columns=["tree", "depth", "function", "time_s"])
    return (df.pivot_table(index=["tree", "depth"],
                           columns="function",
                           values="time_s",
                           aggfunc="sum",
                           fill_value=0.0)
              .reindex(columns=ORDER, fill_value=0.0)
              .reset_index())




PARSER: dict[bool, Callable[[str], pd.DataFrame]] = {False: fast_parse_tree_depth,
                                                     True:  fast_parse_tree_depth}

# ─────────────────────────── writers ───────────────────────────────
def write_csv(table: pd.DataFrame, params: dict[str, object], path: str):
    """Write timing table left-aligned, params block to the right (after 2 blanks)."""
    # build parameter frame
    p_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])

    # row-count so both blocks fit side-by-side
    n_rows = max(len(table), len(p_df))

    tbl = table.reindex(range(n_rows)).fillna("")
    p_df = p_df.reindex(range(n_rows)).fillna("")

    gap = pd.DataFrame({"": [""] * n_rows, "  ": [""] * n_rows})  # two blank cols

    out = pd.concat([tbl, gap, p_df], axis=1)
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    a = get_args()
    out_dir = os.path.join("..", "ariel_results", "per_function_timing",
                           cpu_model(), a.sort_method,
                           f"{a.rows}_x_{a.cols}")
    os.makedirs(out_dir, exist_ok=True)

    for rep in range(a.repeats):
        cmd = [
            "../bazel-bin/examples/train_oblique_forest",
            "--input_mode=synthetic",
            f"--rows={a.rows}", f"--cols={a.cols}",
            f"--num_trees={a.num_trees}",
            f"--tree_depth={a.tree_depth}",
            f"--num_threads={a.num_threads}",
            f"--projection_density_factor={a.projection_density_factor}",
            f"--max_num_projections={a.max_num_projections}",
        ]

        t0 = time.perf_counter()
        log = subprocess.run(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True, check=True).stdout
        t1 = time.perf_counter()

        print(f"\n▶ binary ran in {t1-t0:.3f}s")

        log = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)
        
        table = PARSER[a.verbose](log)
        t2 = time.perf_counter()

        wall = TRAIN_RX.search(log).group(1)   # e.g. "2.28s"

        print(f"parsing output took {t2-t1:.3f}s")

        csv_path = os.path.join(out_dir, f"{wall}.csv")
        params = dict(rows=a.rows, cols=a.cols,
                      num_trees=a.num_trees, tree_depth=a.tree_depth,
                      proj_density_factor=a.projection_density_factor,
                      max_projections=a.max_num_projections,
                      num_threads=a.num_threads,
                      sort_method=a.sort_method,
                      cpu_model=cpu_model(),
                      repeat_index=rep + 1)
        write_csv(table, params, csv_path)
        t3 = time.perf_counter()

        print(f"writing csv took: {t3-t2:.3f}s\n")
        print("CSV written to ",  csv_path)

        if a.save_log:
            with open(os.path.join(out_dir, f"{wall}.log"), "w") as f:
                f.write(log)
