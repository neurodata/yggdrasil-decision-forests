import subprocess
import argparse
import os
import re
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True,
                        help="Number of rows the synthetic input matrix should have")
    parser.add_argument("--cols", type=int, required=True,
                    help="Number of rows the synthetic input matrix should have")
    parser.add_argument("--repeats", type=int, required=True,
                    help="How many experiments to run")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    rows = args.rows
    cols = args.cols
    file_dir = "../ariel_results/per_function_timing/"

    for i in range(args.repeats):
        cmd = [
            "../bazel-bin/examples/train_oblique_forest",
            "--input_mode=synthetic",
            "--max_num_projections=1",
            "--num_trees=20",
            "--num_threads=1",
            "--tree_depth=2",
            f"--rows={rows}",
            f"--cols={cols}",
        ]

        # Run program & capture stdout
        log_output = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,          # gives you a str instead of bytes
            check=True,         # raises CalledProcessError on non-zero exit
        ).stdout

        def parse_train_time(log_output):
            import re

            m = re.search(r"Training wall-time:\s*([\d\.]+s)", log_output)
            if m:
                training_time = m.group(1)  # e.g. "2.71642s"
            else:
                raise LookupError("Training Time string couldn't be found")

            return training_time
        training_time = parse_train_time(log_output) # Get Training Time, to add to filename

        def parse_log_into_table(log: str) -> pd.DataFrame:
                # ── desired column order & renames ────────────────────────────────
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
            RENAMES = {
                "Post-processing after Train": "Post-processing after Training all Trees",
                "FillExampleBucketSet (calls 3 above)": "FillExampleBucketSet (next 3 calls)",
            }

            def _parse(log: str) -> pd.DataFrame:
                TIMING_RE = re.compile(
                    r"^\s*(?:-\s*)*"                 # leading “- ” blocks
                    r"(?P<name>[^:]+?)\s+"           # function name
                    r"(?:Took|took):\s+"             # “took:” (any case)
                    r"(?P<secs>[0-9.eE+-]+)s",
                    re.IGNORECASE,
                )
                            
                rows, cur_tree = [], -1
                for line in log.splitlines():
                    if "Selecting Bootstrapped Samples Took" in line:
                        cur_tree += 1
                    m = TIMING_RE.match(line)
                    if m and cur_tree >= 0:
                        rows.append(
                            {
                                "tree": cur_tree + 1,
                                "function": m.group("name").strip(),
                                "time_s": float(m.group("secs")),
                            }
                        )
                return pd.DataFrame(rows)
            
            df = _parse(log)
            long = df.groupby(["tree", "function"], as_index=False)["time_s"].sum()

            wide = (
                long.pivot(index="tree", columns="function", values="time_s")
                    .rename(columns=RENAMES)          # make names match ORDER list
                    .fillna(0.0)
            )

            # ensure every requested column exists, then apply the order
            wide = wide.reindex(columns=ORDER, fill_value=0.0)

            # make “tree” a visible column instead of the index
            wide = wide.reset_index()

            return wide
        tbl = parse_log_into_table(log_output)


        def save_tbl_to_xlsx(tbl: pd.DataFrame, rows: int, cols: int, file_dir: str, training_time: str):
            dir_path = os.path.join(file_dir, f"{rows}_x_{cols}")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            full_path = os.path.join(dir_path, f"{training_time}.xlsx")
            print("Saving to", full_path)

            tbl.to_excel(full_path, index=False)
        save_tbl_to_xlsx(tbl, rows, cols, file_dir, training_time)
