source benchmarks/.venv/bin/activate


python3 benchmarks/src/per_function_chrono.py --feature_split_type="Oblique" --numerical_split_type="Exact" --experiment_name="Oblique Exact -1 Depth CHRONO 1M rows" --rows=1048576 --cols=1024 --tree_depth=-1 --num_trees=1


python3 benchmarks/src/per_function_chrono.py --feature_split_type="Oblique" --numerical_split_type="Random" --experiment_name="Oblique Random Hist -1 Depth CHRONO 1M rows" --rows=1048576 --cols=1024 --tree_depth=-1 --num_trees=1