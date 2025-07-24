source benchmarks/.venv/bin/activate

python3 benchmarks/src/data_size_vs_train_time.py \
    --feature_split_type="Axis Aligned" \
    --numerical_split_type="Exact" \
    --experiment_name="Axis Aligned Exact Splits - Row Scaling - 48thr" \
    --rows=1048576 \
    --cols=1024 \
    --num_trees=1;
    
    
python3 benchmarks/src/data_size_vs_train_time.py\
    --cols=1024 \
    --rows=1048576 \
    --feature_split_type="Axis Aligned" \
    --numerical_split_type="Random" \
    --experiment_name="Axis Aligned Random Histogram - Row Scaling - 48thr" \
    --num_trees=1