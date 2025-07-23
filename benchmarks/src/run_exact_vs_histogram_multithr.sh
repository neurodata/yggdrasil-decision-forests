source benchmarks/.venv/bin/activate

python3 benchmarks/src/data_size_vs_train_time.py \
    --feature_split_type="Oblique" \
    --numerical_split_type="Exact" \
    --experiment_name="Oblique Exact Splits - Row Scaling - 48thr"\
    --repeats=1 \
    --rows_list 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576\
    --cols=1024 \
    --threads=48 ;
    
    
python3 benchmarks/src/data_size_vs_train_time.py\
    --cols=1024 --rows_list 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576\
    --feature_split_type="Oblique"\
    --numerical_split_type="Random"\
    --experiment_name="Oblique Random Histogram - Row Scaling - 48thr"\
    --threads=48