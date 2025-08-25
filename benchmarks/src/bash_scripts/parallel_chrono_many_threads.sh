#!/bin/bash

source benchmarks/.venv/bin/activate

# Check if we need a password for sudo
if sudo -n true 2>/dev/null; then
    # No password needed, we're good to go
    echo "Passwordless sudo detected"
else
    # Password needed, so ask for it and keep it alive
    sudo -v
    # Keep the timestamp fresh so it never expires.
    while true; do sudo -n true ; sleep 60 ; done &
    SUDO_KEEPALIVE_PID=$!
fi

# So it's found when script is run as sudo
benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=32  --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=16 --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=8 --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=4 --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=2 --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=1 --train_csv=benchmarks/data/trunk_data/10000x4096.csv --label_col=target --tree_depth=8 --num_trees=64

# Only kill the keepalive process if it exists
if [ ! -z "${SUDO_KEEPALIVE_PID}" ]; then
    kill "${SUDO_KEEPALIVE_PID}"
fi