source benchmarks/.venv/bin/activate

# Ask for the password once.
sudo -v
# Keep the timestamp fresh so it never expires.
while true; do sudo -n true ; sleep 60 ; done &
SUDO_KEEPALIVE_PID=$!

# So it's found when script is run as sudo
benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=32 --rows=10000 --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=16 --rows=10000 --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=8 --rows=10000 --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=4 --rows=10000 --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=2 --rows=10000 --tree_depth=8 --num_trees=64

benchmarks/.venv/bin/python benchmarks/src/parallel_chrono.py --num_threads=1 --rows=10000 --tree_depth=8 --num_trees=64

kill "${SUDO_KEEPALIVE_PID}"