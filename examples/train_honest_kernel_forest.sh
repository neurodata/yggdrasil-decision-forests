#!/bin/bash
# train_honest_kernel_forest.sh

set -vex

bazel build //examples:train_honest_kernel_forest

PROJECT="${HOME}/yggdrasil_honest_kernel_forest"
mkdir -p $PROJECT

./bazel-bin/examples/train_honest_kernel_forest \
    --dataset_dir=yggdrasil_decision_forests/test_data/dataset \
    --output_dir=$PROJECT \
    --enable_honest=true \
    --honest_ratio=0.5 \
    --honest_fixed_separation=false \
    --enable_kernel=false \
    --num_trees=100 \
    --winner_take_all=false \
    --alsologtostderr

echo "Honest Forest with Kernel Method completed!"
echo "Results saved to: $PROJECT"
ls -l $PROJECT