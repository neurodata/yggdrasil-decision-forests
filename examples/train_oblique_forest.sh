#!/bin/bash
# train_oblique_forest.sh

set -vex

# 编译
bazel build //examples:train_oblique_forest

# 数据集路径
DATASET_DIR="yggdrasil_decision_forests/test_data/dataset"
TRAIN_CSV="${DATASET_DIR}/adult_train.csv"
TEST_CSV="${DATASET_DIR}/adult_test.csv"

# 输出目录
PROJECT="${HOME}/yggdrasil_oblique_forest_example"
mkdir -p $PROJECT

# 运行oblique forest训练
./bazel-bin/examples/train_oblique_forest \
    --input_mode=csv \
    --train_csv=$TRAIN_CSV \
    --label_col=income \
    --num_trees=100 \
    --tree_depth=16 \
    --max_num_projections=1000 \
    --projection_density_factor=3.0 \
    --num_projections_exponent=1 \
    --num_threads=8 \
    --compute_oob_performances=true \
    --model_out_dir=$PROJECT/oblique_model \
    --alsologtostderr

echo "Oblique Forest training completed!"
echo "Results saved to: $PROJECT"
ls -l $PROJECT