// Deprecated: Surprisingly not faster than loading CSV from Disk

// Usage
// ./bazel-bin/examples/train_from_tfrecord --ds_path=./ariel_test_data/random_csvs/tf_binaries/4096x4096   --label_col=Target   --trees=1 --depth=2 --projection_density_factor=3.0 --max_num_projections=1000

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"   // ← needed for Get/SetBinaryProto & Defaults()

using namespace yggdrasil_decision_forests;

// ── flags ───────────────────────────────────────────────────────────────
ABSL_FLAG(std::string, ds_path,  "", "Path **without extension** of TF-Record file");
ABSL_FLAG(std::string, label_col, "Target", "Label column name");

ABSL_FLAG(int, trees, 1,   "Num trees");
ABSL_FLAG(int, depth, 2,   "Max depth");
ABSL_FLAG(int, threads, 1, "CPU threads");

ABSL_FLAG(int,   max_num_projections,      1000,  "");
ABSL_FLAG(float, projection_density_factor, 3.0, "");
ABSL_FLAG(int,   num_projections_exponent, 1,     "");

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();

  absl::ParseCommandLine(argc, argv);
  const std::string ds_path = absl::GetFlag(FLAGS_ds_path);
  if (ds_path.empty()) { std::cerr << "--ds_path required\n"; return 1; }

  // ── 1. load DataSpec then dataset ────────────────────────────────────
  dataset::proto::DataSpecification spec;
  CHECK_OK(file::GetBinaryProto(ds_path + ".data_spec.pb", &spec,
                                file::Defaults()));               // ← 3rd arg

  dataset::VerticalDataset data;                                  // ← declare
  CHECK_OK(dataset::LoadVerticalDataset("tfrecord:" + ds_path,    // …and load
                                        spec, &data));

  // ── 2. configure learner ────────────────────────────────────────────
  model::proto::TrainingConfig cfg;
  cfg.set_learner("RANDOM_FOREST");
  cfg.set_task(model::proto::CLASSIFICATION);
  cfg.set_label(absl::GetFlag(FLAGS_label_col));

  auto& rf = *cfg.MutableExtension(
      model::random_forest::proto::random_forest_config);
  rf.set_num_trees(absl::GetFlag(FLAGS_trees));
  rf.mutable_decision_tree()->set_max_depth(absl::GetFlag(FLAGS_depth));

  auto* sos = rf.mutable_decision_tree()->mutable_sparse_oblique_split();
  sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
  sos->set_projection_density_factor(
      absl::GetFlag(FLAGS_projection_density_factor));
  sos->set_num_projections_exponent(
      absl::GetFlag(FLAGS_num_projections_exponent));

  model::proto::DeploymentConfig dpl;
  dpl.set_num_threads(absl::GetFlag(FLAGS_threads));

  // ── 3. train & time ───────────────────────────────────────────────────
  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(cfg, &learner, dpl));

  
  auto model_or = learner->TrainWithStatus(data);
  const auto t1 = std::chrono::steady_clock::now();

  CHECK_OK(model_or.status());
  std::cout << "Total wall-time: "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";
}
