// examples/synthetic_matrix_train.cc  (compiles & runs)

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

using namespace yggdrasil_decision_forests;

// ------------------------------------------------------------------
// ‣ Tweak these to taste
// ------------------------------------------------------------------
constexpr int64_t kRows      = 4096;   // NOT 4 096², just 4 096 for now
constexpr int     kCols      = 4096;
constexpr int     kLabelMod  = 3;
constexpr int     kTrees     = 10;
constexpr int     kDepth     = 1;
constexpr int     kThreads   = 4;

// ------------------------------------------------------------------
// 1. Build DataSpecification
// ------------------------------------------------------------------
dataset::proto::DataSpecification MakeSpec() {
  dataset::proto::DataSpecification spec;

  // label
  auto* lbl = spec.add_columns();
  lbl->set_name("y");
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(kLabelMod);
  lbl->mutable_categorical()->set_is_already_integerized(true);

  // numerical features
  for (int c = 0; c < kCols; ++c) {
    auto* col = spec.add_columns();
    col->set_name("x" + std::to_string(c));
    col->set_type(dataset::proto::NUMERICAL);
  }
  spec.set_created_num_rows(kRows);
  return spec;
}

// ------------------------------------------------------------------
// 2. Materialise dataset entirely in RAM
// ------------------------------------------------------------------
dataset::VerticalDataset MakeDataset(
    const dataset::proto::DataSpecification& spec) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());   // ← correct spelling
  ds.Resize(kRows);

  // label values
  {
    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::CategoricalColumn>(0);
    auto* v = col->mutable_values();          // pointer to std::vector<int>
    for (int64_t i = 0; i < kRows; ++i) (*v)[i] = static_cast<int>(i % kLabelMod);
  }

  // numerical columns
  std::mt19937 rng(1234);
  std::normal_distribution<float> N(0.f, 1.f);

  for (int c = 0; c < kCols; ++c) {
    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(1 + c);
    auto* v = col->mutable_values();
    for (int64_t i = 0; i < kRows; ++i) (*v)[i] = N(rng);
  }
  return ds;
}

// ------------------------------------------------------------------
// 3. Configure learner, train, time
// ------------------------------------------------------------------
int main() {
  auto spec = MakeSpec();
  auto ds   = MakeDataset(spec);

  model::proto::TrainingConfig tc;
  tc.set_learner("RANDOM_FOREST");
  tc.set_task(model::proto::CLASSIFICATION);
  tc.set_label("y");

  auto& rf_cfg = *tc.MutableExtension(
      model::random_forest::proto::random_forest_config);
  rf_cfg.set_num_trees(kTrees);
  rf_cfg.mutable_decision_tree()->set_max_depth(kDepth);

  model::proto::DeploymentConfig deploy;
  deploy.set_num_threads(kThreads);

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(tc, &learner, deploy));

  auto t0 = std::chrono::steady_clock::now();
  auto model_or = learner->TrainWithStatus(ds);
  auto t1 = std::chrono::steady_clock::now();

  if (!model_or.ok()) { std::cerr << model_or.status(); return 1; }

  std::cout << "Trained on " << kRows << " × " << kCols
            << " synthetic matrix in "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";
  return 0;
}
