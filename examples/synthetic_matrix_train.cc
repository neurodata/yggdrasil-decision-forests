// synthetic_matrix_train.cc  (oblique + correct label encoding)

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

using namespace yggdrasil_decision_forests;

// ── flags ────────────────────────────────────────────────────────────────
ABSL_FLAG(int64_t, rows, 4096,      "Examples");
ABSL_FLAG(int,     cols, 4096,      "Numerical features");
ABSL_FLAG(int,     label_mod, 2,    "Classes (labels are 1..label_mod)");
ABSL_FLAG(uint32_t,seed, 1234,      "PRNG seed");

ABSL_FLAG(int,     trees, 1,        "Number of trees");
ABSL_FLAG(int,     depth, 1,        "Max depth (root-only = 1)");
ABSL_FLAG(int,     threads, 1,      "Threads");

ABSL_FLAG(int,   max_num_projections,      1000,  "");
ABSL_FLAG(float, projection_density_factor, 3.0, "");
ABSL_FLAG(int,   num_projections_exponent, 1,     "");

ABSL_FLAG(std::string, model_out_dir, "", "Optional save dir");

// ── spec ─────────────────────────────────────────────────────────────────
dataset::proto::DataSpecification MakeSpec(int cols, int64_t rows,
                                           int label_mod) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name("y");
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(label_mod+1);
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

// ── data ────────────────────────────────────────────────────────────────
dataset::VerticalDataset MakeDataset(const dataset::proto::DataSpecification& spec,
                                     int64_t rows, int cols,
                                     int label_mod, uint32_t seed) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.Resize(rows);

  std::mt19937 rng(seed);
  std::normal_distribution<float> N(0.f, 1.f);

  for (int c = 0; c < cols; ++c) {
    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(c);
    auto* v = col->mutable_values();
    for (int64_t i = 0; i < rows; ++i) (*v)[i] = N(rng);
  }

  // Binary labels
  auto* ycol = ds.MutableColumnWithCast<
      dataset::VerticalDataset::CategoricalColumn>(cols);
  auto* yval = ycol->mutable_values();
  for (int64_t i = 0; i < rows; ++i)
    (*yval)[i] = static_cast<int>((i % label_mod)+1);  // 1-based! YDF wants indexing to start from 1

  return ds;
}

// ── main ────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

  std::cout << "\n\n CAREFUL: HARD-CODED 1000 PROJECTIONS FOR PROFILING!!\n\n";

    auto t0 = std::chrono::steady_clock::now();

  absl::ParseCommandLine(argc, argv);

  const int64_t rows  = absl::GetFlag(FLAGS_rows);
  const int     cols  = absl::GetFlag(FLAGS_cols);
  const int     K     = absl::GetFlag(FLAGS_label_mod);
  const uint32_t seed = absl::GetFlag(FLAGS_seed);

  auto spec = MakeSpec(cols, rows, K);
  auto ds   = MakeDataset(spec, rows, cols, K, seed);

  model::proto::TrainingConfig tc;
  tc.set_learner("RANDOM_FOREST");
  tc.set_task(model::proto::CLASSIFICATION);
  tc.set_label("y");

  auto& rf = *tc.MutableExtension(
      model::random_forest::proto::random_forest_config);
  rf.set_num_trees(absl::GetFlag(FLAGS_trees));
  rf.mutable_decision_tree()->set_max_depth(absl::GetFlag(FLAGS_depth));
//   rf.mutable_decision_tree()->set_min_examples(1);
//   rf.mutable_decision_tree()->set_in_split_min_examples_check(false);

  auto* sos = rf.mutable_decision_tree()->mutable_sparse_oblique_split();
  sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
  sos->set_projection_density_factor(
      absl::GetFlag(FLAGS_projection_density_factor));
  sos->set_num_projections_exponent(
      absl::GetFlag(FLAGS_num_projections_exponent));

  model::proto::DeploymentConfig deploy;
  deploy.set_num_threads(absl::GetFlag(FLAGS_threads));

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(tc, &learner, deploy));

  
  auto model_or = learner->TrainWithStatus(ds);
  auto t1 = std::chrono::steady_clock::now();

  if (!model_or.ok()) { std::cerr << model_or.status(); return 1; }

  std::cout << "✓ trained on " << rows << " × " << cols
            << " with depth=" << absl::GetFlag(FLAGS_depth)
            << " in WALL TIME: " << std::chrono::duration<double>(t1-t0).count() << " s\n";

  const auto out_dir = absl::GetFlag(FLAGS_model_out_dir);
  if (!out_dir.empty()) {
    CHECK_OK(model::SaveModel(out_dir, *model_or.value()));
    std::cout << "model saved to " << out_dir << '\n';
  }
}
