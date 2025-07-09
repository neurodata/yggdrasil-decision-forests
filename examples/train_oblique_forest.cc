#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <random>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

#include <random>
#include "absl/random/random.h"          // BitGen (xoshiro)
#include "absl/random/distributions.h"   // Gaussian()


/* #region ABSL Flags */
// Input mode flag: "csv", "synthetic", or "tfrecord"
ABSL_FLAG(std::string, input_mode, "",
          "Data input mode: csv, synthetic, or tfrecord.");
// CSV mode flags
ABSL_FLAG(std::string, train_csv, "/home/ariel/prog/yggdrasil-decision-forests/ariel_test_data/processed_wise1_data.csv",
          "Path to training CSV file (for csv mode). Must include --label_col.");
// TFRecord mode flags
ABSL_FLAG(std::string, ds_path, "",
          "Path (without extension) to TF-Record file (for tfrecord mode).");
// Common flags
ABSL_FLAG(std::string, label_col, "Cancer Status",
          "Name of label column (used in all modes).");
ABSL_FLAG(std::string, model_out_dir, "",
          "Path to output trained model directory (optional)."
          " If empty, model is not saved.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 50, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, -1,
          "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 3.0f,
          "Projection density factor.");
ABSL_FLAG(int, num_projections_exponent, 1,
          "Exponent to determine number of projections.");
ABSL_FLAG(bool, compute_oob_performances, false,
          "Whether to compute out-of-bag performances (only for csv mode).");

// Synthetic mode flags
ABSL_FLAG(int64_t, rows, 4096, "Number of examples (for synthetic mode).");
ABSL_FLAG(int, cols, 4096, "Number of numerical features (for synthetic mode).");
ABSL_FLAG(int, label_mod, 2,
          "Number of classes (labels are 1..label_mod, for synthetic mode).");
ABSL_FLAG(uint32_t, seed, 1234,
          "PRNG seed (for synthetic mode).");

// Histogram-based splits - Updated to match Yggdrasil implementation
ABSL_FLAG(std::string, numerical_split_type, "Exact",
          "Type of histogram splitting: 'Exact (no histogramming)', 'Random', or 'Equal Width'.");
ABSL_FLAG(int, histogram_num_bins, 256,
          "Number of bins for histogram splitting.");

using namespace yggdrasil_decision_forests;

/* #endregion */


/* #region Synthetic Dataset Generation */

// Build a DataSpecification for synthetic data
dataset::proto::DataSpecification MakeSyntheticSpec(
    int cols, int64_t rows, int label_mod, const std::string &label_col) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name(label_col);
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(label_mod + 1);
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

dataset::VerticalDataset MakeSyntheticDataset(
    const dataset::proto::DataSpecification& spec,
    int64_t rows, int cols, int label_mod,
    uint32_t seed) {

  int num_threads = 8;
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.Resize(rows);

  // -------------  NUMERICAL FEATURES  -------------
  // One column is completely independent of another → easy to parallelise.

  // RNG too fast now, this doesn't matter. ~1s for 50k x 2k, algo takes >1min on even AWS 96-thread
  // Haven't managed to get this to work. It compiles, but still runs single-threaded
  // #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int c = 0; c < cols; ++c) {
    // ---- icx-friendly seeding --------------------------------------------
    std::uint32_t s = seed + static_cast<std::uint32_t>(c);
    std::seed_seq seq{ s };           // SeedSequence with one 32-bit word
    absl::BitGen gen(seq);            // OK for every compiler
    // -----------------------------------------------------------------------

    auto* v = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(c)->mutable_values();
    for (auto& x : *v) x = absl::Gaussian<float>(gen);
  }

  // -------------  CATEGORICAL LABELS  -------------
  auto* ycol = ds.MutableColumnWithCast<
      dataset::VerticalDataset::CategoricalColumn>(cols);
  auto* yval = ycol->mutable_values();

  // #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int64_t i = 0; i < rows; ++i) {
    (*yval)[i] = static_cast<int>((i % label_mod) + 1);   // 1-based
  }

  return ds;
}

/* #endregion */

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto mode = absl::GetFlag(FLAGS_input_mode);

  // Validate required input_mode flag
  if (mode.empty()) {
    std::cerr << "Error: --input_mode is required. Use csv, synthetic, or tfrecord.\n";
    return 1;
  }

  // Placeholders for DataSpec and dataset
  dataset::proto::DataSpecification data_spec;
  std::unique_ptr<dataset::VerticalDataset> tf_ds;
  dataset::VerticalDataset* ds_ptr = nullptr;
  std::string csv_path;
  std::string label_col;

  // 1) Prepare data source based on mode
  if (mode == "csv") {
    csv_path = absl::GetFlag(FLAGS_train_csv);
    label_col = absl::GetFlag(FLAGS_label_col);

    // Validate required CSV parameters
    if (csv_path.empty()) {
      std::cerr << "Error: --train_csv is required in csv mode.\n";
      return 1;
    }
    if (label_col.empty()) {
      std::cerr << "Error: --label_col is required in csv mode.\n";
      return 1;
    }

    
    std::cout << "\n\nInferring DataSpec from CSV: " << csv_path << "\n\n" << std::endl;
    dataset::proto::DataSpecificationGuide guide;
    auto* col_guide = guide.add_column_guides();
    col_guide->set_column_name_pattern(label_col);
    col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);

    dataset::CreateDataSpec(
        "csv:" + csv_path,
        /*require_same_dataset_fields=*/false,
        guide,
        &data_spec);
  } else if (mode == "synthetic") {
    std::cout << "\nGenerating synthetic dataset: rows="
              << absl::GetFlag(FLAGS_rows)
              << ", cols=" << absl::GetFlag(FLAGS_cols)
              << "\n" << std::endl;
    data_spec = MakeSyntheticSpec(
        absl::GetFlag(FLAGS_cols),
        absl::GetFlag(FLAGS_rows),
        absl::GetFlag(FLAGS_label_mod),
        "y"); // label_col
    auto ds = MakeSyntheticDataset(
        data_spec,
        absl::GetFlag(FLAGS_rows),
        absl::GetFlag(FLAGS_cols),
        absl::GetFlag(FLAGS_label_mod),
        absl::GetFlag(FLAGS_seed));
    // Ownership
    tf_ds = std::make_unique<dataset::VerticalDataset>(std::move(ds));
    ds_ptr = tf_ds.get();
  } else if (mode == "tfrecord") {
    std::cout << "\n\nReading TFRECORD\n\n";
    
    const std::string path = absl::GetFlag(FLAGS_ds_path);
    if (path.empty()) {
      std::cerr << "--ds_path required in tfrecord mode\n";
      return 1;
    }
    std::cout << "Loading TFRecord dataset from: " << path << std::endl;
    CHECK_OK(file::GetBinaryProto(path + ".data_spec.pb", &data_spec,
                                  file::Defaults()));
    tf_ds = std::make_unique<dataset::VerticalDataset>();
    CHECK_OK(dataset::LoadVerticalDataset("tfrecord:" + path,
                                          data_spec,
                                          tf_ds.get()));
    ds_ptr = tf_ds.get();
  } else {
    std::cerr << "Unknown input_mode: " << mode << ". Use csv, synthetic, or tfrecord.\n";
    return 1;
  }

  // 2) Configure learner
  model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_col);

  model::proto::DeploymentConfig deploy_config;

  /* #region Handle num_threads */
  int num_threads_flag = absl::GetFlag(FLAGS_num_threads);
  if (num_threads_flag > 0) {
    std::cout << "\nRunning with " << num_threads_flag << " threads, as requested.\n";
    deploy_config.set_num_threads(num_threads_flag);

  } else if (num_threads_flag == -1) {
    // Automatically detect number of CPUs
    unsigned int cpu_count = std::thread::hardware_concurrency();
    if (cpu_count == 0) {
      cpu_count = 1;  // fallback if detection fails
    }
    std::cout << "-1 (automatic) threads requested. "
              << cpu_count << " threads set.\n";
    deploy_config.set_num_threads(cpu_count);

  } else {
    std::cerr << "Invalid value for --num_threads: "
              << num_threads_flag
              << ". Must be >0 for fixed threads or -1 for automatic.\n";
    return 1;
  }
  /* #endregion */

  auto& rf = *train_config.MutableExtension(
      model::random_forest::proto::random_forest_config);
  rf.set_num_trees(absl::GetFlag(FLAGS_num_trees));
  rf.mutable_decision_tree()->set_max_depth(
      absl::GetFlag(FLAGS_tree_depth));
  rf.set_bootstrap_training_dataset(true);
  rf.set_bootstrap_size_ratio(1.0);
  auto* sos = rf.mutable_decision_tree()->mutable_sparse_oblique_split();
  sos->set_max_num_projections(
      absl::GetFlag(FLAGS_max_num_projections));
  sos->set_projection_density_factor(
      absl::GetFlag(FLAGS_projection_density_factor));
  sos->set_num_projections_exponent(
      absl::GetFlag(FLAGS_num_projections_exponent));
  rf.set_compute_oob_performances(
      absl::GetFlag(FLAGS_compute_oob_performances));

  // Configure histogram splitting - Updated to match Yggdrasil implementation
  auto* numerical_split = rf.mutable_decision_tree()->mutable_numerical_split();
  
  const std::string hist_type = absl::GetFlag(FLAGS_numerical_split_type);
  if (hist_type == "Exact") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::EXACT);
    std::cout << "Using exact splitting\n";
  } else if (hist_type == "Random") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    std::cout << "Using histogram splitting: Random with " 
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins\n";
  } else if (hist_type == "Equal Width") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    std::cout << "Using histogram splitting: Equal Width with " 
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins\n";
  } else {
    std::cerr << "Unknown histogram type: " << hist_type 
              << ". Use 'Exact', 'Random', or 'Equal Width'.\n";
    return 1;
  }

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(train_config, &learner, deploy_config));

  // 3) Train with timing
  auto start = std::chrono::high_resolution_clock::now();
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or;

  if (mode == "csv") {
    model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
  } else {
    model_or = learner->TrainWithStatus(*ds_ptr);
  }

  if (!model_or.ok()) {
    std::cerr << "Training failed: " << model_or.status().message() << std::endl;
    return 1;
  }
  auto model_ptr = std::move(model_or.value());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur = end - start;
  std::cout << "Training wall-time: " << dur.count() << "s\n";

  // 4) Save model if requested
  const std::string out_dir = absl::GetFlag(FLAGS_model_out_dir);
  if (!out_dir.empty()) {
    auto save_status = model::SaveModel(out_dir, *model_ptr);
    if (!save_status.ok()) {
      std::cerr << "Could not save model: " << save_status.message() << std::endl;
      return 1;
    }
    std::cout << "Model saved to: " << out_dir << std::endl;
  }

  return 0;
}