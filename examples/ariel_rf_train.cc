#include <iostream>
#include <string>
#include <chrono>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// Keywords args
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

// Required arguments
ABSL_FLAG(std::string, train_csv, "", "Path to training CSV file.");
ABSL_FLAG(std::string, label_col, "", "Name of label column.");

// Optional hyperparameters
ABSL_FLAG(std::string, model_out_dir, "", "Path to output trained model directory.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 50, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, -1, "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(int, max_num_projections, 1000, "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 128.0f, "Projection density factor.");
ABSL_FLAG(int, num_projections_exponent, 1, "Exponent to determine number of projections.");
ABSL_FLAG(bool, compute_oob_performances, false, "Whether to compute out-of-bag performances.");


// YDF dataset
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"

// YDF model
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"

// YDF learner
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"

#include "yggdrasil_decision_forests/utils/status_macros.h"

// Ariel - Profiling
// #include <gperftools/profiler.h>

// Ariel 1. Are Honest Forests available in YDF? Yes. See Honest=True param

using namespace yggdrasil_decision_forests;

absl::Status TrainRandomForest(const std::string &csv_path,
                               const std::string &label_column_name,
                               const std::string &output_model_dir,
                               int num_threads,
                               int num_trees,
                               int tree_depth,
                               int max_num_projections,
                               float projection_density_factor,
                               int num_projections_exponent,
                               bool compute_oob_performances)
{
  // 1) Create a data specification
  dataset::proto::DataSpecification data_spec;
  {
    std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
    dataset::proto::DataSpecificationGuide guide;
    auto *col_guide = guide.add_column_guides();
    col_guide->set_column_name_pattern(label_column_name);
    col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);

    dataset::CreateDataSpec(
        "csv:" + csv_path,
        /*require_same_dataset_fields=*/false,
        guide,
        &data_spec);
  }

  // 2) Configure Random Forest
  model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_column_name);

  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(num_threads);

  auto &rf_config = *train_config.MutableExtension(
      model::random_forest::proto::random_forest_config);

  rf_config.set_num_trees(num_trees);
  rf_config.mutable_decision_tree()->set_max_depth(tree_depth);
  rf_config.set_bootstrap_training_dataset(true);
  rf_config.set_bootstrap_size_ratio(1.0);

  // Oblique splits
  rf_config.mutable_decision_tree()->mutable_sparse_oblique_split();
  rf_config.mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_max_num_projections(max_num_projections);
  rf_config.mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_projection_density_factor(projection_density_factor);
  rf_config.mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_num_projections_exponent(num_projections_exponent);

  rf_config.set_compute_oob_performances(compute_oob_performances);

  // 3) Create the learner
  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(train_config, &learner, deployment_config));


  // 4) Train with timing - this also causes data loading!
  //    Can't time training here

  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or =
      learner->TrainWithStatus("csv:" + csv_path, data_spec);


  if (!model_or.ok())
  {
    return absl::InternalError("Training failed: " +
                               std::string(model_or.status().message()));
  }
  std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

  // 5) Save
  if (!output_model_dir.empty()) {
    absl::Status save_status = model::SaveModel(output_model_dir, *model);
    if (!save_status.ok()) {
      return absl::InternalError("Could not save model: " +
                                std::string(save_status.message()));
    }
    std::cout << "Model saved to: " << output_model_dir << std::endl;
  } else {
    std::cout << "Model was trained but not saved (no --model_out_dir provided).\n";
  }

  return absl::OkStatus();
}


int main(int argc, char** argv) {
  auto start_time = std::chrono::high_resolution_clock::now();

  absl::ParseCommandLine(argc, argv);

  std::string train_csv = absl::GetFlag(FLAGS_train_csv);
  std::string label_col = absl::GetFlag(FLAGS_label_col);

  if (train_csv.empty() || label_col.empty()) {
    std::cerr << "WARNING: --train_csv not specified. Defaulting to MIGHT dataset.\n";
    train_csv = "ariel_test_data/processed_wise1_data.csv";  
    label_col = "Cancer Status";
  }

  const std::string model_out_dir = absl::GetFlag(FLAGS_model_out_dir);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const int num_trees = absl::GetFlag(FLAGS_num_trees);
  const int tree_depth = absl::GetFlag(FLAGS_tree_depth);
  const int max_num_projections = absl::GetFlag(FLAGS_max_num_projections);
  const float projection_density_factor = absl::GetFlag(FLAGS_projection_density_factor);
  const int num_projections_exponent = absl::GetFlag(FLAGS_num_projections_exponent);
  const bool compute_oob_performances = absl::GetFlag(FLAGS_compute_oob_performances);

  absl::Status status = TrainRandomForest(
      train_csv, label_col, model_out_dir,
      num_threads, num_trees, tree_depth,
      max_num_projections, projection_density_factor,
      num_projections_exponent, compute_oob_performances);

  if (!status.ok()) {
    std::cerr << "Training failed: " << status.message() << std::endl;
    return 1;
  }

  std::cout << "Training complete.\n";

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "\nAriel Wall time: " << duration.count() << " seconds\n" << std::endl;
  
  return 0;
}

