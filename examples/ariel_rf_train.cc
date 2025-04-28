#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// YDF dataset
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"

// YDF model
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"

// YDF learner
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"

// Ariel - Profiling
// #include <gperftools/profiler.h>


// Ariel 1. Are Honest Forests available in YDF? Yes. See Honest=True param


using namespace yggdrasil_decision_forests;

absl::Status TrainRandomForest(const std::string& csv_path,
                               const std::string& label_column_name,
                               const std::string& output_model_dir) {
  // 1) Create a data specification for the CSV dataset.
    dataset::proto::DataSpecification data_spec;
    {
      std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
      // "csv:" prefix means a CSV dataset recognized by YDF.
      // CreateDataSpec(...) is now a void function (it no longer returns absl::Status).
      // Build a guide that explicitly sets the label column type to CATEGORICAL
      dataset::proto::DataSpecificationGuide guide;
      auto* col_guide = guide.add_column_guides();
      col_guide->set_column_name_pattern(label_column_name);
      col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);

      dataset::CreateDataSpec(
          "csv:" + csv_path,
          /*require_same_dataset_fields=*/false,
          guide,
          &data_spec);
    }

  // 2) Configure a RandomForest learner.

  // NOTE Ariel: TrainingConfig is defined in learner/abstract_learner.proto
  model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_column_name);

  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(1);

  auto& rf_config = *train_config.MutableExtension(
      model::random_forest::proto::random_forest_config);

  rf_config.set_num_trees(10);
  rf_config.mutable_decision_tree()->set_max_depth(-1);  // -1 => unlimited
  rf_config.set_bootstrap_training_dataset(true);
  rf_config.set_bootstrap_size_ratio(1.0);

  // Enable Oblique splits:
  rf_config.mutable_decision_tree()->mutable_sparse_oblique_split();

  rf_config.mutable_decision_tree()
         ->mutable_sparse_oblique_split()
         ->set_max_num_projections(1000); // Ariel - fixed max projections fixes size of proj matrix to max_n_p x n_features. it's max(n_features, max_num_projections), so this should stay 128

  rf_config.mutable_decision_tree()
         ->mutable_sparse_oblique_split()
         ->set_projection_density_factor(128.0f); // 128x p matrix should be fully dense

  rf_config.mutable_decision_tree()
    ->mutable_sparse_oblique_split()
    ->set_num_projections_exponent(1); // Should be n_features=2523^1 > 1000=max_n_projections => n_projections should be = 1000

  rf_config.set_compute_oob_performances(false);

  // 3) Create the learner.
  std::unique_ptr<model::AbstractLearner> learner;
  {
    absl::Status get_learner_status =
        model::GetLearner(train_config, &learner);
    if (!get_learner_status.ok()) {
      return absl::InternalError("Could not create RandomForest learner: " +
                                 std::string(get_learner_status.message()));
    }
  }

  // 4) Train the model from disk-based dataset.
  //    "TrainWithStatus(path, dataspec[, optional_valid])"
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or =
      learner->TrainWithStatus("csv:" + csv_path, data_spec);
  if (!model_or.ok()) {
    return absl::InternalError("Training failed: " +
                               std::string(model_or.status().message()));
  }
  std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

  // 5) Print a short textual description of the trained model
  // {
  //   std::string description;
  //   model->AppendDescriptionAndStatistics(/*full_definition=*/false,
  //                                         &description);
  //   std::cout << "Model trained. Summary:\n" << description << std::endl;
  // }

  // 6) Save the model to disk
  {
    absl::Status save_status = model::SaveModel(output_model_dir, *model);
    if (!save_status.ok()) {
      return absl::InternalError("Could not save model: " +
                                 std::string(save_status.message()));
    }
    std::cout << "Model saved to: " << output_model_dir << std::endl;
  }

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  // ProfilerStart("profile.prof");
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " train_data.csv label_column output_model_dir\n"
              << "Example: " << argv[0]
              << " /tmp/train.csv my_label /tmp/my_rf_model\n";
    return 1;
  }

  const std::string train_csv = argv[1];
  const std::string label_col = argv[2];
  const std::string model_out_dir = argv[3];

  // Train
  absl::Status status =
      TrainRandomForest(train_csv, label_col, model_out_dir);
  if (!status.ok()) {
    std::cerr << "Training failed: " << status.message() << std::endl;
    return 1;
  }

  std::cout << "Training complete.\n";

  // ProfilerStop();
  return 0;
}