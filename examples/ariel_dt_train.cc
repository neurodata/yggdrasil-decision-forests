// single_oblique_decision_tree_train.cc
//
// Example usage:
//   single_oblique_decision_tree_train /path/to/train.csv label_col /path/to/out_model

#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// For reading & writing datasets
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
// For creating & saving the model
#include "yggdrasil_decision_forests/model/model_library.h"
// For the generic "GetLearner" factory function
#include "yggdrasil_decision_forests/learner/learner_library.h"

// For controlling the Decision Tree hyperparams:
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"

using namespace yggdrasil_decision_forests;

absl::Status TrainSingleObliqueDecisionTree(const std::string& csv_path,
                                            const std::string& label_column_name,
                                            const std::string& output_model_dir) {
  // 1) Create a data specification for the CSV dataset.
  dataset::proto::DataSpecification data_spec;
  {
    std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
    dataset::proto::DataSpecificationGuide guide;
    {
      auto* label_guide = guide.add_column_guides();
      label_guide->set_column_name(label_column_name);
      // Example: treat the label as a categorical classification problem.
      label_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);
    }

    // Generate the dataspec:
    dataset::CreateDataSpec("csv:" + csv_path,
                            /*require_same_dataset_fields=*/false,
                            guide,
                            &data_spec);
    // Print the resulting DataSpec (optional).
    std::cout << dataset::PrintHumanReadable(data_spec) << std::endl;
  }

  // 2) Configure a single Decision Tree learner.
  model::proto::TrainingConfig train_config;
  train_config.set_learner("DECISION_TREE");  // Not RANDOM_FOREST
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_column_name);

  // Deployment (e.g. single-thread training if you want):
  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(1);

  // Access the specialized Decision Tree config extension:
  auto& dt_config =
      *train_config.MutableExtension(
          model::decision_tree::proto::decision_tree_config);

  // For an effectively unbounded tree:
  dt_config.set_max_depth(-1);  // -1 => unlimited
  dt_config.set_min_examples(2);

  // Turn on oblique splits (sparse oblique variant):
  dt_config.mutable_sparse_oblique_split();

  // 3) Create the learner
  std::unique_ptr<model::AbstractLearner> learner;
  {
    absl::Status get_learner_status =
        model::GetLearner(train_config, &learner);
    if (!get_learner_status.ok()) {
      return absl::InternalError("Could not create single DT learner: " +
                                 std::string(get_learner_status.message()));
    }
    // Also apply the deployment config
    learner->SetDeploymentConfig(deployment_config);
  }

  // 4) Train the model from the disk-based dataset
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or =
      learner->TrainWithStatus("csv:" + csv_path, data_spec);
  if (!model_or.ok()) {
    return absl::InternalError("Training failed: " +
                               std::string(model_or.status().message()));
  }
  std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

  // 5) Print a short textual description
  {
    std::string description;
    model->AppendDescriptionAndStatistics(/*full_definition=*/false, &description);
    std::cout << "Model trained. Summary:\n" << description << std::endl;
  }

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
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " train_data.csv label_column output_model_dir\n"
              << "Example: " << argv[0]
              << " /tmp/train.csv my_label /tmp/my_dt_model\n";
    return 1;
  }

  const std::string train_csv = argv[1];
  const std::string label_col = argv[2];
  const std::string model_out_dir = argv[3];

  absl::Status status =
      TrainSingleObliqueDecisionTree(train_csv, label_col, model_out_dir);
  if (!status.ok()) {
    std::cerr << "Training failed: " << status.message() << std::endl;
    return 1;
  }

  std::cout << "Single oblique decision tree training complete.\n";
  return 0;
}