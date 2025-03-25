// single_oblique_decision_tree_train.cc
//
// Example usage:
//   bazel build //examples:ariel_dt_train
//   ./bazel-bin/examples/ariel_dt_train /path/to/train.csv label_col /path/to/out_model
//
// (You can omit or modify "bazel" build instructions as appropriate for your setup.)

#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// For reading & writing datasets.
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
// For creating & saving the model.
#include "yggdrasil_decision_forests/model/model_library.h"
// For the generic "GetLearner" factory function
#include "yggdrasil_decision_forests/learner/learner_library.h"

// For specifying HPCs (HyperParameterConfiguration) for single-tree training.
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
// Deployment config is in "abstract_learner.proto".
#include "yggdrasil_decision_forests/model/abstract_learner.pb.h"

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
      // If it's a classification label, mark it as CATEGORICAL.
      auto* label_guide = guide.add_column_guides();
      label_guide->set_column_name(label_column_name);
      label_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);
    }

    // Generate the dataspec:
    dataset::CreateDataSpec(
        "csv:" + csv_path,
        /*require_same_dataset_fields=*/false,
        guide,
        &data_spec);
    // Print (optional)
    std::cout << dataset::PrintHumanReadable(data_spec) << std::endl;
  }

  // 2) Configure a single Decision Tree learner via the main training config.
  model::proto::TrainingConfig train_config;
  train_config.set_learner("DECISION_TREE");  // Single-tree learner name
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_column_name);

  // 3) If you want to limit or configure runtime environment, fill in a
  //    DeploymentConfig. E.g. single-thread training:
  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(1);

  // 4) Create the learner object via the factory
  std::unique_ptr<model::AbstractLearner> learner;
  {
    absl::Status get_learner_status =
        model::GetLearner(train_config, &learner);
    if (!get_learner_status.ok()) {
      return absl::InternalError("Could not create single DT learner: " +
                                 std::string(get_learner_status.message()));
    }
  }

  // 5) Set single-tree hyperparameters. Because “decision_tree_config” is NOT
  //    an extension in modern Yggdrasil, we do it via "GenericHyperParameters."
  //
  // The official single‐tree HPC definitions are found in:
  //   yggdrasil_decision_forests/learner/decision_tree/decision_tree.cc
  //
  // Common HPC keys:
  //   "max_depth" (int)     default=16
  //   "min_examples" (int)  default=5
  //   "split_axis" (cat)    in {"AXIS_ALIGNED","SPARSE_OBLIQUE","MHLD_OBLIQUE"}
  //   "missing_value_policy" (cat) among {"GLOBAL_IMPUTATION", ...}
  //   etc.
  {
    model::proto::GenericHyperParameters hp;

    // Example: unlimited depth
    {
      auto& field = *hp.add_fields();
      field.set_name("max_depth");
      field.mutable_value()->set_integer(-1);
    }

    // Example: minimum of 2 samples per node
    {
      auto& field = *hp.add_fields();
      field.set_name("min_examples");
      field.mutable_value()->set_integer(2);
    }

    // Turn on "SPARSE_OBLIQUE" splits for oblique training
    {
      auto& field = *hp.add_fields();
      field.set_name("split_axis");
      // The possible values are "AXIS_ALIGNED", "SPARSE_OBLIQUE", "MHLD_OBLIQUE"
      field.mutable_value()->set_categorical("SPARSE_OBLIQUE");
    }

    // ... Add more HPC fields if you want. E.g. "store_detailed_label_distribution"

    // Actually set them on the learner
    absl::Status hparam_status = learner->SetHyperParameters(hp);
    if (!hparam_status.ok()) {
      return absl::InternalError(
          "Could not set single-tree hyperparams: " +
          std::string(hparam_status.message()));
    }
  }

  // 6) Train the model from the disk-based dataset + deployment config
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or =
      learner->TrainWithStatus("csv:" + csv_path, data_spec, deployment_config);
  if (!model_or.ok()) {
    return absl::InternalError("Training failed: " +
                               std::string(model_or.status().message()));
  }
  std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

  // 7) Print a short textual description
  {
    std::string description;
    model->AppendDescriptionAndStatistics(/*full_definition=*/false, &description);
    std::cout << "Model trained. Summary:\n" << description << std::endl;
  }

  // 8) Save the model to disk
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