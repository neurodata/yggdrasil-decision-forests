// single_oblique_decision_tree_train.cc
//
// Example usage:
//   single_oblique_decision_tree_train /path/to/train.csv label_col /path/to/out_model

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
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"

// If you want to use oblique splits.
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_split.proto.h"

using namespace yggdrasil_decision_forests;

absl::Status TrainSingleObliqueDecisionTree(const std::string& csv_path,
                                            const std::string& label_column_name,
                                            const std::string& output_model_dir) {
  // 1) Create a data specification for the CSV dataset.
  dataset::proto::DataSpecification data_spec;
  {
    std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
    // "csv:" prefix indicates a CSV dataset recognized by YDF.
    // Optionally configure a DataSpecificationGuide for the label column
    dataset::proto::DataSpecificationGuide guide;
    {
      auto* label_guide = guide.add_column_guides();
      label_guide->set_column_name(label_column_name);
      label_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);
    }

    // CreateDataSpec populates 'data_spec'.
    dataset::CreateDataSpec("csv:" + csv_path,
                            /*require_same_dataset_fields=*/false,
                            guide,
                            &data_spec);
    // Optional: Print data_spec to see what it inferred
    std::cout << dataset::PrintHumanReadable(data_spec) << std::endl;
  }

  // 2) Configure a single DecisionTree learner (NOT a random forest).
  model::proto::TrainingConfig train_config;
  // Pick "DECISION_TREE" instead of "RANDOM_FOREST"
  train_config.set_learner("DECISION_TREE");
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_column_name);

  // If you need single-threaded training, do so in the `DeploymentConfig`.
  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(1);

  // Access the specialized config:
  auto& dt_config =
      *train_config.MutableExtension(
          model::decision_tree::proto::decision_tree_config);

  // For an effectively unbounded tree:
  dt_config.set_max_depth(-1);  // -1 = no limit
  dt_config.set_min_examples(2);

  // If you want "honest" single-tree training, you can do:
  // dt_config.mutable_honest()->set_ratio_leaf_examples(0.5);
  // but that’s more common in an ensemble. For a single tree, it’s less typical.

  // OBLIQUE SPLITS: We can choose "sparse" or "MHLD" oblique.
  // Let’s do Sparse Oblique:
  dt_config.mutable_sparse_oblique_split()
           ->set_num_projections_exponent(1.0f);  // example tweak

  // 3) Create the learner
  std::unique_ptr<model::AbstractLearner> learner;
  {
    absl::Status get_learner_status =
        model::GetLearner(train_config, &learner);
    if (!get_learner_status.ok()) {
      return absl::InternalError("Could not create DecisionTree learner: " +
                                 std::string(get_learner_status.message()));
    }
    // Also set the deployment config (number of threads, etc.)
    learner->SetDeploymentConfig(deployment_config);
  }

  // 4) Train the model from disk-based dataset
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
    model->AppendDescriptionAndStatistics(/*full_definition=*/false,
                                          &description);
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