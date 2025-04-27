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

#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
// #include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.h"


using namespace yggdrasil_decision_forests;

absl::Status TrainRandomForest(const std::string& csv_path,
                               const std::string& label_column_name,
                               const std::string& output_model_dir) {

    // 1) **** Input & Parse CSV **** - Create a data specification for the CSV dataset.
    dataset::proto::DataSpecification data_spec;
    {
    std::cout << "Inferring DataSpec from CSV: " << csv_path << std::endl;
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
    // (Optional) Print the resulting dataspec:
    // std::cout << dataset::PrintHumanReadable(data_spec) << std::endl;
    }

    // 2) **** Configure a RandomForest learner object. No training yet ****
    // Ariel - TrainingConfig is defined in learner/abstract_learner.proto
    model::proto::TrainingConfig train_config;
    // train_config.set_learner("RANDOM_FOREST");

    // Decision Tree
    train_config.set_learner("CART"); // TODO Debug Decision Tree
    train_config.set_task(model::proto::Task::CLASSIFICATION);
    train_config.set_label(label_column_name);

    // 2a) DT Hyperparameters

    // Create the learner.
    std::unique_ptr<model::AbstractLearner> learner;
    {
      // auto& rf_config = *train_config.MutableExtension(
      //     model::random_forest::proto::random_forest_config);
        auto& dt_config = *train_config.MutableExtension(
          model::decision_tree::proto::decision_tree_config);

          dt_config.set_max_depth(-1);  // -1 => unlimited


      // Set number of Projections and Nonzeros
        // Enable oblique splits:
        dt_config.mutable_sparse_oblique_split();//->set_max_num_projections(5000);

        // Fix num projections
        dt_config.mutable_sparse_oblique_split()
            ->set_max_num_projections(1000);

        // Projection Density Factor
        dt_config.mutable_sparse_oblique_split()
            ->set_projection_density_factor(128.0f);

        // Should be n_features=2523^1 (MIGHT) > 1000=max_n_projections => n_projections should be = 1000
        dt_config.mutable_sparse_oblique_split()
          ->set_num_projections_exponent(1);

      // 2c - Create Learner
        {
          absl::Status get_learner_status =
              model::GetLearner(train_config, &learner);
          if (!get_learner_status.ok()) {
            return absl::InternalError("Could not create RandomForest learner: " +
                                      std::string(get_learner_status.message()));
          }
        }

      // 2d) Deployment Config - num. threads
        model::proto::DeploymentConfig deployment_config;
        deployment_config.set_num_threads(1);
  }

    // 3) ******************** Initiate Training ********************

            // Can't use "model" as a object name - it's a class name
    absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);

    // 4) - Post-training cleanup
    if (!model_or.ok()) {    return absl::InternalError("Training failed: " + std::string(model_or.status().message()));  }

    // Ariel: What does this do?
    std::unique_ptr<model::AbstractModel> model = std::move(model_or.value());

    // 5) Print a short description of the trained model
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
