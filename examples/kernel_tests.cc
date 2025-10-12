/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Honest Forest with Kernel Method training example.
//
// This program demonstrates:
//   - Training an Honest Random Forest with Kernel Method
//   - Evaluating the model on test dataset
//   - Saving the trained model
//
// Usage example:
//   bazel build //examples:train_honest_kernel_forest
//   ./bazel-bin/examples/train_honest_kernel_forest --alsologtostderr

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

// Flags
ABSL_FLAG(std::string, dataset_dir,
          "cv_exports",
          "Input directory containing train.csv and test.csv");

ABSL_FLAG(std::string, output_dir, "/tmp/kernel_test",
          "Output directory to save the model and results");



// Kernel Method 
ABSL_FLAG(bool, enable_kernel, false, "Whether to use kernel method");

// Random Forest 
ABSL_FLAG(int, num_trees, 1000, "Number of trees");
ABSL_FLAG(bool, winner_take_all, false, "Winner take all inference");

namespace ydf = yggdrasil_decision_forests;

int main(int argc, char** argv) {
  // Enable the logging 
  InitLogging(argv[0], &argc, &argv, true);

  // Read flags 
  const std::string dataset_dir = absl::GetFlag(FLAGS_dataset_dir);
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  // Training and testing dataset paths 
  const auto train_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "fold1_train.csv"));
  const auto test_path =
      absl::StrCat("csv:", file::JoinPath(dataset_dir, "fold1_test.csv"));

  // Create output directory 
  QCHECK_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));

  LOG(INFO) << "=== Kernel Method Training ===";
  LOG(INFO) << "Training data: " << train_path;
  LOG(INFO) << "Test data: " << test_path;

  // Scan dataset to create a dataspec 
  LOG(INFO) << "Create dataspec";
  const auto dataspec_path = file::JoinPath(output_dir, "dataspec.pbtxt");
  
  ydf::dataset::proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("label");
  col_guide->set_type(ydf::dataset::proto::ColumnType::CATEGORICAL);
  
  const auto dataspec = ydf::dataset::CreateDataSpec(train_path,guide).value();
  QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

  

  // Configure the learner 
  LOG(INFO) << "Configure Kernel Method setting";
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label("label");

  auto& rf_config = *train_config.MutableExtension(
      ydf::model::random_forest::proto::random_forest_config);
  
  rf_config.set_num_trees(absl::GetFlag(FLAGS_num_trees));
  rf_config.set_winner_take_all_inference(absl::GetFlag(FLAGS_winner_take_all));
  rf_config.set_bootstrap_training_dataset(false);

  if (absl::GetFlag(FLAGS_enable_kernel)) {
    LOG(INFO) << "Enabling Kernel Method";
    rf_config.set_kernel_method(true);
  }
  else {
    LOG(INFO) << "Disabling Kernel Method";
    rf_config.set_kernel_method(false);
  }

  
  // Create learner 
  std::unique_ptr<ydf::model::AbstractLearner> learner;
  CHECK_OK(ydf::model::GetLearner(train_config, &learner));

  // Train model 
  LOG(INFO) << "Train model";
  auto model = learner->TrainWithStatus(train_path, dataspec).value();

  // Save the model 
  LOG(INFO) << "Export the model";
  const auto model_path = file::JoinPath(output_dir, "model");
  QCHECK_OK(ydf::model::SaveModel(model_path, *model));

  // Show details about model 
  std::string model_description = model->DescriptionAndStatistics();
  LOG(INFO) << "Model:\n" << model_description;
  QCHECK_OK(
      file::SetContent(absl::StrCat(model_path, ".txt"), model_description));

      

  LOG(INFO) << "===  Kernel Method completed successfully! ===";
  LOG(INFO) << "The results are available in " << output_dir;

  return 0;
}

