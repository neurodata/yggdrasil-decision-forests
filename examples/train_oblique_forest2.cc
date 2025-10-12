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
#include "yggdrasil_decision_forests/metric/report.h"


#include <random>
#include "absl/random/random.h"          // BitGen (xoshiro)
#include "absl/random/distributions.h"   // Gaussian()


/* #region ABSL Flags */
// CSV mode flags
ABSL_FLAG(std::string, train_csv, "",
          "Path to training CSV file (for csv mode). Must include --label_col.");

ABSL_FLAG(std::string, test_csv, "",
          "Path to testing CSV file (for csv mode). Must include --label_col.");

ABSL_FLAG(std::string, label_col, "label",
          "Name of label column (used in all modes).");
ABSL_FLAG(std::string, model_out_dir, "",
          "Path to output trained model directory (optional)."
          " If empty, model is not saved.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 1, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, -1,
          "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");
ABSL_FLAG(int, num_projections_exponent, 0.5,
          "Exponent to determine number of projections.");
ABSL_FLAG(int, min_examples, 1, "Min examples in splits");

ABSL_FLAG(bool, compute_oob_performances, false,
          "Whether to compute out-of-bag performances (only for csv mode).");

// multiple runs
ABSL_FLAG(int, num_runs, 10, "Number of runs with different random seeds.");
ABSL_FLAG(uint32_t, base_seed, 23, "Base seed for random forest training.");

using namespace yggdrasil_decision_forests;

/* #endregion */
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string label_col = absl::GetFlag(FLAGS_label_col);
  uint32_t seed = absl::GetFlag(FLAGS_base_seed);

  dataset::proto::DataSpecification data_spec;
  const auto csv_path_train = absl::GetFlag(FLAGS_train_csv);
  const auto csv_path_test = absl::GetFlag(FLAGS_test_csv);
 
  if (csv_path_train.empty()) {
    std::cerr << "--train_csv required in csv mode\n";
    return 1;
  }
  std::cout << "\n\nInferring DataSpec from train CSV: " << csv_path_train << "\n\n" << std::endl;
  std::cout << "\n\nInferring DataSpec from test CSV: " << csv_path_test << "\n\n" << std::endl;

  dataset::proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern(label_col);
  col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);

  dataset::CreateDataSpec(
      "csv:" + csv_path_train,
      /*require_same_dataset_fields=*/false,
      guide,
      &data_spec);
  dataset::CreateDataSpec(
      "csv:" + csv_path_test,
      /*require_same_dataset_fields=*/false,
      guide,
      &data_spec);
      

  // 2) Configure learner
  model::proto::TrainingConfig train_config;
  train_config.set_learner("RANDOM_FOREST");
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_col);
  train_config.set_random_seed(seed);  
  std::cout << "\nHere seed is " << seed;




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
  rf.mutable_decision_tree()->set_min_examples(
    absl::GetFlag(FLAGS_min_examples));
  rf.set_bootstrap_training_dataset(true);
  rf.set_bootstrap_size_ratio(1.0);
  // sparse oblique setting
  auto* sos = rf.mutable_decision_tree()->mutable_sparse_oblique_split();
  sos->set_max_num_projections(
      absl::GetFlag(FLAGS_max_num_projections));
  sos->set_projection_density_factor(
      absl::GetFlag(FLAGS_projection_density_factor));
  sos->set_num_projections_exponent(
      absl::GetFlag(FLAGS_num_projections_exponent));
  

  rf.set_compute_oob_performances(
      absl::GetFlag(FLAGS_compute_oob_performances));

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(train_config, &learner, deploy_config));

  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or;

  
  model_or = learner->TrainWithStatus("csv:" + csv_path_train, data_spec);

  if (!model_or.ok()) {
    std::cerr << "Training failed: " << model_or.status().message() << std::endl;
    return 1;
  }
  auto model_ptr = std::move(model_or.value());

  // evaluation
  dataset::VerticalDataset test_dataset;
  QCHECK_OK(dataset::LoadVerticalDataset("csv:" +csv_path_test,
                                              model_ptr->data_spec(),
                                              &test_dataset));

  utils::RandomEngine rnd;
  const auto evaluation = model_ptr->Evaluate(test_dataset, {}, &rnd);

  // Save the evaluation in a text file 
  std::string evaluation_report = metric::TextReport(evaluation).value();
  std::string evaluation_path = file::JoinPath("/tmp/obq_forest", "evaluation.pbtxt");
  QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                             evaluation_report));
  LOG(INFO) << "Evaluation:\n" << evaluation_report;

  return 0;
}
