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

// MIGHT (Multidimensional Informed Generalized Hypothesis Testing) Implementation
// This demonstrates MIGHT using YDF's existing Honest Random Forest with Kernel Method
// without any modifications to YDF core functions.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <random>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, dataset_dir, "/might_data/", 
          "Input directory ");
ABSL_FLAG(std::string, output_dir, "./might_results", 
          "Output directory to save results");
ABSL_FLAG(std::string, label_col, "label", "Label column name");

// MIGHT Core Parameters 
ABSL_FLAG(int, num_trees, 1000, "Number of trees (B in MIGHT paper)");
ABSL_FLAG(float, bootstrap_ratio, 1.6f, "Bootstrap ratio (β in MIGHT paper)");
ABSL_FLAG(float, honest_ratio, 0.367f, "Honest ratio (s in MIGHT paper)");
ABSL_FLAG(float, target_specificity, 0.98f, "Target specificity for S@specificity");
// Common Flags
ABSL_FLAG(int, tree_depth, -1,
          "Maximum depth of trees (-1 for unlimited).");
ABSL_FLAG(int, num_threads, 6, "Number of threads to use.");


// Oblique split parameters (only used when feature_split_type = "Oblique")
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");
ABSL_FLAG(float, num_projections_exponent, .5,
          "Exponent to determine number of projections.");




ABSL_FLAG(int, random_seed, 42, "Base random seed");
ABSL_FLAG(int, num_runs, 10, "Number of runs with different random seeds");

namespace ydf = yggdrasil_decision_forests;

// Parse OOB predictions from exported CSV file
std::vector<std::vector<float>> ParseOOBPredictionsFromFile(const std::string& filepath) {
    std::vector<std::vector<float>> predictions;
    std::ifstream file(filepath); // export_oob_prediction_path
    std::string line;
    
    if (std::getline(file, line)) {
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stof(cell));
                } catch (const std::exception& e) {
                    row.push_back(0.0f);  
                }
            }
            predictions.push_back(row);
        }
    }
    
    return predictions;
}

// Convert OOB predictions to posteriors for binary classification
std::vector<float> ConvertToPosteriorsFromOOBFile(
    const std::vector<std::vector<float>>& oob_predictions) {
  std::vector<float> scores;
  scores.reserve(oob_predictions.size());
  for (const auto& row : oob_predictions) {
    if (row.size() < 2) continue;                  // 缺两列，跳过
    const double neg = row[0], pos = row[1];
    const double tot = neg + pos;
    if (tot <= 0.0) continue;                      // 没有OOB投票，跳过
    scores.push_back(static_cast<float>(pos / tot)); // 概率而不是计数
  }
  return scores;
}



float ComputeSensitivityAtSpecificity(
    const std::vector<float>& posteriors,  // 建议已是 pos/(pos+neg)
    const ydf::dataset::VerticalDataset* dataset,
    int label_col_idx,
    float target_specificity) {

  using CatCol = ydf::dataset::VerticalDataset::CategoricalColumn;
  const auto* col = dataset->ColumnWithCast<CatCol>(label_col_idx);
  const auto& yvals = col->values();

  // 1) 选择正类 id：二分类常见为非零中的最大 id（原CSV的“1”）
  int pos_id = 0;
  for (int v : yvals) if (v > pos_id) pos_id = v;
  if (pos_id == 0) return 0.f;  // 全是缺失/单类

  // 2) 只收集“有有效分数 且 标签!=0”的样本
  std::vector<std::pair<float,int>> pairs;
  pairs.reserve(posteriors.size());
  const size_t m = std::min(posteriors.size(), yvals.size());
  for (size_t i = 0; i < m; ++i) {
    float s = posteriors[i];
    int   y = yvals[i];
    if (y == 0 || std::isnan(s)) continue;          // 丢掉缺失与 NaN
    pairs.emplace_back(s, (y == pos_id) ? 1 : 0);
  }
  if (pairs.empty()) return 0.f;

  // 3) （可选）方向自检：若分数方向反了就取反
  {
    int K = std::min<int>(100, pairs.size());
    std::nth_element(pairs.begin(), pairs.begin()+K, pairs.end(),
                     [](auto& a, auto& b){ return a.first > b.first; });
    int pos_top = 0; for (int i=0;i<K;++i) pos_top += pairs[i].second;
    std::nth_element(pairs.begin(), pairs.end()-K, pairs.end(),
                     [](auto& a, auto& b){ return a.first < b.first; });
    int pos_bot = 0; for (auto it=pairs.end()-K; it!=pairs.end(); ++it) pos_bot += it->second;
    if (pos_top < pos_bot) for (auto& p : pairs) p.first = -p.first;
  }

  // 4) 降序 + 按“distinct 分数块的末尾”更新（sklearn一致）
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const auto& a, const auto& b){ return a.first > b.first; });

  int P=0,N=0; for (auto& p : pairs) (p.second? ++P : ++N);
  if (P==0) return 0.f;
  if (N==0) return 1.f;

  const double max_fpr = 1.0 - static_cast<double>(target_specificity);
  double best_tpr = (max_fpr >= 0.0) ? 0.0 : 0.0;   // 起点 (0,0)

  int TP=0, FP=0;
  size_t i=0;
  while (i < pairs.size()) {
    const float v = pairs[i].first;
    size_t j = i;
    while (j < pairs.size() && pairs[j].first == v) ++j;  // 同分一块

    for (size_t k=i; k<j; ++k) (pairs[k].second ? ++TP : ++FP);

    const double fpr = static_cast<double>(FP)/N;
    const double tpr = static_cast<double>(TP)/P;
    if (fpr <= max_fpr && tpr > best_tpr) best_tpr = tpr;

    i = j;
  }

  if (best_tpr < 0.0) best_tpr = 0.0;
  if (best_tpr > 1.0) best_tpr = 1.0;
  return static_cast<float>(best_tpr);
}







float RunSingleMIGHTAnalysis(int run_id, int seed) {
    LOG(INFO) << "=== Run " << (run_id + 1) << " with seed " << seed << " ===";
    
    const std::string dataset_dir = absl::GetFlag(FLAGS_dataset_dir);
    const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
    const std::string label_col = absl::GetFlag(FLAGS_label_col);
    
    const auto train_path = dataset_dir;
    const auto oob_predictions_path = file::JoinPath(output_dir, 
        "oob_predictions_run_" + std::to_string(run_id) + ".csv");
    LOG(INFO) << "OOB Prediction Path " << oob_predictions_path << "===";
    
    ydf::dataset::proto::DataSpecificationGuide guide;
    ydf::dataset::proto::DataSpecification data_spec;
    auto* col_guide = guide.add_column_guides();
    col_guide->set_column_name_pattern(label_col);
    col_guide->set_type(ydf::dataset::proto::ColumnType::CATEGORICAL);
    
    ydf::dataset::CreateDataSpec("csv:" + train_path, false, guide, &data_spec);
    
    // Find label column index
    int label_col_idx = -1;
    for (int i = 0; i < data_spec.columns_size(); ++i) {
        if (data_spec.columns(i).name() == label_col) {
            label_col_idx = i;
            break;
        }
    }
    if (label_col_idx == -1) {
        LOG(ERROR) << "Label column not found: " << label_col;
        return 0.0f;
    }

    // Configure MIGHT Random Forest 
    ydf::model::proto::TrainingConfig train_config;
    train_config.set_learner("RANDOM_FOREST");
    train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
    train_config.set_label(label_col);
    train_config.set_random_seed(seed);  

    ydf::model::proto::DeploymentConfig deploy_config;

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

    auto& rf_config = *train_config.MutableExtension(
        ydf::model::random_forest::proto::random_forest_config);
    
    // MIGHT Core Settings 
    rf_config.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    rf_config.mutable_decision_tree()->set_max_depth(
      absl::GetFlag(FLAGS_tree_depth));
    rf_config.set_bootstrap_training_dataset(true);
    rf_config.set_bootstrap_size_ratio(absl::GetFlag(FLAGS_bootstrap_ratio));
    rf_config.set_kernel_method(true);
    rf_config.set_winner_take_all_inference(false);
    rf_config.set_compute_oob_performances(true);
    rf_config.set_export_oob_prediction_path("csv:" + oob_predictions_path);

    //rf_config.mutable_decision_tree()->mutable_growing_strategy_best_first_global()->set_max_num_nodes(-1);

    //rf_config.mutable_decision_tree()->mutable_growing_strategy_best_first_global();
    // honest trees are not (yet) supported with growing_strategy_best_first_global strategy
    rf_config.mutable_decision_tree()->set_min_examples(1);

    // Oblique and Honest settings
    auto* sos = rf_config.mutable_decision_tree()->mutable_sparse_oblique_split();
    sos->set_max_num_projections(
        absl::GetFlag(FLAGS_max_num_projections));
    sos->set_projection_density_factor(
        absl::GetFlag(FLAGS_projection_density_factor));
    sos->set_num_projections_exponent(
        absl::GetFlag(FLAGS_num_projections_exponent));
    
    auto* dt_config = rf_config.mutable_decision_tree();
    auto* honest_config = dt_config->mutable_honest();
    honest_config->set_ratio_leaf_examples(absl::GetFlag(FLAGS_honest_ratio));
    honest_config->set_fixed_separation(false);

    // Train model
    std::unique_ptr<ydf::model::AbstractLearner> learner;
    if (!ydf::model::GetLearner(train_config, &learner, deploy_config).ok()) {
        LOG(ERROR) << "Failed to create learner for run " << run_id;
        return 0.0f;
    }
    
    auto model_result = learner->TrainWithStatus("csv:" + train_path, data_spec);
    if (!model_result.ok()) {
        LOG(ERROR) << "Training failed for run " << run_id << model_result.status().message() << std::endl;
        return 0.0f;
    }
    
    // Parse OOB predictions
    auto oob_predictions_data = ParseOOBPredictionsFromFile(oob_predictions_path);
    if (oob_predictions_data.empty()) {
        LOG(ERROR) << "No OOB predictions for run " << run_id;
        return 0.0f;
    }
    
    ydf::dataset::VerticalDataset train_dataset;
    if (!ydf::dataset::LoadVerticalDataset("csv:" + train_path, data_spec, &train_dataset).ok()) {
        LOG(ERROR) << "Failed to load dataset for run " << run_id;
        return 0.0f;
    }
    
    // Compute MIGHT statistics
    auto posteriors = ConvertToPosteriorsFromOOBFile(oob_predictions_data);
    float s_at_target = ComputeSensitivityAtSpecificity(
        posteriors, &train_dataset, label_col_idx, absl::GetFlag(FLAGS_target_specificity));
    
    LOG(INFO) << "Run " << (run_id + 1) << " S@" 
              << absl::GetFlag(FLAGS_target_specificity) * 100 
              << "%: " << s_at_target;

    return s_at_target;
}

int main(int argc, char** argv) {
    InitLogging(argv[0], &argc, &argv, true);

    const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
    const int num_runs = absl::GetFlag(FLAGS_num_runs);
    const int base_seed = absl::GetFlag(FLAGS_random_seed);
    
    QCHECK_OK(file::RecursivelyCreateDir(output_dir, file::Defaults()));

    LOG(INFO) << "=== MIGHT Multiple Runs Analysis ===";
    LOG(INFO) << "Number of runs: " << num_runs;
    LOG(INFO) << "Base seed: " << base_seed;
    LOG(INFO) << "Trees per run: " << absl::GetFlag(FLAGS_num_trees);
    LOG(INFO) << "Bootstrap ratio: " << absl::GetFlag(FLAGS_bootstrap_ratio);
    LOG(INFO) << "Honest ratio: " << absl::GetFlag(FLAGS_honest_ratio);

    // Run multiple analyses with different seeds
    std::vector<float> s_at_target_results;
    s_at_target_results.reserve(num_runs);
    
    for (int run = 0; run < num_runs; ++run) {
        int seed = base_seed + run;  
        float result = RunSingleMIGHTAnalysis(run, seed);
        
        if (result > 0.0f) {  
            s_at_target_results.push_back(result);
        }
    }
    
    // Compute summary statistics
    if (!s_at_target_results.empty()) {
        float sum = 0.0f;
        float min_val = s_at_target_results[0];
        float max_val = s_at_target_results[0];
        
        for (float val : s_at_target_results) {
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        float mean = sum / s_at_target_results.size();
        
        // Compute standard deviation
        float sum_sq_diff = 0.0f;
        for (float val : s_at_target_results) {
            float diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        float std_dev = std::sqrt(sum_sq_diff / s_at_target_results.size());
        
        LOG(INFO) << "=== MIGHT Multiple Runs Summary ===";
        LOG(INFO) << "Successful runs: " << s_at_target_results.size() << "/" << num_runs;
        LOG(INFO) << "S@" << absl::GetFlag(FLAGS_target_specificity) * 100 << "% Statistics:";
        LOG(INFO) << "  Mean: " << mean;
        LOG(INFO) << "  Std Dev: " << std_dev;
        LOG(INFO) << "  Min: " << min_val;
        LOG(INFO) << "  Max: " << max_val;
        LOG(INFO) << "  95% CI: [" << (mean - 1.96 * std_dev / std::sqrt(s_at_target_results.size()))
                  << ", " << (mean + 1.96 * std_dev / std::sqrt(s_at_target_results.size())) << "]";
        
        // Save detailed results
        const auto results_path = file::JoinPath(output_dir, "might_multiple_runs_results.txt");
        std::ostringstream report;
        report << "MIGHT Multiple Runs Analysis Results\n";
        report << "===================================\n\n";
        report << "Configuration:\n";
        report << "  Number of runs: " << num_runs << "\n";
        report << "  Trees per run: " << absl::GetFlag(FLAGS_num_trees) << "\n";
        report << "  Bootstrap ratio: " << absl::GetFlag(FLAGS_bootstrap_ratio) << "\n";
        report << "  Honest ratio: " << absl::GetFlag(FLAGS_honest_ratio) << "\n";
        report << "  Target specificity: " << absl::GetFlag(FLAGS_target_specificity) << "\n\n";
        
        report << "Results:\n";
        report << "  Successful runs: " << s_at_target_results.size() << "/" << num_runs << "\n";
        report << "  S@" << absl::GetFlag(FLAGS_target_specificity) * 100 << "% Statistics:\n";
        report << "    Mean: " << mean << "\n";
        report << "    Std Dev: " << std_dev << "\n";
        report << "    Min: " << min_val << "\n";
        report << "    Max: " << max_val << "\n";
        report << "    95% CI: [" << (mean - 1.96 * std_dev / std::sqrt(s_at_target_results.size()))
               << ", " << (mean + 1.96 * std_dev / std::sqrt(s_at_target_results.size())) << "]\n\n";
        
        report << "Individual Results:\n";
        for (size_t i = 0; i < s_at_target_results.size(); ++i) {
            report << "  Run " << (i + 1) << ": " << s_at_target_results[i] << "\n";
        }
        
        QCHECK_OK(file::SetContent(results_path, report.str()));
        LOG(INFO) << "Detailed results saved to: " << results_path;
        
        const auto csv_path = file::JoinPath(output_dir, "might_runs_data.csv");
        std::ostringstream csv;
        csv << "run,seed,s_at_target\n";
        for (size_t i = 0; i < s_at_target_results.size(); ++i) {
            csv << (i + 1) << "," << (base_seed + i) << "," << s_at_target_results[i] << "\n";
        }
        QCHECK_OK(file::SetContent(csv_path, csv.str()));
        LOG(INFO) << "CSV data saved to: " << csv_path;
    } 
    return 0;
}


