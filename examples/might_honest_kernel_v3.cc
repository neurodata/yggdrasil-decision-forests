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
//Oblique setting
ABSL_FLAG(float, num_projections_exponent, 1.5,
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
std::vector<float> ConvertToPosteriorsFromOOBFile(const std::vector<std::vector<float>>& oob_predictions) {
    std::vector<float> posteriors;
    posteriors.reserve(oob_predictions.size());
    
    for (const auto& pred_row : oob_predictions) {
        if (pred_row.size() >= 2) {
            // For binary classification, pred_row[0] = class 0 prob, pred_row[1] = class 1 prob
            float class_1_prob = pred_row[1];
            posteriors.push_back(class_1_prob);
        } else {
            posteriors.push_back(0.0f);
        }
    }
    
    return posteriors;
}


float ComputeSensitivityAtSpecificity(
    const std::vector<float>& posteriors,
    const ydf::dataset::VerticalDataset* dataset,
    int label_col_idx,
    float target_specificity = 0.98f) {
    
    const auto* label_column = dataset->ColumnWithCast<
        ydf::dataset::VerticalDataset::CategoricalColumn>(label_col_idx);
    
    std::vector<std::pair<float, int>> score_label_pairs;
    score_label_pairs.reserve(posteriors.size());
    
    // Build score-label pairs
    for (size_t i = 0; i < posteriors.size(); ++i) {
        if (i < label_column->values().size()) {
            int true_label = label_column->values()[i] - 1;  // Convert to 0/1
            score_label_pairs.emplace_back(posteriors[i], true_label);
        }
    }
    
    std::sort(score_label_pairs.begin(), score_label_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    int P = 0, N = 0;
    for (const auto& pair : score_label_pairs) {
        if (pair.second == 1) P++;
        else N++;
    }
    
    // Find best sensitivity at target specificity
    int TP = 0, FP = 0;
    float best_sensitivity = 0.0f;
    
    for (const auto& pair : score_label_pairs) {
        if (pair.second == 1) TP++;
        else FP++;
        
        float specificity = (N > 0) ? (float)(N - FP) / N : 1.0f;
        if (specificity >= target_specificity) {
            float sensitivity = (P > 0) ? (float)TP / P : 0.0f;
            best_sensitivity = sensitivity;
        } else {
            break;
        }
    }
    
    return best_sensitivity;
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

    auto& rf_config = *train_config.MutableExtension(
        ydf::model::random_forest::proto::random_forest_config);
    
    // MIGHT Core Settings 
    rf_config.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    rf_config.set_bootstrap_training_dataset(true);
    rf_config.set_bootstrap_size_ratio(absl::GetFlag(FLAGS_bootstrap_ratio));
    rf_config.set_kernel_method(true);
    rf_config.set_winner_take_all_inference(false);
    rf_config.set_compute_oob_performances(true);
    rf_config.set_export_oob_prediction_path("csv:" + oob_predictions_path);

    // Oblique and Honest settings
    auto* sos = rf_config.mutable_decision_tree()->mutable_sparse_oblique_split();
    sos->set_num_projections_exponent(absl::GetFlag(FLAGS_num_projections_exponent));
    
    auto* dt_config = rf_config.mutable_decision_tree();
    auto* honest_config = dt_config->mutable_honest();
    honest_config->set_ratio_leaf_examples(absl::GetFlag(FLAGS_honest_ratio));
    honest_config->set_fixed_separation(false);

    // Train model
    std::unique_ptr<ydf::model::AbstractLearner> learner;
    if (!ydf::model::GetLearner(train_config, &learner).ok()) {
        LOG(ERROR) << "Failed to create learner for run " << run_id;
        return 0.0f;
    }
    
    auto model_result = learner->TrainWithStatus("csv:" + train_path, data_spec);
    if (!model_result.ok()) {
        LOG(ERROR) << "Training failed for run " << run_id;
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
        
    } else {
        LOG(ERROR) << "No successful runs completed";
        return 1;
    }

    LOG(INFO) << "MIGHT Multiple Runs Analysis Completed";
    return 0;
}

