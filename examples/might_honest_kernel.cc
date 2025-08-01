/*
 * MIGHT (Multidimensional Informed Generalized Hypothesis Testing) Implementation
 * 
 * This implements MIGHT using YDF's Honest Random Forest with Kernel Method.
 * MIGHT provides accurate ROC curve estimation and statistics like S@98 with 
 * theoretical guarantees.
 */

#include <algorithm>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/distribution.h"

// MIGHT dataset and output parameters
ABSL_FLAG(std::string, data_csv, "", "Path to dataset CSV file");
ABSL_FLAG(std::string, label_col, "label", "Label column name");
ABSL_FLAG(std::string, output_dir, "./might_results", "Output directory");

// MIGHT algorithm parameters (core)
ABSL_FLAG(int, num_trees, 1000, "Number of MIGHTY trees (B in algorithm)");
ABSL_FLAG(float, calibrating_fraction, 0.367, "Calibrating set fraction (s in paper)");
ABSL_FLAG(float, bootstrap_fraction, 1.6f, "Bootstrap fraction (β in paper)");

ABSL_FLAG(float, target_specificity, 0.98f, "Target specificity for S@specificity");
ABSL_FLAG(bool, compute_full_roc, false, "Compute full ROC curve (expensive)");

// Honest Forest + Kernel Method parameters
ABSL_FLAG(bool, enable_kernel, true, "Enable kernel method aggregation");
//ABSL_FLAG(bool, enable_honest, false "Enable honest forest");
// ABSL_FLAG(float, honest_ratio, 0.367f, "Honest ratio for leaf examples");
// ABSL_FLAG(bool, honest_fixed_separation, false, "Fixed separation for honest trees");

// Performance and debugging
ABSL_FLAG(int, random_seed, 42, "Random seed for reproducibility");
ABSL_FLAG(bool, verbose_logging, false, "Enable verbose tree-level logging");

//Oblique setting
ABSL_FLAG(float, num_projections_exponent, 1.5,
          "Exponent to determine number of projections.");

ABSL_FLAG(int, n_runs, 10, "Number of runs for MIGHT analysis");


namespace ydf = yggdrasil_decision_forests;
using UnsignedExampleIdx = ydf::dataset::UnsignedExampleIdx;
using RandomEngine = ydf::utils::RandomEngine;

// Prediction Accumulator for MIGHT
struct MIGHTPredictionAccumulator {
    ydf::utils::IntegerDistribution<float> classification;
    int num_trees = 0;
};

struct MIGHTResults {
    std::vector<float> final_posteriors;
    float sensitivity_at_target_specificity;
    float estimated_auc;
    int num_samples_with_oob;
    
    std::vector<float> roc_tpr;
    std::vector<float> roc_fpr;
};

// Forward declarations
void SampleTrainingExamples(const UnsignedExampleIdx num_examples,
                            const UnsignedExampleIdx num_samples,
                            const bool with_replacement,
                            ydf::utils::RandomEngine* random,
                            std::vector<UnsignedExampleIdx>* selected);

void AddClassificationLeafToAccumulator(
    const bool winner_take_all_inference,
    const bool kernel_method,
    const ydf::model::decision_tree::proto::Node& node,
    ydf::utils::IntegerDistribution<float>* accumulator);

absl::Status UpdateMIGHTOOBPredictions(
    const ydf::dataset::VerticalDataset& train_dataset,
    const ydf::model::proto::TrainingConfig& config,
    const std::vector<UnsignedExampleIdx>& oob_indices,
    const std::vector<UnsignedExampleIdx>& calib_indices, // calculate raw count
    int label_col_idx,
    const bool winner_take_all_inference,
    const bool kernel_method,
    const ydf::model::decision_tree::DecisionTree& new_decision_tree,
    std::vector<MIGHTPredictionAccumulator>* oob_predictions
    );

class MIGHTImplementation {
public:
    explicit MIGHTImplementation(const std::string& data_csv, 
                                const std::string& label_col,
                                const std::string& output_dir)
        : data_path_(data_csv), label_col_(label_col), output_dir_(output_dir) {
        
        num_trees_ = absl::GetFlag(FLAGS_num_trees);
        calibrating_fraction_ = absl::GetFlag(FLAGS_calibrating_fraction);
        bootstrap_fraction_ = absl::GetFlag(FLAGS_bootstrap_fraction);
        target_specificity_ = absl::GetFlag(FLAGS_target_specificity);
        random_seed_ = absl::GetFlag(FLAGS_random_seed);
        
        LOG(INFO) << "MIGHT Configuration:";
        LOG(INFO) << "  Trees: " << num_trees_;
        LOG(INFO) << "  Calibrating fraction: " << calibrating_fraction_;
        LOG(INFO) << "  Kernel method: " << absl::GetFlag(FLAGS_enable_kernel);
        //LOG(INFO) << "  Honest forest: " << absl::GetFlag(FLAGS_enable_honest);
        LOG(INFO) << "  Target specificity: " << target_specificity_;
    }
    
    absl::StatusOr<MIGHTResults> RunMIGHT() {
        //LOG(INFO) << "=== Starting MIGHT Analysis ===";
        
        RETURN_IF_ERROR(LoadData());
        RETURN_IF_ERROR(PrepareAllDataSplits());
        RETURN_IF_ERROR(TrainMIGHTYForest());
        
        ASSIGN_OR_RETURN(auto results, ComputeFinalStatistics());
        RETURN_IF_ERROR(SaveResults(results));
        
        //LOG(INFO) << "=== MIGHT Analysis Completed ===";
        return results;
    }

private:
    // Config
    std::string data_path_, label_col_, output_dir_;
    int num_trees_, random_seed_;
    float calibrating_fraction_, bootstrap_fraction_, target_specificity_;
    
    std::unique_ptr<ydf::dataset::VerticalDataset> dataset_;
    ydf::dataset::proto::DataSpecification data_spec_;
    int label_col_idx_;
    
    // MIGHT state
    std::vector<std::vector<UnsignedExampleIdx>> trees_training_indices_;
    std::vector<std::vector<UnsignedExampleIdx>> trees_calib_indices_;
    std::vector<std::vector<UnsignedExampleIdx>> trees_oob_indices_;
    std::vector<MIGHTPredictionAccumulator> oob_predictions_;
    
    absl::Status LoadData() {
        LOG(INFO) << "Loading dataset: " << data_path_;
        
        const auto typed_path = absl::StrCat("csv:", data_path_);
        
        // Create dataspec with proper label guidance
        ydf::dataset::proto::DataSpecificationGuide guide;
        auto* col_guide = guide.add_column_guides();
        col_guide->set_column_name_pattern(label_col_);
        col_guide->set_type(ydf::dataset::proto::ColumnType::CATEGORICAL);
        
        ASSIGN_OR_RETURN(data_spec_, 
                 ydf::dataset::CreateDataSpec(typed_path, guide));

        dataset_ = std::make_unique<ydf::dataset::VerticalDataset>();
        RETURN_IF_ERROR(ydf::dataset::LoadVerticalDataset(typed_path, data_spec_, dataset_.get()));
        
        // Find label column index
        label_col_idx_ = -1;
        for (int i = 0; i < data_spec_.columns_size(); ++i) {
            if (data_spec_.columns(i).name() == label_col_) {
                label_col_idx_ = i;
                break;
            }
        }
        
        if (label_col_idx_ == -1) {
            return absl::InvalidArgumentError(
                absl::StrCat("Label column '", label_col_, "' not found"));
        }
        
        LOG(INFO) << "Dataset loaded successfully:";
        LOG(INFO) << "  Samples: " << dataset_->nrow();
        LOG(INFO) << "  Features: " << dataset_->ncol()-1;
        LOG(INFO) << "  Label column: " << label_col_ << " (index " << label_col_idx_ << ")";
        
        return absl::OkStatus();
    }
    
    absl::Status PrepareAllDataSplits() {
        LOG(INFO) << "Preparing MIGHT data splits using YDF sampling functions";
        
        trees_training_indices_.clear();
        trees_calib_indices_.clear();
        trees_oob_indices_.clear();

        trees_training_indices_.reserve(num_trees_);
        trees_calib_indices_.reserve(num_trees_);
        trees_oob_indices_.reserve(num_trees_);
        
        // initialize
        oob_predictions_.assign(dataset_->nrow(), MIGHTPredictionAccumulator());
        
        int num_classes = 2; 
        if (label_col_idx_ >= 0 && label_col_idx_ < data_spec_.columns_size()) {
            const auto& label_spec = data_spec_.columns(label_col_idx_);
            if (label_spec.type() == ydf::dataset::proto::ColumnType::CATEGORICAL) {
                num_classes = label_spec.categorical().number_of_unique_values();
            }
        }
        
        for (auto& acc : oob_predictions_) {
            acc.classification.SetNumClasses(num_classes);
        }
        
        ydf::utils::RandomEngine global_random(random_seed_);
        
        for (int tree_idx = 0; tree_idx < num_trees_; ++tree_idx) {
            ydf::utils::RandomEngine tree_random(global_random());
            
            std::vector<UnsignedExampleIdx> bootstrap_indices;
            int bootstrap_size = static_cast<int>(bootstrap_fraction_ * dataset_->nrow()); // 1.6*
            
            SampleTrainingExamples(dataset_->nrow(), bootstrap_size, true, // with_replacement=true
                                 &tree_random, &bootstrap_indices); // indices should with replication
            // bootstrap output: inbag indices
            
            // use inbag indices to calculate oob indices
            auto oob_indices = ComputeOOBIndices(bootstrap_indices);
            
            // honest splitting-> split indices, calib indices
            auto [train_indices, calib_indices] = PerformHonestSplit(bootstrap_indices, tree_random);
            
            trees_training_indices_.push_back(train_indices);
            trees_calib_indices_.push_back(calib_indices);
            trees_oob_indices_.push_back(oob_indices);

            
            if (tree_idx % 5000 == 0) {
                LOG(INFO) << "Prepared splits for tree " << tree_idx << "/" << num_trees_;
            }
        }
        
        if (!trees_training_indices_.empty()) {
            LOG(INFO) << "First tree statistics:";
            LOG(INFO) << "  Training samples: " << trees_training_indices_[0].size();
            LOG(INFO) << "Calibrated samples: " << trees_calib_indices_[0].size();
            LOG(INFO) << "  OOB samples: " << trees_oob_indices_[0].size();
        }
        
        return absl::OkStatus();
    }
    
    std::vector<UnsignedExampleIdx> ComputeOOBIndices(
        const std::vector<UnsignedExampleIdx>& bootstrap_indices) {
        
        std::vector<bool> in_bootstrap(dataset_->nrow(), false);
        for (auto idx : bootstrap_indices) {
            in_bootstrap[idx] = true;
        }
        
        std::vector<UnsignedExampleIdx> oob_indices;
        for (UnsignedExampleIdx i = 0; i < static_cast<UnsignedExampleIdx>(dataset_->nrow()); ++i) {
            if (!in_bootstrap[i]) {
                oob_indices.push_back(i);
            }
        }
        
        return oob_indices;
    }
    
    std::pair<std::vector<UnsignedExampleIdx>,
                std::vector<UnsignedExampleIdx>>
    PerformHonestSplit(const std::vector<UnsignedExampleIdx>& bootstrap_indices,
                    ydf::utils::RandomEngine& random) {
        
        std::vector<UnsignedExampleIdx> shuffled_bootstrap = bootstrap_indices;
        std::shuffle(shuffled_bootstrap.begin(), shuffled_bootstrap.end(), random);
        
        int calib_size = static_cast<int>(calibrating_fraction_ * shuffled_bootstrap.size());
        
        std::vector<UnsignedExampleIdx> calib_indices(
            shuffled_bootstrap.begin(), 
            shuffled_bootstrap.begin() + calib_size);
        
        std::vector<UnsignedExampleIdx> train_indices(
            shuffled_bootstrap.begin() + calib_size, 
            shuffled_bootstrap.end());
        
        return {train_indices, calib_indices};
    }
    
    absl::Status TrainMIGHTYForest() {
        LOG(INFO) << "Training MIGHTY Forest using YDF functions";
        
        for (int tree_idx = 0; tree_idx < num_trees_; ++tree_idx) {
            RETURN_IF_ERROR(TrainSingleMIGHTYTreeWithYDF(tree_idx));
            
            if (tree_idx % 1000 == 0) {
                LOG(INFO) << "Trained tree " << tree_idx << "/" << num_trees_;
            }
        }
        
        LOG(INFO) << "MIGHTY Forest training completed";
        return absl::OkStatus();
    }
    
    absl::Status TrainSingleMIGHTYTreeWithYDF(int tree_idx) {
        const auto& train_indices = trees_training_indices_[tree_idx]; //split indices
        const auto& calib_indices = trees_calib_indices_[tree_idx]; // calib indices
        const auto& oob_indices = trees_oob_indices_[tree_idx];
        (void)oob_indices;  
        
        ASSIGN_OR_RETURN(auto train_subset, CreateDataSubset(train_indices)); // use train indices
        //ASSIGN_OR_RETURN(auto calib_subset, CreateDataSubset(calib_indices)); // use calib indices

        
        ydf::model::proto::TrainingConfig config;
        config.set_learner("RANDOM_FOREST");
        config.set_task(ydf::model::proto::Task::CLASSIFICATION);
        config.set_label(label_col_);
        
        auto& rf_config = *config.MutableExtension(
            ydf::model::random_forest::proto::random_forest_config);
        
        rf_config.set_num_trees(1);
        rf_config.set_winner_take_all_inference(false);
        rf_config.set_kernel_method(absl::GetFlag(FLAGS_enable_kernel));
        rf_config.set_bootstrap_training_dataset(false);
        rf_config.set_compute_oob_performances(false);

        auto* sos = rf_config.mutable_decision_tree()->mutable_sparse_oblique_split();
        sos->set_num_projections_exponent(
            absl::GetFlag(FLAGS_num_projections_exponent));

        // if (absl::GetFlag(FLAGS_enable_honest)) {
        //     auto* dt_config = rf_config.mutable_decision_tree();
        //     auto* honest_config = dt_config->mutable_honest();
        //     honest_config->set_ratio_leaf_examples(absl::GetFlag(FLAGS_honest_ratio));
        //     honest_config->set_fixed_separation(absl::GetFlag(FLAGS_honest_fixed_separation));
        // }
        // else {
        //     continue;
        // }

    
        
        ASSIGN_OR_RETURN(const auto learner, ydf::model::GetLearner(config));
        ASSIGN_OR_RETURN(auto model, learner->TrainWithStatus(*train_subset));
        
        const auto* rf_model = dynamic_cast<const ydf::model::random_forest::RandomForestModel*>(model.get());
        if (!rf_model || rf_model->decision_trees().empty()) {
            return absl::InternalError("Failed to get decision tree from model");
        }
        const auto& decision_tree = *rf_model->decision_trees()[0]; //finish training tree
        
        
        
        RETURN_IF_ERROR(UpdateMIGHTOOBPredictions(
            *dataset_, config, oob_indices,calib_indices, label_col_idx_,
            false, 
            absl::GetFlag(FLAGS_enable_kernel), 
            decision_tree, &oob_predictions_));
        
        return absl::OkStatus();
    }
    
    absl::StatusOr<std::unique_ptr<ydf::dataset::VerticalDataset>> 
    CreateDataSubset(const std::vector<UnsignedExampleIdx>& indices) {
        if (indices.empty()) {
            return absl::InvalidArgumentError("Cannot create subset with empty indices");
        }
        
        auto subset = std::make_unique<ydf::dataset::VerticalDataset>();
        subset->set_data_spec(data_spec_);
        RETURN_IF_ERROR(subset->CreateColumnsFromDataspec());
        subset->Resize(indices.size());
        
        // get data for specified indices
        for (int col = 0; col < dataset_->ncol(); ++col) {
            const auto* src_column = dataset_->column(col);
            auto* dst_column = subset->mutable_column(col);
            
            if (src_column->type() == ydf::dataset::proto::ColumnType::NUMERICAL) {
                const auto* src_num = static_cast<const ydf::dataset::VerticalDataset::NumericalColumn*>(src_column);
                auto* dst_num = static_cast<ydf::dataset::VerticalDataset::NumericalColumn*>(dst_column);

                
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= src_num->values().size()) {
                        return absl::OutOfRangeError(
                            absl::StrCat("Index ", indices[i], " out of range for column ", col));
                    }
                    dst_num->mutable_values()->at(i) = src_num->values().at(indices[i]);
                }
                
            } else if (src_column->type() == ydf::dataset::proto::ColumnType::CATEGORICAL) {
                const auto* src_cat = static_cast<const ydf::dataset::VerticalDataset::CategoricalColumn*>(src_column);
                auto* dst_cat = static_cast<ydf::dataset::VerticalDataset::CategoricalColumn*>(dst_column);
                
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= src_cat->values().size()) {
                        return absl::OutOfRangeError(
                            absl::StrCat("Index ", indices[i], " out of range for column ", col));
                    }
                    dst_cat->mutable_values()->at(i) = src_cat->values().at(indices[i]);
                }
            }
        }
        
        return subset;
    }
    
    absl::StatusOr<MIGHTResults> ComputeFinalStatistics() {
        LOG(INFO) << "Computing final MIGHT statistics using YDF aggregation";
        
        MIGHTResults results;
        
        auto posteriors_with_index = ExtractFinalPosteriorsWithIndex(); // <posterior, original_sample_index>

        // 保存 posterior,index 到 csv
        std::ofstream fout("./might_data/posteriors_with_index.csv");
        fout << "posterior,index\n";  // header
        for (const auto& [posterior, idx] : posteriors_with_index) {
            fout << posterior << "," << idx << "\n";
        }
        fout.close();

        
        // seperate posterior and idx
        results.final_posteriors.clear();
        std::vector<int> sample_indices;
        for (const auto& [posterior, idx] : posteriors_with_index) {
            results.final_posteriors.push_back(posterior);
            sample_indices.push_back(idx);
        }
        
        results.num_samples_with_oob = results.final_posteriors.size();
        
        if (results.final_posteriors.empty()) {
            return absl::InternalError("No samples have OOB predictions");
        }
        
        results.sensitivity_at_target_specificity = 
            ComputeSensitivityAtSpecificityWithIndices(results.final_posteriors, sample_indices,0.98);
        
        return results;
    }
    
    

    std::vector<std::pair<float, int>> ExtractFinalPosteriorsWithIndex() {
        std::vector<std::pair<float, int>> posteriors_with_index;  // <posterior, original_sample_index>
        posteriors_with_index.reserve(dataset_->nrow());

        
        for (size_t sample_idx = 0; sample_idx < oob_predictions_.size(); ++sample_idx) {
            const auto& acc = oob_predictions_[sample_idx];
            if (acc.num_trees == 0) {
                continue;  
            }

            
            float count_pos = acc.classification.count(2);  // label=2
            float count_neg = acc.classification.count(1);  // label=1
            float count_total = count_pos + count_neg;
            float posterior = 0.0f;
            if (count_total > 0) {
                posterior = count_pos / count_total;
            }
            
            posteriors_with_index.push_back({posterior, static_cast<int>(sample_idx)});
        }
        
        return posteriors_with_index;
    }

    
    
    float ComputeSensitivityAtSpecificityWithIndices(
        const std::vector<float>& posteriors, 
        const std::vector<int>& sample_indices,
        float target_specificity = 0.98) {
        const auto* label_column = dataset_->ColumnWithCast
            <ydf::dataset::VerticalDataset::CategoricalColumn>(label_col_idx_);
        
        std::vector<std::pair<float, int>> score_label_pairs;
        for (size_t i = 0; i < posteriors.size() && i < sample_indices.size(); ++i) {
            int sample_idx = sample_indices[i]; 
            int true_label = label_column->values()[sample_idx];  
            true_label = true_label - 1;  
            float posterior = posteriors[i];  
            score_label_pairs.push_back({posterior, true_label});
        }
       
        std::sort(score_label_pairs.begin(), score_label_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });
        int P = 0, N = 0;
        for (const auto& p : score_label_pairs) {
            if (p.second == 1) P++;
            else N++;
        }
        int TP = 0, FP = 0, TN = N, FN = P;
        float best_sens = 0.0, best_spec = 0.0, best_thresh = 0.0;

        for (size_t i = 0; i < score_label_pairs.size(); ++i) {
            int y = score_label_pairs[i].second;
            if (y == 1) {
                TP++; FN--;
            } else {
                FP++; TN--;
            }
            float specificity = (N > 0) ? (float)TN / (TN + FP) : 1.0;
            float sensitivity = (P > 0) ? (float)TP / (TP + FN) : 0.0;
            if (specificity >= target_specificity) {
                best_sens = sensitivity;
                best_spec = specificity;
                best_thresh = score_label_pairs[i].first;
            } else {
                break;
            }
        }
        LOG(INFO) << "S@" << target_specificity*100 << "%: " << best_sens
                << ", threshold=" << best_thresh
                << ", specificity=" << best_spec
                << ", P=" << P << ", N=" << N;
        return best_sens;
    }

    





    absl::Status SaveResults(const MIGHTResults& results) {
        try {
            std::filesystem::create_directories(output_dir_);
        } catch (const std::exception& e) {
            return absl::InternalError("Failed to create directory: " + std::string(e.what()));
        }
        
        std::string report = absl::StrFormat(
            "MIGHT Analysis Results\n"
            "======================\n\n"
            "Dataset: %s\n"
            "Label column: %s\n"
            "Total samples: %d\n"
            "Samples with OOB predictions: %d (%.1f%%)\n\n"
            "Algorithm Parameters:\n"
            "  Number of trees: %d\n"
            "  Calibrating fraction: %.3f\n"
            "  Bootstrap fraction: %.3f\n"
            "  Target specificity: %.3f\n"
            "  Kernel method: %s\n"
            "  Honest forest: %s\n"
            "  Honest ratio: %.3f\n\n"
            "Results:\n"
            "  S@%.2f: %.4f\n"
            "  Estimated AUC: %.4f\n\n"
            "Random seed: %d\n",
            data_path_, label_col_, dataset_->nrow(),
            results.num_samples_with_oob,
            100.0 * results.num_samples_with_oob / dataset_->nrow(),
            num_trees_, calibrating_fraction_, bootstrap_fraction_, target_specificity_,
            absl::GetFlag(FLAGS_enable_kernel) ? "enabled" : "disabled",
            //absl::GetFlag(FLAGS_enable_honest) ? "enabled" : "disabled",
            //absl::GetFlag(FLAGS_honest_ratio),
            target_specificity_, results.sensitivity_at_target_specificity,
            results.estimated_auc,
            random_seed_);
        
        const auto results_path = std::filesystem::path(output_dir_) / "might_results.txt";
        
        std::ofstream file(results_path);
        if (!file.is_open()) {
            return absl::InternalError("Failed to open results file: " + results_path.string());
        }
        file << report;
        file.close();
        
        LOG(INFO) << "Results saved to: " << results_path.string();
        return absl::OkStatus();
    }
};

// ============ Origin helper functions ============

void SampleTrainingExamples(const UnsignedExampleIdx num_examples,
                            const UnsignedExampleIdx num_samples,
                            const bool with_replacement,
                            ydf::utils::RandomEngine* random,
                            std::vector<UnsignedExampleIdx>* selected) {
    selected->resize(num_samples);

    if (with_replacement) {
        // bootstrapping with replacement
        std::uniform_int_distribution<UnsignedExampleIdx> example_idx_distrib(
            0, num_examples - 1);
        for (UnsignedExampleIdx sample_idx = 0; sample_idx < num_samples;
             sample_idx++) {
            (*selected)[sample_idx] = example_idx_distrib(*random);
        }
        std::sort(selected->begin(), selected->end());
    } else {
        selected->clear();
        selected->reserve(num_samples);
        // without replacement.
        std::uniform_real_distribution<float> dist_01;
        for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
             example_idx++) {
            const float proba_select =
                static_cast<float>(num_samples - selected->size()) /
                (num_examples - example_idx);
            if (dist_01(*random) < proba_select) {
                selected->push_back(example_idx);
            }
        }
    }
}

void AddClassificationLeafToAccumulator(
    const bool winner_take_all_inference,
    const bool kernel_method,
    const ydf::model::decision_tree::proto::Node& node,
    ydf::utils::IntegerDistribution<float>* accumulator) {
    if (winner_take_all_inference) {
        accumulator->Add(node.classifier().top_value());
    } else {
        DCHECK(node.classifier().has_distribution());
        if (kernel_method) {
            accumulator->AddProto(node.classifier().distribution());
        } else {
            accumulator->AddNormalizedProto(node.classifier().distribution());
        }
    }
}

absl::Status UpdateMIGHTOOBPredictions(
    const ydf::dataset::VerticalDataset& train_dataset, // original dataset
    const ydf::model::proto::TrainingConfig& config,
    const std::vector<UnsignedExampleIdx>& oob_indices,
    const std::vector<UnsignedExampleIdx>& calib_indices,
    int label_col_idx,
    const bool winner_take_all_inference,
    const bool kernel_method,
    const ydf::model::decision_tree::DecisionTree& new_decision_tree,
    std::vector<MIGHTPredictionAccumulator>* oob_predictions
    ) {
        // Build leaf_id -> label -> count from calibration samples
        std::unordered_map<int, std::unordered_map<int, int>> leaf_label_count;

        for (auto calib_idx : calib_indices) {
            const auto& leaf = new_decision_tree.GetLeafAlt(train_dataset, calib_idx);
            if (leaf.leaf_idx() < 0) {
                return absl::InternalError("Invalid leaf index in calibration");
            }
            int leaf_id = leaf.leaf_idx(); 
            int label = train_dataset
            .ColumnWithCast<ydf::dataset::VerticalDataset::CategoricalColumn>(label_col_idx)
            ->values()[calib_idx];
            leaf_label_count[leaf_id][label]++;
        }

        // OOB prediction aggregation
        for (auto oob_idx : oob_indices) {
            //LOG(INFO) << "oob_idx: " << oob_idx;
            const auto& leaf = new_decision_tree.GetLeafAlt(train_dataset, oob_idx);
            if (leaf.leaf_idx() < 0) {
                continue; 
            }
            int leaf_id = leaf.leaf_idx();

            // temperory node for leaf raw class distribution
            ydf::model::decision_tree::proto::Node node;
            auto* dist = node.mutable_classifier()->mutable_distribution();
            if (leaf_label_count.count(leaf_id)) {
                int total_count = 0;
                for (const auto& [label, count] : leaf_label_count[leaf_id]) {
                    dist->set_counts(label, count);
                    total_count += count;
                }
                dist->set_sum(total_count);
            } 

            // added to accumulator
            auto& accumulator = (*oob_predictions)[oob_idx];
            accumulator.num_trees++;
            AddClassificationLeafToAccumulator(
            winner_take_all_inference, kernel_method, node, &accumulator.classification);
        }

        return absl::OkStatus();
            
}


absl::Status ExportOOBPredictions(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& dataspec,
    const std::vector<PredictionAccumulator>& oob_predictions,
    absl::string_view typed_path) {
  // Create the dataspec that describes the exported prediction dataset.
  dataset::proto::DataSpecification pred_dataspec;

  // Buffer example.
  dataset::proto::Example example;

  // Number of classification classes. Unused if the label is not categorical.
  int num_label_classes = -1;

  const auto& label_spec = dataspec.columns(config_link.label());
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      num_label_classes = label_spec.categorical().number_of_unique_values();
      for (int i = 1 /skip the OOV/; i < num_label_classes; i++) {
        auto* col = pred_dataspec.add_columns();
        //col->set_name(dataset::CategoricalIdxToRepresentation(label_spec, i));
        col->set_name(absl::StrCat("rawcount", 
             dataset::CategoricalIdxToRepresentation(label_spec, i)));
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        example.add_attributes()->set_numerical(0);
      }
    } break;

    case model::proto::Task::REGRESSION: {
      auto* col = pred_dataspec.add_columns();
      col->set_name(label_spec.name());
      col->set_type(dataset::proto::ColumnType::NUMERICAL);
      example.add_attributes()->set_numerical(0);
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      num_label_classes = label_spec.categorical().number_of_unique_values();
      for (int i = 2 /skip the OOV and treatment/; i < num_label_classes;
           i++) {
        auto* col = pred_dataspec.add_columns();
        col->set_name(dataset::CategoricalIdxToRepresentation(label_spec, i));
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        example.add_attributes()->set_numerical(0);
      }
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      auto* col = pred_dataspec.add_columns();
      col->set_name(label_spec.name());
      col->set_type(dataset::proto::ColumnType::NUMERICAL);
      example.add_attributes()->set_numerical(0);
    } break;

    default:
      return absl::InvalidArgumentError(
          "Exporting oob-predictions not supported for this task");
  }

  ASSIGN_OR_RETURN(auto writer,
                   dataset::CreateExampleWriter(typed_path, pred_dataspec));

  // Write the predictions one by one.
  for (const auto& pred : oob_predictions) {
    switch (config.task()) {
      case model::proto::Task::CLASSIFICATION:
        DCHECK_EQ(pred.classification.NumClasses(), num_label_classes);
        for (int i = 1 /skip the OOV/; i < num_label_classes; i++) {
          // example.mutable_attributes(i - 1)->set_numerical(
          //     pred.classification.NumObservations() > 0
          //         ? pred.classification.SafeProportionOrMinusInfinity(i)
          //         : 0);
          example.mutable_attributes(i - 1)->set_numerical(
              pred.classification.NumObservations() > 0
                  ? static_cast<float>(pred.classification.count(i))  // directly use raw count
                  : 0);
        }
        break;

      case model::proto::Task::REGRESSION:
        example.mutable_attributes(0)->set_numerical(pred.regression);
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT:
        DCHECK_EQ(pred.uplift.size(), num_label_classes - 2);
        for (int i = 2; i < num_label_classes; i++) {
          example.mutable_attributes(i - 2)->set_numerical(pred.uplift[i - 2]);
        }
        break;

      case model::proto::Task::NUMERICAL_UPLIFT:
        DCHECK_EQ(pred.uplift.size(), 1);
        example.mutable_attributes(0)->set_numerical(pred.uplift[0]);
        break;

      default:
        return absl::InvalidArgumentError("Unsupported task");
    }
    RETURN_IF_ERROR(writer->Write(example));
  }

  return absl::OkStatus();
}







int main(int argc, char** argv) {
    InitLogging(argv[0], &argc, &argv, true);

    const std::string data_csv = absl::GetFlag(FLAGS_data_csv);
    const std::string label_col = absl::GetFlag(FLAGS_label_col);
    const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
    const int n_runs = absl::GetFlag(FLAGS_n_runs);

    if (data_csv.empty()) {
        LOG(ERROR) << "Please specify --data_csv";
        return 1;
    }

    LOG(INFO) << "Starting MIGHT analysis with " << n_runs << " runs";

    std::vector<double> s_at_target_list;
    
    for (int run = 0; run < n_runs; ++run) {
        absl::SetFlag(&FLAGS_random_seed, absl::GetFlag(FLAGS_random_seed) + run);
        MIGHTImplementation might(data_csv, label_col, output_dir);
        auto results_or = might.RunMIGHT();

        if (!results_or.ok()) {
            LOG(ERROR) << "MIGHT analysis failed: " << results_or.status();
            return 1;
        }

        const auto& results = results_or.value();
        s_at_target_list.push_back(results.sensitivity_at_target_specificity);
        
        LOG(INFO) << "Run " << (run + 1) << " S@" << absl::GetFlag(FLAGS_target_specificity) * 100 
                  << ": " << results.sensitivity_at_target_specificity;
    }

    double avg = 0.0;
    for (double val : s_at_target_list) avg += val;
    avg /= s_at_target_list.size();

    std::ofstream csv(output_dir + "might_data/s_at_target_results.csv");
    csv << "run,s_at_target\n";
    for (int i = 0; i < s_at_target_list.size(); ++i) {
        csv << (i + 1) << "," << s_at_target_list[i] << "\n";
    }
    csv << "average," << avg << "\n";
    csv.close();

    LOG(INFO) << "Average S@" << absl::GetFlag(FLAGS_target_specificity) * 100 << ": " << avg;

    return 0;
}