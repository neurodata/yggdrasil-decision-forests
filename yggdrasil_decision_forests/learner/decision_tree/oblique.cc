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

#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include <fstream>
#include <iomanip>

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {
using std::is_same;

using Projection = internal::Projection;
using ProjectionEvaluator = internal::ProjectionEvaluator;
using LDACache = internal::LDACache;

}


static constexpr bool ENABLE_PROJECTION_MATRIX_LOGGING = false;
// Already in splitter_scanner.h . Apparently it carries from all the way there
// static constexpr bool MEASURE_CHRONO_TIMES = true;

// TODO Ariel - important for access pattern
/* #region Extract() - select out vector Indices at positions */

template <typename T>
std::vector<T> Extract(const std::vector<T>& values,
                       const absl::Span<const UnsignedExampleIdx> selected) {
  if (values.empty()) {
    return {};
  }
  std::vector<T> extracted(selected.size());
  for (size_t selected_idx = 0; selected_idx < selected.size();
       selected_idx++) {
    extracted[selected_idx] = values[selected[selected_idx]];
  }
  return extracted;
}

template std::vector<int32_t> Extract<int32_t>(
    const std::vector<int32_t>& values,
    const absl::Span<const UnsignedExampleIdx> selected);

template std::vector<float> Extract<float>(
    const std::vector<float>& values,
    const absl::Span<const UnsignedExampleIdx> selected);

std::vector<int32_t> ExtractLabels(
    const ClassificationLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return Extract(labels.label_data, selected);
}

std::vector<float> ExtractLabels(
    const RegressionLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return Extract(labels.label_data, selected);
}

GradientAndHessian ExtractLabels(
    const RegressionHessianLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return {/*.gradient_data =*/Extract(labels.gradient_data, selected),
          /*.hessian_data =*/Extract(labels.hessian_data, selected)};
}

/* #endregion */


int GetNumProjections(const proto::DecisionTreeTrainingConfig& dt_config,
                      const int num_numerical_features) {
  if (num_numerical_features <= 1) {
    // Note: if there is only one feature, all the projections are the same.
    return 1;
  }
  const int max_num_projections =
      dt_config.sparse_oblique_split().max_num_projections();

  const int min_num_projections =
      std::min(dt_config.sparse_oblique_split().min_num_projections(),
               num_numerical_features);

  const int target_num_projections =
      0.5 + std::ceil(std::pow(
                num_numerical_features,
                dt_config.sparse_oblique_split().num_projections_exponent()));

  return std::max(std::min(target_num_projections, max_num_projections),
                  min_num_projections);
}


// Ariel - actual work, not wrapper. Loop over Projections
template <typename LabelStats>
absl::StatusOr<bool> FindBestConditionSparseObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent,
    const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints,
    proto::NodeCondition* best_condition,
    utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {

  /* #region Initializations */
    
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> dur;
  if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
      start = std::chrono::high_resolution_clock::now();
  }

  // ---------- basic sanity‑checks --------------------
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), train_dataset.nrow());
  }
  if (config_link.numerical_features().empty()) {
    return false;
  }

  // Decide Num Projections
  int num_projections = override_num_projections.has_value()
      ? override_num_projections.value()
      : GetNumProjections(dt_config, config_link.numerical_features_size());

  const float projection_density =
      dt_config.sparse_oblique_split().projection_density_factor() /
      config_link.numerical_features_size();

  // This is a class object
  ProjectionEvaluator projection_evaluator(train_dataset,
                                           config_link.numerical_features());
  /* #endregion */                                         

  // TODO Understand Memory Access Pattern. Is this heavy enough to be important?
  // Bunch of memory accesses here - how long do they take?
  // BEWARE: These are executed once per fn. call - may be irrelevant
  const auto selected_labels = ExtractLabels(label_stats, selected_examples);

  // TODO Are weights relevant to the basic Oblique case?
  std::vector<float> selected_weights;
  if (!weights.empty()) {
    selected_weights = Extract(weights, selected_examples);
  }

  std::vector<UnsignedExampleIdx> dense_idxs(selected_examples.size());
  std::iota(dense_idxs.begin(), dense_idxs.end(), 0);

  auto& projection_values = cache->projection_values;

  /* #region Initializations pt. 2 */
  Projection best_projection, current_projection;
  float best_threshold = 0.f;
  const int num_features = config_link.numerical_features_size();

  static bool first_call = true;
  static int node_counter = 0;
  std::ofstream log;
  std::vector<std::vector<float>> matrix;

  if constexpr (ENABLE_PROJECTION_MATRIX_LOGGING) {
    if (first_call) {
      log.open("ariel_results/projection_matrices.txt", std::ios::trunc);
      first_call = false;
    } else {
      log.open("ariel_results/projection_matrices.txt", std::ios::app);
    }
    matrix.resize(num_projections, std::vector<float>(num_features, 0.f));
  }

  /* #endregion */

  if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
    end = std::chrono::high_resolution_clock::now();
    dur = end - start;
    std::cout << "\n - Initialization of FindBestCondOblique Took: " << dur.count() << "s\n";
  }

  // std::cout << "Num projections: " << num_projections << "\n";
  // remove this outside of profiling!!
  // num_projections = 1000;
  // std::cout << "Hard coded 1000 projections! " << num_projections << "\n";

  /* #region ----------  MAIN LOOP  ------------------ */
  // 1. Declare accumulators
  std::chrono::duration<double> total_sample_proj_time{0};
  std::chrono::duration<double> total_apply_proj_time{0};
  std::chrono::duration<double> total_eval_proj_time{0};

  for (int proj_idx = 0; proj_idx < num_projections; ++proj_idx) {
      int8_t monotonic = 0;

      // 2a. SampleProjection timing
      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
        start = std::chrono::high_resolution_clock::now();
      }
      SampleProjection(config_link.numerical_features(), dt_config,
                      train_dataset.data_spec(), config_link, projection_density,
                      &current_projection, &monotonic, random);
      
      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
        dur = std::chrono::high_resolution_clock::now() - start;
        total_sample_proj_time += dur;

        // 2b. ApplyProjection timing
        start = std::chrono::high_resolution_clock::now();
      }
      RETURN_IF_ERROR(
        projection_evaluator.Evaluate(current_projection, selected_examples, &projection_values)
      );
      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
        dur = std::chrono::high_resolution_clock::now() - start;
        total_apply_proj_time += dur;

        // 2c. EvaluateProjection timing
        start = std::chrono::high_resolution_clock::now();
      }
      ASSIGN_OR_RETURN(
          const auto split_result,
          EvaluateProjection(dt_config, label_stats, dense_idxs, selected_weights,
                            selected_labels, projection_values, internal_config,
                            current_projection.front().attribute_idx,
                            constraints, monotonic,
                            best_condition, cache)
      );
      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
        dur = std::chrono::high_resolution_clock::now() - start;
        total_eval_proj_time += dur;
      }

      if (split_result == SplitSearchResult::kBetterSplitFound) {
        best_projection = current_projection;
        best_threshold =
            best_condition->condition().higher_condition().threshold();
      }
  }

  if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
  std::cout << "\n=== Timing summary for " << num_projections << " projections ===\n"
            << "SampleProjection took:  " << total_sample_proj_time.count() << "s\n"
            << "ApplyProjection took:   " << total_apply_proj_time.count()  << "s\n"
            << "EvaluateProjection took:  " << total_eval_proj_time.count()  << "s\n";
  }

  /* #endregion */

  // start = std::chrono::high_resolution_clock::now();
  /* #region Post-Processing - unimportant for runtime */

  // Save projection matrix to file if desired
  if constexpr (ENABLE_PROJECTION_MATRIX_LOGGING) {
    log << "Node " << node_counter++ << " | "
        << num_projections << " projections × "
        << num_features << " features\n";

    log << "      ";
    for (const int feat_idx : config_link.numerical_features()) {
      log << std::setw(6) << feat_idx;
    }
    log << '\n';

    for (int r = 0; r < num_projections; ++r) {
      log << "proj " << r << ' ';
      for (int c = 0; c < num_features; ++c) {
        log << std::setw(6) << matrix[r][c];
      }
      log << '\n';
    }
    log << '\n';
    log.close();
  }

  if (!best_projection.empty()) {
    RETURN_IF_ERROR(SetCondition(best_projection, best_threshold,
                                 train_dataset.data_spec(), best_condition));

    // e-06 for 512k x 1k
    // end = std::chrono::high_resolution_clock::now();
    // dur = end - start;
    // std::cout << "\nFindBestCondOblique::SetCond() Took: " << dur.count() << "s\n";
    return true;
  }

  if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
    end = std::chrono::high_resolution_clock::now();
    dur = end - start;
    std::cout << "\n FindBestCondOblique - non-SetCond Took: " << dur.count() << "s\n";
  }
  
  return false;

  /* #endregion */
}


absl::Status SolveLDA(const proto::DecisionTreeTrainingConfig& dt_config,
                      const ProjectionEvaluator& projection_evaluator,
                      const std::vector<int>& selected_features,
                      const int num_classes, const std::vector<int32_t>& labels,
                      const std::vector<float>& weights, Projection* projection,
                      utils::RandomEngine* random) {
  // TODO: Cache.
  LDACache lda_cache;
  RETURN_IF_ERROR(lda_cache.ComputeClassification(
      dt_config, projection_evaluator, selected_features, num_classes, labels,
      weights));
  const auto& sb = lda_cache.FullSB();
  const auto& sw = lda_cache.FullSW();
  const int num_features = selected_features.size();
  const Eigen::Map<const Eigen::MatrixXd> eg_sw(sw.data(), num_features,
                                                num_features);
  const Eigen::Map<const Eigen::MatrixXd> eg_sb(sb.data(), num_features,
                                                num_features);

  // Inverse the SW matrice.
  Eigen::PartialPivLU<Eigen::MatrixXd> invert_solver(eg_sw);
  if (invert_solver.determinant() == 0) {
    // The matrix is not invertible.
    return absl::OkStatus();
  }
  const auto eg_w = invert_solver.inverse() * eg_sb;

  // Get the eigenvalues / vectors.
  Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(eg_w, true);

  if (eigen_solver.info() != Eigen::Success) {
    return absl::OkStatus();
  }

  const auto& eigenvalues = eigen_solver.eigenvalues();
  const auto& eigenvectors = eigen_solver.eigenvectors();

  // Get the largest eigenvalue / vector.
  int arg_abs_max = -1;
  double abs_max = 0;
  for (int i = 0; i < num_features; i++) {
    const auto value = std::abs(eigenvalues(i).real());
    if (value > abs_max) {
      arg_abs_max = i;
      abs_max = value;
    }
  }
  if (arg_abs_max == -1) {
    return absl::OkStatus();
  }

  // Convert the top eigen vector into a projection.
  projection->clear();
  for (int i = 0; i < num_features; i++) {
    const float vector = eigenvectors(i, arg_abs_max).real();
    if (vector == 0) {
      continue;
    }
    projection->push_back({selected_features[i], vector});
  }

  return absl::OkStatus();
}

struct ScoreAndThreshold {
  float score;
  float threhsold;
};

/* #region EvaluateProjection() and Templates */

// TODO ariel understand projection eval
template <typename LabelStats, typename Labels>
absl::StatusOr<SplitSearchResult> EvaluateProjection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  InternalTrainConfig effective_internal_config = internal_config;

  // Choose sorting strategy
  effective_internal_config.override_sorting_strategy =
      proto::DecisionTreeTrainingConfig::Internal::SortingStrategy::
          DecisionTreeTrainingConfig_Internal_SortingStrategy_IN_NODE;

  const UnsignedExampleIdx min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  // Projection are never missing.
  const float na_replacement = 0;
#ifndef NDEBUG
// Ariel - this grows linearly w/ proj_vals, but not executed in production! Only shows up in Intel Profiler - Ignore
  for (const float v : projection_values) {
    DCHECK(!std::isnan(v));
  }
#endif

  // Find a good split in the current_projection.
  SplitSearchResult result;
  if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
    ASSIGN_OR_RETURN(
        result,
        FindSplitLabelClassificationFeatureNumericalCart(
            dense_example_idxs, selected_weights, projection_values,
            selected_labels, label_stats.num_label_classes, na_replacement,
            min_num_obs, dt_config, label_stats.label_distribution,
            first_attribute_idx, effective_internal_config, condition, cache));
  } else if constexpr (is_same<LabelStats,
                               RegressionHessianLabelStats>::value) {
    if (!selected_weights.empty()) {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/true>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels.gradient_data, selected_labels.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights,
              first_attribute_idx, effective_internal_config, constraints,
              monotonic_direction, condition, cache));

    } else {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/false>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels.gradient_data, selected_labels.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights,
              first_attribute_idx, effective_internal_config, constraints,
              monotonic_direction, condition, cache));
    }
  } else if constexpr (is_same<LabelStats, RegressionLabelStats>::value) {
    if (!selected_weights.empty()) {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/true>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, na_replacement, min_num_obs, dt_config,
              label_stats.label_distribution, first_attribute_idx,
              effective_internal_config, condition, cache));
    } else {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/false>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, na_replacement, min_num_obs, dt_config,
              label_stats.label_distribution, first_attribute_idx,
              effective_internal_config, condition, cache));
    }
  } else {
    static_assert(!is_same<LabelStats, LabelStats>::value, "Not implemented.");
  }

  return result;
}

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<ClassificationLabelStats, std::vector<int32_t>>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const ClassificationLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const std::vector<int32_t>& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<RegressionLabelStats, std::vector<float>>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const RegressionLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const std::vector<float>& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<RegressionHessianLabelStats, GradientAndHessian>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const RegressionHessianLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const GradientAndHessian& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);


template <typename LabelStats, typename Labels>
absl::Status EvaluateProjectionAndSetCondition(
    const dataset::proto::DataSpecification& dataspec,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const absl::Span<const float> projection_values,
    const Projection& projection, const InternalTrainConfig& internal_config,
    const int first_attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  ASSIGN_OR_RETURN(
      const auto result,
      EvaluateProjection(dt_config, label_stats, dense_example_idxs,
                         selected_weights, selected_labels, projection_values,
                         internal_config, first_attribute_idx,
                         /*constraints=*/{}, /*monotonic_direction=*/0,
                         condition, cache));

  if (result == SplitSearchResult::kBetterSplitFound) {
    RETURN_IF_ERROR(SetCondition(
        projection, condition->condition().higher_condition().threshold(),
        dataspec, condition));
  }
  return absl::OkStatus();
}

template <typename LabelStats, typename Labels>
absl::Status EvaluateMHLDCandidates(
    const dataset::proto::DataSpecification& dataspec,
    const std::vector<std::vector<int>>& candidates,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const InternalTrainConfig& internal_config,
    const ProjectionEvaluator& projection_evaluator,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<proto::NodeCondition>* conditions,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  // TODO: Multi-thread
  conditions->assign(candidates.size(), {});
  auto& projection_values = cache->projection_values;

  for (int candidate_idx = 0; candidate_idx < candidates.size();
       candidate_idx++) {
    const auto& candidate = candidates[candidate_idx];
    auto& condition = (*conditions)[candidate_idx];

    if (candidate.empty()) {
      return absl::InternalError("No candidates");
    } else if (candidate.size() == 1) {
      const auto attribute_idx = candidate.front();

      // Extract attribute value
      RETURN_IF_ERROR(projection_evaluator.ExtractAttribute(
          attribute_idx, selected_examples, &projection_values));

      RETURN_IF_ERROR(EvaluateProjectionAndSetCondition(
          dataspec, dt_config, label_stats, dense_example_idxs,
          selected_weights, selected_labels, projection_values,
          {{attribute_idx, 1.f}}, internal_config, attribute_idx, &condition,
          cache));
    } else {
      // Find best projection
      Projection projection;
      if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
        RETURN_IF_ERROR(SolveLDA(dt_config, projection_evaluator, candidate,
                                 label_stats.num_label_classes, selected_labels,
                                 selected_weights, &projection, random));
      } else {
        return absl::InvalidArgumentError(
            "MHLD Oblique splits only available on classification. Use sparse "
            "oblique splits for other tasks.");
      }

      if (projection.empty()) {
        continue;
      }

      // Compute projection
      RETURN_IF_ERROR(projection_evaluator.Evaluate(
          projection, selected_examples, &projection_values));

      // Evaluate projection quality
      RETURN_IF_ERROR(EvaluateProjectionAndSetCondition(
          dataspec, dt_config, label_stats, dense_example_idxs,
          selected_weights, selected_labels, projection_values, projection,
          internal_config, candidate.front(), &condition, cache));
    }
  }

  return absl::OkStatus();
}


/* #endregion */

absl::StatusOr<std::vector<int>> SampleAttributes(
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::TrainingConfig& config,
    const proto::DecisionTreeTrainingConfig& dt_config,
    utils::RandomEngine* random) {
      // Is this the Bootstrapped data per-feature access? Or MHLD specific?
  std::vector<int> candidate_attributes{
      config_link.numerical_features().begin(),
      config_link.numerical_features().end()};

  if (dt_config.mhld_oblique_split().sample_attributes()) {
    std::shuffle(candidate_attributes.begin(), candidate_attributes.end(),
                 *random);

    const int num_attributes_to_test = NumAttributesToTest(
        dt_config, config_link.numerical_features_size(), config.task());
    if (num_attributes_to_test < 0 ||
        num_attributes_to_test > candidate_attributes.size()) {
      return absl::InternalError("Wrong number of attributes to test");
    }

    candidate_attributes.resize(num_attributes_to_test);
    std::sort(candidate_attributes.begin(), candidate_attributes.end());
  }

  return candidate_attributes;
}

template <typename LabelStats>
absl::StatusOr<bool> FindBestConditionMHLDObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  if (config_link.numerical_features().empty()) {
    return false;
  }

  ProjectionEvaluator projection_evaluator(train_dataset,
                                           config_link.numerical_features());

  // TODO: Cache.
  const auto selected_labels = ExtractLabels(label_stats, selected_examples);
  std::vector<float> selected_weights;
  if (!weights.empty()) {
    selected_weights = Extract(weights, selected_examples);
  }

  std::vector<UnsignedExampleIdx> dense_example_idxs(selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);

  std::vector<int> selected_features;
  ASSIGN_OR_RETURN(std::vector<int> candidate_features,
                   SampleAttributes(config_link, config, dt_config, random));
  std::vector<std::vector<int>> round_candidates;
  std::vector<float> round_scores;
  std::vector<proto::NodeCondition> round_conditions;

  float global_best_score = best_condition->split_score();
  bool found_better_global = false;

  const int num_rounds =
      std::min(static_cast<int>(candidate_features.size()),
               dt_config.mhld_oblique_split().max_num_attributes());

  for (int round_idx = 0; round_idx < num_rounds; round_idx++) {
    if (candidate_features.empty()) {
      // No more features to try.
      continue;
    }

    // Compute the sets of set of features to evaluate.
    round_candidates.clear();
    for (const auto candidate_feature : candidate_features) {
      round_candidates.push_back(selected_features);
      round_candidates.back().push_back(candidate_feature);
      std::sort(round_candidates.back().begin(), round_candidates.back().end());
    }

    // Evaluate
    RETURN_IF_ERROR(EvaluateMHLDCandidates(
        train_dataset.data_spec(), round_candidates, dt_config, label_stats,
        dense_example_idxs, selected_weights, selected_labels, internal_config,
        projection_evaluator, selected_examples, &round_conditions, cache,
        random));
    DCHECK_EQ(round_conditions.size(), round_candidates.size());

    // Find the best local and global projection.
    float round_best_score = 0;
    int round_best_candidate_idx = -1;
    for (int candidate_idx = 0; candidate_idx < round_candidates.size();
         candidate_idx++) {
      const float score = round_conditions[candidate_idx].split_score();

      if (std::isnan(score)) {
        continue;
      }
      if (score > round_best_score) {
        round_best_score = score;
        round_best_candidate_idx = candidate_idx;
      }
      if (score > global_best_score) {
        global_best_score = score;
        *best_condition = round_conditions[candidate_idx];
        found_better_global = true;
      }
    }

    if (round_best_candidate_idx == -1) {
      // No local improvement.
      continue;
    }

    selected_features.push_back(candidate_features[round_best_candidate_idx]);
    candidate_features.erase(candidate_features.begin() +
                             round_best_candidate_idx);
  }

  return found_better_global;
}


/* #region NonInteresting semaphore functions for choosing MHLD or Regular Oblique, based on user call */
absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      // a.k.a SPORF
      return FindBestConditionSparseObliqueTemplate<ClassificationLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, {}, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<ClassificationLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      return FindBestConditionSparseObliqueTemplate<
          RegressionHessianLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, constraints, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<RegressionHessianLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      return FindBestConditionSparseObliqueTemplate<RegressionLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, {}, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<RegressionLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

/* #endregion */

namespace internal {

void SampleProjection(const absl::Span<const int>& features,
  const proto::DecisionTreeTrainingConfig& dt_config,
  const dataset::proto::DataSpecification& data_spec,
  const model::proto::TrainingConfigLinking& config_link,
  const float projection_density,
  internal::Projection* projection,
  int8_t* monotonic_direction,
  utils::RandomEngine* random) {
  // ----- RNG helpers --------------------------------------------------------
  std::uniform_real_distribution<float>  unif01;
  std::uniform_real_distribution<float>  unif1m1(-1.f, 1.f);

  // ----- reset output -------------------------------------------------------
  *monotonic_direction = 0;
  projection->clear();

  const auto& oblique_cfg = dt_config.sparse_oblique_split();
  const int   p          = features.size();
  int         k          = std::ceil(projection_density * p);

  // Always pick at least one feature.
  k = std::max(1, std::min(k, p));

  projection->reserve(k);

  // ----- helper to draw a weight -------------------------------------------
  auto gen_weight = [&](int feature)->float {
  float w = unif1m1(*random);

  switch (oblique_cfg.weights_case()) {
  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::kBinary:
  w = (w >= 0.f) ? 1.f : -1.f; break;

  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::kPowerOfTwo: {
  const float sign = (w >= 0.f) ? 1.f : -1.f;
  const int   exp  = absl::Uniform<int>(absl::IntervalClosed,
                              *random,
                              oblique_cfg.power_of_two().min_exponent(),
                              oblique_cfg.power_of_two().max_exponent());
  w = sign * std::pow(2.f, exp);
  } break;

  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::kInteger:
  w = absl::Uniform(absl::IntervalClosed, *random,
          oblique_cfg.integer().minimum(),
          oblique_cfg.integer().maximum());
  break;

  default: /* continuous */ break;
  }

  if (config_link.per_columns_size() > 0 &&
  config_link.per_columns(feature).has_monotonic_constraint()) {
  const bool inc =
  config_link.per_columns(feature).monotonic_constraint().direction() ==
  model::proto::MonotonicConstraint::INCREASING;
  if (inc == (w < 0)) w = -w;
  *monotonic_direction = 1;
  }

  const auto& spec = data_spec.columns(feature).numerical();
  switch (oblique_cfg.normalization()) {
  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
  return w;
  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::STANDARD_DEVIATION:
  return w / std::max(1e-6, spec.standard_deviation());
  case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::MIN_MAX:
  return w / std::max(1e-6f, spec.max_value() - spec.min_value());
  }
  };

  // =================  PARTIAL FISHER–YATES  =================
  // Copy indices locally so we can shuffle.
  std::vector<int> pool(features.begin(), features.end());

  // TODO pick 2xp one-time - it's what Treeple does
  for (int i = 0; i < k; ++i) {
      // Pick j uniformly in [i, p)
      const int j = absl::Uniform<int>(*random, i, p);
      std::swap(pool[i], pool[j]);

      const int feature = pool[i];
      DCHECK_EQ(data_spec.columns(feature).type(), dataset::proto::NUMERICAL);

      projection->push_back({feature, gen_weight(feature)});
  }
  // ===========================================================

  // --------- Optional post-processing: max_num_features ---------------
  const int max_feat = dt_config.sparse_oblique_split().max_num_features();
  if (max_feat > 0 && projection->size() > max_feat) {
  internal::Projection trimmed;
  trimmed.reserve(max_feat);

  // Reservoir-sample max_feat indices out of [0 .. projection->size())
  absl::btree_set<size_t> kept;
  for (size_t j = projection->size() - max_feat; j < projection->size(); ++j) {
  size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
  if (!kept.insert(t).second) kept.insert(j);
  }
  for (size_t idx : kept) trimmed.push_back((*projection)[idx]);
  *projection = std::move(trimmed);
  }

  // Ensure single-feature projection gets weight 1
  if (projection->size() == 1) projection->front().weight = 1.f;
}


absl::Status SetCondition(const Projection& projection, const float threshold,
                          const dataset::proto::DataSpecification& dataspec,
                          proto::NodeCondition* condition) {
  if (projection.empty()) {
    return absl::InternalError("Empty projection");
  }
  auto& oblique_condition =
      *condition->mutable_condition()->mutable_oblique_condition();
  oblique_condition.set_threshold(threshold);
  oblique_condition.clear_attributes();
  oblique_condition.clear_weights();
  for (const auto& item : projection) {
    oblique_condition.add_attributes(item.attribute_idx);
    oblique_condition.add_weights(item.weight);
    oblique_condition.add_na_replacements(
        dataspec.columns(item.attribute_idx).numerical().mean());
  }
  condition->set_attribute(projection.front().attribute_idx);
  condition->set_na_value(false);
  return absl::OkStatus();
}


/* #region LDA-Specific f() */

absl::Status LDACache::ComputeClassification(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const ProjectionEvaluator& projection_evaluator,
    const std::vector<int>& selected_features, const int num_classes,
    const std::vector<int32_t>& labels, const std::vector<float>& weights,
    const bool index_features) {
  // Solve a LDA (Linear Discriminant Analysis) using the Eigenvalue
  // decomposition approach.
  //
  // Based on the section 2 of "Revisiting Classical Multiclass Linear
  // Discriminant Analysis with a Novel Prototype-based Interpretable
  // Solution".
  //
  // TODO: Experiment with other approaches. For instance, the singular
  // value decomposition approach.

  const int num_features = selected_features.size();
  size_ = num_features;

  // Compute the mean of the features (globally and per class).
  const int shifted_num_classes = num_classes - 1;  // Ignore class 0.
  DCHECK_GE(shifted_num_classes, 2);
  // TODO: Cache.
  std::vector<double> mean_per_feature(selected_features.size(), 0);
  std::vector<double> mean_per_feature_and_class(
      selected_features.size() * shifted_num_classes, 0);
  std::vector<double> weight_per_class(shifted_num_classes, 0);
  double sum_weights = 0;

  for (std::size_t example_idx = 0; example_idx < labels.size();
       example_idx++) {
    const int32_t shifted_class = labels[example_idx] - 1;
    DCHECK_GE(shifted_class, 0);
    DCHECK_LT(shifted_class, shifted_num_classes);
    const float weight = (weights.empty()) ? 1.f : weights[example_idx];

    sum_weights += weight;
    weight_per_class[shifted_class] += weight;

    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      const float feature_value = projection_evaluator.AttributeValues(
          selected_features[feature_idx])[example_idx];

      mean_per_feature[feature_idx] += feature_value * weight;
      mean_per_feature_and_class[feature_idx + shifted_class * num_features] +=
          feature_value * weight;
    }
  }

  if (sum_weights == 0) {
    return absl::InvalidArgumentError("Null weight");
  }

  // Normalize the sums into means.
  DCHECK_GT(sum_weights, 0);
  const double inv_sum_weights = 1. / sum_weights;
  for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
    mean_per_feature[feature_idx] *= inv_sum_weights;

    for (int shifted_class = 0; shifted_class < shifted_num_classes;
         shifted_class++) {
      if (weight_per_class[shifted_class] == 0) {
        continue;
      }
      mean_per_feature_and_class[feature_idx + shifted_class * num_features] /=
          weight_per_class[shifted_class];
    }
  }

  // Compute Sb
  sb_.assign(num_features * num_features, 0);
  for (int shifted_class = 0; shifted_class < shifted_num_classes;
       shifted_class++) {
    internal::SubtractTransposeMultiplyAdd(
        weight_per_class[shifted_class],
        absl::MakeSpan(mean_per_feature_and_class)
            .subspan(shifted_class * num_features, num_features),
        absl::MakeSpan(mean_per_feature), sb_);
  }

  // Compute Sw
  sw_.assign(num_features * num_features, 0);
  for (std::size_t example_idx = 0; example_idx < labels.size();
       example_idx++) {
    const int shifted_class = labels[example_idx] - 1;
    DCHECK_GE(shifted_class, 0);
    const float weight = (weights.empty()) ? 1.f : weights[example_idx];

    internal::SubtractTransposeMultiplyAdd(
        weight, example_idx, selected_features, projection_evaluator,
        absl::MakeSpan(mean_per_feature_and_class)
            .subspan(shifted_class * num_features, num_features),
        sw_);
  }

  // Help the matrix to be invertible.
  for (int i = 0; i < num_features; i++) {
    sw_[i + num_features * i] += 0.001;
  }

  if (index_features) {
    // Index features.
    const int max_num_features =
        *std::max_element(selected_features.begin(), selected_features.end());
    feature_to_idx_.assign(max_num_features + 1, -1);
    for (int i = 0; i < num_features; i++) {
      feature_to_idx_[selected_features[i]] = i;
    }
  }

  return absl::OkStatus();
}

// Builds the feature mapping. "mapping[i]" is the index in "sw_" and "sb_" of
// feature "selected_features[i]".
absl::Status LDACache::BuildMapping(const std::vector<int>& selected_features,
                                    std::vector<int>* mapping) const {
  mapping->resize(selected_features.size());
  for (size_t i = 0; i < selected_features.size(); i++) {
    const int j = feature_to_idx_[selected_features[i]];
    if (j == -1) {
      return absl::InternalError("Non indexed feature");
    }
    (*mapping)[i] = j;
  }
  return absl::OkStatus();
}

absl::Status LDACache::Extract(const std::vector<int>& selected_features,
                               const std::vector<double>& in,
                               std::vector<double>* out) const {
  // TODO: Cache.
  std::vector<int> mapping;
  RETURN_IF_ERROR(BuildMapping(selected_features, &mapping));
  const int num_features = selected_features.size();

  out->resize(num_features * num_features);
  for (int col = 0; col < num_features; col++) {
    for (int row = 0; row < num_features; row++) {
      (*out)[row + col * num_features] =
          in[mapping[row] + size_ * mapping[col]];
    }
  }
  return absl::OkStatus();
}

absl::Status LDACache::GetSB(const std::vector<int>& selected_features,
                             std::vector<double>* out) const {
  return Extract(selected_features, sb_, out);
}

absl::Status LDACache::GetSW(const std::vector<int>& selected_features,
                             std::vector<double>* out) const {
  return Extract(selected_features, sw_, out);
}

/* #endregion */


ProjectionEvaluator::ProjectionEvaluator(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features) {
  DCHECK(!numerical_features.empty());

  const int max_feature_idx =
      *std::max_element(numerical_features.begin(), numerical_features.end());

  numerical_attributes_.assign(max_feature_idx + 1, nullptr);
  na_replacement_value_.assign(max_feature_idx + 1, 0.f);

  // Loop over numerical features - select out columns
  for (const auto attribute_idx : numerical_features) {
    const auto column_or = train_dataset.ColumnWithCastWithStatus<
        dataset::VerticalDataset::NumericalColumn>(attribute_idx);
    constructor_status_.Update(column_or.status());
    if (!constructor_status_.ok()) { break; }

    numerical_attributes_[attribute_idx] = &column_or.value()->values();
    na_replacement_value_[attribute_idx] =
        train_dataset.data_spec().columns(attribute_idx).numerical().mean();
  }
}


// Takes ~12% of multicore runtime | 4096 x various synthetic n_cols
// Skip optimizing for now - bigger fish to fry
absl::Status ProjectionEvaluator::Evaluate(
    const Projection& projection,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);
  values->resize(selected_examples.size());
  // std::cout << "Projection vector: " << projection;

  // TODO make these loops in a vector operation
  // TODO start w/ full dataE

  // TODO Ariel PRIORITY: Show selected_idx and examples_idx - type, domain
  //    Maybe plot selected_examples as a mask
  //    selected_examples should be length n at first iteration, then ~n/2, n/4 ...

  // Ariel UPDATE: This consumes <5% of runtime - skip

  // What is this - should be bag indices
  // Optimization experiment: do this as a vector , then mask out the rows out of bag - values = bag_mask(A+B)
  for (size_t selected_idx = 0; selected_idx < selected_examples.size(); selected_idx++) {
    float value = 0;
    // Ariel wtf is this?? Is this dense
    const auto example_idx = selected_examples[selected_idx];

    // std::cout << "example_idx: " << example_idx;

    // Ariel: This is one-per-nonzero in projection vector - optimal
    for (const auto& item : projection) {
      // Sanity checks
      DCHECK_LT(item.attribute_idx, numerical_attributes_.size());
      DCHECK_GE(item.attribute_idx, 0);

      // TODO Ariel: Move the indirection outside of the loop.
      const auto* attribute_values = numerical_attributes_[item.attribute_idx];

      DCHECK(attribute_values != nullptr);

      float attribute_value = (*attribute_values)[example_idx];
      if (std::isnan(attribute_value)) {
        attribute_value = na_replacement_value_[item.attribute_idx];
      }
      value += attribute_value * item.weight; // e.g. x1 - x2 + x3
    }
    (*values)[selected_idx] = value;
  }
  return absl::OkStatus();
}

// TODO Important - Loop over Bag, what happens?
absl::Status ProjectionEvaluator::ExtractAttribute(
    const int attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);
  values->resize(selected_examples.size());
  const std::vector<float>& src_values = *numerical_attributes_[attribute_idx];
  const float na_replacement_value = na_replacement_value_[attribute_idx];
  for (size_t selected_idx = 0; selected_idx < selected_examples.size();
       selected_idx++) {
    const auto example_idx = selected_examples[selected_idx];
    float value = src_values[example_idx];
    if (std::isnan(value)) {
      value = na_replacement_value;
    }
    (*values)[selected_idx] = value;
  }
  return absl::OkStatus();
}


/* #region SubtractTransposeMultiplyAdd Interesting Kernels! */
void SubtractTransposeMultiplyAdd(double weight, absl::Span<double> a,
                                  absl::Span<double> b,
                                  std::vector<double>& output) {
  DCHECK_EQ(a.size(), b.size());
  DCHECK_EQ(b.size() * b.size(), output.size());

  const int n = a.size();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      output[j + i * n] += weight * (a[i] - b[i]) * (a[j] - b[j]);
    }
  }
}

void SubtractTransposeMultiplyAdd(
    double weight, std::size_t example_idx,
    const std::vector<int>& selected_features,
    const ProjectionEvaluator& projection_evaluator, absl::Span<double> b,
    std::vector<double>& output) {
  DCHECK_EQ(selected_features.size(), b.size());
  DCHECK_EQ(b.size() * b.size(), output.size());

  const int n = b.size();
  for (int i = 0; i < n; i++) {
    const double x_i =
        projection_evaluator.AttributeValues(selected_features[i])[example_idx];
    for (int j = 0; j < n; j++) {
      const double x_j = projection_evaluator.AttributeValues(
          selected_features[j])[example_idx];
      output[j + i * n] += weight * (x_i - b[i]) * (x_j - b[j]);
    }
  }
}

/* #endregion */

}  // namespace internal
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
