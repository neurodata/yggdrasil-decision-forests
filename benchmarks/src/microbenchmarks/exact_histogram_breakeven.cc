#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Abseil libraries
#include "absl/status/statusor.h"
#include "absl/types/span.h"

// Yggdrasil Decision Forests specific includes
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

// For the internal functions and types
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"

// Standard library for benchmarking
#include <benchmark/benchmark.h> // If using Google Benchmark
// OR
#include <chrono>
#include <random>



absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalHistogram(
const absl::Span<const UnsignedExampleIdx> selected_examples,
const std::vector<float> &weights, const absl::Span<const float> attributes,
const std::vector<int32_t> &labels, const int32_t num_label_classes,
float na_replacement, const UnsignedExampleIdx min_num_obs,
const proto::DecisionTreeTrainingConfig &dt_config,
const utils::IntegerDistributionDouble &label_distribution,
const int32_t attribute_idx, utils::RandomEngine *random,
      proto::NodeCondition *condition)
{
/* #region Checks */
DCHECK(condition != nullptr);
if (!weights.empty())
{
DCHECK_EQ(weights.size(), labels.size());
}

if (dt_config.missing_value_policy() ==
    proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION)
{
  LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
}
    /* #endregion */

// Determine the minimum and maximum values of the attribute.
// Ariel which attribute? Each feature?
float min_value, max_value;

    /* #region Basic Validity Checks */
if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                              &max_value))
{ return SplitSearchResult::kInvalidAttribute; }
// There should be at least two different unique values.
if (min_value == max_value)
{ return SplitSearchResult::kInvalidAttribute; }
/* #endregion */

struct CandidateSplit
{
  float threshold;
  utils::IntegerDistributionDouble pos_label_distribution;
  int64_t num_positive_examples_without_weights = 0;
  bool operator<(const CandidateSplit &other) const
  {
    return threshold < other.threshold;
  }
};
std::chrono::high_resolution_clock::time_point start, end;
std::chrono::duration<double> dur;

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) { start = std::chrono::high_resolution_clock::now(); }

// Randomly select some threshold values.
ASSIGN_OR_RETURN(
    const auto bins,
    internal::GenHistogramBins(dt_config.numerical_split().type(),
                               dt_config.numerical_split().num_candidates(),
                               attributes, min_value, max_value, random));
if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>1) {
end = std::chrono::high_resolution_clock::now();
dur = end - start;
std::cout << " - - Initializing Histogram Bins took: " << dur.count() << "s" << std::endl;
}

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) { start = std::chrono::high_resolution_clock::now(); }

std::vector<CandidateSplit> candidate_splits(bins.size());
for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++)
{
  auto &candidate_split = candidate_splits[split_idx];
  candidate_split.pos_label_distribution.SetNumClasses(num_label_classes);
  candidate_split.threshold = bins[split_idx];
}

  if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>1) {
end = std::chrono::high_resolution_clock::now();
dur = end - start;
std::cout << " - - Setting Split Distributions took: " << dur.count() << "s" << std::endl;
}

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) { start = std::chrono::high_resolution_clock::now(); }

// Compute the split score of each threshold.
for (const auto example_idx : selected_examples)
{
  const int32_t label = labels[example_idx];
  const float weight = weights.empty() ? 1.f : weights[example_idx];
  float attribute = attributes[example_idx];
  if (std::isnan(attribute))
  {
    attribute = na_replacement;
  }
  auto it_split = std::upper_bound(
      candidate_splits.begin(), candidate_splits.end(), attribute,
      [](const float a, const CandidateSplit &b)
      { return a < b.threshold; });
  if (it_split == candidate_splits.begin())
  {
    continue;
  }
  --it_split;
  it_split->num_positive_examples_without_weights++;
  it_split->pos_label_distribution.Add(label, weight);
}

      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>1) {
end = std::chrono::high_resolution_clock::now();
dur = end - start;
std::cout << " - - Looping over samples took: " << dur.count() << "s" << std::endl;
}

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) { start = std::chrono::high_resolution_clock::now(); }


for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
     split_idx--)
{
  const auto &src = candidate_splits[split_idx + 1];
  auto &dst = candidate_splits[split_idx];
  dst.num_positive_examples_without_weights +=
      src.num_positive_examples_without_weights;
  dst.pos_label_distribution.Add(src.pos_label_distribution);
}

      if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>1) {
end = std::chrono::high_resolution_clock::now();
dur = end - start;
std::cout << " - - Looping over splits took: " << dur.count() << "s" << std::endl;
}

const double initial_entropy = label_distribution.Entropy();
utils::BinaryToIntegerConfusionMatrixDouble confusion;
confusion.SetNumClassesIntDim(num_label_classes);

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) { start = std::chrono::high_resolution_clock::now(); }

// Select the best threshold.
bool found_split = false;
for (auto &candidate_split : candidate_splits)
{
  if (selected_examples.size() -
              candidate_split.num_positive_examples_without_weights <
          min_num_obs ||
      candidate_split.num_positive_examples_without_weights < min_num_obs)
  {
    continue;
  }

  confusion.mutable_neg()->Set(label_distribution);
  confusion.mutable_neg()->Sub(candidate_split.pos_label_distribution);
  confusion.mutable_pos()->Set(candidate_split.pos_label_distribution);

  const double final_entropy = confusion.FinalEntropy();
  const double information_gain = initial_entropy - final_entropy;
  if (information_gain > condition->split_score())
  {
    condition->set_split_score(information_gain);
    condition->mutable_condition()->mutable_higher_condition()->set_threshold(
        candidate_split.threshold);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_training_examples_with_weight(
        confusion.NumObservations());
    condition->set_num_pos_training_examples_without_weight(
        candidate_split.num_positive_examples_without_weights);
    condition->set_num_pos_training_examples_with_weight(
        confusion.pos().NumObservations());
    condition->set_na_value(na_replacement >= candidate_split.threshold);
    found_split = true;
  }
}

if constexpr (CHRONO_MEASUREMENTS_LOG_LEVEL>0) {
end = std::chrono::high_resolution_clock::now();
dur = end - start;
std::cout << " - - Finding best threshold (Computing Entropies) took: " << dur.count() << "s" << std::endl;
}

return found_split ? SplitSearchResult::kBetterSplitFound
                   : SplitSearchResult::kNoBetterSplitFound;
}

  // Ariel - train oblique rf w/ MIGHT hits this, not Histogram
  // ~13% of multicore runtime
  absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureNumericalCart(
      const absl::Span<const UnsignedExampleIdx> selected_examples,
      const std::vector<float> &weights,
      const absl::Span<const float> attributes, // This is called projection_values in oblique.cc
      const std::vector<int32_t> &labels, const int32_t num_label_classes,
      float na_replacement, const UnsignedExampleIdx min_num_obs,
      const proto::DecisionTreeTrainingConfig &dt_config,
      const utils::IntegerDistributionDouble &label_distribution,
      const int32_t attribute_idx, const InternalTrainConfig &internal_config,
      proto::NodeCondition *condition, SplitterPerThreadCache *cache)
  {
    /* #region Deal w/ empty weights */
    if (!weights.empty()) { DCHECK_EQ(weights.size(), labels.size()); }
    // If Weights empty Deal w/ missing values
    if (dt_config.missing_value_policy() == proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION)
    { LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                           &na_replacement); }
    /* #endregion */

    // TODO Ariel Optimize - possibly why this fn. takes ~13% of runtime
    FeatureNumericalBucket::Filler feature_filler(selected_examples.size(), na_replacement, attributes);

    const auto sorting_strategy = EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

    
    if (num_label_classes == 3) { // Binary classification.
    // "Why ==3" ?
    // Categorical attributes always have one class reserved for
    // "out-of-vocabulary" items. The "num_label_classes" takes into account this
    // class. In case of binary classification, "num_label_classes" is 3 (OOB,
    // False, True).

      if (weights.empty()) // Ariel: This is our case. Idk what weights mean
      {
        // No significant Memory access here.
        LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Filler
            label_filler(labels, weights);
        LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Initializer
            initializer(label_distribution);

        // Irrelevant
        if (sorting_strategy == proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED)
        {
          const auto &sorted_attributes = internal_config.preprocessing->presorted_numerical_features()[attribute_idx];

          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler, initializer,
              min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
        else if (sorting_strategy == proto::DecisionTreeTrainingConfig::Internal::IN_NODE)
        {
          // This is what's done by MIGHT, w/ default settings
          return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
              selected_examples, feature_filler, label_filler, initializer,
              min_num_obs, attribute_idx, condition, &cache->cache_v2);
        }
        else { return absl::InvalidArgumentError("Non supported strategy."); }
      }
      else // Weights not empty - Irrelevant to Ariel
      {
        LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Filler
            label_filler(labels, weights);
        LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Initializer
            initializer(label_distribution);
        if (sorting_strategy ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED)
        {
          const auto &sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler, initializer,
              min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
        else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE)
        {
          return FindBestSplit_LabelBinaryClassificationFeatureNumerical(
              selected_examples, feature_filler, label_filler, initializer,
              min_num_obs, attribute_idx, condition, &cache->cache_v2);
        }
        else
        {
          return absl::InvalidArgumentError("Non supported strategy");
        }
      }
    }
  }
