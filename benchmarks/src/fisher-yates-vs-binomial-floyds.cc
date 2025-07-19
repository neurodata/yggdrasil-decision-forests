#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

// Yggdrasil Decision Forests includes
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "absl/types/span.h"
#include "absl/container/btree_set.h"
#include "absl/random/random.h"

using namespace yggdrasil_decision_forests;
using namespace yggdrasil_decision_forests::model::decision_tree;

// Toggle between implementations - set to true for Floyd's, false for Fisher-Yates
#define USE_FLOYDS_SAMPLER true

void SampleProjection_floyds(const absl::Span<const int>& features,
                      const proto::DecisionTreeTrainingConfig& dt_config,
                      const dataset::proto::DataSpecification& data_spec,
                      const model::proto::TrainingConfigLinking& config_link,
                      const float projection_density,
                      internal::Projection* projection,
                      int8_t* monotonic_direction,
                      utils::RandomEngine* random) {
  *monotonic_direction = 0;
  projection->clear();
  std::uniform_real_distribution<float> unif01;
  std::uniform_real_distribution<float> unif1m1(-1.f, 1.f);
  const auto& oblique_config = dt_config.sparse_oblique_split();

  const auto gen_weight = [&](const int feature) -> float {
    float weight = unif1m1(*random);
    switch (oblique_config.weights_case()) {
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kBinary): {
        weight = (weight >= 0) ? 1.f : -1.f;
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kPowerOfTwo): {
        float sign = (weight >= 0) ? 1.f : -1.f;
        int exponent =
            absl::Uniform<int>(absl::IntervalClosed, *random,
                               oblique_config.power_of_two().min_exponent(),
                               oblique_config.power_of_two().max_exponent());
        weight = sign * std::pow(2, exponent);
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kInteger): {
        weight = absl::Uniform(absl::IntervalClosed, *random,
                               oblique_config.integer().minimum(),
                               oblique_config.integer().maximum());
        break;
      }
      default: {
        // Return continuous weights.
        break;
      }
    }

    if (config_link.per_columns_size() > 0 &&
        config_link.per_columns(feature).has_monotonic_constraint()) {
      const bool direction_increasing =
          config_link.per_columns(feature).monotonic_constraint().direction() ==
          model::proto::MonotonicConstraint::INCREASING;
      if (direction_increasing == (weight < 0)) {
        weight = -weight;
      }
      // As soon as one selected feature is monotonic, the oblique split
      // becomes monotonic.
      *monotonic_direction = 1;
    }

    const auto& spec = data_spec.columns(feature).numerical();
    switch (oblique_config.normalization()) {
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
        return weight;
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
          STANDARD_DEVIATION:
        return weight / std::max(1e-6, spec.standard_deviation());
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::MIN_MAX:
        return weight / std::max(1e-6f, spec.max_value() - spec.min_value());
    }
  };

  #ifdef NDEBUG  // Keep DCHECK_EQ from for feature : features
  for (const auto feature : features) {
    DCHECK_EQ(data_spec.columns(feature).type(), dataset::proto::NUMERICAL);
  }
  #endif

  // Checks for 1 and >p are unnecessary due to Binomial dist.
  std::binomial_distribution<size_t> binom(features.size(), projection_density);

  // Expectation[Binomial(p,projection_density)] = num_selected_features
  const size_t num_selected_features = binom(*random);

  absl::btree_set<size_t> picked_idx;

  // Floyd's sampler to select k indices uniformly
  for (size_t j = features.size() - num_selected_features; j < features.size(); ++j) {
    size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
    if (!picked_idx.insert(t).second) picked_idx.insert(j);
  }

  projection->reserve(num_selected_features);
  // O(k) minimal pass to fill in those indices
  for (const auto idx : picked_idx) {
      projection->push_back({features[idx], gen_weight(features[idx])});
  }

  if (projection->empty()) {
    std::uniform_int_distribution<int> unif_feature_idx(0, features.size() - 1);
    projection->push_back(
        {/*.attribute_idx =*/features[unif_feature_idx(*random)],
         /*.weight =*/1.f});
  } else if (projection->size() == 1) {
    projection->front().weight = 1.f;
  }

  int max_num_features = dt_config.sparse_oblique_split().max_num_features();
  int cur_num_projections = projection->size();

  if (max_num_features > 0 && cur_num_projections > max_num_features) {
    internal::Projection resampled_projection;
    resampled_projection.reserve(max_num_features);
    // For a small number of features, a boolean vector is more efficient.
    // Re-evaluate if this becomes a bottleneck.
    absl::btree_set<size_t> sampled_features;
    // Floyd's sampling algorithm. TODO could reuse this
    for (size_t j = cur_num_projections - max_num_features;
         j < cur_num_projections; j++) {
      size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
      if (!sampled_features.insert(t).second) {
        // t was already sampled, so insert j instead.
        sampled_features.insert(j);
        resampled_projection.push_back((*projection)[j]);
      } else {
        // t was not yet sampled.
        resampled_projection.push_back((*projection)[t]);
      }
    }
    *projection = std::move(resampled_projection);
  }
}


void SampleProjection_fisher_yates(const absl::Span<const int>& features,
  const proto::DecisionTreeTrainingConfig& dt_config,
  const dataset::proto::DataSpecification& data_spec,
  const model::proto::TrainingConfigLinking& config_link,
  const float projection_density,
  internal::Projection* projection,
  int8_t* monotonic_direction,
  utils::RandomEngine* random) {
  *monotonic_direction = 0;
  projection->clear();
  std::uniform_real_distribution<float>  unif01;
  std::uniform_real_distribution<float>  unif1m1(-1.f, 1.f);
  const auto& oblique_config = dt_config.sparse_oblique_split();

  const int   p          = features.size();
  // Always pick at least one feature.
  // const int         k = std::max(1, std::min(std::ceil(projection_density * p), p));
  
  std::binomial_distribution<size_t> binom(features.size(), projection_density);

  // Expectation[Binomial(p,projection_density)] = num_selected_features
  const size_t k = binom(*random);


  const auto gen_weight = [&](const int feature) -> float {
    float weight = unif1m1(*random);
    switch (oblique_config.weights_case()) {
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kBinary): {
        weight = (weight >= 0) ? 1.f : -1.f;
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kPowerOfTwo): {
        float sign = (weight >= 0) ? 1.f : -1.f;
        int exponent =
            absl::Uniform<int>(absl::IntervalClosed, *random,
                               oblique_config.power_of_two().min_exponent(),
                               oblique_config.power_of_two().max_exponent());
        weight = sign * std::pow(2, exponent);
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kInteger): {
        weight = absl::Uniform(absl::IntervalClosed, *random,
                               oblique_config.integer().minimum(),
                               oblique_config.integer().maximum());
        break;
      }
      default: {
        // Return continuous weights.
        break;
      }
    }

    if (config_link.per_columns_size() > 0 &&
        config_link.per_columns(feature).has_monotonic_constraint()) {
      const bool direction_increasing =
          config_link.per_columns(feature).monotonic_constraint().direction() ==
          model::proto::MonotonicConstraint::INCREASING;
      if (direction_increasing == (weight < 0)) {
        weight = -weight;
      }
      // As soon as one selected feature is monotonic, the oblique split
      // becomes monotonic.
      *monotonic_direction = 1;
    }

    const auto& spec = data_spec.columns(feature).numerical();
    switch (oblique_config.normalization()) {
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
        return weight;
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
          STANDARD_DEVIATION:
        return weight / std::max(1e-6, spec.standard_deviation());
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::MIN_MAX:
        return weight / std::max(1e-6f, spec.max_value() - spec.min_value());
    }
  };

  // =================  PARTIAL FISHER–YATES  =================
  // Copy indices locally so we can shuffle. features should remain const
  std::vector<int> pool(features.begin(), features.end());
  projection->reserve(k);

  for (int i = 0; i < k; ++i) {
      const int j = absl::Uniform<int>(*random, i, p);
      std::swap(pool[i], pool[j]);

      const int feature = pool[i];
      // This costs nothing if not compiled w/ -c dbg
      DCHECK_EQ(data_spec.columns(feature).type(), dataset::proto::NUMERICAL);

      projection->push_back({feature, gen_weight(feature)});
  }

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

// Wrapper function to easily switch between implementations
void SampleProjection_wrapper(const absl::Span<const int>& features,
                             const proto::DecisionTreeTrainingConfig& dt_config,
                             const dataset::proto::DataSpecification& data_spec,
                             const model::proto::TrainingConfigLinking& config_link,
                             const float projection_density,
                             internal::Projection* projection,
                             int8_t* monotonic_direction,
                             utils::RandomEngine* random) {
#if USE_FLOYDS_SAMPLER
    SampleProjection_floyds(features, dt_config, data_spec, config_link, 
                            projection_density, projection, monotonic_direction, random);
#else
    SampleProjection_fisher_yates(features, dt_config, data_spec, config_link, 
                                  projection_density, projection, monotonic_direction, random);
#endif
}

absl::Status SetCondition(const internal::Projection& projection, const float threshold,
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



int main() {
    const std::uintmax_t NUM_ITERATIONS = 10'000'000;
    const int NUM_FEATURES = 1000;
    const float PROJECTION_DENSITY_FACTOR = 3.0;
    const float PROJECTION_DENSITY = PROJECTION_DENSITY_FACTOR / NUM_FEATURES;
    
    // Setup test data - feature indices
    std::vector<int> features(NUM_FEATURES);
    std::iota(features.begin(), features.end(), 0);
    
    // Create realistic config objects
    proto::DecisionTreeTrainingConfig dt_config;
    auto* sparse_oblique = dt_config.mutable_sparse_oblique_split();

    sparse_oblique->set_projection_density_factor(PROJECTION_DENSITY_FACTOR);
    sparse_oblique->set_normalization(
        proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE);
    
    // Create data specification with numerical columns
    dataset::proto::DataSpecification data_spec;
    for (int i = 0; i < NUM_FEATURES; ++i) {
        auto* column = data_spec.add_columns();
        column->set_type(dataset::proto::NUMERICAL);
        auto* numerical = column->mutable_numerical();
        numerical->set_min_value(0.0);
        numerical->set_max_value(10.0);
        numerical->set_standard_deviation(1.0);
    }
    
    // Create training config linking (empty for basic case)
    model::proto::TrainingConfigLinking config_link;
    
    // Random engine
    utils::RandomEngine random(42); // Fixed seed for reproducibility
    
    // Output variables
    internal::Projection projection;
    int8_t monotonic_direction;
    
    std::cout << "Starting benchmark with " << NUM_ITERATIONS << " iterations..." << std::endl;
    std::cout << "Features: " << NUM_FEATURES << ", Projection density factor: " << PROJECTION_DENSITY_FACTOR << std::endl;
    std::cout << "Using implementation: " << (USE_FLOYDS_SAMPLER ? "Floyd's Sampler" : "Fisher-Yates") << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (std::uintmax_t i = 0; i < NUM_ITERATIONS; ++i) {
        SampleProjection_wrapper(
            absl::MakeConstSpan(features), 
            dt_config, 
            data_spec, 
            config_link,
            PROJECTION_DENSITY, 
            &projection, 
            &monotonic_direction, 
            &random
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double total_time_ms = duration.count() / 1000.0;
    double avg_time_ns = (duration.count() * 1000.0) / NUM_ITERATIONS;
    
    std::cout << std::endl;
    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average time per call: " << avg_time_ns << " ns" << std::endl;
    std::cout << "Calls per second: " << (NUM_ITERATIONS * 1000.0) / total_time_ms << std::endl;
    
    // Show sample output
    std::cout << std::endl << "Sample projection (last call):" << std::endl;
    for (const auto& elem : projection) {
        std::cout << "  Feature " << elem.attribute_idx << " -> weight " << elem.weight << std::endl;
    }
    
    return 0;
}