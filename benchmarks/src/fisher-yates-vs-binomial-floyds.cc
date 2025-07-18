#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>

// Yggdrasil Decision Forests includes
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
// #include "yggdrasil_decision_forests/model/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "absl/types/span.h"

using namespace yggdrasil_decision_forests;
using namespace yggdrasil_decision_forests::model::decision_tree;

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
    std::cout << "Using real Yggdrasil Decision Forests library" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        internal::SampleProjection(
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