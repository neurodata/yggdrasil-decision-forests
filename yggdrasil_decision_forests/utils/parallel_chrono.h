#pragma once
#include <atomic>
#include <chrono>

// TODO Replace this back with CHRONO_MEASUREMENTS_LOG_LEVEL
#ifdef CHRONO_ENABLED
namespace yggdrasil_decision_forests::chrono_prof {

enum FuncId { kTreeTrain = 0, kNumFuncs };

extern std::array<std::atomic<uint64_t>, kNumFuncs> global_stats;

class ScopedTimerTop {
 public:
  explicit ScopedTimerTop(FuncId id) : id_(id),
      start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimerTop() {
    auto dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_).count();
    global_stats[id_].fetch_add(dt, std::memory_order_relaxed);
  }
 private:
  FuncId id_;
  std::chrono::steady_clock::time_point start_;
};

#define CHRONO_SCOPE_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimerTop \
      CONCAT(__timer__, __LINE__)(ID)
}  // namespace yggdrasil_decision_forests::chrono_prof
#else
#define CHRONO_SCOPE_TOP(ID)
#endif