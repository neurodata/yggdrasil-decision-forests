#ifdef CHRONO_ENABLED
namespace yggdrasil_decision_forests::chrono_prof {

// 1.  Add a new id
enum FuncId {
  kTreeTrain = 0,
  kSampleProjection,
  kProjectionEvaluate,     // projection_evaluator.Evaluate
  kEvaluateProjection,     // EvaluateProjection(...)
  kNumFuncs
};

// 2.  Global atomic counters (already defined in some .cc)
inline std::array<std::atomic<uint64_t>, kNumFuncs> global_stats;

// 3.  Generic scoped timer that writes to the atomic array
class ScopedTimer {
 public:
  explicit ScopedTimer(FuncId id) : id_(id),
      start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    const auto dt = std::chrono::steady_clock::now() - start_;
    global_stats[id_].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
        std::memory_order_relaxed);
  }
 private:
  FuncId id_;
  std::chrono::steady_clock::time_point start_;
};

// 4.  Macros
#define YDF_PP_CAT_INNER(a, b) a##b
#define YDF_PP_CAT(a, b) YDF_PP_CAT_INNER(a, b)

#define CHRONO_SCOPE(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_timer_, __LINE__)(ID)

#define CHRONO_SCOPE_TOP(ID) CHRONO_SCOPE(ID)   // still available

}  // namespace yggdrasil_decision_forests::chrono_prof
#else
#define CHRONO_SCOPE(ID)
#define CHRONO_SCOPE_TOP(ID)
#endif