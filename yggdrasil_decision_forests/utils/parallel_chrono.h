#ifdef CHRONO_ENABLED
namespace yggdrasil_decision_forests::chrono_prof {

// ── existing enum and global_stats … ─────────────────────────────────
enum FuncId { kTreeTrain = 0, kSampleProjection, kProjectionEvaluate,
              kEvaluateProjection, kNumFuncs };

inline std::array<std::atomic<uint64_t>, kNumFuncs> global_stats{};

// ── NEW: per-thread context ─────────────────────────────────────────
struct TlsCtx {
  int cur_tree  = -1;   // set in pool lambda
  int cur_depth = -1;   // maintained by NodeTrain recursion
};
inline thread_local TlsCtx tls_ctx;

// ── NEW: 2-D global accumulator  time_ns[tree][depth][func] ─────────
// No need for atomic. Always run by 1 thread before ThreadPool in random_forest.cc
using FuncArray = std::array<uint64_t, kNumFuncs>;   // plain 64-bit counters
using DepthVec  = std::vector<FuncArray>;            // one per depth
inline std::vector<DepthVec> time_ns;                // one per tree

// Add dt_ns to the right bucket.
inline void add_time(int tree, int depth, FuncId id, uint64_t dt_ns) {
  if (tree < 0 || tree >= static_cast<int>(time_ns.size())) {
    // Fallback / global bucket: atomic because many threads can hit it.
    global_stats[id].fetch_add(dt_ns, std::memory_order_relaxed);
    return;
  }
  auto& by_depth = time_ns[tree];
  if (depth >= static_cast<int>(by_depth.size()))
    by_depth.resize(depth + 1);                      // FuncArray is movable
  by_depth[depth][id] += dt_ns;                      // single-threaded write
}

// ---------- scoped timer ----------------------------------------------------
class ScopedTimer {
 public:
  explicit ScopedTimer(FuncId id)
      : id_(id), start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count();
    add_time(tls_ctx.cur_tree, tls_ctx.cur_depth, id_, dt_ns);
  }
 private:
  FuncId id_;
  std::chrono::steady_clock::time_point start_;
};

// ── light helpers to set tree / depth ───────────────────────────────
struct TreeScope {
  explicit TreeScope(int tree) { tls_ctx.cur_tree = tree; tls_ctx.cur_depth = -1; }
  ~TreeScope() { tls_ctx.cur_tree = -1; }
};

struct DepthScope {
  DepthScope() { ++tls_ctx.cur_depth; }
  ~DepthScope() { --tls_ctx.cur_depth; }
};

// macros unchanged
#define YDF_PP_CAT_INNER(a,b) a##b
#define YDF_PP_CAT(a,b) YDF_PP_CAT_INNER(a,b)

#define CHRONO_SCOPE(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_timer_, __LINE__)(ID)

#define CHRONO_SCOPE_TOP(ID) CHRONO_SCOPE(ID)

}  // namespace yggdrasil_decision_forests::chrono_prof
#else
#define CHRONO_SCOPE(ID)
#define CHRONO_SCOPE_TOP(ID)
#endif