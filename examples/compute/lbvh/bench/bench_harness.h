// Host-side plumbing of the software-LBVH performance benchmark: the command
// line, wall-clock timing samples, the device-memory estimate the budget check
// uses, and the least-squares fit of the stress sweeps.
//
// Everything here is plain host code.  It deliberately knows nothing about the
// DSL or about the device, because the driver has to answer two questions
// *before* it creates a single device resource:
//
//   * how many device bytes this measurement will allocate (the benchmark must
//     never exceed its memory budget, see `--budget-gib`), and
//   * how long a measurement may take before it is aborted (a debug/ASan run
//     must not hang the machine, see `--max-seconds`).
//
// All timings are host-observed wall times around `stream << ... << synchronize()`.
// They therefore include the submission of the recorded work and the fence, so
// only release-mode numbers are meaningful; debug/ASan runs are for correctness.

#pragma once

#include "../lbvh_common.h"
#include "../lbvh_storage.h"

#include <cstddef>
#include <cstdint>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct BenchOptions {
    luisa::string backend;     // argv[1], mandatory unless --list/--help
    luisa::string scene{"all"};// scene name, "all" or "worst"
    luisa::string mesh_path;   // --mesh <file.obj>
    size_t triangles{0u};      // 0 = the scene's own default
    size_t instances{0u};      // 0 = the scene's own default
    size_t rays{0u};           // 0 = the scene's own default
    uint64_t seed{0u};
    uint32_t iterations{5u};
    uint32_t warmup{1u};
    double budget_gib{5.0};
    size_t max_triangles{8388608u};// 1 << 23
    double max_seconds{30.0};
    // Longest single device submission the benchmark is allowed to predict.  A
    // multi-second dispatch makes the Windows driver reset the device (TDR,
    // ~2 s), which is exactly what `DXGI_ERROR_DEVICE_REMOVED` /
    // `VK_ERROR_DEVICE_LOST` are, so the traversal is split into chunks and the
    // build is pre-flighted against this limit.  A single submission can never
    // be aborted from the host, so the only safe answer is to not submit it.
    double dispatch_budget_ms{1000.0};
    // Escape hatch for the pre-flight: submitting a scene whose build is
    // predicted to blow through the dispatch budget may remove the device.
    bool force_oversize{false};
    bool stress_build{false};
    bool stress_traversal{false};
    bool validate{false};
    bool repeat_check{false};
    bool list{false};
    bool help{false};
    // Storage compaction (the software analogue of the RTX compacted copy, see
    // `SoftwareLbvh::compact`): off by default so the existing numbers stay
    // comparable, `--compact[=as-built]` turns it on, and `--headroom <factor>`
    // sizes the storage (and hence the loose slack the compaction reclaims) from
    // a budget instead of the exact scene.  The `subtree_contiguous` policy is
    // reserved but not implemented (`--compact=subtrees` fails the parse; see
    // bench/README.md).
    bool compact{false};
    // Also retire the build scratch through the compaction's completion callback
    // (opt-in; after it the storage is traverse-only - see
    // `LbvhStorage::release_build_scratch`).
    bool release_scratch{false};
    double headroom{2.0};
};

// Result of parsing the command line.  A parse error is *recoverable*: the
// driver prints the usage and returns a non-zero exit code (project code must
// not raise, and an argument mistake is not a fatal invariant violation), so the
// message travels in `error` instead of through LUISA_ERROR.
struct BenchOptionParse {
    BenchOptions options;
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

[[nodiscard]] BenchOptionParse parse_bench_options(int argc, char *const *argv) noexcept;
void print_bench_usage(const char *executable) noexcept;

// ---------------------------------------------------------------------------
// Timing samples
// ---------------------------------------------------------------------------

// Wall-clock samples (milliseconds) of one measurement.  The headline is the
// *minimum* (the run least polluted by the OS and by the driver), the median
// and the mean are reported next to it because the spread of a GPU measurement
// is itself a result: a large mean/min ratio means the measurement is noisy.
struct BenchTiming {
    luisa::vector<double> samples_ms;

    void add(double ms) noexcept { samples_ms.emplace_back(ms); }
    [[nodiscard]] size_t count() const noexcept { return samples_ms.size(); }
    [[nodiscard]] double min_ms() const noexcept;
    [[nodiscard]] double median_ms() const noexcept;
    [[nodiscard]] double mean_ms() const noexcept;
};

// The build measurement: the total plus the per-stage breakdown of the library
// timing hook (`LbvhBuildTimings`), which is what tells the optimizer *where*
// the build time goes.
struct BenchBuildTiming {
    BenchTiming total_ms;
    BenchTiming prim_ms;// triangle / instance AABB kernel
    BenchTiming morton_ms;
    BenchTiming sort_ms;  // the 4 LSD radix passes
    BenchTiming node_ms;  // Karras radix-tree construction
    size_t primitives{0u};// BLAS triangles + TLAS instances (for Mprim/s)
    size_t nodes{0u};
    size_t blas_count{0u};
    size_t instance_count{0u};

    [[nodiscard]] double mprim_per_s() const noexcept {
        auto ms = total_ms.min_ms();
        return ms > 0.0 ? static_cast<double>(primitives) * 1.0e-6 / (ms * 1.0e-3) : 0.0;
    }
};

// The traversal measurement: the plain two-level traversal of the library plus
// the counters of the instrumented walk (bench_stats.h).
struct BenchTraceTiming {
    BenchTiming total_ms;
    size_t rays{0u};
    size_t hits{0u};
    size_t misses{0u};

    [[nodiscard]] double mray_per_s() const noexcept {
        auto ms = total_ms.min_ms();
        return ms > 0.0 ? static_cast<double>(rays) * 1.0e-6 / (ms * 1.0e-3) : 0.0;
    }
    [[nodiscard]] double ns_per_ray() const noexcept {
        auto ms = total_ms.min_ms();
        return rays > 0u ? ms * 1.0e6 / static_cast<double>(rays) : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Device-memory estimate
// ---------------------------------------------------------------------------

// Exact byte count of everything one measurement allocates on the device.
//
// `estimate()` is called before any allocation and compared against the budget;
// the numbers are *exact* (not upper bounds) except for the deliberate
// over-allocation of the shared storage, which reserves the two-node-per-
// primitive budget for every tree instead of `2 * primitives - 1` for the last
// one - so the estimate is a small over-estimate of the truth by construction,
// which is the safe direction for a budget.
struct BenchMemoryEstimate {
    size_t geometry_bytes{0u};// shared vertex + triangle buffers
    size_t lbvh_bytes{0u};    // every buffer of `LbvhStorage`
    size_t stats_bytes{0u};   // tree-statistics scratch (leaf ranges, counters)
    size_t ray_bytes{0u};     // rays + hits + per-ray counters

    [[nodiscard]] size_t total_bytes() const noexcept {
        return geometry_bytes + lbvh_bytes + stats_bytes + ray_bytes;
    }
};

// Byte estimate of one measurement of `triangles` triangles over `instances`
// instances in `blas_count` meshes, `vertices` shared vertices and `rays` rays.
// With `with_compaction` it also counts the transient peak of
// `SoftwareLbvh::compact()`: the loose node buffer and the dense one exist at the
// same time until the loose one is retired, so one more node buffer of room is
// required.  `headroom` is the factor the storage is deliberately over-sized by
// (`--headroom`), applied to the triangle/instance capacities exactly like
// `SceneResources` applies it - the budget check has to see the storage the
// measurement actually allocates, not the nominal scene size.
[[nodiscard]] BenchMemoryEstimate estimate_bench_memory(size_t triangles, size_t instances,
                                                        size_t blas_count, size_t vertices,
                                                        size_t rays,
                                                        bool with_rtx_reference,
                                                        bool with_compaction = false,
                                                        double headroom = 1.0) noexcept;

// Bytes the shared LBVH storage allocates for a scene with these capacities.
// `Sizes::key_bytes` (and hence `Sizes::total_bytes()`) accounts for one of the
// two ping-pong Morton-key buffers while the storage allocates both, so the
// budget counts the second one explicitly.  `with_compaction` adds the dense node
// buffer `compact()` allocates while the loose one is still alive.
[[nodiscard]] size_t lbvh_storage_bytes(const LbvhStorage::Sizes &sizes,
                                        bool with_compaction = false) noexcept;

[[nodiscard]] luisa::string human_bytes(size_t bytes) noexcept;

// ---------------------------------------------------------------------------
// Stress-sweep fit
// ---------------------------------------------------------------------------

// Least-squares fit of `log(timing) = exponent * log(size) + c` over a sweep.
// The exponent is what identifies the complexity of a stage: ~1.0 is linear,
// ~1.0 + 1/ln(N) is N log N, anything above is super-linear.
struct BenchScaling {
    double exponent{0.0};
    double intercept{0.0};
    double r2{0.0};
    size_t count{0u};
    [[nodiscard]] luisa::string to_string(const char *stage) const noexcept;
};

[[nodiscard]] BenchScaling fit_log_log(luisa::span<const double> sizes,
                                       luisa::span<const double> timings) noexcept;

}// namespace luisa::example::lbvh
