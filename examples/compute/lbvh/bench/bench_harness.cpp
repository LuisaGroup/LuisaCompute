// Host-side plumbing of the software-LBVH performance benchmark: the command
// line, wall-clock timing samples, the device-memory estimate the budget check
// uses, and the least-squares fit of the stress sweeps.

#include "bench_harness.h"

#include "bench_stats.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <cmath>

namespace luisa::example::lbvh {

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

namespace {

// Splits `--name=value` / `--name value` out of `argv`.  Returns false when
// `arg` is neither form of `name`; sets `error` when the name matches but the
// value is missing or empty.
[[nodiscard]] bool take_value(int argc, char *const *argv, int &i, luisa::string_view arg,
                              luisa::string_view name, luisa::string_view &value,
                              luisa::string &error) noexcept {
    if (arg == name) {
        if (i + 1 >= argc || argv[i + 1] == nullptr || argv[i + 1][0] == '\0') {
            error = luisa::format("missing value for {}", name);
            return true;
        }
        value = luisa::string_view{argv[++i]};
        return true;
    }
    if (arg.size() > name.size() && arg.substr(0u, name.size()) == name &&
        arg[name.size()] == '=') {
        value = arg.substr(name.size() + 1u);
        if (value.empty()) { error = luisa::format("missing value for {}", name); }
        return true;
    }
    return false;
}

[[nodiscard]] bool parse_u64(luisa::string_view text, uint64_t &value) noexcept {
    if (text.empty() || text.size() > 20u) { return false; }
    uint64_t result = 0u;
    for (auto c : text) {
        if (c < '0' || c > '9') { return false; }
        result = result * 10u + static_cast<uint64_t>(c - '0');
    }
    value = result;
    return true;
}

[[nodiscard]] bool parse_f64(luisa::string_view text, double &value) noexcept {
    if (text.empty()) { return false; }
    auto buffer = luisa::string{text};
    char *end = nullptr;
    auto result = std::strtod(buffer.c_str(), &end);
    if (end == nullptr || end != buffer.c_str() + buffer.size()) { return false; }
    if (!std::isfinite(result)) { return false; }
    value = result;
    return true;
}

}// namespace

BenchOptionParse parse_bench_options(int argc, char *const *argv) noexcept {
    BenchOptionParse result;
    auto &options = result.options;
    auto set_error = [&result](luisa::string_view message) noexcept {
        result.error = luisa::string{message};
    };
    for (auto i = 1; i < argc; i++) {
        if (argv[i] == nullptr) { continue; }
        luisa::string_view arg{argv[i]};
        luisa::string_view value;
        // the boolean flags come first: they are the ones that also have to work
        // without a backend (`--list`, `--help`).
        if (arg == "-h" || arg == "--help") {
            options.help = true;
            continue;
        }
        if (arg == "--list") {
            options.list = true;
            continue;
        }
        if (arg == "--stress-build") {
            options.stress_build = true;
            continue;
        }
        if (arg == "--stress-traversal") {
            options.stress_traversal = true;
            continue;
        }
        if (arg == "--validate") {
            options.validate = true;
            continue;
        }
        if (arg == "--repeat-check") {
            options.repeat_check = true;
            continue;
        }
        if (arg == "--force-oversize") {
            // documented as dangerous: it disables the pre-flight that keeps a
            // single submission short enough for the driver not to reset
            options.force_oversize = true;
            continue;
        }
        // then the options that take a value
        if (take_value(argc, argv, i, arg, "--scene", value, result.error)) {
            if (result.error.empty()) { options.scene = luisa::string{value}; }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--mesh", value, result.error)) {
            if (result.error.empty()) { options.mesh_path = luisa::string{value}; }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--triangles", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && !parse_u64(value, n)) {
                set_error(luisa::format("invalid --triangles value '{}'", value));
            } else if (result.error.empty()) {
                options.triangles = static_cast<size_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--instances", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && !parse_u64(value, n)) {
                set_error(luisa::format("invalid --instances value '{}'", value));
            } else if (result.error.empty()) {
                options.instances = static_cast<size_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--rays", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && !parse_u64(value, n)) {
                set_error(luisa::format("invalid --rays value '{}'", value));
            } else if (result.error.empty()) {
                options.rays = static_cast<size_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--seed", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && !parse_u64(value, n)) {
                set_error(luisa::format("invalid --seed value '{}'", value));
            } else if (result.error.empty()) {
                options.seed = n;
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--iters", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && (!parse_u64(value, n) || n == 0u || n > 10000u)) {
                set_error(luisa::format("invalid --iters value '{}'", value));
            } else if (result.error.empty()) {
                options.iterations = static_cast<uint32_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--warmup", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && (!parse_u64(value, n) || n > 1000u)) {
                set_error(luisa::format("invalid --warmup value '{}'", value));
            } else if (result.error.empty()) {
                options.warmup = static_cast<uint32_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--budget-gib", value, result.error)) {
            double f = 0.0;
            if (result.error.empty() && (!parse_f64(value, f) || f <= 0.0)) {
                set_error(luisa::format("invalid --budget-gib value '{}'", value));
            } else if (result.error.empty()) {
                options.budget_gib = f;
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--max-triangles", value, result.error)) {
            uint64_t n = 0u;
            if (result.error.empty() && (!parse_u64(value, n) || n == 0u)) {
                set_error(luisa::format("invalid --max-triangles value '{}'", value));
            } else if (result.error.empty()) {
                options.max_triangles = static_cast<size_t>(n);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--dispatch-budget-ms", value, result.error)) {
            double f = 0.0;
            if (result.error.empty() && (!parse_f64(value, f) || f <= 0.0)) {
                set_error(luisa::format("invalid --dispatch-budget-ms value '{}'", value));
            } else if (result.error.empty()) {
                options.dispatch_budget_ms = f;
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--max-seconds", value, result.error)) {
            double f = 0.0;
            if (result.error.empty() && (!parse_f64(value, f) || f <= 0.0)) {
                set_error(luisa::format("invalid --max-seconds value '{}'", value));
            } else if (result.error.empty()) {
                options.max_seconds = f;
            }
            continue;
        }
        if (arg.size() > 1u && arg[0] == '-') {
            set_error(luisa::format("unknown option '{}'", arg));
            continue;
        }
        // positional: the backend device name
        if (!options.backend.empty()) {
            set_error(luisa::format("unexpected extra argument '{}'", arg));
        } else {
            options.backend = luisa::string{arg};
        }
    }
    return result;
}

void print_bench_usage(const char *executable) noexcept {
    std::printf(
        "Usage: %s <backend> [options]\n"
        "\n"
        "Performance benchmark of the two-level software LBVH: it drives adversarial\n"
        "(worst-case) scenes through both the build and the traversal, and reports a\n"
        "per-stage breakdown so the implementation can be optimized.\n"
        "\n"
        "  --scene <name|all|worst>   scene(s) to run; 'all' (default) runs the catalogue,\n"
        "                             'worst' runs only the scenes marked worst-case\n"
        "  --triangles <n>            override the scene's triangle count\n"
        "  --instances <k>            override the scene's instance count (K meshes)\n"
        "  --rays <n>                 override the ray count\n"
        "  --seed <n>                 scene/ray seed (default 0)\n"
        "  --iters <n>                timed iterations per measurement (default 5)\n"
        "  --warmup <n>               untimed warm-up iterations (default 1)\n"
        "  --budget-gib <f>           device-memory budget in GiB (default 5.0); a scene or\n"
        "                             sweep step whose estimate exceeds it is skipped\n"
        "  --max-triangles <n>        cap used by the stress sweeps (default 8388608)\n"
        "  --max-seconds <f>          stop the sweeps / reduce the ray count when a\n"
        "                             measurement would take longer (default 30)\n"
        "  --dispatch-budget-ms <f>   longest single device submission the benchmark\n"
        "                             may predict, in ms (default 1000); the traversal\n"
        "                             grows its slice by measurement until one slice\n"
        "                             reaches ~3/4 of it, and an oversized build is\n"
        "                             refused instead of hanging the driver\n"
        "  --force-oversize           skip the pre-flight and submit anyway: a single\n"
        "                             submission over ~2 s removes the device (TDR)\n"
        "  --stress-build             in addition, sweep the triangle count in log2 steps\n"
        "  --stress-traversal         in addition, sweep the ray count in log2 steps\n"
        "  --validate                 run the structural self-check and compare the hits\n"
        "                             with the Luisa RTX reference (RTX-capable backend)\n"
        "  --repeat-check             verify the traversal result is bit-identical across\n"
        "                             the timed iterations\n"
        "  --mesh <file.obj>          run a scene loaded from an OBJ file (v/f only) instead\n"
        "                             of the catalogue: any real asset works (Sponza, the\n"
        "                             Stanford dragon, ...); the mesh becomes one BLAS\n"
        "  --list                     print the scene catalogue and exit\n"
        "  -h, --help                 print this help and exit\n"
        "\n"
        "Notes:\n"
        "  * every timing is a host-observed wall time around submission + synchronize,\n"
        "    so only release-mode numbers are meaningful;\n"
        "  * a sliced traversal reports trace_ms as the *sum* over its slices, i.e. the\n"
        "    time of the whole traversal and not of one submission; the slice size is\n"
        "    grown by measurement until one slice reaches ~3/4 of --dispatch-budget-ms;\n"
        "  * a traversal that would exceed --max-seconds is traced over a smaller,\n"
        "    re-generated ray frustum; the record prints both ray counts (rays and\n"
        "    rays_requested);\n"
        "  * all scenes are deterministic for a given (scene, triangles, instances, seed)\n"
        "    on every backend.\n",
        executable);
}

// ---------------------------------------------------------------------------
// Timing samples
// ---------------------------------------------------------------------------

double BenchTiming::min_ms() const noexcept {
    return samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end());
}

double BenchTiming::mean_ms() const noexcept {
    if (samples_ms.empty()) { return 0.0; }
    auto sum = 0.0;
    for (auto sample : samples_ms) { sum += sample; }
    return sum / static_cast<double>(samples_ms.size());
}

double BenchTiming::median_ms() const noexcept {
    if (samples_ms.empty()) { return 0.0; }
    luisa::vector<double> sorted{samples_ms};
    std::sort(sorted.begin(), sorted.end());
    auto n = sorted.size();
    return n % 2u == 0u ? 0.5 * (sorted[n / 2u - 1u] + sorted[n / 2u]) : sorted[n / 2u];
}

// ---------------------------------------------------------------------------
// Device-memory estimate
// ---------------------------------------------------------------------------

size_t lbvh_storage_bytes(const LbvhStorage::Sizes &sizes) noexcept {
    // `total_bytes()` is the storage's own query and already includes the scratch
    // of the parallel radix sort; the benchmark must budget for exactly what the
    // storage allocates, so it asks the storage instead of re-adding the parts.
    return sizes.total_bytes();
}

BenchMemoryEstimate estimate_bench_memory(size_t triangles, size_t instances,
                                          size_t blas_count, size_t vertices,
                                          size_t rays, bool with_rtx_reference) noexcept {
    // The storage sizes are the same host-side query the demo uses; the estimate
    // is therefore exact for everything the LBVH allocates.
    auto sizes = LbvhStorage::estimate(triangles, instances, blas_count);
    BenchMemoryEstimate estimate;
    estimate.geometry_bytes = vertices * sizeof(float3) + triangles * sizeof(Triangle);
    estimate.lbvh_bytes = lbvh_storage_bytes(sizes);
    // The tree statistics own one leaf-range record (uint2) per node slot and one
    // encoded leaf depth per primitive slot (see bench_stats.h).
    estimate.stats_bytes = sizes.node_capacity * sizeof(uint2) +
                           sizes.primitive_capacity * sizeof(uint);
    // rays + the hits of the plain walk + the hits of the instrumented walk
    // (+ the RTX reference of `--validate`) + the per-ray counters
    auto hit_buffers = with_rtx_reference ? 3u : 2u;
    estimate.ray_bytes = rays * (sizeof(LbvhRay) + hit_buffers * sizeof(LbvhHit) +
                                 sizeof(LbvhRayStats));
    return estimate;
}

luisa::string human_bytes(size_t bytes) noexcept {
    constexpr auto kibi = 1024.0;
    char buffer[64];
    if (bytes < 1024u) {
        std::snprintf(buffer, sizeof(buffer), "%llu B", static_cast<unsigned long long>(bytes));
    } else if (bytes < 1024u * 1024u) {
        std::snprintf(buffer, sizeof(buffer), "%.1f KiB", static_cast<double>(bytes) / kibi);
    } else if (bytes < 1024u * 1024u * 1024u) {
        std::snprintf(buffer, sizeof(buffer), "%.1f MiB",
                      static_cast<double>(bytes) / (kibi * kibi));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f GiB",
                      static_cast<double>(bytes) / (kibi * kibi * kibi));
    }
    return luisa::string{buffer};
}

// ---------------------------------------------------------------------------
// Stress-sweep fit
// ---------------------------------------------------------------------------

BenchScaling fit_log_log(luisa::span<const double> sizes,
                         luisa::span<const double> timings) noexcept {
    BenchScaling fit;
    if (sizes.size() != timings.size() || sizes.size() < 2u) { return fit; }
    auto n = 0.0;
    auto sum_x = 0.0;
    auto sum_y = 0.0;
    auto sum_xx = 0.0;
    auto sum_xy = 0.0;
    for (auto i = 0u; i < sizes.size(); i++) {
        if (sizes[i] <= 0.0 || timings[i] <= 0.0) { continue; }
        auto x = std::log(sizes[i]);
        auto y = std::log(timings[i]);
        n += 1.0;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    fit.count = static_cast<size_t>(n);
    if (n < 2.0) { return fit; }
    auto denominator = n * sum_xx - sum_x * sum_x;
    if (std::abs(denominator) < 1.0e-30) { return fit; }
    fit.exponent = (n * sum_xy - sum_x * sum_y) / denominator;
    fit.intercept = (sum_y - fit.exponent * sum_x) / n;
    auto mean_y = sum_y / n;
    auto ss_res = 0.0;
    auto ss_tot = 0.0;
    for (auto i = 0u; i < sizes.size(); i++) {
        if (sizes[i] <= 0.0 || timings[i] <= 0.0) { continue; }
        auto x = std::log(sizes[i]);
        auto y = std::log(timings[i]);
        auto predicted = fit.exponent * x + fit.intercept;
        ss_res += (y - predicted) * (y - predicted);
        ss_tot += (y - mean_y) * (y - mean_y);
    }
    fit.r2 = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 1.0;
    return fit;
}

luisa::string BenchScaling::to_string(const char *stage) const noexcept {
    if (count < 2u) {
        return luisa::format("{}: not enough points to fit", stage);
    }
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%s: p=%.3f r2=%.4f (%llu points)", stage,
                  exponent, r2, static_cast<unsigned long long>(count));
    return luisa::string{buffer};
}

}// namespace luisa::example::lbvh
