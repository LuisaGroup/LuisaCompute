// Correctness and performance harness of the LBVH radix sort (`lbvh_sort.h`).
//
// The sort is the dominant stage of an LBVH build and the one build stage whose
// *output order* is semantically load-bearing (four LSD passes only add up to a
// sorted array if every pass is stable), so this driver does two jobs:
//
//   * `--check` verifies the sort against a host reference on every backend:
//     a `std::stable_sort` over `(code, input index)` of the same input, for
//     every count of interest, every key distribution, a non-zero `base` (with
//     garbage before and after the range, which the sort must not touch), and
//     both implementations.  It checks the result element by element *and*
//     independently: sorted by code, ascending slots inside an equal-code run,
//     the slots are a permutation of the range, and two runs (and both
//     implementations) are bit-identical.
//
//   * `--measure` times the implementations against each other (`min` of N
//     samples, the same convention as `bench_harness.h`) and reports the
//     speedup against the current single-work-group sort.
//
// Both phases are guarded the way `benchmark_lbvh.cpp` is: the device-memory
// estimate is checked against `--budget-gib` before anything is allocated and a
// size whose *predicted* submission time exceeds `--dispatch-budget-ms` is
// refused (a multi-second submission removes the device on Windows) instead of
// being measured.  Only release builds produce meaningful timings.
//
// Usage: example_software_lbvh_sort_bench <backend> [options]   (see --help)

#include "bench_harness.h"

#include "../lbvh_sort.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::lbvh;

namespace {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

// Everything the sort benchmark adds on top of the harness command line.  The
// harness options are parsed by `parse_bench_options` (see bench_harness.h), so
// the ones this driver does not know are passed through unchanged - the table of
// options that take a value mirrors `parse_bench_options`.
struct SortOptions {
    BenchOptions harness;
    bool check{false};
    bool measure{false};
    bool breakdown{false};
    bool items_sweep{false};
    luisa::string method{"all"};// all | single | multi | auto
    uint64_t base{0u};
    luisa::vector<size_t> counts;      // empty: the benchmark's own list
    luisa::vector<size_t> check_counts;// empty: the correctness catalogue
    luisa::string error;
};

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

// `a,b,c` (or `a`) of decimal counts.
[[nodiscard]] bool parse_counts(luisa::string_view text, luisa::vector<size_t> &counts) noexcept {
    counts.clear();
    while (!text.empty()) {
        auto end = text.find(',');
        auto piece = end == luisa::string_view::npos ? text : text.substr(0u, end);
        uint64_t value = 0u;
        if (!parse_u64(piece, value) || value > 0xffffffffull) { return false; }
        counts.emplace_back(static_cast<size_t>(value));
        if (end == luisa::string_view::npos) { return true; }
        text = text.substr(end + 1u);
    }
    return !counts.empty();
}

// The harness options that consume the following argument, i.e. the ones whose
// value must be forwarded too.
[[nodiscard]] bool harness_takes_argument(luisa::string_view arg) noexcept {
    for (auto name : {"--scene", "--mesh", "--triangles", "--instances", "--rays",
                      "--seed", "--iters", "--warmup", "--budget-gib", "--max-triangles",
                      "--dispatch-budget-ms", "--max-seconds"}) {
        if (arg == name) { return true; }
    }
    return false;
}

[[nodiscard]] SortOptions parse_options(int argc, char *const *argv) noexcept {
    SortOptions options;
    luisa::vector<char *> forwarded;
    forwarded.emplace_back(argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : const_cast<char *>("example_software_lbvh_sort_bench"));
    for (auto i = 1; i < argc; i++) {
        if (argv[i] == nullptr) { continue; }
        luisa::string_view arg{argv[i]};
        luisa::string_view value;
        if (arg == "--check") {
            options.check = true;
            continue;
        }
        if (arg == "--measure" || arg == "--bench") {
            options.measure = true;
            continue;
        }
        if (arg == "--breakdown") {
            options.breakdown = true;
            continue;
        }
        if (arg == "--items-sweep") {
            options.items_sweep = true;
            continue;
        }
        if (take_value(argc, argv, i, arg, "--method", value, options.error)) {
            if (options.error.empty()) {
                if (value != "all" && value != "single" && value != "multi" && value != "auto") {
                    options.error = luisa::format("invalid --method value '{}'", value);
                } else {
                    options.method = luisa::string{value};
                }
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--base", value, options.error)) {
            uint64_t n = 0u;
            if (options.error.empty() && (!parse_u64(value, n) || n > 0xffffffffull)) {
                options.error = luisa::format("invalid --base value '{}'", value);
            } else if (options.error.empty()) {
                options.base = n;
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--counts", value, options.error)) {
            if (options.error.empty() && !parse_counts(value, options.counts)) {
                options.error = luisa::format("invalid --counts value '{}'", value);
            }
            continue;
        }
        if (take_value(argc, argv, i, arg, "--check-counts", value, options.error)) {
            if (options.error.empty() && !parse_counts(value, options.check_counts)) {
                options.error = luisa::format("invalid --check-counts value '{}'", value);
            }
            continue;
        }
        // not ours: hand it to the harness parser, together with its value
        forwarded.emplace_back(argv[i]);
        if (harness_takes_argument(arg) && i + 1 < argc) {
            forwarded.emplace_back(argv[++i]);
        }
    }
    if (!options.error.empty()) { return options; }
    auto parsed = parse_bench_options(static_cast<int>(forwarded.size()), forwarded.data());
    if (!parsed.ok()) {
        options.error = parsed.error;
        return options;
    }
    options.harness = parsed.options;
    return options;
}

void print_usage(const char *executable) noexcept {
    std::printf(
        "Usage: %s <backend> [options]\n"
        "\n"
        "Correctness and performance harness of the LBVH radix sort:\n"
        "  --check            verify the sort against a host reference (all methods,\n"
        "                     every count/distribution/base of the catalogue)\n"
        "  --measure          time single- and multi-block sorts (min of N samples)\n"
        "  (neither: both)\n"
        "\n"
        "  --method <all|single|multi|auto>  which implementation to run during\n"
        "                     --measure (default all)\n"
        "  --counts <a,b,c>   comma-separated element counts of the timing sweep\n"
        "                     (default 4096,16384,65536,262144,1048576,2097152,4194304,8388608)\n"
        "  --check-counts <a,b,c>  restrict the correctness catalogue to these counts\n"
        "                     (a debugging aid; the default is the full catalogue)\n"
        "  --base <n>         sort at a non-zero offset into the buffers, i.e. with\n"
        "                     garbage before the range (default 0)\n"
        "  --breakdown        also time 1, 2, 3 and 4 passes to attribute the cost\n"
        "  --items-sweep      also sweep the elements per chunk of the parallel path\n"
        "\n"
        "plus the options of the LBVH benchmark harness (--iters, --warmup,\n"
        "--budget-gib, --dispatch-budget-ms, --max-seconds, --force-oversize, ...):\n"
        "  --iters <n>        timed samples per measurement (default 5)\n"
        "  --warmup <n>       untimed samples first (default 1)\n"
        "  --dispatch-budget-ms <f>  refuse a size whose predicted submission is longer\n"
        "                     (default 1000; a submission over ~2 s removes the device)\n"
        "  --budget-gib <f>   device-memory budget of one measurement (default 5)\n"
        "  --max-seconds <f>  stop the sweep after this long (default 30)\n"
        "  -h, --help         this help\n",
        executable);
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

// The expected sorted range: a stable sort of the input indices by code, which
// is exactly what a stable LSD sort over `(code, input order)` produces.  The
// slot of the input element `i` is `base + i`, so the element-wise comparison
// against this reference also proves that equal codes keep their input order.
[[nodiscard]] luisa::vector<LbvhKey> reference_sort(luisa::span<const uint> codes,
                                                    uint base) noexcept {
    luisa::vector<uint> order(codes.size());
    for (auto i = 0u; i < order.size(); i++) { order[i] = static_cast<uint>(i); }
    std::stable_sort(order.begin(), order.end(),
                     [codes](uint a, uint b) noexcept { return codes[a] < codes[b]; });
    luisa::vector<LbvhKey> sorted(order.size());
    for (auto i = 0u; i < order.size(); i++) {
        sorted[i] = LbvhKey{codes[order[i]], base + order[i]};
    }
    return sorted;
}

// ---------------------------------------------------------------------------
// Key distributions
// ---------------------------------------------------------------------------

// xorshift64*, so every backend and every run gets the same keys.
struct Rng {
    uint64_t state;
    [[nodiscard]] uint32_t next() noexcept {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return static_cast<uint32_t>(state >> 32u);
    }
};

enum class Distribution : uint {
    random,      // uniform 32-bit codes
    all_equal,   // one code: every pass must keep the order untouched
    two_values,  // two codes far apart: only one pass moves anything
    ascending,   // already sorted: no pass may reorder anything
    descending,  // reverse sorted
    top_bits,    // codes differing only in the top 2 bits: catches a wrong last pass
    top_bits_low,// the same, with varying low bits: same catch, less degenerate
    mostly_equal,// one code with a few outliers
    count_
};

[[nodiscard]] const char *distribution_name(Distribution d) noexcept {
    switch (d) {
        case Distribution::random: return "random";
        case Distribution::all_equal: return "all_equal";
        case Distribution::two_values: return "two_values";
        case Distribution::ascending: return "ascending";
        case Distribution::descending: return "descending";
        case Distribution::top_bits: return "top_bits";
        case Distribution::top_bits_low: return "top_bits_low";
        case Distribution::mostly_equal: return "mostly_equal";
        default: break;
    }
    return "?";
}

void generate_codes(Distribution d, uint count, uint64_t seed,
                    luisa::vector<uint> &codes) noexcept {
    codes.resize(count);
    Rng rng{seed | 1u};
    for (auto i = 0u; i < count; i++) {
        auto v = uint{};
        switch (d) {
            case Distribution::random: v = rng.next(); break;
            case Distribution::all_equal: v = 0x5a5a5a5au; break;
            case Distribution::two_values: v = (i & 1u) != 0u ? 0xf8000000u : 0x0000001fu; break;
            case Distribution::ascending: v = i; break;
            case Distribution::descending: v = count - i; break;
            case Distribution::top_bits: v = (i & 3u) << 30u; break;
            case Distribution::top_bits_low: v = ((i & 3u) << 30u) | (i >> 2u); break;
            case Distribution::mostly_equal:
                v = (i % 1000u) == 0u ? rng.next() : 0x0badf00du;
                break;
            default: break;
        }
        codes[i] = v;
    }
}

// ---------------------------------------------------------------------------
// One case: buffers of `base + count` elements plus the garbage around them
// ---------------------------------------------------------------------------

// Elements of garbage around the range of every case; the sort must not touch
// one byte of it (the buffers of the shared LBVH storage hold several trees).
constexpr uint kPad = 64u;

struct SortCase {
    Stream &stream;
    LbvhRadixSort &sort;
    Buffer<LbvhKey> a;
    Buffer<LbvhKey> b;
    luisa::vector<LbvhKey> input_a;
    luisa::vector<LbvhKey> input_b;
    luisa::vector<LbvhKey> out_a;
    luisa::vector<LbvhKey> out_b;
    size_t total{0u};
    uint base{0u};
    uint count{0u};

    SortCase(Device &device, Stream &s, LbvhRadixSort &radix_sort, uint offset, uint n) noexcept
        : stream{s}, sort{radix_sort},
          a{device.create_buffer<LbvhKey>(static_cast<size_t>(offset) + n + kPad)},
          b{device.create_buffer<LbvhKey>(static_cast<size_t>(offset) + n + kPad)},
          input_a{a.size()}, input_b{b.size()},
          out_a{a.size()}, out_b{b.size()},
          total{static_cast<size_t>(offset) + n + kPad}, base{offset}, count{n} {
        // garbage: a distinct pattern per buffer so a stray write between the two
        // ping-pong buffers is visible as well
        for (auto i = 0u; i < input_a.size(); i++) {
            input_a[i] = LbvhKey{0xa5a5a5a5u + static_cast<uint>(i), 0x5a5a5a5au + static_cast<uint>(i)};
            input_b[i] = LbvhKey{0xdead0000u + static_cast<uint>(i), 0xbeef0000u + static_cast<uint>(i)};
        }
    }

    // Uploads `codes` (slot = base + i) on top of the garbage.
    void upload(luisa::span<const uint> codes) noexcept {
        for (auto i = 0u; i < count; i++) {
            input_a[base + i] = LbvhKey{codes[i], base + i};
            input_b[base + i] = LbvhKey{0x11111111u + i, 0x22222222u + i};// overwritten by the sort
        }
        stream << a.view(0u, total).copy_from(luisa::span{input_a})
               << b.view(0u, total).copy_from(luisa::span{input_b})
               << synchronize();
    }

    // One sort, followed by the download of both buffers; returns the sorted
    // range of `keys_a`.
    [[nodiscard]] luisa::vector<LbvhKey> run(LbvhRadixSort::Method method, uint items,
                                             uint variant, bool batched) noexcept {
        sort.set_items(items);
        sort.set_variant(variant);
        sort.set_batched(batched);
        stream << a.view(0u, total).copy_from(luisa::span{input_a})
               << b.view(0u, total).copy_from(luisa::span{input_b})
               << synchronize();
        sort.sort(stream, a, b, base, count, method);
        stream << a.view(0u, total).copy_to(luisa::span{out_a})
               << b.view(0u, total).copy_to(luisa::span{out_b})
               << synchronize();
        luisa::vector<LbvhKey> result(count);
        for (auto i = 0u; i < count; i++) { result[i] = out_a[base + i]; }
        return result;
    }

    // Every property the sort must have: the two buffers outside the range are
    // untouched, the range is sorted by code, equal codes keep their input order,
    // the slots are a permutation of [base, base + count), and every element is
    // bit-identical to the host reference.
    [[nodiscard]] size_t check(const char *label, luisa::span<const LbvhKey> result,
                               luisa::span<const LbvhKey> reference,
                               bool verbose) noexcept {
        auto problems = size_t{0u};
        auto report = [&](const char *what, size_t index, LbvhKey got,
                          LbvhKey want) noexcept {
            if (problems < 3u) {
                std::printf("    [%s] %s at %llu: got (code=%08x slot=%u) want (code=%08x slot=%u)\n",
                            label, what, static_cast<unsigned long long>(index),
                            got.code, got.slot, want.code, want.slot);
            }
            problems++;
        };
        for (auto i = size_t{0u}; i < total; i++) {
            auto inside = i >= base && i < static_cast<size_t>(base) + count;
            if (inside) { continue; }
            if (out_a[i].code != input_a[i].code || out_a[i].slot != input_a[i].slot) {
                report("keys_a garbage", i, out_a[i], input_a[i]);
            }
            if (out_b[i].code != input_b[i].code || out_b[i].slot != input_b[i].slot) {
                report("keys_b garbage", i, out_b[i], input_b[i]);
            }
        }
        for (auto i = 0u; i < count; i++) {
            if (result[i].code != reference[i].code || result[i].slot != reference[i].slot) {
                report("reference", i, result[i], reference[i]);
            }
        }
        for (auto i = 1u; i < count; i++) {
            if (result[i].code < result[i - 1u].code) {
                report("code order", i, result[i], result[i - 1u]);
            }
            if (result[i].code == result[i - 1u].code && result[i].slot <= result[i - 1u].slot) {
                report("stability", i, result[i], result[i - 1u]);
            }
        }
        // the slots are a permutation of [base, base + count): no element is
        // dropped, duplicated or invented
        luisa::vector<uint8_t> seen(count, 0u);
        for (auto i = 0u; i < count; i++) {
            auto slot = result[i].slot;
            if (slot < base || slot - base >= count) {
                report("slot range", i, result[i], result[i]);
            } else if (seen[slot - base] != 0u) {
                report("slot duplicate", i, result[i], result[i]);
                seen[slot - base] = 2u;
            } else {
                seen[slot - base] = 1u;
            }
        }
        for (auto i = 0u; i < count; i++) {
            if (seen[i] != 1u) {
                report("slot missing", i, LbvhKey{}, LbvhKey{});
                break;
            }
        }
        if (verbose && problems == 0u) {
            std::printf("    [%s] ok (count=%u base=%u)\n", label, count, base);
        }
        return problems;
    }

    [[nodiscard]] static bool same(luisa::span<const LbvhKey> lhs,
                                   luisa::span<const LbvhKey> rhs) noexcept {
        if (lhs.size() != rhs.size()) { return false; }
        for (auto i = 0u; i < lhs.size(); i++) {
            if (lhs[i].code != rhs[i].code || lhs[i].slot != rhs[i].slot) { return false; }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Correctness catalogue
// ---------------------------------------------------------------------------

// Every count the sort has to survive: the boundaries of the work-group, the
// chunk and the scan group, the DirectX grid limit and the sizes of the real
// trees.
constexpr size_t kCheckCounts[] = {
    0u, 1u, 2u, 3u, 31u, 32u, 33u, 255u, 256u, 257u, 1023u, 1024u, 1025u,
    4095u, 4096u, 4097u, 65535u, 65536u, 65537u, 262144u, 1048575u,
    1u << 20u, 1u << 21u, (1u << 20u) + 1u};

[[nodiscard]] luisa::vector<Distribution> distributions_for(size_t count) noexcept {
    if (count <= 1025u) {
        return {Distribution::random, Distribution::all_equal, Distribution::two_values,
                Distribution::ascending, Distribution::descending, Distribution::top_bits,
                Distribution::top_bits_low, Distribution::mostly_equal};
    }
    if (count <= 65537u) {
        return {Distribution::random, Distribution::all_equal, Distribution::descending,
                Distribution::top_bits};
    }
    return {Distribution::random, Distribution::top_bits};
}

[[nodiscard]] luisa::vector<uint> bases_for(size_t count, uint offset) noexcept {
    if (offset != 0u) { return {offset}; }
    if (count <= 65537u) { return {0u, 1237u}; }
    return {0u};
}

[[nodiscard]] luisa::vector<uint> items_for(size_t count) noexcept {
    if (count >= 4096u) { return {4u, 8u, 16u}; }
    return {8u};
}

[[nodiscard]] size_t run_checks(Device &device, Stream &stream, LbvhRadixSort &sort,
                                const SortOptions &options) noexcept {
    auto failures = size_t{0u};
    auto cases = size_t{0u};
    luisa::vector<uint> codes;
    luisa::vector<LbvhKey> reference;
    luisa::vector<size_t> counts{kCheckCounts, kCheckCounts + sizeof(kCheckCounts) / sizeof(kCheckCounts[0])};
    if (!options.check_counts.empty()) { counts = options.check_counts; }
    for (auto count : counts) {
        for (auto base : bases_for(count, static_cast<uint>(options.base))) {
            for (auto items : items_for(count)) {
                auto failures_before = failures;
                SortCase test{device, stream, sort, base, static_cast<uint>(count)};
                for (auto d : distributions_for(count)) {
                    generate_codes(d, static_cast<uint>(count),
                                   options.harness.seed + 0x9e3779b97f4a7c15ull *
                                                              (static_cast<uint64_t>(count) + 1u),
                                   codes);
                    reference = reference_sort(codes, base);
                    test.upload(codes);
                    cases++;
                    auto label = [&](const char *method) noexcept {
                        return luisa::format("{} count={} base={} dist={} items={}",
                                             method, count, base, distribution_name(d), items);
                    };
                    // the reference implementation (the current library sort) and
                    // its own repeatability: two runs must be bit-identical
                    auto single = test.run(LbvhRadixSort::Method::single_block, items, 0u, true);
                    auto problems = test.check(label("single").c_str(), single, reference, false);
                    auto again = test.run(LbvhRadixSort::Method::single_block, items, 0u, true);
                    if (!SortCase::same(single, again)) {
                        problems++;
                        std::printf("    [single] a repeated run is not bit-identical\n");
                    }
                    // ... and once without the command list, which is the exact
                    // shape of the four recorded passes of `LbvhStorage`
                    if (count >= 4096u && count <= 262144u) {
                        auto legacy = test.run(LbvhRadixSort::Method::single_block, items, 0u, false);
                        if (!SortCase::same(single, legacy)) {
                            problems++;
                            std::printf("    [single] batching changed the result\n");
                        }
                    }
                    // the parallel implementation, every variant and chunk size
                    for (auto variant : {0u, 1u}) {
                        auto multi = test.run(LbvhRadixSort::Method::multi_block, items,
                                              variant, true);
                        auto multi_label = label(variant == 0u ? "multi0" : "multi1");
                        problems += test.check(multi_label.c_str(), multi, reference, false);
                        if (!SortCase::same(single, multi)) {
                            problems++;
                            std::printf("    [%s] differs from the single-block reference\n",
                                        multi_label.c_str());
                        }
                    }
                    // and the automatic choice
                    auto automatic = test.run(LbvhRadixSort::Method::automatic, items, 1u, true);
                    problems += test.check(label("auto").c_str(), automatic, reference, false);
                    if (!SortCase::same(single, automatic)) {
                        problems++;
                        std::printf("    [auto] differs from the single-block reference\n");
                    }
                    if (problems != 0u) {
                        failures += problems;
                        std::printf("  FAIL %s (count=%llu base=%u dist=%s items=%u): %llu problem(s)\n",
                                    options.harness.backend.c_str(),
                                    static_cast<unsigned long long>(count), base,
                                    distribution_name(d), items,
                                    static_cast<unsigned long long>(problems));
                    }
                }
                std::printf("  check count=%llu base=%u items=%u: %s\n",
                            static_cast<unsigned long long>(count), base, items,
                            failures == failures_before ? "ok" : "FAILED");
                std::fflush(stdout);
            }
        }
    }
    std::printf("bench_check backend=%s cases=%llu failures=%llu\n",
                options.harness.backend.c_str(),
                static_cast<unsigned long long>(cases),
                static_cast<unsigned long long>(failures));
    return failures;
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct SortMeasurement {
    luisa::string label;
    size_t count{0u};
    BenchTiming timing;
    double relative{1.0};// vs single_block at the same count
};

[[nodiscard]] double now_ms() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct MeasureEntry {
    luisa::string label;
    LbvhRadixSort::Method method{};
    uint items{8u};
    uint variant{0u};
    bool batched{true};
    uint pass_count{4u};
};

[[nodiscard]] size_t sort_bytes(size_t total, size_t scratch) noexcept {
    return 2u * total * sizeof(LbvhKey) + scratch;
}

// Times one configuration: `--warmup` untimed samples, then `--iters` samples of
// `sort + synchronize`, i.e. the host-observed wall time of the whole submission
// (the same convention as the LBVH benchmark, so only release builds are
// meaningful).
void measure(Stream &stream, LbvhRadixSort &sort, const Buffer<LbvhKey> &a,
             const Buffer<LbvhKey> &b, uint base, uint count, const MeasureEntry &entry,
             const BenchOptions &options, BenchTiming &timing) noexcept {
    sort.set_items(entry.items);
    sort.set_variant(entry.variant);
    sort.set_batched(entry.batched);
    for (auto i = 0u; i < options.warmup; i++) {
        sort.sort_passes(stream, a, b, base, count, entry.pass_count, entry.method);
        stream << synchronize();
    }
    for (auto i = 0u; i < options.iterations; i++) {
        auto begin = now_ms();
        sort.sort_passes(stream, a, b, base, count, entry.pass_count, entry.method);
        stream << synchronize();
        timing.add(now_ms() - begin);
    }
}

// The measurement sweep.  Everything is pre-flighted: the device-memory
// estimate of one measurement must fit `--budget-gib`, and a size whose
// submission time is *predicted* (from the previous, smaller size of the same
// configuration) to exceed `--dispatch-budget-ms` is refused rather than
// submitted - a submission of more than ~2 s removes the device on Windows and
// cannot be aborted from the host.
void run_measurements(Device &device, Stream &stream, LbvhRadixSort &sort,
                      const SortOptions &options) noexcept {
    auto base = static_cast<uint>(options.base);
    luisa::vector<size_t> counts = options.counts;
    if (counts.empty()) {
        for (auto n : {4096ull, 16384ull, 65536ull, 262144ull, 1048576ull,
                       2097152ull, 4194304ull, 8388608ull}) {
            counts.emplace_back(static_cast<size_t>(n));
        }
    }
    luisa::vector<MeasureEntry> entries;
    if (options.method == "all" || options.method == "single") {
        entries.push_back({"single", LbvhRadixSort::Method::single_block, 8u, 0u, true, 4u});
        entries.push_back({"single_unbatched", LbvhRadixSort::Method::single_block, 8u, 0u, false, 4u});
    }
    if (options.method == "all" || options.method == "multi") {
        entries.push_back({"multi0", LbvhRadixSort::Method::multi_block, 8u, 0u, true, 4u});
        entries.push_back({"multi1", LbvhRadixSort::Method::multi_block, 8u, 1u, true, 4u});
    }
    if (options.method == "auto") {
        entries.push_back({"auto", LbvhRadixSort::Method::automatic, 8u, 1u, true, 4u});
    }
    if (options.items_sweep) {
        for (auto items : {4u, 16u}) {
            entries.push_back({luisa::format("multi0_i{}", items),
                               LbvhRadixSort::Method::multi_block, items, 0u, true, 4u});
            entries.push_back({luisa::format("multi1_i{}", items),
                               LbvhRadixSort::Method::multi_block, items, 1u, true, 4u});
        }
    }
    // what the previous size of the same configuration measured, for the
    // submission-size pre-flight
    luisa::vector<double> last_ms(entries.size(), 0.0);
    luisa::vector<size_t> last_count(entries.size(), 0u);
    auto budget_bytes = options.harness.budget_gib * 1024.0 * 1024.0 * 1024.0;
    auto started = now_ms();
    luisa::vector<SortMeasurement> results;
    for (auto count : counts) {
        if (count == 0u) { continue; }
        auto total = static_cast<size_t>(base) + count + kPad;
        auto bytes = sort_bytes(total, sort.scratch_bytes());
        if (static_cast<double>(bytes) > budget_bytes) {
            std::printf("bench_sort_skip backend=%s count=%llu reason=memory bytes=%llu budget=%llu\n",
                        options.harness.backend.c_str(),
                        static_cast<unsigned long long>(count),
                        static_cast<unsigned long long>(bytes),
                        static_cast<unsigned long long>(static_cast<size_t>(budget_bytes)));
            continue;
        }
        Buffer<LbvhKey> a = device.create_buffer<LbvhKey>(total);
        Buffer<LbvhKey> b = device.create_buffer<LbvhKey>(total);
        // Input: random 32-bit codes with `slot = base + i`; the garbage around
        // the range (a non-zero `--base`) is the same pattern the check phase
        // uses, so a stray write outside the range shows up as a wrong timing too.
        {
            luisa::vector<LbvhKey> host(total);
            Rng rng{options.harness.seed | 1u};
            for (auto i = 0u; i < total; i++) {
                host[i] = LbvhKey{0xa5a5a5a5u + static_cast<uint>(i),
                                  0x5a5a5a5au + static_cast<uint>(i)};
            }
            for (auto i = 0u; i < count; i++) {
                host[base + i] = LbvhKey{rng.next(), base + i};
            }
            stream << a.view(0u, total).copy_from(luisa::span{host})
                   << b.view(0u, total).copy_from(luisa::span{host})
                   << synchronize();
        }
        std::printf("\n-- count %llu (%s for the two key buffers + %s scratch) --\n",
                    static_cast<unsigned long long>(count),
                    human_bytes(2u * total * sizeof(LbvhKey)).c_str(),
                    human_bytes(sort.scratch_bytes()).c_str());
        std::printf("  %-18s %11s %11s %11s %9s %10s\n", "method", "ms(min)", "ms(median)",
                    "ms(mean)", "Melem/s", "vs single");
        for (auto e = 0u; e < entries.size(); e++) {
            auto &entry = entries[e];
            if (last_count[e] != 0u) {
                auto predicted = last_ms[e] * static_cast<double>(count) /
                                     static_cast<double>(last_count[e]) * 1.25 +
                                 0.5;
                if (predicted > options.harness.dispatch_budget_ms &&
                    !options.harness.force_oversize) {
                    std::printf("  %-18s refused: predicted %.1f ms > --dispatch-budget-ms %.1f\n",
                                entry.label.c_str(), predicted,
                                options.harness.dispatch_budget_ms);
                    continue;
                }
            }
            BenchTiming timing;
            measure(stream, sort, a, b, base, static_cast<uint>(count), entry, options.harness,
                    timing);
            last_ms[e] = timing.min_ms();
            last_count[e] = count;
            results.push_back(SortMeasurement{entry.label, count, timing, 1.0});
            auto megapixels = static_cast<double>(count) * 1.0e-6 /
                              (timing.min_ms() * 1.0e-3);
            std::printf("  %-18s %11.4f %11.4f %11.4f %9.1f\n", entry.label.c_str(),
                        timing.min_ms(), timing.median_ms(), timing.mean_ms(), megapixels);
            std::printf("bench_sort backend=%s method=%s items=%u variant=%u batched=%d count=%llu base=%u iters=%llu ms_min=%.6f ms_median=%.6f ms_mean=%.6f melem_per_s=%.3f\n",
                        options.harness.backend.c_str(), entry.label.c_str(), entry.items,
                        entry.variant, entry.batched ? 1 : 0,
                        static_cast<unsigned long long>(count), base,
                        static_cast<unsigned long long>(timing.count()), timing.min_ms(),
                        timing.median_ms(), timing.mean_ms(), megapixels);
            std::fflush(stdout);
        }
        // the per-pass breakdown: the difference between pass p and p-1 is what
        // one pass costs (an odd pass count ends in keys_b)
        if (options.breakdown) {
            for (auto pass_count : {1u, 2u, 3u, 4u}) {
                for (auto entry : {MeasureEntry{"single", LbvhRadixSort::Method::single_block, 8u, 0u, true, pass_count},
                                   MeasureEntry{"multi0", LbvhRadixSort::Method::multi_block, 8u, 0u, true, pass_count},
                                   MeasureEntry{"multi1", LbvhRadixSort::Method::multi_block, 8u, 1u, true, pass_count}}) {
                    BenchTiming timing;
                    measure(stream, sort, a, b, base, static_cast<uint>(count), entry,
                            options.harness, timing);
                    std::printf("bench_passes backend=%s method=%s passes=%u count=%llu ms_min=%.6f\n",
                                options.harness.backend.c_str(), entry.label.c_str(), pass_count,
                                static_cast<unsigned long long>(count), timing.min_ms());
                }
            }
        }
        std::fflush(stdout);
        if ((now_ms() - started) * 1.0e-3 > options.harness.max_seconds) {
            std::printf("bench_sort_stop backend=%s reason=max-seconds elapsed_ms=%.1f\n",
                        options.harness.backend.c_str(), now_ms() - started);
            break;
        }
    }
    // the summary: every measurement against the batched single-block sort of the
    // same count (the only comparison that isolates the algorithm)
    std::printf("\n==== %s: speedup against the single-block sort ====\n",
                options.harness.backend.c_str());
    std::printf("  %-18s %11s %11s %9s %10s\n", "count", "method", "ms(min)", "Melem/s", "vs single");
    for (auto const &entry : results) {
        auto baseline = 0.0;
        for (auto const &other : results) {
            if (other.count == entry.count && other.label == "single") {
                baseline = other.timing.min_ms();
                break;
            }
        }
        auto relative = (baseline > 0.0 && entry.timing.min_ms() > 0.0) ? baseline / entry.timing.min_ms() : 0.0;
        auto megapixels = static_cast<double>(entry.count) * 1.0e-6 /
                          (entry.timing.min_ms() * 1.0e-3);
        std::printf("  %-11llu %-18s %11.4f %9.1f %9.2fx\n",
                    static_cast<unsigned long long>(entry.count), entry.label.c_str(),
                    entry.timing.min_ms(), megapixels, relative);
    }
}

// The capacity one sorter has to serve: the largest count of either phase.
[[nodiscard]] size_t bench_capacity(const SortOptions &options) noexcept {
    auto capacity = size_t{1u};
    for (auto count : kCheckCounts) {
        capacity = std::max(capacity, count + options.base + kPad);
    }
    for (auto count : options.counts) {
        capacity = std::max(capacity, count + options.base + kPad);
    }
    if (options.counts.empty()) {
        capacity = std::max(capacity, (1u << 23u) + options.base + kPad);
    }
    return capacity;
}

}// namespace

int main(int argc, char *argv[]) {
    // The records printed below are the result of the benchmark: they must not be
    // lost and they must not interleave with the console logger.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    auto options = parse_options(argc, argv);
    if (!options.error.empty()) {
        std::printf("error: %s\n\n", options.error.c_str());
        print_usage(executable);
        return 1;
    }
    if (options.harness.help) {
        print_usage(executable);
        return 0;
    }
    if (options.harness.backend.empty()) {
        print_usage(executable);
        return 1;
    }
    if (!options.check && !options.measure) {
        options.check = true;
        options.measure = true;
    }
    Context context{executable};
    Device device = context.create_device(options.harness.backend);
    Stream stream = device.create_stream();
    auto capacity = bench_capacity(options);
    LbvhRadixSort sort{device, capacity};
    std::printf("sort_config backend=%s capacity=%llu chunk=%u scratch_bytes=%llu warp_size=%u iters=%u warmup=%u seed=%llu budget_gib=%.2f dispatch_budget_ms=%.1f max_seconds=%.1f\n",
                options.harness.backend.c_str(), static_cast<unsigned long long>(capacity),
                sort.chunk_size(),
                static_cast<unsigned long long>(sort.scratch_bytes()),
                device.compute_warp_size(), options.harness.iterations, options.harness.warmup,
                static_cast<unsigned long long>(options.harness.seed), options.harness.budget_gib,
                options.harness.dispatch_budget_ms, options.harness.max_seconds);
    auto failures = size_t{0u};
    if (options.check) {
        std::printf("\n==== %s: correctness against the host reference ====\n",
                    options.harness.backend.c_str());
        failures = run_checks(device, stream, sort, options);
        if (failures != 0u) {
            std::printf("\nCHECK FAILED: %llu problem(s) on %s\n",
                        static_cast<unsigned long long>(failures),
                        options.harness.backend.c_str());
            return 1;
        }
        std::printf("CHECK PASSED on %s\n", options.harness.backend.c_str());
    }
    if (options.measure) {
        std::printf("\n==== %s: timing ====\n", options.harness.backend.c_str());
        run_measurements(device, stream, sort, options);
    }
    std::printf("\n%s: PASS\n", options.harness.backend.c_str());
    return 0;
}
