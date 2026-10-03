#include "ut/ut.hpp"
#include "cuda_tile_scan_cost.h"
#include <bit>
#include <limits>
#include <luisa/tile/algorithms.h>

namespace {
using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace boost::ut;
namespace cost = luisa::compute::cuda::native_tile;

[[nodiscard]] auto device_facts() noexcept {
    return cost::ScanCostDevice{true, 89u, 24u, 32u, 1536u, 13040u, 13040u, 130400u};
}

template<typename T>
[[nodiscard]] auto capture_prefix(int64_t rows, int64_t columns, int64_t block_rows = 1,
                                   ReductionPolicy policy = reduction::unordered_tree) noexcept {
    auto width = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    return tile_kernel("an_arbitrary_function_name", [=](TensorView<const T, 2> unused,
                                                          TensorView<const T, 2> input,
                                                          TensorView<T, 2> output) {
               auto r = axis("independent", block_rows);
               auto c = axis("contribution", width);
               for (auto &p : parallel(shape((rows + block_rows - 1) / block_rows))) {
                   auto row = p.index() * block_rows;
                   auto x = cast<float>(input.tile(coord(row, 0), shape(r, c)).load());
                   output(coord(row, 0), shape(r, c)).store(cast<T>(inclusive_sum(x, c, policy)));
               }
           })
        .capture(tensor_shape(rows, columns), tensor_shape(rows, columns), tensor_shape(rows, columns));
}

struct TestCandidates {
    std::array<cost::CubScanArtifact, 4u> artifacts;
    std::array<cost::CompiledScanCandidate, 4u> observations;
};

void make_candidates(const tile::Kernel &kernel, const cost::Artifact &original, TestCandidates &out) {
    constexpr std::array capacities{12, 6, 3, 1};
    for (auto i = size_t{0u}; i < out.artifacts.size(); i++) {
        auto threads = cost::kScanCostThreads[i];
        out.artifacts[i] = cost::generate_cub_scan(kernel.function(), original, threads);
        expect(out.artifacts[i].ok()) << out.artifacts[i].error;
        out.observations[i].artifact = &out.artifacts[i];
        out.observations[i].compile_key_known = true;
        out.observations[i].compile_key = i; // Zero can be a known 64-bit key.
        // Synthetic query facts, not an assertion about the real device.
        out.observations[i].resources = cost::CompiledScanResources{
            true, true, true, 32, 64, 0, 1024, capacities[i], threads, 0u};
    }
}

void score_arithmetic_and_frozen_policy() {
    auto device = device_facts();
    constexpr auto bytes = uint64_t{128u} * 8192u * 2u;
    std::array<double, 3u> original{};
    expect(cost::scan_cost_detail::tile_features(128u, 8192u, bytes, device, original));
    expect(original == std::array{1.0, 1.0 / 6.0, 49152.0});
    constexpr std::array capacities{12u, 6u, 3u, 1u};
    constexpr std::array local{64.0, 32.0, 32.0, 48.0};
    constexpr std::array group{32.0, 32.0, 64.0, 192.0};
    for (auto i = size_t{0u}; i < capacities.size(); i++) {
        std::array<double, 4u> features{};
        expect(cost::scan_cost_detail::cub_features(128u, 8192u, bytes, cost::kScanCostThreads[i], capacities[i], device, features));
        expect(features == std::array{1.0, 1.0 / 6.0, local[i], group[i]});
        double score{};
        expect(cost::scan_cost_detail::score(cost::kScanCostCubCoefficients, features, score));
        auto reference = 0.6027021529393383 + 12.227593513162688 * (1.0 / 6.0) +
                         0.027854484240835357 * local[i] + 0.022923736019806327 * group[i];
        expect(std::abs(score - reference) <= 8.0 * std::numeric_limits<double>::epsilon() * reference);
    }
    expect(!cost::scan_cost_detail::predicts_saving(0.95, 1.0));
    expect(cost::scan_cost_detail::predicts_saving(std::nextafter(0.95, 0.0), 1.0));
    expect(!cost::scan_cost_detail::predicts_saving(1.0, 1.0));
    expect(!cost::scan_cost_detail::predicts_saving(0.0, 1.0));
    expect(!cost::scan_cost_detail::predicts_saving(std::numeric_limits<double>::quiet_NaN(), 1.0));
    double result{};
    auto bad = cost::kScanCostCubCoefficients;
    bad[0u] = -1.0;
    expect(!cost::scan_cost_detail::score(bad, std::array{1.0, 1.0, 1.0, 1.0}, result));
    bad[0u] = std::numeric_limits<double>::infinity();
    expect(!cost::scan_cost_detail::score(bad, std::array{1.0, 1.0, 1.0, 1.0}, result));
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    uint64_t integer{};
    expect(!cost::scan_cost_detail::multiply(max, 2u, integer));
    expect(!cost::scan_cost_detail::multiply(1u, 0u, integer));
    expect(!cost::scan_cost_detail::add(max, 1u, integer));
    expect(cost::scan_cost_detail::ceiling(max, max, integer) && integer == 1u);
    expect(!cost::scan_cost_detail::ceiling(1u, 0u, integer));
    std::array<double, 4u> features{};
    expect(!cost::scan_cost_detail::cub_features(128u, 8192u, bytes, 256u, max, device, features));
    expect(!cost::scan_cost_detail::cub_features(max, max, bytes, 128u, 1u, device, features));
    expect(!cost::scan_cost_detail::cub_features(128u, 8192u, max, 256u, 6u, device, features));
    expect(!cost::scan_cost_detail::cub_features(128u, 8191u, bytes, 256u, 6u, device, features));
}

template<typename T>
void real_ir_and_resource_gates() {
    auto kernel = capture_prefix<T>(128, 8192);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto proof = analyze_closed_prefix(kernel.function());
    auto original = cost::generate(kernel.function());
    expect(proof.ok()) << proof.error;
    expect(original.ok()) << original.error;
    if (!proof.ok() || !original.ok()) { return; }
    // This is deliberately not benchmark slots 0/3.
    expect(proof.disjoint.input.argument_index == 1u && proof.disjoint.output.argument_index == 2u);
    auto untouched = original.source;
    TestCandidates candidates;
    make_candidates(kernel, original, candidates);
    auto choose = [&]() {
        return cost::choose_cub_scan_cost(proof, original, candidates.observations, device_facts(), false, false);
    };
    auto baseline = choose();
    expect(baseline.has_score && baseline.selected_threads == 256u);
    expect(baseline.status == "selected" && baseline.reason == "predicted-saving");
    expect(original.source == untouched);
    for (auto &record : baseline.candidates) { expect(record.has_score); }

    for (auto mutation = 0u; mutation < 11u; mutation++) {
        auto good = candidates.observations[1u];
        auto &bad = candidates.observations[1u];
        switch (mutation) {
            case 0u: bad.resources.loaded_entry_verified = false; break;
            case 1u: bad.resources.function_query_ok = false; break;
            case 2u: bad.resources.registers = -1; break;
            case 3u: bad.resources.static_shared_bytes = -1; break;
            case 4u: bad.resources.local_bytes = 4; break;
            case 5u: bad.resources.maximum_threads = 128; break;
            case 6u: bad.resources.capacity_query_ok = false; break;
            case 7u: bad.resources.resident_cta_capacity = 0; break;
            case 8u: bad.resources.capacity_threads = 128u; break;
            case 9u: bad.resources.capacity_dynamic_shared_bytes = 16u; break;
            case 10u: bad.compile_key_known = false; break;
        }
        auto rejected = choose();
        expect(!rejected.candidates[1u].has_score);
        expect(rejected.selected_threads != 256u);
        candidates.observations[1u] = good;
    }
    auto valid = candidates.artifacts[1u];
    candidates.artifacts[1u].guard.input_bytes -= 2u;
    expect(!choose().candidates[1u].has_score);
    candidates.artifacts[1u] = valid;
    candidates.artifacts[1u].grid[0u]--;
    expect(!choose().candidates[1u].has_score);
    candidates.artifacts[1u] = valid;
    for (auto &observation : candidates.observations) { observation.resources.function_query_ok = false; }
    auto retained = choose();
    expect(retained.has_score && retained.selected_threads == 0u);
    expect(retained.selected_score == retained.original_score && retained.status == "retained");
    expect(original.source == untouched);

    auto invalid_original = original;
    invalid_original.arguments[1u].minimum_size_bytes -= 2u;
    expect(!cost::choose_cub_scan_cost(proof, invalid_original, candidates.observations, device_facts(), false, false).has_score);
    expect(!cost::choose_cub_scan_cost(proof, original, candidates.observations, device_facts(), true, false).has_score);
    expect(!cost::choose_cub_scan_cost(proof, original, candidates.observations, device_facts(), false, true).has_score);
    auto device = device_facts();
    device.nvrtc_version = 13040u; // Wrong encoding, despite identical human major/minor.
    expect(!cost::choose_cub_scan_cost(proof, original, candidates.observations, device, false, false).has_score);
    device = device_facts();
    device.resident_threads = 2048u;
    expect(!cost::choose_cub_scan_cost(proof, original, candidates.observations, device, false, false).has_score);
}

void semantic_and_target_rejections() {
    auto reject = [](const tile::Kernel &kernel) {
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto proof = analyze_closed_prefix(kernel.function());
        auto original = cost::generate(kernel.function());
        std::array<cost::CompiledScanCandidate, 4u> candidates{};
        expect(!cost::choose_cub_scan_cost(proof, original, candidates, device_facts(), false, false).has_score);
    };
    reject(capture_prefix<half>(17, 2051)); // Padded contribution tail stays original.
    reject(capture_prefix<half>(17, 4096, 2)); // Multiple logical rows/program is outside this recipe family.
    reject(capture_prefix<float>(17, 4096));
    reject(capture_prefix<half>(17, 4096, 1, reduction::ordered_tree));
    for (auto field = 0u; field < 8u; field++) {
        auto device = device_facts();
        switch (field) {
            case 0u: device.query_ok = false; break;
            case 1u: device.compute_capability = 90u; break;
            case 2u: device.processors = 25u; break;
            case 3u: device.subgroup_width = 64u; break;
            case 4u: device.resident_threads = 2048u; break;
            case 5u: device.driver_api_version = 13030u; break;
            case 6u: device.toolkit_version = 13030u; break;
            case 7u: device.nvrtc_version = 130300u; break;
        }
        expect(!cost::scan_cost_detail::supports_device(device));
    }
}
}// namespace

static auto scan_cost_registration = [] {
    using namespace boost::ut;
    "tile_cuda_scan_cost_arithmetic"_test = [] { score_arithmetic_and_frozen_policy(); };
    "tile_cuda_scan_cost_real_ir"_test = [] {
        real_ir_and_resource_gates<luisa::half>();
        real_ir_and_resource_gates<luisa::compute::tile::bfloat16>();
    };
    "tile_cuda_scan_cost_rejections"_test = [] { semantic_and_target_rejections(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
