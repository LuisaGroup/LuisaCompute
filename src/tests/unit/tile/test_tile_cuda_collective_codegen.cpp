#include "ut/ut.hpp"
#include "cuda_tile_codegen.h"
#include "cuda_tile_streaming_scan.h"
#include "cuda_tile_collective_cost.h"
#include "cuda_tile_partition_codegen.h"
#include "cuda_tile_partition_cost.h"
#include <array>
#include <cmath>
#include <bit>
#include <limits>
#include <luisa/tile/algorithms.h>
#include <luisa/core/stl/format.h>

// Exercise the production emitter without invoking a CUDA compiler or device.
namespace {

size_t chunk_scan_occurrences(luisa::string_view text, luisa::string_view needle) {
    auto count = size_t{0u}, position = size_t{0u};
    while ((position = text.find(needle, position)) != luisa::string_view::npos) {
        count++;
        position += needle.size();
    }
    return count;
}

void test_native_pure_chunk_scan_sources() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    auto capture_scan = [](int64_t width, ReductionPolicy policy) {
        return tile_kernel("unrelated_middle_axis", [=](TensorView<const float, 3> input,
                                                        TensorView<float, 3> output) {
                   auto a = axis("a", 2), b = axis("b", width), c = axis("c", 4);
                   for (auto &p : parallel(shape(3))) {
                       auto x = input.tile(coord(p.index() * int64_t{2}, 0, 0), shape(a, b, c)).load();
                       auto y = inclusive_sum(x, b, policy);
                       output(coord(p.index() * int64_t{2}, 0, 0), shape(a, b, c)).store(y);
                   }
               })
            .capture(tensor_shape(6, width, 4), tensor_shape(6, width, 4));
    };
    auto kernel = capture_scan(8192, reduction::unordered_tree);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    for (auto fast : {false, true}) {
        auto baseline = cuda::native_tile::generate(kernel.function(), fast);
        auto explicit_zero = cuda::native_tile::generate(kernel.function(), fast, false, 0u, 0u, 0u);
        expect(baseline.ok()) << baseline.error;
        expect(explicit_zero.ok()) << explicit_zero.error;
        expect(baseline.source == explicit_zero.source);
        for (auto chunk : {1024u, 2048u}) {
            auto candidate = cuda::native_tile::generate(kernel.function(), fast, false, 0u, 0u, chunk);
            expect(candidate.ok()) << candidate.error;
            if (!candidate.ok()) { continue; }
            auto parts = 8192u / chunk;
            expect(candidate.grid == baseline.grid);
            expect(candidate.block == baseline.block);
            expect(candidate.arguments.size() == baseline.arguments.size());
            expect(candidate.scan_chunk_extent == chunk);
            expect(candidate.chunked_scan_operations == 1u);
            expect(chunk_scan_occurrences(candidate.source, "ct::partial_sum(") == parts);
            expect(chunk_scan_occurrences(candidate.source, "ct::cat(") == parts - 1u);
            expect(chunk_scan_occurrences(candidate.source, "ct::add(") ==
                   chunk_scan_occurrences(baseline.source, "ct::add(") + parts - 1u);
            for (auto memory : {"ct::load(", "ct::load_masked(", "ct::store(", "ct::store_masked(", "ct::assume_aligned<"}) {
                expect(chunk_scan_occurrences(candidate.source, memory) ==
                       chunk_scan_occurrences(baseline.source, memory));
            }
            expect(candidate.source.find("ct::shape<2, 1, 4>") != luisa::string::npos);
            expect(candidate.source.find("ct::integral_constant<1>") != luisa::string::npos);
        }
    }
    for (auto invalid_chunk : {1u, 512u, 4096u}) {
        auto rejected = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, invalid_chunk);
        expect(!rejected.ok());
        expect(rejected.source.empty());
    }
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto ordered = capture_scan(8192, policy);
        auto rejected = cuda::native_tile::generate(ordered.function(), false, false, 0u, 0u, 1024u);
        expect(!rejected.ok());
        expect(rejected.source.empty());
    }
    auto small = capture_scan(1024, reduction::unordered_tree);
    auto rejected = cuda::native_tile::generate(small.function(), false, false, 0u, 0u, 1024u);
    expect(!rejected.ok());
    expect(rejected.source.empty());
}
}// namespace

namespace {
size_t independent_occurrences(luisa::string_view text, luisa::string_view needle) {
    auto count = size_t{0u}, position = size_t{0u};
    while ((position = text.find(needle, position)) != luisa::string_view::npos) {
        count++;
        position += needle.size();
    }
    return count;
}

void test_native_independent_collective_sources() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    enum class Kind { SCAN,
                      SUM,
                      MINIMUM,
                      MAXIMUM };
    auto capture = [](Kind kind, int64_t rows, ReductionPolicy policy) {
        return tile_kernel("arbitrary_independent_values", [=](TensorView<const float, 2> input,
                                                               TensorView<float, 2> output) {
                   auto r = axis("renamed", rows), c = axis("contribution", 64);
                   for (auto &p : parallel(shape(3))) {
                       auto origin = p.index() * rows;
                       auto x = input.tile(coord(origin, 0), shape(r, c)).load();
                       if (kind == Kind::SCAN) {
                           auto y = inclusive_sum(x, c, policy);
                           output(coord(origin, 0), shape(r, c)).store(y);
                       } else {
                           auto y = kind == Kind::SUM     ? reduce(x, c, add) :
                                    kind == Kind::MINIMUM ? reduce(x, c, minimum) :
                                                            reduce(x, c, maximum);
                           output(coord(origin, 0), shape(r, axis("single", 1))).store(y);
                       }
                   }
               })
            .capture(tensor_shape(3 * rows, 64), tensor_shape(3 * rows, 64));
    };
    for (auto kind : {Kind::SCAN, Kind::SUM, Kind::MINIMUM, Kind::MAXIMUM}) {
        auto kernel = capture(kind, 8, reduction::unordered_tree);
        expect(kernel.valid());
        if (!kernel.valid()) { continue; }
        for (auto fast : {false, true}) {
            auto baseline = cuda::native_tile::generate(kernel.function(), fast);
            auto zero = cuda::native_tile::generate(kernel.function(), fast, false, 0u, 0u, 0u, 0u);
            expect(baseline.ok()) << baseline.error;
            expect(zero.ok()) << zero.error;
            expect(baseline.source == zero.source);
            for (auto extent : {1u, 2u, 4u}) {
                auto candidate = cuda::native_tile::generate(kernel.function(), fast, false, 0u, 0u, 0u, extent);
                expect(candidate.ok()) << candidate.error;
                if (!candidate.ok()) { continue; }
                auto parts = 8u / extent;
                expect(candidate.grid == baseline.grid);
                expect(candidate.block == baseline.block);
                expect(candidate.arguments.size() == baseline.arguments.size());
                for (auto i = size_t{0u}; i < candidate.arguments.size(); i++) {
                    auto a = candidate.arguments[i], b = baseline.arguments[i];
                    expect(a.element == b.element && a.minimum_size_bytes == b.minimum_size_bytes && a.read == b.read && a.written == b.written);
                }
                expect(candidate.independent_axis_extent == extent);
                expect(candidate.partitioned_collective_operations == 1u);
                auto name = kind == Kind::SCAN ? "ct::partial_sum(" : kind == Kind::SUM ? "ct::sum(" :
                                                                  kind == Kind::MINIMUM ? "ct::reduce_min(" :
                                                                                          "ct::reduce_max(";
                expect(independent_occurrences(candidate.source, name) == parts);
                expect(independent_occurrences(candidate.source, "ct::cat(") == parts - 1u);
                expect(independent_occurrences(candidate.source, "ct::extract(") == independent_occurrences(baseline.source, "ct::extract(") + parts);
                for (auto memory : {"ct::load(", "ct::load_masked(", "ct::store(", "ct::store_masked(", "ct::assume_aligned<", "ct::add("}) {
                    expect(independent_occurrences(candidate.source, memory) == independent_occurrences(baseline.source, memory));
                }
                expect(candidate.source.find(luisa::format("ct::shape<{}, 64>", extent)) != luisa::string::npos);
            }
        }
        for (auto extent : {3u, 8u, 16u}) {
            auto rejected = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, 0u, extent);
            expect(!rejected.ok());
            expect(rejected.source.empty());
        }
    }
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto kernel = capture(Kind::SCAN, 8, policy);
        auto rejected = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, 0u, 1u);
        expect(!rejected.ok());
        expect(rejected.source.empty());
    }
    for (auto rows : {int64_t{1}, int64_t{32}}) {
        auto kernel = capture(Kind::SCAN, rows, reduction::unordered_tree);
        auto rejected = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, 0u, 1u);
        expect(!rejected.ok());
        expect(rejected.source.empty());
    }
    auto middle = tile_kernel("independent_last_axis", [](TensorView<const float, 3> input,
                                                          TensorView<float, 3> output) {
                      auto a = axis("a", 2), b = axis("b", 64), c = axis("c", 4);
                      for (auto &p : parallel(shape(3))) {
                          auto x = input.tile(coord(p.index() * int64_t{2}, 0, 0), shape(a, b, c)).load();
                          output(coord(p.index() * int64_t{2}, 0, 0), shape(a, b, c)).store(inclusive_sum(x, b, reduction::unordered_tree));
                      }
                  }).capture(tensor_shape(6, 64, 4), tensor_shape(6, 64, 4));
    auto candidate = cuda::native_tile::generate(middle.function(), false, false, 0u, 0u, 0u, 1u);
    expect(candidate.ok()) << candidate.error;
    expect(candidate.source.find("ct::shape<2, 64, 1>") != luisa::string::npos);
    expect(independent_occurrences(candidate.source, "ct::partial_sum(") == 4u);
    expect(independent_occurrences(candidate.source, "ct::cat(") == 3u);
}
}// namespace

namespace {
void test_native_collective_configuration() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    auto capture = [](int64_t width) {
        return tile_kernel("configuration_not_workload_name", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                   auto r = axis("r", 8), c = axis("c", width);
                   for (auto &p : parallel(shape(1))) {
                       auto x = input.tile(coord(0, 0), shape(r, c)).load();
                       output(coord(0, 0), shape(r, c)).store(inclusive_sum(x, c, reduction::unordered_tree));
                   }
               })
            .capture(tensor_shape(8, width), tensor_shape(8, width));
    };
    auto kernel = capture(8192);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto conflict = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, 1024u, 1u);
    expect(!conflict.ok());
    expect(conflict.source.empty());
    auto baseline = cuda::native_tile::generate(kernel.function());
    auto zeros = cuda::native_tile::generate(kernel.function(), false, false, 0u, 0u, 0u, 0u);
    expect(baseline.ok()) << baseline.error;
    expect(baseline.source == zeros.source);
    expect(zeros.chunked_scan_operations == 0u);
    expect(zeros.partitioned_collective_operations == 0u);
    for (auto worker : {4u, 8u}) {
        auto hinted = cuda::native_tile::generate(kernel.function(), false, false, worker, 89u, 1024u, 0u);
        expect(hinted.ok()) << hinted.error;
        expect(hinted.chunked_scan_operations == 1u);
        expect(hinted.source.find(luisa::format("num_worker_warps_per_cta = {}", worker)) != luisa::string::npos);
    }
    auto large = capture(32768);
    auto over_budget = cuda::native_tile::generate(large.function(), false, false, 0u, 0u, 1024u, 0u);
    expect(!over_budget.ok());
    expect(over_budget.source.empty());
    auto at_budget = cuda::native_tile::generate(large.function(), false, false, 0u, 0u, 2048u, 0u);
    expect(at_budget.ok()) << at_budget.error;
    expect(chunk_scan_occurrences(at_budget.source, "ct::partial_sum(") == 16u);
}
}// namespace

namespace {
template<typename T>
[[nodiscard]] luisa::compute::tile::Kernel capture_streaming_source(int64_t rows, int64_t columns, int64_t block_rows,
                                                                    int mode = 0) {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    auto width = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    return tile_kernel("unrelated_prefix_identity", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto r = axis("independent", block_rows), c = axis("contribution", width);
               for (auto &p : parallel(shape((rows + block_rows - 1) / block_rows))) {
                   auto row = mode == 1 ? Scalar<int64_t>{0} : p.index() * block_rows;
                   auto loaded = cast<float>(input.tile(coord(row, 0), shape(r, c)).load());
                   // Normal front ends can leave an unused pure coordinate map.
                   auto unused_mask = iota(c) < columns;
                   static_cast<void>(unused_mask);
                   auto policy = mode == 2 ? reduction::fold_left : reduction::unordered_tree;
                   auto result = inclusive_sum(loaded, c, policy);
                   if (mode == 3) { result = result + loaded; }
                   output(coord(row, 0), shape(r, c)).store(cast<T>(result));
                   if (mode == 4) { output(coord(row, 0), shape(r, c)).store(cast<T>(loaded)); }
               }
           })
        .capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
}

void test_native_streaming_scan_plan() {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::native_tile;
    using namespace boost::ut;
    auto check = []<typename T>() {
        for (auto rows : {int64_t{3}, int64_t{8}}) {
            for (auto columns : {int64_t{2049}, int64_t{8192}}) {
                auto kernel = capture_streaming_source<T>(rows, columns, 4);
                expect(kernel.valid());
                if (!kernel.valid()) { continue; }
                for (auto fast : {false, true}) {
                    auto original = generate(kernel.function(), fast);
                    expect(original.ok()) << original.error;
                    if (!original.ok()) { continue; }
                    auto zero = original;
                    append_streaming_scan(zero, kernel.function(), 0u);
                    expect(zero.source == original.source);
                    expect(zero.streaming_scan_entry.empty());
                    for (auto chunk : {1024u, 2048u}) {
                        auto plan = match_streaming_scan(kernel.function(), original, chunk);
                        expect(plan.ok()) << plan.error;
                        if (!plan.ok()) { continue; }
                        expect(plan.rows == static_cast<uint64_t>(rows));
                        expect(plan.columns == static_cast<uint64_t>(columns));
                        expect(plan.rows_per_program == 4u);
                        expect(plan.minimum_bytes == static_cast<uint64_t>(rows * columns * sizeof(T)));
                        auto candidate = original;
                        append_streaming_scan(candidate, kernel.function(), chunk);
                        expect(candidate.ok()) << candidate.error;
                        expect(candidate.source.starts_with(original.source + '\n'));
                        expect(candidate.grid == original.grid);
                        expect(candidate.block == original.block);
                        expect(candidate.streaming_scan_entry == "luisa_tile_stream_scan");
                        expect(candidate.streaming_scan_source_offset == original.source.size());
                        expect(candidate.streaming_scan_guard.input_slot == 0u);
                        expect(candidate.streaming_scan_guard.output_slot == 1u);
                        auto extra = candidate.source.substr(original.source.size());
                        expect(chunk_scan_occurrences(extra, "ct::partial_sum(") == 1u);
                        expect(chunk_scan_occurrences(extra, "ct::extract(") == 1u);
                        expect(extra.find("ct::shape<4, 1>") != string::npos);
                        expect(extra.find("ct::round_subnormals_to_zero") == string::npos);
                        expect((extra.find("ct::store(") != string::npos) == (rows == 8 && columns == 8192));
                        expect((extra.find("ct::store_masked(") != string::npos) == (rows != 8 || columns != 8192));
                    }
                }
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<half>();
    check.template operator()<tile::bfloat16>();
    for (auto mode : {1, 2, 3, 4}) {
        auto kernel = capture_streaming_source<float>(8, 4096, 4, mode);
        auto original = generate(kernel.function());
        expect(original.ok()) << original.error;
        if (!original.ok()) { continue; }
        auto candidate = original;
        append_streaming_scan(candidate, kernel.function(), 1024u);
        expect(candidate.ok());
        expect(candidate.streaming_scan_entry.empty());
        expect(!candidate.streaming_scan_diagnostic.empty());
        expect(candidate.source == original.source);
    }
    auto kernel = capture_streaming_source<float>(8, 8192, 4);
    auto original = generate(kernel.function());
    auto bad = match_streaming_scan(kernel.function(), original, 512u);
    expect(!bad.ok());
    auto transformed = generate(kernel.function(), false, false, 0u, 0u, 1024u, 0u);
    expect(transformed.ok()) << transformed.error;
    expect(!match_streaming_scan(kernel.function(), transformed, 1024u).ok());

    std::array<uint64_t, 2u> pointers{0x1000u, 0x2000u};
    StreamingScanGuard guard{0u, 1u, 0x1000u, 0x1000u};
    auto admitted = [&] { return streaming_scan_disjoint(guard, span<const uint64_t>{pointers.data(), pointers.size()}); };
    expect(admitted());
    pointers[1] = 0x1fffu;
    expect(!admitted());
    pointers[1] = pointers[0];
    expect(!admitted());
    pointers[0] = std::numeric_limits<uint64_t>::max() - 0xfffu;
    pointers[1] = 0x1000u;
    expect(!admitted());
}
}// namespace

namespace {
struct ScheduleParity {
    uint64_t programs;
    uint64_t elementwork;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t live_bytes;
    uint64_t largest_tile;
    std::array<luisa::compute::tile::CollectiveWork, 2u> collectives;
    size_t collective_count;
    std::array<double, 9u> features;
    uint32_t workers;
    bool has_score;
    double score;
};

[[nodiscard]] auto schedule_parity_cases() {
    namespace tile = luisa::compute::tile;
    // Frozen training-only logical facts. No fixture names or performance
    // results enter the evaluator; the expected values are Python parity.
    // Profile: 357c15e0c745797a75e8d11c9de9fafc0ab204f71d1619fedd9251323e7abb35.
    return std::array<ScheduleParity, 32u>{{{128u, 32768u, 16384u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {}}}, 1u, {2.662965012722429, 13.000176099486442, 9.702172685365548, 2.321928094887362, 8.005624549193879, 1.0, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {128u, 32768u, 16384u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {}}}, 1u, {2.662965012722429, 13.000176099486442, 9.702172685365548, 2.321928094887362, 8.005624549193879, 1.0, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {128u, 32768u, 32768u, 32768u, 106496u, 8192u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {}}}, 1u, {2.662965012722429, 13.000176099486442, 9.702172685365548, 2.321928094887362, 8.005624549193879, 1.0, 0.0, 0.0, 1.584962500721156}, 0u, false, 0.0},
                                            {128u, 385u, 512u, 4u, 1664u, 128u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 128u, 1u, 128u}, {}}}, 1u, {2.662965012722429, 7.011227255423254, 3.807354922057604, 2.002815015607054, 2.321928094887362, 1.0, 1.0, 0.0, 1.005624549193878}, 0u, true, 0.09904688984331009},
                                            {32u, 773u, 2048u, 16u, 4096u, 512u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 128u, 4u, 512u}, {}}}, 1u, {1.2223924213364477, 9.002815015607053, 5.044394119358453, 1.3275526440812404, 2.321928094887362, 2.321928094887362, 1.0, 0.0, 1.005624549193878}, 0u, true, 0.09904688984331009},
                                            {17u, 3073u, 2048u, 2u, 13312u, 1024u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 1024u, 1u, 1024u}, {}}}, 1u, {0.7725895038969276, 10.001408194392809, 6.714245517666122, 2.000352177480301, 5.044394119358453, 1.0, 1.0, 0.0, 0.5854320515929623}, 0u, true, 0.014389211452413706},
                                            {5u, 6149u, 8192u, 8u, 25600u, 4096u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 7.651051691178929, 1.322632363898609, 5.044394119358453, 2.321928094887362, 1.0, 0.0, 0.5854320515929623}, 8u, true, -0.2564738724463534},
                                            {1u, 73731u, 32768u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {}}}, 1u, {0.058893689053568614, 13.000176099486442, 9.702172685365548, 3.3219809269903284, 8.005624549193879, 1.0, 1.0, 0.0, 1.3219280948873624}, 8u, true, -0.2564738724463534},
                                            {1u, 98308u, 49152u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {1u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}}}, 2u, {0.058893689053568614, 13.000176099486442, 9.702172685365548, 2.8074052383900145, 8.005624549193879, 1.584962500721156, 1.0, 0.0, 1.0}, 8u, true, -0.2564738724463534},
                                            {128u, 73731u, 32768u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {}}}, 1u, {2.662965012722429, 13.000176099486442, 9.702172685365548, 3.3219809269903284, 8.005624549193879, 1.0, 1.0, 0.0, 1.3219280948873624}, 0u, true, 0.014389211452413706},
                                            {128u, 98308u, 49152u, 16384u, 106496u, 8192u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}, {1u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 8192u, 1u, 8192u}}}, 2u, {2.662965012722429, 13.000176099486442, 9.702172685365548, 2.8074052383900145, 8.005624549193879, 1.584962500721156, 1.0, 0.0, 1.0}, 0u, true, 0.014389211452413706},
                                            {1024u, 4608u, 1024u, 1024u, 6656u, 512u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 512u, 1u, 512u}, {1u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 512u, 1u, 512u}}}, 2u, {5.448460500816294, 9.002815015607053, 5.727920454563199, 2.4594316186372973, 4.087462841250339, 1.584962500721156, 0.5, 0.5, 0.5849625007211562}, 0u, true, 0.3131874161611729},
                                            {17u, 4096u, 2048u, 2048u, 13312u, 1024u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 1024u, 1u, 1024u}, {}}}, 1u, {0.7725895038969276, 10.001408194392809, 6.714245517666122, 2.321928094887362, 5.044394119358453, 1.0, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {5u, 10241u, 8192u, 8192u, 32768u, 4096u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 8.005624549193879, 1.8074555529676222, 5.044394119358453, 2.321928094887362, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {17u, 4097u, 2048u, 2u, 13312u, 1024u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 1024u, 1u, 1024u}, {}}}, 1u, {0.7725895038969276, 10.001408194392809, 6.714245517666122, 2.3222098437488943, 5.044394119358453, 1.0, 0.0, 1.0, 0.5854320515929623}, 0u, true, 0.014389211452413706},
                                            {5u, 10245u, 8192u, 8u, 33792u, 4096u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 8.049848549450562, 1.807858006430275, 5.044394119358453, 2.321928094887362, 0.0, 1.0, 0.5854320515929623}, 8u, true, -0.2564738724463534},
                                            {32u, 773u, 2048u, 16u, 4096u, 512u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 128u, 4u, 512u}, {}}}, 1u, {1.2223924213364477, 9.002815015607053, 5.044394119358453, 1.3275526440812404, 2.321928094887362, 2.321928094887362, 1.0, 0.0, 1.005624549193878}, 0u, true, 0.09904688984331009},
                                            {9u, 2569u, 8192u, 32u, 16384u, 2048u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 256u, 8u, 2048u}, {}}}, 1u, {0.45943161863729726, 11.000704269011246, 7.011227255423254, 1.1727400170493665, 3.169925001442312, 3.169925001442312, 1.0, 0.0, 1.002815015607054}, 0u, true, 0.09904688984331009},
                                            {32u, 3077u, 4096u, 8u, 12800u, 2048u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 512u, 4u, 2048u}, {}}}, 1u, {1.2223924213364477, 11.000704269011246, 6.658211482751795, 1.3233362892801708, 4.087462841250339, 2.321928094887362, 1.0, 0.0, 0.5859014496907713}, 0u, true, 0.3131874161611729},
                                            {5u, 6149u, 8192u, 8u, 25600u, 4096u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 7.651051691178929, 1.322632363898609, 5.044394119358453, 2.321928094887362, 1.0, 0.0, 0.5854320515929623}, 8u, true, -0.2564738724463534},
                                            {9u, 20489u, 32768u, 16u, 98304u, 16384u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 2048u, 8u, 16384u}, {}}}, 1u, {0.45943161863729726, 14.000088052430122, 9.586839787961827, 1.1702771789226134, 6.022367813028454, 3.169925001442312, 1.0, 0.0, 0.585197295260024}, 8u, true, -0.2564738724463534},
                                            {32u, 24581u, 65536u, 16u, 131072u, 16384u, {{{0u, tile::CollectiveKind::SUM, tile::ScalarType::FLOAT32, 4096u, 4u, 16384u}, {}}}, 1u, {1.2223924213364477, 14.000088052430122, 10.001408194392809, 1.3221041943738048, 7.011227255423254, 2.321928094887362, 1.0, 0.0, 1.0001760994864426}, 0u, true, 0.014389211452413706},
                                            {32u, 2565u, 2048u, 8u, 8448u, 1024u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 256u, 4u, 1024u}, {}}}, 1u, {1.2223924213364477, 10.001408194392809, 6.066089190457772, 1.8093662078160775, 3.169925001442312, 2.321928094887362, 0.0, 1.0, 0.5868397879618266}, 0u, true, 0.09904688984331009},
                                            {9u, 9225u, 16384u, 32u, 33280u, 4096u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 512u, 8u, 4096u}, {}}}, 1u, {0.45943161863729726, 12.0003521774803, 8.027905996569885, 1.701414768331626, 4.087462841250339, 3.169925001442312, 0.0, 1.0, 1.0014081943928084}, 0u, true, 0.09904688984331009},
                                            {5u, 10245u, 8192u, 8u, 33792u, 4096u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 8.049848549450562, 1.807858006430275, 5.044394119358453, 2.321928094887362, 0.0, 1.0, 0.5854320515929623}, 8u, true, -0.2564738724463534},
                                            {16u, 36873u, 32768u, 16u, 133120u, 16384u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 2048u, 8u, 16384u}, {}}}, 1u, {0.736965594166206, 14.000088052430122, 10.023754353299417, 1.7006835424760793, 6.022367813028454, 3.169925001442312, 0.0, 1.0, 0.585197295260024}, 0u, true, 0.014389211452413706},
                                            {5u, 40965u, 65536u, 16u, 135168u, 16384u, {{{0u, tile::CollectiveKind::MAXIMUM, tile::ScalarType::FLOAT32, 4096u, 4u, 16384u}, {}}}, 1u, {0.27301849440641585, 14.000088052430122, 10.045759661382682, 1.8074807095984133, 7.011227255423254, 2.321928094887362, 0.0, 1.0, 1.0001760994864426}, 8u, true, -0.2564738724463534},
                                            {3u, 4609u, 4096u, 4096u, 16384u, 2048u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 256u, 8u, 2048u}, {}}}, 1u, {0.16992500144231237, 11.000704269011246, 7.011227255423254, 1.7006564529181676, 3.169925001442312, 3.169925001442312, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {17u, 5121u, 4096u, 4096u, 16384u, 2048u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 512u, 4u, 2048u}, {}}}, 1u, {0.7725895038969276, 11.000704269011246, 7.011227255423254, 1.8075561768589195, 4.087462841250339, 2.321928094887362, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {5u, 10241u, 8192u, 8192u, 32768u, 4096u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 1024u, 4u, 4096u}, {}}}, 1u, {0.27301849440641585, 12.0003521774803, 8.005624549193879, 1.8074555529676222, 5.044394119358453, 2.321928094887362, 0.0, 0.0, 1.0}, 0u, false, 0.0},
                                            {16u, 36865u, 65536u, 65536u, 131072u, 16384u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 2048u, 8u, 16384u}, {}}}, 1u, {0.736965594166206, 14.000088052430122, 10.001408194392809, 1.7004668117689115, 6.022367813028454, 3.169925001442312, 0.0, 0.0, 1.584962500721156}, 0u, false, 0.0},
                                            {17u, 40961u, 32768u, 32768u, 131072u, 16384u, {{{0u, tile::CollectiveKind::INCLUSIVE_SUM, tile::ScalarType::FLOAT32, 4096u, 4u, 16384u}, {}}}, 1u, {0.7725895038969276, 14.000088052430122, 10.001408194392809, 1.8073800804431672, 7.011227255423254, 2.321928094887362, 0.0, 0.0, 1.0}, 0u, false, 0.0}}};
}

[[nodiscard]] luisa::compute::tile::CollectiveWorkAnalysis schedule_parity_work(const ScheduleParity &row) {
    luisa::compute::tile::CollectiveWorkAnalysis work;
    work.programs = row.programs;
    work.elementwise_elements_per_program = row.elementwork;
    work.global_read_bytes_per_program = row.read_bytes;
    work.global_write_bytes_per_program = row.write_bytes;
    work.materialized_tile_peak_bytes = row.live_bytes;
    work.largest_materialized_tile_elements = row.largest_tile;
    for (auto i = size_t{0u}; i < row.collective_count; i++) { work.collectives.emplace_back(row.collectives[i]); }
    return work;
}

void test_native_collective_schedule_parity() {
    using namespace luisa::compute;
    using namespace boost::ut;
    auto cases = schedule_parity_cases();
    auto selected = size_t{0u};
    for (auto i = size_t{0u}; i < cases.size(); i++) {
        auto &&row = cases[i];
        auto work = schedule_parity_work(row);
        auto features = tile::collective_cost_features(work, {24u, 32u});
        expect(features.ok()) << features.error;
        if (!features.ok()) { continue; }
        for (auto j = size_t{0u}; j < row.features.size(); j++) {
            expect(std::abs(features.values[j] - row.features[j]) < 2e-14) << i << j;
        }
        auto choice = cuda::native_tile::choose_collective_schedule(work, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
        expect(choice.worker_warps == row.workers) << i;
        expect(choice.has_score == row.has_score) << i;
        if (row.has_score) {
            expect(std::abs(choice.log_score - row.score) < 2e-14) << i;
            expect(choice.status == (row.workers == 8u ? "selected" : "default"));
            expect(choice.reason == (row.workers == 8u ? "predicted-saving" : "predicted-default"));
        } else {
            expect(choice.status == "default");
            expect(choice.reason == "prefix");
        }
        selected += choice.worker_warps != 0u;
    }
    expect(selected == 8u);
}

void test_native_collective_schedule_gates() {
    using namespace luisa::compute;
    using namespace boost::ut;
    auto cases = schedule_parity_cases();
    auto positive = size_t{0u};
    while (positive < cases.size() && cases[positive].workers == 0u) { positive++; }
    expect(positive < cases.size());
    if (positive == cases.size()) { return; }
    auto work = schedule_parity_work(cases[positive]);
    auto choose = [](const tile::CollectiveWorkAnalysis &facts) {
        return cuda::native_tile::choose_collective_schedule(facts, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
    };
    expect(choose(work).worker_warps == 8u);
    // Version-family boundary only: API/header versions are not cryptographic
    // proof that runtime compiler files match the recorded tool receipts.
    constexpr std::array<uint32_t, 6u> target{89u, 24u, 32u, 1536u, 13040u, 13040u};
    for (auto i = size_t{0u}; i < target.size(); i++) {
        auto incompatible = target;
        incompatible[i]++;
        auto choice = cuda::native_tile::choose_collective_schedule(work, incompatible[0u], incompatible[1u],
                                                                    incompatible[2u], incompatible[3u], incompatible[4u], incompatible[5u], false);
        expect(choice.worker_warps == 0u && !choice.has_score);
        expect(choice.status == "ineligible" && choice.reason == "target-profile");
    }
    auto fast = cuda::native_tile::choose_collective_schedule(work, 89u, 24u, 32u, 1536u, 13040u, 13040u, true);
    expect(fast.worker_warps == 0u && !fast.has_score && fast.reason == "fast-math");
    auto bad = work;
    bad.error = "analysis rejected";
    expect(choose(bad).worker_warps == 0u && !choose(bad).has_score && choose(bad).reason == "analysis");
    bad = work;
    bad.collectives.clear();
    expect(choose(bad).reason == "analysis");
    bad = work;
    bad.collectives[0u].kind = tile::CollectiveKind::MINIMUM;
    expect(choose(bad).worker_warps == 0u && !choose(bad).has_score && choose(bad).reason == "unsupported-algebra");
    bad = work;
    bad.collectives[0u].kind = tile::CollectiveKind::INCLUSIVE_SUM;
    expect(choose(bad).worker_warps == 0u && !choose(bad).has_score && choose(bad).reason == "prefix");
    expect(choose(bad).status == "default");
    bad = work;
    bad.collectives[0u].kind = static_cast<tile::CollectiveKind>(255u);
    expect(choose(bad).reason == "unsupported-algebra");
    bad = work;
    bad.collectives[0u].input_elements++;
    expect(choose(bad).worker_warps == 0u && !choose(bad).has_score && choose(bad).reason == "features");
    bad = work;
    bad.collectives[0u].element = tile::ScalarType::FLOAT16;
    expect(choose(bad).reason == "features");
    bad = work;
    bad.programs = 0u;
    expect(choose(bad).reason == "features");
    bad = work;
    bad.global_read_bytes_per_program = std::numeric_limits<uint64_t>::max();
    expect(choose(bad).reason == "features");
}
}// namespace

namespace {
template<typename T>
[[nodiscard]] luisa::compute::tile::Kernel capture_partition_source(int64_t rows, int64_t columns,
                                                                    int64_t block_rows, bool maximum,
                                                                    bool masked = false, bool transposed = false,
                                                                    bool epilogue = false) {
    using namespace luisa::compute::tile;
    auto width = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    return tile_kernel("independent_program_geometry", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto r = axis("unrelated_rows", block_rows), c = axis("unrelated_contributions", width);
               for (auto &p : parallel(shape((rows + block_rows - 1) / block_rows))) {
                   auto row = p.index() * block_rows;
                   auto x = cast<float>(input.tile(transposed ? coord(0, row) : coord(row, 0),
                                                   transposed ? shape(c, r) : shape(r, c))
                                            .load());
                   if (masked) { x = ite(iota(c) < columns, x, maximum ? -std::numeric_limits<float>::infinity() : 0.0f); }
                   auto y = maximum ? reduce(x, c, luisa::compute::tile::maximum) : reduce(x, c, add);
                   if (epilogue) { y = y + 1.0f; }
                   auto single = axis("output_singleton", 1);
                   output(transposed ? coord(0, row) : coord(row, 0),
                          transposed ? shape(single, r) : shape(r, single))
                       .store(cast<T>(y));
               }
           })
        .capture(transposed ? tensor_shape(columns, rows) : tensor_shape(rows, columns), transposed ? tensor_shape(1, rows) : tensor_shape(rows, 1));
}

void test_native_program_partition_sources() {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::native_tile;
    using namespace boost::ut;
    auto check = []<typename T>() {
        for (auto old_rows : {int64_t{4}, int64_t{8}}) {
            for (auto geometry : {std::array<int64_t, 2>{16, 64}, std::array<int64_t, 2>{17, 33}}) {
                auto rows = geometry[0u], columns = geometry[1u];
                for (auto maximum : {false, true}) {
                    for (auto masked : {false, true}) {
                        auto kernel = capture_partition_source<T>(rows, columns, old_rows, maximum, masked);
                        expect(kernel.valid());
                        if (!kernel.valid()) { continue; }
                        for (auto fast : {false, true}) {
                            auto original = generate(kernel.function(), fast);
                            expect(original.ok()) << original.error;
                            if (!original.ok()) { continue; }
                            auto disabled = original;
                            append_program_partition(disabled, kernel.function(), 0u);
                            expect(disabled.source == original.source);
                            expect(disabled.partition_entry.empty() && disabled.partition_diagnostic.empty());
                            expect(disabled.grid == original.grid && disabled.block == original.block);
                            for (auto target : {1u, 2u, 4u}) {
                                if (target >= static_cast<uint32_t>(old_rows)) { continue; }
                                auto candidate = original;
                                append_program_partition(candidate, kernel.function(), target);
                                expect(candidate.ok()) << candidate.error;
                                expect(candidate.partition_diagnostic.empty()) << candidate.partition_diagnostic;
                                expect(candidate.partition_entry == "luisa_tile_partition");
                                if (candidate.partition_entry.empty()) { continue; }
                                auto plan = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = target});
                                expect(plan.ok()) << plan.error;
                                expect(candidate.source.starts_with(original.source + '\n'));
                                expect(candidate.partition_source_offset == original.source.size());
                                expect(candidate.entry == original.entry);
                                expect(candidate.grid == original.grid && candidate.block == original.block);
                                expect(candidate.partition_rows == target);
                                expect(candidate.partition_original_rows == static_cast<uint32_t>(old_rows));
                                auto programs = static_cast<uint32_t>((rows + target - 1u) / target);
                                expect(candidate.partition_grid == std::array<uint32_t, 3>{programs, 1u, 1u});
                                expect(candidate.partition_grid[0u] == plan.candidate.programs);
                                expect(candidate.grid[0u] == plan.original.programs);
                                expect(candidate.arguments.size() == original.arguments.size());
                                for (auto i = size_t{0}; i < original.arguments.size(); i++) {
                                    auto a = candidate.arguments[i], b = original.arguments[i];
                                    expect(a.element == b.element && a.minimum_size_bytes == b.minimum_size_bytes && a.read == b.read && a.written == b.written);
                                }
                                auto guard = candidate.partition_guard;
                                expect(guard.input_slot == 0u && guard.output_slot == 1u);
                                expect(guard.input_bytes == static_cast<uint64_t>(rows * columns) * sizeof(T));
                                expect(guard.output_bytes == static_cast<uint64_t>(rows) * sizeof(T));
                                auto extra = candidate.source.substr(original.source.size());
                                expect(chunk_scan_occurrences(extra, "ct::sum(") == (maximum ? 0u : 1u));
                                expect(chunk_scan_occurrences(extra, "ct::reduce_max(") == (maximum ? 1u : 0u));
                                expect(chunk_scan_occurrences(extra, "ct::add(0.0f,") == (maximum ? 0u : 1u));
                                expect(chunk_scan_occurrences(extra, "ct::max(identity,") == (maximum ? 1u : 0u));
                                expect(chunk_scan_occurrences(extra, "input = ct::select(column <") == (masked ? 1u : 0u));
                                expect(extra.find(format("ct::shape<{}, 64>", target)) != string::npos);
                                expect(extra.find(format("ct::bid().x) * {}ll", target)) != string::npos);
                                auto full_rows = rows % target == 0;
                                auto full_input = full_rows && columns == 64;
                                expect(chunk_scan_occurrences(extra, "ct::load(") == static_cast<size_t>(full_input));
                                expect(chunk_scan_occurrences(extra, "ct::load_masked(") == static_cast<size_t>(!full_input));
                                expect(chunk_scan_occurrences(extra, "ct::store(") == static_cast<size_t>(full_rows));
                                expect(chunk_scan_occurrences(extra, "ct::store_masked(") == static_cast<size_t>(!full_rows));
                                expect(extra.find("ct::assume_aligned<") == string::npos);
                                expect(extra.find("ct::round_subnormals_to_zero") == string::npos);
                                expect(extra.find("ct::extract(") == string::npos);
                            }
                        }
                    }
                }
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<half>();
    check.template operator()<tile::bfloat16>();
}

void test_native_program_partition_fallbacks() {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::native_tile;
    using namespace boost::ut;
    auto reject = [](const tile::Kernel &kernel, Artifact original, uint32_t target) {
        auto before = original.source;
        auto grid = original.grid;
        append_program_partition(original, kernel.function(), target);
        expect(original.ok()) << original.error;
        expect(original.source == before && original.grid == grid);
        expect(original.partition_entry.empty());
        expect(!original.partition_diagnostic.empty());
    };
    auto kernel = capture_partition_source<float>(17, 33, 4, false);
    auto original = generate(kernel.function());
    expect(original.ok()) << original.error;
    if (!original.ok()) { return; }
    for (auto target : {3u, 4u, 8u}) { reject(kernel, original, target); }
    for (auto old_rows : {int64_t{1}, int64_t{2}}) {
        auto narrow = capture_partition_source<float>(17, 33, old_rows, false);
        reject(narrow, generate(narrow.function()), 1u);
    }
    auto transposed = capture_partition_source<float>(17, 33, 4, false, false, true);
    auto transposed_plan = tile::plan_independent_collective(transposed.function());
    expect(transposed_plan.ok()) << transposed_plan.error;
    reject(transposed, generate(transposed.function()), 1u);
    auto epilogue = capture_partition_source<float>(17, 33, 4, false, false, false, true);
    reject(epilogue, generate(epilogue.function()), 1u);
    for (auto changed : {0u, 1u, 2u, 3u, 4u}) {
        auto invalid = original;
        switch (changed) {
            case 0u: invalid.grid[0u]++; break;
            case 1u: invalid.arguments[0u].minimum_size_bytes--; break;
            case 2u: invalid.arguments[1u].read = true; break;
            case 3u: invalid.scan_chunk_extent = 1024u; break;
            case 4u: invalid.aligned16_entry = "independent_existing_entry"; break;
        }
        reject(kernel, std::move(invalid), 1u);
    }
    auto candidate = original;
    append_program_partition(candidate, kernel.function(), 1u);
    expect(!candidate.partition_entry.empty()) << candidate.partition_diagnostic;
    auto guard = candidate.partition_guard;
    auto disjoint = [&](uint64_t input, uint64_t output) {
        std::array<uint64_t, 2> pointers{input, output};
        return streaming_scan_disjoint(guard, span<const uint64_t>{pointers});
    };
    expect(disjoint(4096u, 4096u + guard.input_bytes));
    expect(disjoint(4096u + guard.output_bytes, 4096u));
    expect(!disjoint(4096u, 4096u + guard.input_bytes - 1u));
    expect(!disjoint(4096u + guard.output_bytes - 1u, 4096u));
    expect(!disjoint(4096u, 4096u));
    expect(!disjoint(0u, 65536u));
    expect(!disjoint(std::numeric_limits<uint64_t>::max() - guard.input_bytes + 1u, 4096u));
    expect(!disjoint(4096u, std::numeric_limits<uint64_t>::max() - guard.output_bytes + 1u));
    expect(!streaming_scan_disjoint(guard, span<const uint64_t>{}));
}
void test_native_partition_cost_decisions() {
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::native_tile;
    using namespace boost::ut;
    // Frozen training parity; no heldout data or observed timings.
    struct PartitionCostParity {
        int64_t rows, columns, original_rows;
        tile::ScalarType storage;
        bool maximum;
        uint32_t selected_rows;
        double original_score, rows1_score, rows2_score;
    };
    constexpr std::array<PartitionCostParity, 16u> partition_cost_parity{{
        {1024ll, 512ll, 4ll, tile::ScalarType::FLOAT16, true, 0u, 3.454643538706734, 3.3966491395253198, 3.454643538706734},
        {128ll, 2048ll, 8ll, tile::ScalarType::BFLOAT16, true, 1u, 2.7587107485297633, 2.2947555550784497, 2.2947555550784497},
        {17ll, 1024ll, 4ll, tile::ScalarType::FLOAT16, true, 1u, 1.3668451681758222, 1.0188787730873368, 1.1348675714501653},
        {256ll, 128ll, 8ll, tile::ScalarType::BFLOAT16, true, 1u, 1.1348675714501653, 1.0623745724733975, 1.076873172268751},
        {37ll, 257ll, 4ll, tile::ScalarType::FLOAT32, true, 1u, 1.1348675714501653, 1.0188787730873368, 1.0188787730873368},
        {3ll, 8191ll, 4ll, tile::ScalarType::BFLOAT16, true, 1u, 4.614531522335018, 1.8308003616271358, 2.7587107485297633},
        {65ll, 2048ll, 8ll, tile::ScalarType::FLOAT16, true, 1u, 2.7587107485297633, 1.598822764901479, 1.8308003616271358},
        {65ll, 512ll, 8ll, tile::ScalarType::FLOAT32, true, 1u, 1.3668451681758222, 1.076873172268751, 1.1348675714501653},
        {1024ll, 512ll, 4ll, tile::ScalarType::FLOAT16, false, 0u, 3.454643538706734, 3.3966491395253198, 3.454643538706734},
        {128ll, 2048ll, 8ll, tile::ScalarType::BFLOAT16, false, 1u, 2.7587107485297633, 2.2947555550784497, 2.2947555550784497},
        {17ll, 1024ll, 4ll, tile::ScalarType::FLOAT16, false, 1u, 1.3668451681758222, 1.0188787730873368, 1.1348675714501653},
        {256ll, 128ll, 8ll, tile::ScalarType::BFLOAT16, false, 1u, 1.1348675714501653, 1.0623745724733975, 1.076873172268751},
        {37ll, 257ll, 4ll, tile::ScalarType::FLOAT32, false, 1u, 1.1348675714501653, 1.0188787730873368, 1.0188787730873368},
        {3ll, 8191ll, 4ll, tile::ScalarType::BFLOAT16, false, 1u, 4.614531522335018, 1.8308003616271358, 2.7587107485297633},
        {65ll, 2048ll, 8ll, tile::ScalarType::FLOAT16, false, 1u, 2.7587107485297633, 1.598822764901479, 1.8308003616271358},
        {65ll, 512ll, 8ll, tile::ScalarType::FLOAT32, false, 1u, 1.3668451681758222, 1.076873172268751, 1.1348675714501653},
    }};
    auto check_parity = []<typename T>(const PartitionCostParity &row) {
        auto kernel = capture_partition_source<T>(row.rows, row.columns, row.original_rows, row.maximum, true);
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto one = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 1u});
        auto two = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 2u});
        auto choice = choose_program_partition(one, two, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
        expect(choice.has_score);
        expect(choice.original_rows == row.original_rows && choice.target_rows == row.selected_rows);
        expect(std::abs(choice.original_score - row.original_score) < 1e-12);
        auto selected_score = row.selected_rows == 0u ? row.original_score :
                              row.selected_rows == 1u ? row.rows1_score :
                                                        row.rows2_score;
        expect(std::abs(choice.selected_score - selected_score) < 1e-12);
        double score1{}, score2{};
        expect(partition_cost_score(tile::analyze_independent_collective_candidate(one), score1));
        expect(partition_cost_score(tile::analyze_independent_collective_candidate(two), score2));
        expect(std::abs(score1 - row.rows1_score) < 1e-12);
        expect(std::abs(score2 - row.rows2_score) < 1e-12);
    };
    for (auto &&row : partition_cost_parity) {
        switch (row.storage) {
            case tile::ScalarType::FLOAT16: check_parity.template operator()<luisa::half>(row); break;
            case tile::ScalarType::BFLOAT16: check_parity.template operator()<tile::bfloat16>(row); break;
            default: check_parity.template operator()<float>(row); break;
        }
    }
    auto check = []<typename T>() {
        for (auto maximum : {false, true}) {
            for (auto old_rows : {int64_t{4}, int64_t{8}}) {
                auto kernel = capture_partition_source<T>(128, 8192, old_rows, maximum, true);
                expect(kernel.valid());
                if (!kernel.valid()) { continue; }
                auto one = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 1u});
                auto two = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 2u});
                expect(one.ok()) << one.error;
                expect(two.ok()) << two.error;
                auto choice = choose_program_partition(one, two, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
                expect(choice.has_score && choice.target_rows == 1u);
                expect(choice.original_rows == old_rows);
                expect(choice.status == "selected" && choice.reason == "predicted-saving");
                // Both original BR4/8 have demand 8*8192; rows1/2 tie at 6*8192.
                auto original_score = kPartitionCostConstant + kPartitionCostVolume * (8.0 * 8192.0);
                auto selected_score = kPartitionCostConstant + kPartitionCostVolume * (6.0 * 8192.0);
                expect(std::abs(choice.original_score - original_score) < 1e-12);
                expect(std::abs(choice.selected_score - selected_score) < 1e-12);
                auto original = generate(kernel.function(), false);
                expect(original.ok()) << original.error;
                auto candidate = original;
                append_program_partition(candidate, kernel.function(), choice.target_rows);
                expect(candidate.source.starts_with(original.source + '\n'));
                expect(candidate.partition_rows == 1u && candidate.partition_grid[0u] == 128u);
                expect(candidate.grid == original.grid && candidate.block == original.block);
                // If only rows2 is legal, it remains a candidate; the algorithm
                // never evaluates unmeasured rows4 as an automatic choice.
                one.error = "test-unavailable";
                auto only_two = choose_program_partition(one, two, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
                expect(only_two.target_rows == 2u && only_two.selected_score == selected_score);
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<luisa::half>();
    check.template operator()<tile::bfloat16>();
    for (auto geometry : {std::array<int64_t, 2u>{384, 8192}, std::array<int64_t, 2u>{128, 64}}) {
        auto kernel = capture_partition_source<float>(geometry[0u], geometry[1u], 4, false);
        auto one = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 1u});
        auto two = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 2u});
        auto choice = choose_program_partition(one, two, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
        expect(choice.has_score && choice.target_rows == 0u && choice.original_rows == 4u);
        expect(choice.status == "retained" && choice.reason == "predicted-original");
        expect(choice.selected_score == choice.original_score);
        auto original = generate(kernel.function());
        auto retained = original;
        append_program_partition(retained, kernel.function(), choice.target_rows);
        expect(retained.source == original.source && retained.grid == original.grid);
    }
}

void test_native_partition_cost_gates() {
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::native_tile;
    using namespace boost::ut;
    auto kernel = capture_partition_source<float>(128, 8192, 4, false);
    auto one = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 1u});
    auto two = tile::plan_independent_collective(kernel.function(), {.target_extent_per_program = 2u});
    for (auto changed = 0u; changed < 7u; changed++) {
        std::array<uint32_t, 6u> target{89u, 24u, 32u, 1536u, 13040u, 13040u};
        if (changed < target.size()) { target[changed]++; }
        auto choice = choose_program_partition(one, two, target[0u], target[1u], target[2u], target[3u], target[4u], target[5u], changed == 6u);
        expect(!choice.has_score && choice.target_rows == 0u && choice.original_rows == 0u);
        expect(choice.status == "ineligible");
        expect(choice.reason == (changed == 6u ? "fast-math" : "target-profile"));
    }
    for (auto changed = 0u; changed < 6u; changed++) {
        auto first = one, second = two;
        for (auto plan : {&first, &second}) {
            switch (changed) {
                case 0u: plan->kind = tile::CollectiveKind::MINIMUM; break;
                case 1u:
                    plan->input_independent_axis = 1u;
                    plan->input_contribution_axis = 0u;
                    break;
                case 2u: plan->output_storage = tile::ScalarType::FLOAT16; break;
                case 3u: plan->original.independent_extent_per_program = 2u; break;
                case 4u: plan->candidate.programs = uint64_t{1u} << 32u; break;
                case 5u: plan->logical_contribution_extent = std::numeric_limits<uint64_t>::max(); break;
            }
        }
        auto choice = choose_program_partition(first, second, 89u, 24u, 32u, 1536u, 13040u, 13040u, false);
        expect(!choice.has_score && choice.target_rows == 0u);
        expect(choice.reason == "analysis-or-layout");
    }
    auto invalid = one;
    invalid.logical_independent_extent++;
    auto facts = tile::analyze_independent_collective_candidate(invalid);
    expect(!facts.ok());
    double score{};
    expect(!partition_cost_score(facts, score));
    tile::IndependentCollectiveWorkFacts overflow;
    overflow.geometry.programs = std::numeric_limits<uint64_t>::max();
    overflow.collective_input_elements_per_program = 25u;
    expect(!partition_cost_score(overflow, score));
}
}// namespace

namespace {
template<typename T>
void check_native_aligned_partition_view_source() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    auto kernel = tile_kernel("unrelated_input_representation", [](TensorView<const T, 2> input,
                                                                     TensorView<T, 2> output) {
                      auto m = axis("independent", 8), n = axis("contribution", 128);
                      for (auto &p : parallel(shape(32))) {
                          auto origin = p.index() * int64_t{8};
                          auto x = cast<float>(input.tile(coord(origin, 0), shape(m, n)).load());
                          output(coord(origin, 0), shape(m, axis("out", 1))).store(cast<T>(reduce(x, n, add)));
                      }
                  }).capture(tensor_shape(256, 128), tensor_shape(256, 1));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    for (auto fast : {false, true}) {
        auto original = cuda::native_tile::generate(kernel.function(), fast, false);
        auto candidate = cuda::native_tile::generate(kernel.function(), fast, true);
        expect(original.ok()) << original.error;
        expect(candidate.ok()) << candidate.error;
        if (!original.ok() || !candidate.ok()) { continue; }
        expect(candidate.source.starts_with(original.source + '\n'));
        expect(candidate.aligned16_buffer_mask == 1u); // output shape<8,1> is not eligible
        expect(candidate.aligned16_partition_loads == 1u);
        expect(original.aligned16_partition_loads == 0u);
        expect(candidate.grid == original.grid);
        expect(candidate.block == original.block);
        expect(chunk_scan_occurrences(original.source, "ct::partition_view") == 0u);
        expect(chunk_scan_occurrences(candidate.source, "ct::partition_view") == 1u);
        auto alternate = candidate.source.substr(original.source.size());
        expect(alternate.find("ct::extents<long long, 256, 128>{}") != luisa::string::npos);
        expect(alternate.find("buffer0 = ct::assume_aligned<16>(buffer0)") != luisa::string::npos);
        expect(alternate.find("buffer1 = ct::assume_aligned") == luisa::string::npos);
        expect(chunk_scan_occurrences(alternate, "ct::sum(") == chunk_scan_occurrences(original.source, "ct::sum("));
        expect(chunk_scan_occurrences(alternate, "ct::store(") == chunk_scan_occurrences(original.source, "ct::store("));
    }
}

void test_native_aligned_view_sources() {
    check_native_aligned_partition_view_source<luisa::half>();
    check_native_aligned_partition_view_source<luisa::compute::tile::bfloat16>();
}

void test_native_aligned_view_fallbacks() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    // case0 full but row-origin not Tile aligned; case1 partial column tail;
    // case2 negative row; case3 later unproved/alignment-ineligible use of root.
    for (auto changed : {0u, 1u, 2u, 3u}) {
        auto row_origin = changed == 0u ? int64_t{1} : changed == 2u ? int64_t{-1} : int64_t{0};
        auto input_rows = changed == 0u ? int64_t{9} : int64_t{8};
        auto input_columns = changed == 1u ? int64_t{127} : changed == 3u ? int64_t{136} : int64_t{128};
        auto kernel = tile_kernel("no_name_based_gate", [=](TensorView<const luisa::half, 2> input,
                                                             TensorView<luisa::half, 2> output) {
                          auto m = axis("a", 8), n = axis("b", 128);
                          for (auto &p : parallel(shape(1))) {
                              auto x = input.tile(coord(row_origin, 0), shape(m, n)).load();
                              if (changed == 3u) {
                                  auto y = input.tile(coord(0, 1), shape(m, n)).load();
                                  x = cast<luisa::half>(cast<float>(x) + cast<float>(y));
                              }
                              output(coord(0, 0), shape(m, n)).store(x);
                          }
                      }).capture(tensor_shape(input_rows, input_columns), tensor_shape(8, 128));
        expect(kernel.valid());
        if (!kernel.valid()) { continue; }
        auto original = cuda::native_tile::generate(kernel.function(), false, false);
        auto candidate = cuda::native_tile::generate(kernel.function(), false, true);
        expect(original.ok()) << original.error;
        expect(candidate.ok()) << candidate.error;
        if (!original.ok() || !candidate.ok()) { continue; }
        expect(candidate.source.starts_with(original.source + '\n'));
        expect(candidate.aligned16_partition_loads == 0u);
        expect(candidate.aligned16_buffer_mask == (changed == 0u ? 3u : 2u));
        expect(chunk_scan_occurrences(candidate.source, "ct::partition_view") == 0u);
    }
}

}// namespace

namespace {
void test_native_aligned_view_unknown_overflow() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    auto check = [](const tile::Kernel &kernel, uint32_t output_mask, uint32_t input_slot) {
        // These must reach source generation successfully: rejection by an
        // earlier verifier is not evidence that the load proof failed closed.
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto original = cuda::native_tile::generate(kernel.function(), false, false);
        auto candidate = cuda::native_tile::generate(kernel.function(), false, true);
        expect(original.ok()) << original.error;
        expect(candidate.ok()) << candidate.error;
        if (!original.ok() || !candidate.ok()) { return; }
        expect(candidate.source.starts_with(original.source + '\n'));
        expect(candidate.aligned16_buffer_mask == output_mask);
        expect(candidate.aligned16_partition_loads == 0u);
        expect(chunk_scan_occurrences(candidate.source, "ct::partition_view") == 0u);
        expect(chunk_scan_occurrences(original.source, "ct::load_masked(") >= 1u);
        auto alternate = candidate.source.substr(original.source.size());
        expect(alternate.find(luisa::format("buffer{} = ct::assume_aligned", input_slot)) == luisa::string::npos);
        expect(chunk_scan_occurrences(alternate, "ct::load_masked(") == chunk_scan_occurrences(original.source, "ct::load_masked("));
        expect(candidate.grid == original.grid);
        expect(candidate.block == original.block);
    };
    auto unknown = tile_kernel("unproved_loaded_origin", [](TensorView<const int64_t, 1> offsets,
                                                             TensorView<const luisa::half, 1> input,
                                                             TensorView<luisa::half, 1> output) {
        for (auto &program : parallel(shape(1))) {
            static_cast<void>(program);
            auto origin = offsets.tile(coord(0), shape(1)).load().at(coord(0));
            auto domain = shape(axis("element", 64));
            auto x = input.tile(coord(origin), domain).load();
            output.tile(coord(0), domain).store(x);
        }
    }).capture(tensor_shape(1), tensor_shape(64), tensor_shape(64));
    check(unknown, 4u, 1u);

    // Deliberately exercise the interval-overflow rejection in source analysis.
    // The captured IR is verifier-valid, but this fixture is never compiled or
    // launched on a GPU: no claim about execution of overflowing signed math.
    auto overflow = tile_kernel("overflowing_origin_interval", [](TensorView<const luisa::half, 1> input,
                                                                   TensorView<luisa::half, 1> output) {
        for (auto &program : parallel(shape(3))) {
            auto origin = program.index() * std::numeric_limits<int64_t>::max();
            auto domain = shape(axis("element", 64));
            auto x = input.tile(coord(origin), domain).load();
            output.tile(coord(program.index() * int64_t{64}), domain).store(x);
        }
    }).capture(tensor_shape(64), tensor_shape(192));
    check(overflow, 2u, 0u);
}
}// namespace

namespace {
template<typename T>
void check_native_masked_partition_sources() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    constexpr std::array shapes{std::array<int64_t, 2u>{257, 128},
                                std::array<int64_t, 2u>{9, 136},
                                std::array<int64_t, 2u>{7, 128}};
    for (auto extents : shapes) {
        auto rows = extents[0], columns = extents[1];
        auto row_chunks = (rows + 7) / 8, column_chunks = (columns + 127) / 128;
        auto kernel = tile_kernel("partial_partition_representation", [=](TensorView<const T, 2> input,
                                                                           TensorView<T, 2> output) {
            auto pr = axis("program_row", row_chunks), pc = axis("program_column", column_chunks);
            auto r = axis("row", 8), c = axis("column", 128);
            for (auto &program : parallel(shape(pr, pc))) {
                auto origin = coord(program.index(pr) * int64_t{8}, program.index(pc) * int64_t{128});
                output.tile(origin, shape(r, c)).store(input.tile(origin, shape(r, c)).load());
            }
        }).capture(tensor_shape(rows, columns), tensor_shape(row_chunks * 8, column_chunks * 128));
        expect(kernel.valid());
        if (!kernel.valid()) { continue; }
        for (auto fast : {false, true}) {
            auto original = cuda::native_tile::generate(kernel.function(), fast, false);
            auto candidate = cuda::native_tile::generate(kernel.function(), fast, true);
            expect(original.ok()) << original.error;
            expect(candidate.ok()) << candidate.error;
            if (!original.ok() || !candidate.ok()) { continue; }
            expect(candidate.source.starts_with(original.source + '\n'));
            expect(candidate.grid == original.grid && candidate.block == original.block);
            expect(candidate.aligned16_buffer_mask == 3u);
            expect(candidate.aligned16_partition_loads == 1u);
            auto alternate = candidate.source.substr(original.source.size());
            expect(chunk_scan_occurrences(alternate, "_partition.load_masked(ct::view_padding_zero_t{}, ") == 1u);
            expect(chunk_scan_occurrences(original.source, "ct::load_masked(") == 1u);
            expect(chunk_scan_occurrences(alternate, "ct::store(") == chunk_scan_occurrences(original.source, "ct::store("));
        }
    }
    // A frontend MAX identity is its own explicit operation. Both the variant
    // with this select and the one whose zero tail participates are preserved.
    for (auto neutral_select : {false, true}) {
        auto kernel = capture_partition_source<T>(9, 136, 8, true, neutral_select);
        expect(kernel.valid());
        if (!kernel.valid()) { continue; }
        auto original = cuda::native_tile::generate(kernel.function(), false, false);
        auto candidate = cuda::native_tile::generate(kernel.function(), false, true);
        expect(original.ok()) << original.error;
        expect(candidate.ok()) << candidate.error;
        if (!original.ok() || !candidate.ok()) { continue; }
        expect(candidate.source.starts_with(original.source + '\n'));
        expect(candidate.aligned16_buffer_mask == 1u);
        expect(candidate.aligned16_partition_loads == 1u);
        auto alternate = candidate.source.substr(original.source.size());
        expect(chunk_scan_occurrences(alternate, "ct::select(") == chunk_scan_occurrences(original.source, "ct::select("));
        expect(chunk_scan_occurrences(alternate, "ct::max(") == chunk_scan_occurrences(original.source, "ct::max("));
        expect(chunk_scan_occurrences(alternate, "ct::store_masked(") == chunk_scan_occurrences(original.source, "ct::store_masked("));
    }
}

void test_native_masked_partition_sources() {
    check_native_masked_partition_sources<luisa::half>();
    check_native_masked_partition_sources<luisa::compute::tile::bfloat16>();
}

template<typename T>
void check_native_masked_partition_fallbacks() {
    using namespace luisa::compute;
    using namespace luisa::compute::tile;
    using namespace boost::ut;
    auto check = [](const tile::Kernel &kernel, uint32_t mask) {
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto original = cuda::native_tile::generate(kernel.function(), false, false);
        auto candidate = cuda::native_tile::generate(kernel.function(), false, true);
        expect(original.ok()) << original.error;
        expect(candidate.ok()) << candidate.error;
        if (!original.ok() || !candidate.ok()) { return; }
        expect(candidate.source.starts_with(original.source + '\n'));
        expect(candidate.aligned16_buffer_mask == mask);
        expect(candidate.aligned16_partition_loads == 0u);
        expect(candidate.grid == original.grid && candidate.block == original.block);
    };
    // Negative / fully out-of-range / nonchunk origins remain legal ordinary
    // masked loads. The first eligible partial load must not leak a root proof.
    for (auto bad_row : {int64_t{-8}, int64_t{16}, int64_t{4}}) {
        auto kernel = tile_kernel("mixed_partial_partition_rejection", [=](TensorView<const T, 2> input,
                                                                            TensorView<T, 2> output) {
            for (auto &program : parallel(shape(1))) {
                static_cast<void>(program);
                auto domain = shape(axis("row", 8), axis("column", 128));
                auto first = input.tile(coord(8, 128), domain).load();
                auto second = input.tile(coord(bad_row, 128), domain).load();
                output.tile(coord(0, 0), domain).store(first);
                output.tile(coord(8, 0), domain).store(second);
            }
        }).capture(tensor_shape(9, 136), tensor_shape(16, 128));
        check(kernel, 2u);
    }
    for (auto fill : {0.0f, -0.0f, 1.0f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto kernel = tile_kernel("explicit_partial_fill", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
            for (auto &program : parallel(shape(1))) {
                static_cast<void>(program);
                auto domain = shape(axis("row", 8), axis("column", 128));
                output.tile(coord(0, 0), domain).store(input.tile(coord(8, 128), domain).load(T{fill}));
            }
        }).capture(tensor_shape(9, 136), tensor_shape(8, 128));
        check(kernel, 2u);
    }
    // Same-root partial stores still cancel its earlier eligible partial read.
    auto tail_store = tile_kernel("partial_store_keeps_conservative_gate", [](TensorView<T, 2> inout,
                                                                               TensorView<T, 2> output) {
        for (auto &program : parallel(shape(1))) {
            static_cast<void>(program);
            auto domain = shape(axis("row", 8), axis("column", 128));
            auto before = inout.tile(coord(8, 128), domain).load();
            inout.tile(coord(8, 128), domain).store(before);
            auto after = inout.tile(coord(8, 128), domain).load();
            output.tile(coord(0, 0), domain).store(before);
            output.tile(coord(8, 0), domain).store(after);
        }
    }).capture(tensor_shape(9, 136), tensor_shape(16, 128));
    check(tail_store, 2u);
    // Existing unknown-origin and interval-overflow source tests remain intact.
    // Tile-valued fill is not tested here: the public verifier already rejects
    // it, so it would not test this backend proof boundary.
}

void test_native_masked_partition_fallbacks() {
    check_native_masked_partition_fallbacks<luisa::half>();
    check_native_masked_partition_fallbacks<luisa::compute::tile::bfloat16>();
}
}// namespace

int main(int argc, char *argv[]) {
    using namespace boost::ut;
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_cuda_collective_chunk_sources"_test = [] { test_native_pure_chunk_scan_sources(); };
    "tile_cuda_collective_independent_sources"_test = [] { test_native_independent_collective_sources(); };
    "tile_cuda_collective_configuration"_test = [] { test_native_collective_configuration(); };
    "tile_cuda_streaming_scan_plan"_test = [] { test_native_streaming_scan_plan(); };
    "tile_cuda_collective_schedule_parity"_test = [] { test_native_collective_schedule_parity(); };
    "tile_cuda_collective_schedule_gates"_test = [] { test_native_collective_schedule_gates(); };
    "tile_cuda_program_partition_sources"_test = [] { test_native_program_partition_sources(); };
    "tile_cuda_program_partition_fallbacks"_test = [] { test_native_program_partition_fallbacks(); };
    "tile_cuda_partition_cost_decisions"_test = [] { test_native_partition_cost_decisions(); };
    "tile_cuda_partition_cost_gates"_test = [] { test_native_partition_cost_gates(); };
    "tile_cuda_aligned_view_sources"_test = [] { test_native_aligned_view_sources(); test_native_masked_partition_sources(); };
    "tile_cuda_aligned_view_fallbacks"_test = [] { test_native_aligned_view_fallbacks(); test_native_aligned_view_unknown_overflow(); test_native_masked_partition_fallbacks(); };
}
