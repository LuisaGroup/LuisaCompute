#include "ut/ut.hpp"
#include "cuda_tile_codegen.h"
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

int main(int argc, char *argv[]) {
    using namespace boost::ut;
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_cuda_collective_chunk_sources"_test = [] { test_native_pure_chunk_scan_sources(); };
    "tile_cuda_collective_independent_sources"_test = [] { test_native_independent_collective_sources(); };
    "tile_cuda_collective_configuration"_test = [] { test_native_collective_configuration(); };
}
