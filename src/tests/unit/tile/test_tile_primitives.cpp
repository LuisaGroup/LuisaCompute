// Portable primitive conformance through the real Tile capture/runtime API.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/platform.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr size_t kPad = 17u;
constexpr float kFloatGuard = -917.25f;
constexpr int64_t kIndexGuard = -719;
constexpr std::array<int64_t, 5> kWidths{1, 2, 32, 64, 128};

[[nodiscard]] uint32_t bits(float value) noexcept { return std::bit_cast<uint32_t>(value); }

struct Route {
    Device *device{nullptr};
    tile::CompileOptions options;
    bool cuda_native{false};
    bool host_only() const noexcept { return device == nullptr; }
};

[[nodiscard]] bool check_shader(const tile::Shader &shader, const Route &route, bool optional_prefix) {
    if (route.cuda_native && optional_prefix && !shader) {
        expect(shader.metadata().error.find("powers of two") != luisa::string::npos) << shader.metadata().error;
        return false;
    }
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return false; }
    if (route.cuda_native) {
        expect(shader.metadata().realization.starts_with("CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin; no cache"));
    } else if (route.options.lowering == tile::Lowering::TIRX) {
        expect(shader.metadata().realization.find("TIRx") != luisa::string::npos) << shader.metadata().realization;
    }
    return true;
}

[[nodiscard]] tile::Kernel rank_kernel(int64_t width, int64_t count, bool descending, bool interior) {
    using namespace tile;
    return tile_kernel("primitive_stable_rank", [=](TensorView<const float, 3> x,
                                                    TensorView<float, 3> values,
                                                    TensorView<int64_t, 3> indices) {
               auto a = axis("a", interior ? 2 : 4);
               auto b = axis("b", interior ? width : 1);
               auto c = axis("c", interior ? 4 : width);
               for (auto &item : parallel(shape(1))) {
                   static_cast<void>(item);
                   auto value = x.tile(coord(0, 0, 0), shape(a, b, c)).load();
                   auto dimension = interior ? b : c;
                   auto ranked = count == width ? tile::sort(value, dimension, descending) : topk(value, dimension, count, descending);
                   expect(ranked.values.space() == ranked.indices.space()) << "rank values/indices axis order";
                   if (count == width) { expect(ranked.values.space() == value.space()) << "full sort preserves source axis order"; }
                   values(coord(0, 0, 0), ranked.values.space()).store(ranked.values);
                   indices(coord(0, 0, 0), ranked.indices.space()).store(ranked.indices);
               }
           })
        .capture(tensor_shape(interior ? 2 : 4, interior ? width : 1, interior ? 4 : width), tensor_shape(interior ? 2 : 4, interior ? count : 1, interior ? 4 : count), tensor_shape(interior ? 2 : 4, interior ? count : 1, interior ? 4 : count));
}

void ranking(const Route &route, int64_t width, int64_t count, bool descending, bool interior) {
    auto kernel = rank_kernel(width, count, descending, interior);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    if (route.host_only()) { return; }
    auto shader = tile::compile(*route.device, kernel, route.options, {.enable_fast_math = false});
    if (!check_shader(shader, route, count == 7 || !std::has_single_bit(static_cast<uint64_t>(width)))) { return; }
    auto outer = interior ? int64_t{2} : int64_t{4};
    auto inner = interior ? int64_t{4} : int64_t{1};
    auto input_size = static_cast<size_t>(outer * width * inner);
    auto output_size = static_cast<size_t>(outer * count * inner);
    std::vector<float> input(input_size + 2 * kPad, kFloatGuard), actual(output_size + 2 * kPad, kFloatGuard);
    std::vector<float> expected(output_size);
    std::vector<int64_t> indices(output_size + 2 * kPad, kIndexGuard), expected_indices(output_size);
    for (auto a = int64_t{0}; a < outer; a++) {
        for (auto c = int64_t{0}; c < inner; c++) {
            auto row = a * inner + c;
            for (auto b = int64_t{0}; b < width; b++) {
                auto value = static_cast<float>((b * 17 + row * 3) % 11 - 5) * .25f;
                if (row % 4 == 1) { value = b % 2 == 0 ? 0.0f : -0.0f; }
                if (row % 4 == 2) { value = -2.5f; }
                if (row % 4 == 3) { value = static_cast<float>(width - b) * .125f; }
                input[kPad + static_cast<size_t>((a * width + b) * inner + c)] = value;
            }
            std::vector<int64_t> order(static_cast<size_t>(width));
            std::iota(order.begin(), order.end(), int64_t{0});
            std::stable_sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
                auto x = input[kPad + static_cast<size_t>((a * width + lhs) * inner + c)];
                auto y = input[kPad + static_cast<size_t>((a * width + rhs) * inner + c)];
                return descending ? x > y : x < y;
            });
            for (auto b = int64_t{0}; b < count; b++) {
                auto offset = static_cast<size_t>((a * count + b) * inner + c);
                expected_indices[offset] = order[static_cast<size_t>(b)];
                expected[offset] = input[kPad + static_cast<size_t>((a * width + expected_indices[offset]) * inner + c)];
            }
        }
    }
    auto original = input;
    std::fill_n(actual.begin() + kPad, output_size, std::numeric_limits<float>::quiet_NaN());
    auto x = route.device->create_buffer<float>(input.size());
    auto y = route.device->create_buffer<float>(actual.size());
    auto ix = route.device->create_buffer<int64_t>(indices.size());
    auto stream = route.device->create_stream(StreamTag::COMPUTE);
    stream << x.copy_from(input.data()) << y.copy_from(actual.data()) << ix.copy_from(indices.data())
           << shader(x.view(kPad, input_size), y.view(kPad, output_size), ix.view(kPad, output_size)).dispatch()
           << x.copy_to(input.data()) << y.copy_to(actual.data()) << ix.copy_to(indices.data()) << synchronize();
    for (auto i = size_t{0}; i < input.size(); i++) { expect(bits(input[i]) == bits(original[i])) << "readonly index=" << i; }
    for (auto i = size_t{0}; i < output_size; i++) {
        expect(bits(actual[kPad + i]) == bits(expected[i])) << "rank value N=" << width << " K=" << count << " offset=" << i;
        expect(indices[kPad + i] == expected_indices[i]) << "rank index N=" << width << " K=" << count << " offset=" << i;
    }
    for (auto i = size_t{0}; i < kPad; i++) {
        expect(bits(actual[i]) == bits(kFloatGuard));
        expect(bits(actual[kPad + output_size + i]) == bits(kFloatGuard));
        expect(indices[i] == kIndexGuard);
        expect(indices[kPad + output_size + i] == kIndexGuard);
    }
}

[[nodiscard]] tile::Kernel scan_kernel(int64_t width, bool interior, tile::ReductionPolicy policy) {
    using namespace tile;
    return tile_kernel("primitive_inclusive_sum", [=](TensorView<const float, 3> x, TensorView<float, 3> y) {
               auto a = axis("a", 2), b = axis("b", interior ? width : 4), c = axis("c", interior ? 4 : width);
               for (auto &item : parallel(shape(1))) {
                   static_cast<void>(item);
                   auto value = x.tile(coord(0, 0, 0), shape(a, b, c)).load();
                   y(coord(0, 0, 0), value.space()).store(inclusive_sum(value, interior ? b : c, policy));
               }
           })
        .capture(tensor_shape(2, interior ? width : 4, interior ? 4 : width), tensor_shape(2, interior ? width : 4, interior ? 4 : width));
}

size_t policy_count(const tile::Region &region, tile::ReductionPolicy policy) {
    auto count = size_t{0};
    for (auto block : region.blocks()) {
        for (auto op : block->operations()) {
            if (op->kind() == tile::OperationKind::REDUCE && op->reduction_policy() == policy) { count++; }
            for (auto &&child : op->regions()) { count += policy_count(*child, policy); }
        }
    }
    return count;
}

void scan(const Route &route, int64_t width, bool interior, tile::ReductionPolicy policy) {
    auto kernel = scan_kernel(width, interior, policy);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    expect(policy_count(kernel.function().body(), policy) > 0u);
    if (route.host_only()) { return; }
    auto shader = tile::compile(*route.device, kernel, route.options, {.enable_fast_math = false});
    if (!check_shader(shader, route, false)) { return; }
    auto inner = interior ? int64_t{4} : int64_t{1};
    auto outer = int64_t{8} / inner;
    auto size = static_cast<size_t>(8 * width);
    std::vector<float> input(size + 2 * kPad, kFloatGuard), actual(size + 2 * kPad, kFloatGuard), ordered(size);
    std::vector<double> expected(size), bounds(size);
    constexpr std::array<float, 8> cancellation{0x1p24f, 1.0f, -0x1p24f, 1.0f, -.5f, .25f, -.125f, .0625f};
    for (auto a = int64_t{0}; a < outer; a++) {
        for (auto c = int64_t{0}; c < inner; c++) {
            auto sum = 0.0;
            auto magnitude = 0.0;
            auto serial = 0.0f;
            for (auto b = int64_t{0}; b < width; b++) {
                auto offset = static_cast<size_t>((a * width + b) * inner + c);
                auto value = (a * inner + c) % 2 == 0 ? static_cast<float>((b * 13 + a + c) % 17 - 8) * .125f : cancellation[static_cast<size_t>(b) % cancellation.size()];
                input[kPad + offset] = value;
                // Volatile materializes each binary32 step; no host reassociation.
                volatile float step = serial + value;
                serial = step;
                sum += static_cast<double>(value);
                magnitude += std::abs(static_cast<double>(value));
                ordered[offset] = serial;
                expected[offset] = sum;
                auto nu = static_cast<double>(width) * 0x1p-24;
                bounds[offset] = nu / (1.0 - nu) * magnitude;
            }
        }
    }
    auto original = input;
    std::fill_n(actual.begin() + kPad, size, std::numeric_limits<float>::quiet_NaN());
    auto x = route.device->create_buffer<float>(input.size());
    auto y = route.device->create_buffer<float>(actual.size());
    auto stream = route.device->create_stream(StreamTag::COMPUTE);
    stream << x.copy_from(input.data()) << y.copy_from(actual.data())
           << shader(x.view(kPad, size), y.view(kPad, size)).dispatch()
           << x.copy_to(input.data()) << y.copy_to(actual.data()) << synchronize();
    for (auto i = size_t{0}; i < input.size(); i++) { expect(bits(input[i]) == bits(original[i])); }
    for (auto i = size_t{0}; i < size; i++) {
        expect(std::isfinite(actual[kPad + i]));
        if (policy == tile::reduction::fold_left) {
            expect(bits(actual[kPad + i]) == bits(ordered[i])) << "ordered scan N=" << width << " offset=" << i;
        } else {
            expect(std::abs(static_cast<double>(actual[kPad + i]) - expected[i]) <= bounds[i]) << "tree scan N=" << width << " offset=" << i;
        }
    }
    for (auto i = size_t{0}; i < kPad; i++) {
        expect(bits(actual[i]) == bits(kFloatGuard));
        expect(bits(actual[kPad + size + i]) == bits(kFloatGuard));
    }
}

void run(const Route &route) {
    "tile_primitives_sort_stable_power2"_test = [&] {
        for (auto width : kWidths) {
            for (auto descending : {false, true}) { ranking(route, width, width, descending, false); }
        }
    };
    "tile_primitives_topk_stable_prefix"_test = [&] {
        for (auto width : kWidths) {
            for (auto count : {int64_t{1}, int64_t{7}, int64_t{16}}) {
                if (count <= width) { ranking(route, width, count, true, false); }
            }
        }
    };
    "tile_primitives_sort_interior_axis"_test = [&] {
        for (auto descending : {false, true}) { ranking(route, 32, 32, descending, true); }
        ranking(route, 32, 16, true, true);
        for (auto descending : {false, true}) {
            ranking(route, 33, 33, descending, true);
            ranking(route, 33, 7, descending, true);
        }
    };
    "tile_primitives_scan_ordered"_test = [&] {
        for (auto width : kWidths) { scan(route, width, false, tile::reduction::fold_left); }
        scan(route, 32, true, tile::reduction::fold_left);
    };
    "tile_primitives_scan_tree"_test = [&] {
        for (auto width : kWidths) { scan(route, width, false, tile::reduction::unordered_tree); }
        scan(route, 32, true, tile::reduction::unordered_tree);
    };
    "tile_primitives_capture_validation"_test = [] {
        using namespace tile;
        auto invalid = tile_kernel("wrong_scan_axis", [](TensorView<const float, 1> x, TensorView<float, 1> y) {
                           auto a = axis("a", 32), other = axis("other", 32);
                           auto value = x.tile(coord(0), shape(a)).load();
                           static_cast<void>(y);
                           static_cast<void>(inclusive_sum(value, other));
                       }).capture(tensor_shape(32), tensor_shape(32));
        expect(!invalid.valid());
        expect(!invalid.diagnostics().empty());
    };
}

}// namespace

int main(int argc, char *argv[]) {
    auto usage = [] { LUISA_INFO("Usage: test_tile_primitives host [UT filter] | <backend> <native|tirx> [UT filter]"); return 2; };
    if (argc < 2) { return usage(); }
    if (argv == nullptr) { return usage(); }
    if (argv[0] == nullptr) { return usage(); }
    if (argv[1] == nullptr) { return usage(); }
    auto host = std::strcmp(argv[1], "host") == 0;
    Route route;
    if (!host) {
        if (argc < 3) { return usage(); }
        if (argv[2] == nullptr) { return usage(); }
        if (std::strcmp(argv[2], "native") == 0) {
            route.options.lowering = tile::Lowering::NATIVE;
        } else if (std::strcmp(argv[2], "tirx") == 0) {
            route.options.lowering = tile::Lowering::TIRX;
        } else {
            return usage();
        }
        if (std::strcmp(argv[1], "cuda") == 0) {
            route.cuda_native = route.options.lowering == tile::Lowering::NATIVE;
            auto environment = get_environment_variable("LUISA_CUDA_TILE_IR");
            auto enabled = false;
            if (environment) { enabled = std::strcmp(environment->c_str(), "1") == 0; }
            if (enabled != route.cuda_native) {
                LUISA_INFO("CUDA route disagrees with LUISA_CUDA_TILE_IR; refusing ambiguous execution.");
                return 2;
            }
        }
    }
    std::vector<const char *> arguments{argv[0]};
    for (auto i = host ? 2 : 3; i < argc; i++) {
        if (argv[i] == nullptr) { return usage(); }
        arguments.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(arguments.size()), arguments.data());
    if (host) {
        run(route);
    } else {
        auto [context, device] = test::create_device(argc, argv);
        route.device = &device;
        run(route);
    }
    return 0;
}
