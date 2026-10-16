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

[[nodiscard]] tile::Kernel rank_kernel(int64_t width, int64_t count, bool descending, bool interior, tile::SortAlgorithm algorithm = tile::SortAlgorithm::DEFAULT) {
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
                   auto ranked = count == width ? tile::sort(value, dimension, descending, algorithm) : topk(value, dimension, count, descending, algorithm);
                   expect(ranked.values.space() == ranked.indices.space()) << "rank values/indices axis order";
                   if (count == width) { expect(ranked.values.space() == value.space()) << "full sort preserves source axis order"; }
                   values(coord(0, 0, 0), ranked.values.space()).store(ranked.values);
                   indices(coord(0, 0, 0), ranked.indices.space()).store(ranked.indices);
               }
           })
        .capture(tensor_shape(interior ? 2 : 4, interior ? width : 1, interior ? 4 : width), tensor_shape(interior ? 2 : 4, interior ? count : 1, interior ? 4 : count), tensor_shape(interior ? 2 : 4, interior ? count : 1, interior ? 4 : count));
}

void ranking(const Route &route, int64_t width, int64_t count, bool descending, bool interior, tile::SortAlgorithm algorithm = tile::SortAlgorithm::DEFAULT) {
    auto kernel = rank_kernel(width, count, descending, interior, algorithm);
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
                if (algorithm == tile::SortAlgorithm::PACKED_FP32) {
                    constexpr uint32_t special[]{0u, 0x80000000u, 0x7f800000u, 0xff800000u, 1u, 0x80000001u,
                                                 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                                                 0x7f7fffffu, 0xff7fffffu, 0x3f800001u, 0xbf800001u, 0u, 0x80000000u};
                    value = std::bit_cast<float>(special[(b + row * 3) % std::size(special)]);
                }
                input[kPad + static_cast<size_t>((a * width + b) * inner + c)] = value;
            }
            std::vector<int64_t> order(static_cast<size_t>(width));
            std::iota(order.begin(), order.end(), int64_t{0});
            std::stable_sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
                auto x = input[kPad + static_cast<size_t>((a * width + lhs) * inner + c)];
                auto y = input[kPad + static_cast<size_t>((a * width + rhs) * inner + c)];
                if (algorithm == tile::SortAlgorithm::PACKED_FP32) {
                    // Decode through normal FP64 arithmetic, avoiding host DAZ/FTZ
                    // effects on comparison of the original FP32 subnormal values.
                    auto numeric = [](uint32_t word) noexcept {
                        auto exponent = (word / 0x800000u) % 256u;
                        auto mantissa = word % 0x800000u;
                        auto magnitude = exponent == 255u ? std::numeric_limits<double>::infinity() :
                                         std::ldexp(static_cast<double>(mantissa + (exponent == 0u ? 0u : 0x800000u)),
                                                    exponent == 0u ? -149 : static_cast<int>(exponent) - 150);
                        return word >= 0x80000000u ? -magnitude : magnitude;
                    };
                    auto dx = numeric(bits(x)), dy = numeric(bits(y));
                    return descending ? dx > dy : dx < dy;
                }
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
    stream << x.copy_from(luisa::span{input}) << y.copy_from(luisa::span{actual}) << ix.copy_from(luisa::span{indices})
           << shader(x.view(kPad, input_size), y.view(kPad, output_size), ix.view(kPad, output_size)).dispatch()
           << x.copy_to(luisa::span{input}) << y.copy_to(luisa::span{actual}) << ix.copy_to(luisa::span{indices}) << synchronize();
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
    stream << x.copy_from(luisa::span{input}) << y.copy_from(luisa::span{actual})
           << shader(x.view(kPad, size), y.view(kPad, size)).dispatch()
           << x.copy_to(luisa::span{input}) << y.copy_to(luisa::span{actual}) << synchronize();
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

void bitcast_storage(const Route &route) {
    using namespace tile;
    constexpr auto count = size_t{32};
    auto kernel = tile_kernel("bitcast_storage", [](TensorView<const uint32_t, 1> a, TensorView<const float, 1> b,
                                                    TensorView<uint32_t, 1> x, TensorView<float, 1> y) {
                      auto n = axis("n", count);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto words = a.tile(coord(0), shape(n)).load();
                          auto floats = b.tile(coord(0), shape(n)).load();
                          auto scalar_floats = map<float>(shape(n), [&](const Nest &element) { return tile::bitcast<float>(words.at(element)); });
                          y(coord(0), shape(n)).store(ite(iota(n) % 2 == 0, tile::bitcast<float>(words), scalar_floats));
                          auto mapped = map<uint32_t>(shape(n), [&](const Nest &element) { return tile::bitcast<uint32_t>(floats.at(element)); });
                          x(coord(0), shape(n)).store(ite(iota(n) % 2 == 0, mapped, tile::bitcast<uint32_t>(floats)));
                      }
                  }).capture(tensor_shape(count), tensor_shape(count), tensor_shape(count), tensor_shape(count));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    if (route.host_only()) { return; }
    constexpr uint32_t special[]{0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
                                 0x00800000u, 0x80800000u, 0x7f800000u, 0xff800000u, 0x7fc12345u,
                                 0xffc54321u, 0x7f812345u, 0xff812345u, 0x7f7fffffu, 0xff7fffffu};
    std::vector<uint32_t> a(count + 2 * kPad, 0x7139a25du), x(count + 2 * kPad, 0x7139a25du);
    std::vector<float> b(count + 2 * kPad, kFloatGuard), y(count + 2 * kPad, kFloatGuard);
    for (auto i = size_t{0}; i < count; i++) {
        a[kPad + i] = special[(i / 2) % std::size(special)];
        b[kPad + i] = std::bit_cast<float>(special[(i / 2 + 7) % std::size(special)]);
    }
    auto original_a = a;
    auto original_b = b;
    auto ga = route.device->create_buffer<uint32_t>(a.size()), gx = route.device->create_buffer<uint32_t>(x.size());
    auto gb = route.device->create_buffer<float>(b.size()), gy = route.device->create_buffer<float>(y.size());
    auto stream = route.device->create_stream(StreamTag::COMPUTE);
    for (auto fast : {false, true}) {
        auto shader = tile::compile(*route.device, kernel, route.options, {.enable_fast_math = fast});
        if (!check_shader(shader, route, false)) { return; }
        std::fill(x.begin(), x.end(), 0x7139a25du); std::fill(y.begin(), y.end(), kFloatGuard);
        stream << ga.copy_from(luisa::span{a}) << gb.copy_from(luisa::span{b}) << gx.copy_from(luisa::span{x}) << gy.copy_from(luisa::span{y})
               << shader(ga.view(kPad, count), gb.view(kPad, count), gx.view(kPad, count), gy.view(kPad, count)).dispatch()
               << ga.copy_to(luisa::span{a}) << gb.copy_to(luisa::span{b}) << gx.copy_to(luisa::span{x}) << gy.copy_to(luisa::span{y}) << synchronize();
        for (auto i = size_t{0}; i < a.size(); i++) {
            expect(a[i] == original_a[i]); expect(bits(b[i]) == bits(original_b[i]));
            auto in_bounds = i >= kPad && i < kPad + count;
            expect(x[i] == (in_bounds ? bits(original_b[i]) : 0x7139a25du));
            expect(bits(y[i]) == (in_bounds ? original_a[i] : bits(kFloatGuard)));
        }
    }
}

void run(const Route &route) {
    "tile_primitives_bitcast_storage"_test = [&] { bitcast_storage(route); };
    "tile_primitives_packed_fp32"_test = [&] {
        for (auto width : {int64_t{1}, int64_t{2}, int64_t{32}, int64_t{128}}) {
            for (auto descending : {false, true}) { ranking(route, width, width, descending, false, tile::SortAlgorithm::PACKED_FP32); }
        }
        for (auto descending : {false, true}) {
            ranking(route, 32, 32, descending, true, tile::SortAlgorithm::PACKED_FP32);
            ranking(route, 32, 16, descending, true, tile::SortAlgorithm::PACKED_FP32);
        }
    };
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
        // Rejected captures must not invoke rank_kernel's positive axis and
        // store assertions on the deliberately empty RankedTile result.
        auto rejected_packed = [](int64_t width, uint64_t count) {
            return tile_kernel("rejected_packed_shape", [=](TensorView<const float, 1> x) {
                       auto n = axis("n", width);
                       static_cast<void>(topk(x.tile(coord(0), shape(n)).load(), n, count, true, SortAlgorithm::PACKED_FP32));
                   }).capture(tensor_shape(width));
        };
        auto expect_packed_rejected = [](const tile::Kernel &kernel) {
            expect(!kernel.valid());
            expect(std::any_of(kernel.diagnostics().begin(), kernel.diagnostics().end(), [](auto &&message) {
                return message.find("packed sort requires float32, a power-of-two axis and N <= 2^31") != string::npos;
            }));
        };
        expect_packed_rejected(rejected_packed(33, 33u));
        expect_packed_rejected(rejected_packed(int64_t{1} << 32u, 1u));
        auto wrong_type = tile_kernel("packed_sort_requires_float", [](TensorView<const uint32_t, 1> x) {
                              auto n = axis("n", 16);
                              static_cast<void>(tile::sort(x.tile(coord(0), shape(n)).load(), n, true, SortAlgorithm::PACKED_FP32));
                          }).capture(tensor_shape(16));
        expect(!wrong_type.valid());
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
