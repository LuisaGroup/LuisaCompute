// End-to-end native Tile IR tests. Every positive shader is captured by the
// Luisa Tile DSL and compiled through tile::compile; no embedded CUDA source.
#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_xir_test_utils.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <luisa/core/platform.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/stream.h>

#ifndef LUISA_TEST_CUDA_TILE_IR_ENABLED
#define LUISA_TEST_CUDA_TILE_IR_ENABLED 0
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto kRows = 31u;
constexpr auto kColumns = 37u;
constexpr auto kTerms = 19u;
constexpr auto kPad = 13u;
constexpr auto kGuard = -12345.25f;
constexpr auto kInitial = 0.25f;

[[nodiscard]] uint32_t bits(float value) noexcept { return std::bit_cast<uint32_t>(value); }

struct GuardedBuffer {
    size_t count;
    luisa::vector<float> host;
    Buffer<float> buffer;
    GuardedBuffer(Device &device, size_t n)
        : count{n}, host(n + 2u * kPad, kGuard), buffer{device.create_buffer<float>(host.size())} {}
    [[nodiscard]] auto view() const noexcept { return buffer.view(kPad, count); }
    [[nodiscard]] float &operator[](size_t i) noexcept { return host[kPad + i]; }
    [[nodiscard]] float operator[](size_t i) const noexcept { return host[kPad + i]; }
    void poison() noexcept { std::fill_n(host.begin() + kPad, count, std::numeric_limits<float>::quiet_NaN()); }
    void check_guards() const noexcept {
        for (auto i = 0u; i < kPad; i++) {
            expect(bits(host[i]) == bits(kGuard)) << "prefix guard=" << i;
            expect(bits(host[kPad + count + i]) == bits(kGuard)) << "suffix guard=" << i;
        }
    }
};

[[nodiscard]] float random_float(uint32_t &state) noexcept {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return static_cast<float>(state >> 8u) * 0x1p-23f - 1.0f;
}

void check_readonly(const GuardedBuffer &buffer, luisa::span<const float> previous) noexcept {
    expect(buffer.host.size() == previous.size());
    if (buffer.host.size() != previous.size()) { return; }
    for (auto i = 0u; i < previous.size(); i++) { expect(bits(buffer.host[i]) == bits(previous[i])) << "readonly word=" << i; }
}

[[nodiscard]] bool check_native(const tile::Shader &shader, uint3 grid, size_t argument_count) noexcept {
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return false; }
    auto &&metadata = shader.metadata();
    expect(metadata.realization.starts_with("CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin; no cache")) << metadata.realization;
    expect(metadata.source.find("__tile_global__") != luisa::string::npos);
    expect(metadata.source.find("luisa_tile_main") != luisa::string::npos);
    expect(all(metadata.dispatch_size == grid));
    expect(all(shader.block_size() == make_uint3(1u)));
    expect(metadata.arguments.size() == argument_count);
    expect(!metadata.disjoint_writes);
    return true;
}

[[nodiscard]] tile::Kernel matrix(bool transpose_a, bool transpose_b, bool reassociate) {
    using namespace tile;
    return tile_kernel("unrelated_contraction_name", [=](TensorView<const float, 2> a,
                                                        TensorView<const float, 2> b,
                                                        TensorView<float, 2> c) {
               auto gm = axis("gm", 2), gn = axis("gn", 3);
               auto m = axis("m", 16), n = axis("n", 16), k = axis("k", 8);
               for (auto &nest : parallel(shape(gm, gn))) {
                   auto m0 = nest.index(gm) * 16, n0 = nest.index(gn) * 16;
                   auto accumulator = full<float>(shape(m, n), kInitial);
                   for (auto &step : nest.pipeline(shape(3), {.window = 2u, .interval = 1u})) {
                       step.stage("load");
                       auto k0 = step.index() * 8;
                       auto lhs = transpose_a ? a.tile(coord(k0, m0), shape(k, m)).load() : a.tile(coord(m0, k0), shape(m, k)).load();
                       auto rhs = transpose_b ? b.tile(coord(n0, k0), shape(n, k)).load() : b.tile(coord(k0, n0), shape(k, n)).load();
                       step.stage("contract");
                       accumulator = mma(lhs, rhs, accumulator, MmaPolicy{reassociate});
                   }
                   c(coord(m0, n0), shape(m, n)).store(accumulator);
               }
           })
        .capture(tensor_shape(transpose_a ? kTerms : kRows, transpose_a ? kRows : kTerms),
                 tensor_shape(transpose_b ? kColumns : kTerms, transpose_b ? kTerms : kColumns),
                 tensor_shape(kRows, kColumns));
}

void matrix_oracle(Device &device, bool transpose_a, bool transpose_b, bool reassociate) {
    auto kernel = matrix(transpose_a, transpose_b, reassociate);
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 3u)) { return; }
    auto &&arguments = shader.metadata().arguments;
    if (arguments.size() != 3u) { return; }
    for (auto i = 0u; i < 3u; i++) { expect(arguments[i].element == tile::ScalarType::FLOAT32); }
    expect(arguments[0].minimum_size_bytes == kRows * kTerms * sizeof(float));
    expect(arguments[1].minimum_size_bytes == kTerms * kColumns * sizeof(float));
    expect(arguments[2].minimum_size_bytes == kRows * kColumns * sizeof(float));
    expect(arguments[0].usage == Usage::READ && arguments[1].usage == Usage::READ && arguments[2].usage == Usage::WRITE);
    GuardedBuffer a{device, kRows * kTerms}, b{device, kTerms * kColumns}, c{device, kRows * kColumns};
    auto a_index = [=](uint32_t m, uint32_t k) noexcept { return transpose_a ? k * kRows + m : m * kTerms + k; };
    auto b_index = [=](uint32_t k, uint32_t n) noexcept { return transpose_b ? n * kTerms + k : k * kColumns + n; };
    auto stream = device.create_stream(StreamTag::COMPUTE);
    for (auto flavor = 0u; flavor < 3u; flavor++) {
        auto state = uint32_t{0x61c88647u};
        for (auto i = 0u; i < a.count; i++) { a[i] = random_float(state); }
        for (auto i = 0u; i < b.count; i++) { b[i] = random_float(state); }
        if (flavor == 1u) {
            for (auto k = 0u; k + 1u < kTerms; k += 2u) {
                for (auto m = 0u; m < kRows; m++) { a[a_index(m, k + 1u)] = -a[a_index(m, k)] + 0x1p-20f; }
                for (auto n = 0u; n < kColumns; n++) { b[b_index(k + 1u, n)] = b[b_index(k, n)]; }
            }
        } else if (flavor == 2u) {
            // A contraction-order witness: ((.25 + 2^24) + 1) - 2^24 is
            // sensitive to grouping. Every row/column still uses a real Tile.
            for (auto m = 0u; m < kRows; m++) {
                for (auto k = 0u; k < kTerms; k++) { a[a_index(m, k)] = k == 0u ? 0x1p24f : k == 1u ? 1.0f : k == 2u ? -0x1p24f : 0.0f; }
            }
            for (auto i = 0u; i < b.count; i++) { b[i] = 1.0f; }
        }
        auto original_a = a.host, original_b = b.host;
        c.poison();
        stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
               << c.buffer.copy_from(luisa::span{c.host}) << shader(a.view(), b.view(), c.view()).dispatch()
               << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
               << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
        check_readonly(a, original_a);
        check_readonly(b, original_b);
        c.check_guards();
        constexpr auto unit = 0x1p-24;
        constexpr auto operations = 2.0 * (kTerms + 1u);
        constexpr auto gamma = operations * unit / (1.0 - operations * unit);
        for (auto m = 0u; m < kRows; m++) {
            for (auto n = 0u; n < kColumns; n++) {
                auto expected = static_cast<double>(kInitial), magnitude = static_cast<double>(kInitial);
                auto ordered = kInitial;
                for (auto k = 0u; k < kTerms; k++) {
                    auto lhs = a[a_index(m, k)], rhs = b[b_index(k, n)];
                    auto product = static_cast<double>(lhs) * static_cast<double>(rhs);
                    expected += product;
                    magnitude += std::abs(product);
                    ordered = std::fma(lhs, rhs, ordered);
                }
                auto actual = c[m * kColumns + n];
                auto bound = magnitude * (gamma + 8.0 * (kTerms + 1u) * 0x1p-53);
                expect(std::isfinite(actual) && std::abs(static_cast<double>(actual) - expected) <= bound)
                    << "full FP64 oracle m=" << m << " n=" << n << " flavor=" << flavor;
                // MmaPolicy(false) retains ascending-K fused contraction order.
                if (!reassociate) { expect(bits(actual) == bits(ordered)) << "ordered contraction m=" << m << " n=" << n << " flavor=" << flavor; }
            }
        }
    }
}

[[nodiscard]] tile::Kernel pointwise() {
    using namespace tile;
    return tile_kernel("scalar_branch_broadcast", [](TensorView<const float, 2> a,
                                                     TensorView<const float, 2> b,
                                                     TensorView<float, 2> c) {
               auto gm = axis("gm", 2), gn = axis("gn", 3), m = axis("m", 16), n = axis("n", 16);
               for (auto &nest : parallel(shape(gm, gn))) {
                   auto origin = coord(nest.index(gm) * 16, nest.index(gn) * 16);
                   auto domain = shape(m, n);
                   auto x = a.tile(origin, domain).load(), y = b.tile(origin, domain).load();
                   c(origin, domain).store(ite(x > 0.0f, 2.0f, -2.0f) * x + y);
               }
           })
        .capture(tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns));
}

void pointwise_alias(Device &device) {
    auto kernel = pointwise();
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 3u)) { return; }
    GuardedBuffer a{device, kRows * kColumns}, b{device, kRows * kColumns};
    auto state = uint32_t{0x9e3779b9u};
    for (auto i = 0u; i < a.count; i++) { a[i] = random_float(state); b[i] = random_float(state); }
    auto original_a = a.host, original_b = b.host;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
           << shader(a.view(), b.view(), a.view()).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host}) << synchronize();
    a.check_guards();
    check_readonly(b, original_b);
    for (auto i = 0u; i < a.count; i++) {
        auto x = original_a[kPad + i], y = original_b[kPad + i];
        auto expected = (x > 0.0f ? 2.0 : -2.0) * static_cast<double>(x) + static_cast<double>(y);
        auto bound = (2.0 * std::abs(static_cast<double>(x)) + std::abs(static_cast<double>(y))) * 0x1p-21;
        expect(std::isfinite(a[i]) && std::abs(static_cast<double>(a[i]) - expected) <= bound) << "alias output=" << i;
    }
}

void snapshot_ordering(Device &device) {
    using namespace tile;
    auto kernel = tile_kernel("load_store_snapshot", [](TensorView<float, 2> a,
                                                        TensorView<float, 2> c) {
                      auto gm = axis("gm", 2), gn = axis("gn", 3), m = axis("m", 16), n = axis("n", 16);
                      for (auto &nest : parallel(shape(gm, gn))) {
                          auto origin = coord(nest.index(gm) * 16, nest.index(gn) * 16);
                          auto domain = shape(m, n);
                          auto old_value = a.tile(origin, domain).load();
                          a(origin, domain).store(old_value + 3.0f);
                          auto new_value = a.tile(origin, domain).load();
                          c(origin, domain).store(old_value * 2.0f + new_value);
                      }
                  }).capture(tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 2u)) { return; }
    if (shader.metadata().arguments.size() != 2u) { return; }
    expect(shader.metadata().arguments[0].usage == Usage::READ_WRITE);
    expect(shader.metadata().arguments[1].usage == Usage::WRITE);
    GuardedBuffer a{device, kRows * kColumns}, c{device, kRows * kColumns};
    auto state = uint32_t{7654321u};
    for (auto i = 0u; i < a.count; i++) { a[i] = random_float(state); }
    auto original = a.host;
    c.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << c.buffer.copy_from(luisa::span{c.host})
           << shader(a.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
           << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
    a.check_guards();
    c.check_guards();
    for (auto i = 0u; i < a.count; i++) {
        auto before = original[kPad + i];
        auto after = static_cast<float>(static_cast<double>(before) + 3.0);
        expect(bits(a[i]) == bits(after)) << "updated input=" << i;
        auto expected = static_cast<double>(before) * 2.0 + static_cast<double>(after);
        auto bound = (2.0 * std::abs(static_cast<double>(before)) + std::abs(static_cast<double>(after))) * 0x1p-22;
        expect(std::isfinite(c[i]) && std::abs(static_cast<double>(c[i]) - expected) <= bound) << "snapshot output=" << i;
    }
}

[[nodiscard]] tile::Kernel shifted_copy() {
    using namespace tile;
    return tile_kernel("negative_view_origin", [](TensorView<const float, 2> a, TensorView<float, 2> c) {
               auto gm = axis("gm", 2), gn = axis("gn", 3), m = axis("m", 16), n = axis("n", 16);
               for (auto &nest : parallel(shape(gm, gn))) {
                   auto m0 = nest.index(gm) * 16, n0 = nest.index(gn) * 16;
                   auto domain = shape(m, n);
                   c(coord(m0, n0), domain).store(a.tile(coord(m0 - 3, n0 - 5), domain).load());
               }
           })
        .capture(tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns));
}

void shifted_bitwise_copy(Device &device) {
    auto kernel = shifted_copy();
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 2u)) { return; }
    GuardedBuffer a{device, kRows * kColumns}, c{device, kRows * kColumns};
    constexpr std::array patterns{0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                                 0x007fffffu, 0x00800000u, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x3f812345u};
    for (auto i = 0u; i < a.count; i++) { a[i] = std::bit_cast<float>(patterns[i % patterns.size()]); }
    auto original = a.host;
    c.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << c.buffer.copy_from(luisa::span{c.host})
           << shader(a.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
           << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
    check_readonly(a, original);
    c.check_guards();
    for (auto m = 0u; m < kRows; m++) {
        for (auto n = 0u; n < kColumns; n++) {
            auto expected = m >= 3u && n >= 5u ? bits(a[(m - 3u) * kColumns + n - 5u]) : 0u;
            expect(bits(c[m * kColumns + n]) == expected) << "masked bitwise copy m=" << m << " n=" << n;
        }
    }
}

[[nodiscard]] tile::Kernel swap_kernel() {
    using namespace tile;
    return tile_kernel("temporal_permutation", [](TensorView<const float, 2> a, TensorView<const float, 2> b,
                                                TensorView<float, 2> c, TensorView<float, 2> d) {
               auto gm = axis("gm", 2), gn = axis("gn", 3), m = axis("m", 16), n = axis("n", 16);
               for (auto &nest : parallel(shape(gm, gn))) {
                   auto origin = coord(nest.index(gm) * 16, nest.index(gn) * 16);
                   auto domain = shape(m, n);
                   auto x = a.tile(origin, domain).load(), y = b.tile(origin, domain).load();
                   for (auto &step : nest.serial(shape(3))) {
                       static_cast<void>(step);
                       auto previous = x;
                       x = y;
                       y = previous;
                   }
                   c(origin, domain).store(x);
                   d(origin, domain).store(y);
               }
           })
        .capture(tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns), tensor_shape(kRows, kColumns));
}

void swap_oracle(Device &device) {
    auto kernel = swap_kernel();
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 4u)) { return; }
    GuardedBuffer a{device, kRows * kColumns}, b{device, kRows * kColumns}, c{device, kRows * kColumns}, d{device, kRows * kColumns};
    auto state = uint32_t{1234567u};
    for (auto i = 0u; i < a.count; i++) { a[i] = random_float(state); b[i] = random_float(state); }
    auto original_a = a.host, original_b = b.host;
    c.poison();
    d.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
           << c.buffer.copy_from(luisa::span{c.host}) << d.buffer.copy_from(luisa::span{d.host})
           << shader(a.view(), b.view(), c.view(), d.view()).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
           << c.buffer.copy_to(luisa::span{c.host}) << d.buffer.copy_to(luisa::span{d.host}) << synchronize();
    check_readonly(a, original_a);
    check_readonly(b, original_b);
    c.check_guards();
    d.check_guards();
    for (auto i = 0u; i < a.count; i++) {
        expect(bits(c[i]) == bits(b[i])) << "first carry=" << i;
        expect(bits(d[i]) == bits(a[i])) << "second carry=" << i;
    }
}

void expect_rejected(const tile::Shader &shader, luisa::string_view reason) noexcept {
    expect(!shader);
    expect(!shader.metadata().error.empty());
    expect(shader.metadata().error.find(reason) != luisa::string::npos) << shader.metadata().error;
}

void rejected_options(Device &device) {
    auto kernel = pointwise();
    tile::CompileOptions tile_options;
    tile_options.threads_per_group = 1u;
    expect_rejected(tile::compile(device, kernel, tile_options), "threads_per_group");
    ShaderOption option{.enable_fast_math = false};
    option.max_registers = 64u;
    expect_rejected(tile::compile(device, kernel, {}, option), "register");
    option.max_registers = 0u;
    option.enable_fast_math = true;
    expect_rejected(tile::compile(device, kernel, {}, option), "fast_math");
    option.enable_fast_math = false;
    option.compile_only = true;
    expect_rejected(tile::compile(device, kernel, {}, option), "compile-only");
    option.compile_only = false;
    option.name = "native_tile_archive_not_implemented";
    expect_rejected(tile::compile(device, kernel, {}, option), "name");
    option.name.clear();
    option.native_include = "// unsupported native source";
    expect_rejected(tile::compile(device, kernel, {}, option), "native_include");
}

void rejected_shapes_and_constraints(Device &device) {
    ShaderOption strict{.enable_fast_math = false};
    auto non_power_two = test::tile_xir::gemm({.m = 31, .n = 37, .k = 19, .bm = 3, .bn = 16, .bk = 8});
    expect_rejected(tile::compile(device, non_power_two, {}, strict), "powers of two");
    auto constrained = pointwise();
    for (auto op : constrained.function().body().block(0u)->operations()) {
        if (op->kind() == tile::OperationKind::PARALLEL) { op->set_execution_scope_constraint("warp"); }
    }
    expect_rejected(tile::compile(device, constrained, {}, strict), "constraints");
    using namespace tile;
    auto unsupported = tile_kernel("explicit_double_buffer", [](TensorView<const double, 1> a, TensorView<double, 1> b) {
                           for (auto &nest : parallel(shape(1))) {
                               static_cast<void>(nest);
                               b(coord(0), shape(16)).store(a.tile(coord(0), shape(16)).load());
                           }
                       }).capture(tensor_shape(16), tensor_shape(16));
    expect_rejected(tile::compile(device, unsupported, {}, strict), "FP32");
}

}// namespace

int main(int argc, char *argv[]) {
    const char *executable = "test_tile_cuda_ir";
    if (argv != nullptr) {
        if (argc > 0) {
            if (argv[0] != nullptr) { executable = argv[0]; }
        }
    }
    auto usage = [&] {
        LUISA_INFO("Usage: {} cuda <--require-native|--expect-disabled|--expect-unavailable> [UT filter]", executable);
        return 2;
    };
    if (argc < 3) { return usage(); }
    if (argv == nullptr) { return usage(); }
    if (argv[1] == nullptr) { return usage(); }
    if (argv[2] == nullptr) { return usage(); }
    if (std::strcmp(argv[1], "cuda") != 0) { return usage(); }
    enum struct Mode { NATIVE, DISABLED, UNAVAILABLE };
    Mode mode;
    if (std::strcmp(argv[2], "--require-native") == 0) { mode = Mode::NATIVE; }
    else if (std::strcmp(argv[2], "--expect-disabled") == 0) { mode = Mode::DISABLED; }
    else if (std::strcmp(argv[2], "--expect-unavailable") == 0) { mode = Mode::UNAVAILABLE; }
    else { return usage(); }
    auto environment = luisa::get_environment_variable("LUISA_CUDA_TILE_IR");
    auto enabled = false;
    if (environment) { enabled = std::strcmp(environment->c_str(), "1") == 0; }
    if (enabled == (mode == Mode::DISABLED)) {
        LUISA_INFO("CLI mode disagrees with LUISA_CUDA_TILE_IR; refusing an ambiguous test.");
        return 2;
    }
#if LUISA_TEST_CUDA_TILE_IR_ENABLED
    if (mode == Mode::UNAVAILABLE) { return usage(); }
#else
    if (mode == Mode::NATIVE) {
        LUISA_INFO("This test target has no native CUDA Tile IR helper; positive verification cannot be skipped.");
        return 2;
    }
#endif
    std::vector<const char *> ut_arguments{argv[0]};
    for (auto i = 3; i < argc; i++) {
        if (argv[i] == nullptr) { return usage(); }
        ut_arguments.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(ut_arguments.size()), ut_arguments.data());
    auto [context, device] = test::create_device(argc, argv);
    if (mode != Mode::NATIVE) {
        "tile_cuda_ir_capability_guard"_test = [&] {
            auto kernel = pointwise();
            auto shader = tile::compile(device, kernel);
            expect_rejected(shader, mode == Mode::DISABLED ? "LUISA_CUDA_TILE_IR" : "CUDA Tile IR");
        };
        return 0;
    }
    "tile_cuda_ir_ragged_transposed_gemm"_test = [&] {
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) { matrix_oracle(device, ta, tb, true); }
        }
    };
    "tile_cuda_ir_ordered_contraction"_test = [&] { matrix_oracle(device, false, false, false); };
    "tile_cuda_ir_pointwise_exact_alias"_test = [&] { pointwise_alias(device); };
    "tile_cuda_ir_read_before_write_snapshot"_test = [&] { snapshot_ordering(device); };
    "tile_cuda_ir_negative_origin_bitwise_copy"_test = [&] { shifted_bitwise_copy(device); };
    "tile_cuda_ir_simultaneous_carry_swap"_test = [&] { swap_oracle(device); };
    "tile_cuda_ir_rejects_unsupported_options"_test = [&] { rejected_options(device); };
    "tile_cuda_ir_rejects_shapes_constraints_and_types"_test = [&] { rejected_shapes_and_constraints(device); };
    return 0;
}
