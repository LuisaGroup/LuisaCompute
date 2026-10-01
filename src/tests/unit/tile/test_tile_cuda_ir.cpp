// End-to-end native Tile IR tests. Every positive shader is captured by the
// Luisa Tile DSL and compiled through tile::compile; no embedded CUDA source.
#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_xir_test_utils.h"
#include "tile_llm_test_utils.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
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

void row_operations(Device &device) {
    using test::tile_llm::RowOp;
    for (auto op : {RowOp::RMS_NORM, RowOp::LAYER_NORM, RowOp::SWIGLU, RowOp::ROPE, RowOp::MASKED_SOFTMAX, RowOp::GELU_RESIDUAL}) {
        auto fixture = test::tile_llm::rows(op, 7, 32);
        auto shader = tile::compile(device, fixture.kernel);
        if (!check_native(shader, make_uint3(7u, 1u, 1u), 4u)) { continue; }
        if (op == RowOp::RMS_NORM || op == RowOp::LAYER_NORM || op == RowOp::MASKED_SOFTMAX) {
            expect(shader.metadata().source.find("ct::sum(") != luisa::string::npos);
        }
        if (op == RowOp::MASKED_SOFTMAX) { expect(shader.metadata().source.find("ct::reduce_max(") != luisa::string::npos); }
        GuardedBuffer x{device, fixture.inputs[0].size()}, u{device, fixture.inputs[1].size()},
            v{device, fixture.inputs[2].size()}, y{device, fixture.expected.size()};
        for (auto i = 0u; i < x.count; i++) { x[i] = fixture.inputs[0][i]; }
        for (auto i = 0u; i < u.count; i++) { u[i] = fixture.inputs[1][i]; }
        for (auto i = 0u; i < v.count; i++) { v[i] = fixture.inputs[2][i]; }
        y.poison();
        auto original_x = x.host, original_u = u.host, original_v = v.host;
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << x.buffer.copy_from(luisa::span{x.host}) << u.buffer.copy_from(luisa::span{u.host})
               << v.buffer.copy_from(luisa::span{v.host}) << y.buffer.copy_from(luisa::span{y.host})
               << shader(x.view(), u.view(), v.view(), y.view()).dispatch()
               << x.buffer.copy_to(luisa::span{x.host}) << u.buffer.copy_to(luisa::span{u.host})
               << v.buffer.copy_to(luisa::span{v.host}) << y.buffer.copy_to(luisa::span{y.host}) << synchronize();
        check_readonly(x, original_x);
        check_readonly(u, original_u);
        check_readonly(v, original_v);
        y.check_guards();
        for (auto i = 0u; i < y.count; i++) {
            expect(std::isfinite(y[i]));
            auto error = std::abs(static_cast<double>(y[i]) - fixture.expected[i]);
            expect(error <= 2e-5 * (1.0 + std::abs(fixture.expected[i]))) << "op=" << static_cast<int>(op) << " index=" << i << " error=" << error;
        }
    }
}

void named_axis_broadcast(Device &device) {
    using namespace tile;
    auto kernel = tile_kernel("permuted_named_dimensions", [](TensorView<const float, 2> a, TensorView<float, 2> b) {
                      auto m = axis("m", 4), n = axis("n", 8);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0, 0), shape(n, m)).load();
                          auto total = reduce(x, shape(n, m), add);
                          auto y = full<float>(shape(m, n), 2.0f) + x + cast<float>(iota(n));
                          auto mapped = map<float>(shape(m, n), [&](const Nest &index) {
                              auto value = y.at(index);
                              return ite(index.index(m) < 2, value, Scalar<float>{-3.0f});
                          });
                          b(coord(0, 0), shape(m, n)).store(mapped + total);
                      }
                  }).capture(tensor_shape(8, 4), tensor_shape(4, 8));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 2u)) { return; }
    expect(shader.metadata().source.find("ct::permute(") != luisa::string::npos);
    expect(shader.metadata().source.find("ct::shape<>") != luisa::string::npos);
    GuardedBuffer a{device, 32u}, b{device, 32u};
    auto total = 0.0f;
    for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>(i) * .125f - 2.0f; total += a[i]; }
    auto original = a.host;
    b.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
           << shader(a.view(), b.view()).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host}) << synchronize();
    check_readonly(a, original);
    b.check_guards();
    for (auto m = 0u; m < 4u; m++) {
        for (auto n = 0u; n < 8u; n++) {
            auto expected = (m < 2u ? 2.0f + a[n * 4u + m] + static_cast<float>(n) : -3.0f) + total;
            expect(bits(b[m * 8u + n]) == bits(expected)) << "m=" << m << " n=" << n;
        }
    }
}

struct WeightedFold {
    template<typename T>
    [[nodiscard]] static constexpr T identity() noexcept { return T{1}; }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return a * .5f + b; }
};

void ordered_reductions(Device &device) {
    using namespace tile;
    auto kernel = tile_kernel("ordered_non_associative_reducers", [](TensorView<const float, 2> a, TensorView<float, 2> b) {
                      auto m = axis("m", 4), n = axis("n", 8);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0, 0), shape(m, n)).load();
                          auto left = reduce(x, n, WeightedFold{}, reduction::fold_left);
                          auto right = reduce(x, n, WeightedFold{}, reduction::fold_right);
                          auto ordered = reduce(x, n, WeightedFold{}, reduction::ordered_tree);
                          b(coord(0, 0), shape(m, n)).store(left + right * 16.0f + ordered * 256.0f);
                      }
                  }).capture(tensor_shape(4, 8), tensor_shape(4, 8));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 2u)) { return; }
    expect(shader.metadata().source.find("ct::sum(") == luisa::string::npos);
    expect(shader.metadata().source.find("-- > 0ll") != luisa::string::npos);
    GuardedBuffer a{device, 32u}, b{device, 32u};
    for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>(i % 11u) * .125f; }
    auto original = a.host;
    b.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
           << shader(a.view(), b.view()).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host}) << synchronize();
    check_readonly(a, original);
    b.check_guards();
    for (auto m = 0u; m < 4u; m++) {
        auto left = 1.0f, right = 1.0f;
        for (auto n = 0u; n < 8u; n++) { left = left * .5f + a[m * 8u + n]; }
        for (auto n = 8u; n-- > 0u;) { right = a[m * 8u + n] * .5f + right; }
        auto expected = left + right * 16.0f + left * 256.0f;
        for (auto n = 0u; n < 8u; n++) { expect(bits(b[m * 8u + n]) == bits(expected)) << "row=" << m; }
    }
}

void elementary_math(Device &device) {
    using namespace tile;
    auto kernel = tile_kernel("explicit_fp32_math", [](TensorView<const float, 1> a, TensorView<float, 1> b) {
                      auto n = axis("n", 16);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0), shape(n)).load();
                          b(coord(0), shape(n)).store(abs(x));
                          b(coord(16), shape(n)).store(tile::log(abs(x)));
                          b(coord(32), shape(n)).store(sqrt(abs(x)));
                          b(coord(48), shape(n)).store(exp(x));
                          b(coord(64), shape(n)).store(tanh(x));
                          b(coord(80), shape(n)).store(min(x, 0.25f));
                          b(coord(96), shape(n)).store(max(x, -0.25f));
                          b(coord(112), shape(n)).store(broadcast_to(reduce(x, n, minimum), shape(n)));
                          b(coord(128), shape(n)).store(broadcast_to(reduce(x, n, maximum), shape(n)));
                      }
                  }).capture(tensor_shape(16), tensor_shape(144));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 2u)) { return; }
    expect(shader.metadata().source.find("ct::reduce_min(") != luisa::string::npos);
    GuardedBuffer a{device, 16u}, b{device, 144u};
    std::array<float, 16u> values{0.0f, -0.0f, 0x1p-149f, -0x1p-149f, 0x1p-126f, -0x1p-126f,
                                .125f, -.125f, 1.0f, -1.0f, 8.0f, -8.0f,
                                std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN(), .25f};
    for (auto i = 0u; i < a.count; i++) { a[i] = values[i]; }
    auto original = a.host;
    b.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
           << shader(a.view(), b.view()).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host}) << synchronize();
    check_readonly(a, original);
    b.check_guards();
    for (auto i = 0u; i < a.count; i++) {
        auto x = static_cast<double>(a[i]);
        std::array<double, 9u> expected{std::abs(x), std::log(std::abs(x)), std::sqrt(std::abs(x)), std::exp(x),
                                      std::tanh(x), std::fmin(x, .25), std::fmax(x, -.25),
                                      -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        for (auto op = 0u; op < expected.size(); op++) {
            auto actual = b[op * 16u + i];
            if (std::isnan(expected[op])) { expect(std::isnan(actual)) << "math op=" << op << " index=" << i; }
            else if (std::isinf(expected[op])) { expect(actual == expected[op]) << "math op=" << op << " index=" << i; }
            else if (op == 0u || op == 5u || op == 6u) { expect(bits(actual) == bits(static_cast<float>(expected[op]))) << "math op=" << op << " index=" << i; }
            else {
                expect(std::isfinite(actual));
                auto error = std::abs(static_cast<double>(actual) - expected[op]);
                expect(error <= 2e-6 * std::abs(expected[op]) + 1e-44) << "math op=" << op << " index=" << i << " error=" << error;
            }
        }
    }
}

template<typename T>
void typed_buffer_copy(Device &device) {
    using namespace tile;
    constexpr auto count = 19u;
    auto kernel = tile_kernel("typed_views_and_negative_origin", [](TensorView<const T, 1> a, TensorView<T, 1> b) {
                      auto n = axis("n", 32);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          b(coord(0), shape(n)).store(a.tile(coord(-3), shape(n)).load());
                      }
                  }).capture(tensor_shape(count), tensor_shape(count));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 2u)) { return; }
    for (auto &&argument : shader.metadata().arguments) {
        expect(argument.element == scalar_type_v<T>);
        expect(argument.minimum_size_bytes == count * sizeof(T));
    }
    expect(shader.metadata().arguments[0].usage == Usage::READ);
    expect(shader.metadata().arguments[1].usage == Usage::WRITE);
    std::array<T, count + 2u * kPad> a{}, b{};
    for (auto i = 0u; i < a.size(); i++) {
        if constexpr (std::is_same_v<T, bool>) { a[i] = i % 3u == 0u; b[i] = true; }
        else {
            using U = std::make_unsigned_t<T>;
            auto word = static_cast<U>(0x80b19d5e7f3c2a01ull ^ (static_cast<uint64_t>(i) * 0x1f45d78b9ull));
            a[i] = std::bit_cast<T>(word);
            b[i] = static_cast<T>(37);
        }
    }
    auto original_a = a, original_b = b;
    auto gpu_a = device.create_buffer<T>(a.size()), gpu_b = device.create_buffer<T>(b.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << gpu_a.copy_from(luisa::span{a}) << gpu_b.copy_from(luisa::span{b})
           << shader(gpu_a.view(kPad, count), gpu_b.view(kPad, count)).dispatch()
           << gpu_a.copy_to(luisa::span{a}) << gpu_b.copy_to(luisa::span{b}) << synchronize();
    for (auto i = 0u; i < a.size(); i++) {
        expect(a[i] == original_a[i]) << "readonly typed word=" << i;
        auto expected = i < kPad || i >= kPad + count ? original_b[i] : i < kPad + 3u ? T{} : original_a[i - 3u];
        expect(b[i] == expected) << "typed word=" << i;
    }
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
    auto gather = tile_kernel("lane_dependent_gather", [](TensorView<const float, 1> a, TensorView<float, 1> b) {
                      auto n = axis("n", 16);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0), shape(n)).load();
                          auto shifted = map<float>(shape(n), [&](const Nest &index) {
                              Scalar<int64_t> coordinates[]{(index.index(n) + 1) % 16};
                              return x.at(coordinates);
                          });
                          b(coord(0), shape(n)).store(shifted);
                      }
                  }).capture(tensor_shape(16), tensor_shape(16));
    expect_rejected(tile::compile(device, gather, {}, strict), "lane-dependent gather");
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
    "tile_cuda_ir_llm_rows"_test = [&] { row_operations(device); };
    "tile_cuda_ir_named_axis_broadcast"_test = [&] { named_axis_broadcast(device); };
    "tile_cuda_ir_ordered_reductions"_test = [&] { ordered_reductions(device); };
    "tile_cuda_ir_elementary_math"_test = [&] { elementary_math(device); };
    "tile_cuda_ir_typed_buffer_views"_test = [&] {
        typed_buffer_copy<bool>(device);
        typed_buffer_copy<int32_t>(device);
        typed_buffer_copy<uint32_t>(device);
        typed_buffer_copy<int64_t>(device);
        typed_buffer_copy<uint64_t>(device);
    };
    "tile_cuda_ir_rejects_unsupported_options"_test = [&] { rejected_options(device); };
    "tile_cuda_ir_rejects_shapes_constraints_and_types"_test = [&] { rejected_shapes_and_constraints(device); };
    return 0;
}
