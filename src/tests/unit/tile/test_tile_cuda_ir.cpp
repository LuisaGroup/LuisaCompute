// End-to-end native Tile IR tests. Every positive shader is captured by the
// Luisa Tile DSL and compiled through tile::compile; no embedded CUDA source.
#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_xir_test_utils.h"
#include "tile_llm_test_utils.h"
#include "tile_selection_test_utils.h"
#include "tile_argmax_test_utils.h"
#include "tile_embedding_test_utils.h"
#include "tile_sort_pipeline_test_utils.h"
#include "tile_workload_test_utils.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include <luisa/core/platform.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/algorithms.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

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

[[nodiscard]] tile::Kernel matrix(bool transpose_a, bool transpose_b, bool reassociate,
                                 uint32_t tile_k = 8u, uint32_t terms = kTerms) {
    using namespace tile;
    return tile_kernel("unrelated_contraction_name", [=](TensorView<const float, 2> a,
                                                        TensorView<const float, 2> b,
                                                        TensorView<float, 2> c) {
               auto gm = axis("gm", 2), gn = axis("gn", 3);
               auto m = axis("m", 16), n = axis("n", 16), k = axis("k", tile_k);
               for (auto &nest : parallel(shape(gm, gn))) {
                   auto m0 = nest.index(gm) * 16, n0 = nest.index(gn) * 16;
                   auto accumulator = full<float>(shape(m, n), kInitial);
                   for (auto &step : nest.pipeline(shape((terms + tile_k - 1u) / tile_k), {.window = 2u, .interval = 1u})) {
                       step.stage("load");
                       auto k0 = step.index() * tile_k;
                       auto lhs = transpose_a ? a.tile(coord(k0, m0), shape(k, m)).load() : a.tile(coord(m0, k0), shape(m, k)).load();
                       auto rhs = transpose_b ? b.tile(coord(n0, k0), shape(n, k)).load() : b.tile(coord(k0, n0), shape(k, n)).load();
                       step.stage("contract");
                       accumulator = mma(lhs, rhs, accumulator, MmaPolicy{reassociate});
                   }
                   c(coord(m0, n0), shape(m, n)).store(accumulator);
               }
           })
        .capture(tensor_shape(transpose_a ? terms : kRows, transpose_a ? kRows : terms),
                 tensor_shape(transpose_b ? kColumns : terms, transpose_b ? terms : kColumns),
                 tensor_shape(kRows, kColumns));
}

void matrix_oracle(Device &device, bool transpose_a, bool transpose_b, bool reassociate,
                   uint32_t tile_k = 8u, uint32_t terms = kTerms) {
    auto kernel = matrix(transpose_a, transpose_b, reassociate, tile_k, terms);
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(2u, 3u, 1u), 3u)) { return; }
    expect((shader.metadata().source.find("ct::mma(") != luisa::string::npos) == reassociate);
    if (reassociate) { expect(shader.metadata().source.find("ct::fma(") == luisa::string::npos); }
    if (!reassociate && (tile_k == 32u || tile_k == 64u || tile_k == 128u || tile_k == 256u)) {
        // Validate the bounded source transform as well as its numerical result.
        // At least one valid contribution follows the first contraction block.
        auto &&source = shader.metadata().source;
        auto count = size_t{0u}, position = size_t{0u};
        constexpr luisa::string_view call = "ct::fma(";
        while ((position = source.find(call, position)) != luisa::string::npos) {
            count++;
            position += call.size();
        }
        expect(count == (tile_k <= 128u ? tile_k : 1u)) << "contraction BK=" << tile_k;
        auto dynamic = source.find("for (unsigned mma") != luisa::string::npos;
        expect(dynamic == (tile_k > 128u)) << "dynamic contraction BK=" << tile_k;
    }
    auto &&arguments = shader.metadata().arguments;
    if (arguments.size() != 3u) { return; }
    for (auto i = 0u; i < 3u; i++) { expect(arguments[i].element == tile::ScalarType::FLOAT32); }
    expect(arguments[0].minimum_size_bytes == kRows * terms * sizeof(float));
    expect(arguments[1].minimum_size_bytes == terms * kColumns * sizeof(float));
    expect(arguments[2].minimum_size_bytes == kRows * kColumns * sizeof(float));
    expect(arguments[0].usage == Usage::READ && arguments[1].usage == Usage::READ && arguments[2].usage == Usage::WRITE);
    GuardedBuffer a{device, kRows * terms}, b{device, terms * kColumns}, c{device, kRows * kColumns};
    auto a_index = [=](uint32_t m, uint32_t k) noexcept { return transpose_a ? k * kRows + m : m * terms + k; };
    auto b_index = [=](uint32_t k, uint32_t n) noexcept { return transpose_b ? n * terms + k : k * kColumns + n; };
    auto stream = device.create_stream(StreamTag::COMPUTE);
    for (auto flavor = 0u; flavor < 3u; flavor++) {
        auto state = uint32_t{0x61c88647u};
        for (auto i = 0u; i < a.count; i++) { a[i] = random_float(state); }
        for (auto i = 0u; i < b.count; i++) { b[i] = random_float(state); }
        if (flavor == 1u) {
            for (auto k = 0u; k + 1u < terms; k += 2u) {
                for (auto m = 0u; m < kRows; m++) { a[a_index(m, k + 1u)] = -a[a_index(m, k)] + 0x1p-20f; }
                for (auto n = 0u; n < kColumns; n++) { b[b_index(k + 1u, n)] = b[b_index(k, n)]; }
            }
        } else if (flavor == 2u) {
            // A contraction-order witness: ((.25 + 2^24) + 1) - 2^24 is
            // sensitive to grouping. Every row/column still uses a real Tile.
            for (auto m = 0u; m < kRows; m++) {
                for (auto k = 0u; k < terms; k++) { a[a_index(m, k)] = k == 0u ? 0x1p24f : k == 1u ? 1.0f : k == 2u ? -0x1p24f : 0.0f; }
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
        const auto operations = 2.0 * (terms + 1u);
        const auto gamma = operations * unit / (1.0 - operations * unit);
        for (auto m = 0u; m < kRows; m++) {
            for (auto n = 0u; n < kColumns; n++) {
                auto expected = static_cast<double>(kInitial), magnitude = static_cast<double>(kInitial);
                auto ordered = kInitial;
                for (auto k = 0u; k < terms; k++) {
                    auto lhs = a[a_index(m, k)], rhs = b[b_index(k, n)];
                    auto product = static_cast<double>(lhs) * static_cast<double>(rhs);
                    expected += product;
                    magnitude += std::abs(product);
                    ordered = std::fma(lhs, rhs, ordered);
                }
                auto actual = c[m * kColumns + n];
                auto bound = magnitude * (gamma + 8.0 * (terms + 1u) * 0x1p-53);
                expect(std::isfinite(actual) && std::abs(static_cast<double>(actual) - expected) <= bound)
                    << "full FP64 oracle m=" << m << " n=" << n << " flavor=" << flavor;
                // MmaPolicy(false) retains ascending-K fused contraction order.
                if (!reassociate) { expect(bits(actual) == bits(ordered)) << "ordered contraction m=" << m << " n=" << n << " flavor=" << flavor; }
            }
        }
    }
}

// Same typed FP32 inputs for ordered and reassociated contractions. Exact
// identities expose precision loss/FTZ, while the FMA witness uses the original
// forward-error bound under reassociation instead of demanding the same bits.
void fp32_mma_precision_boundaries(Device &device) {
    using namespace tile;
    constexpr auto extent = 16u;
    for (auto reassociate : {false, true}) {
        auto kernel = tile_kernel("typed_contraction_precision_boundaries", [=](TensorView<const float, 2> a,
                                                                               TensorView<const float, 2> b,
                                                                               TensorView<const float, 2> initial,
                                                                               TensorView<float, 2> output) {
                          auto m = axis("m", extent), n = axis("n", extent), k = axis("k", extent);
                          for (auto &nest : parallel(shape(1))) {
                              static_cast<void>(nest);
                              auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                              auto rhs = b.tile(coord(0, 0), shape(k, n)).load();
                              auto seed = initial.tile(coord(0, 0), shape(m, n)).load();
                              output(coord(0, 0), shape(m, n)).store(mma(lhs, rhs, seed, MmaPolicy{reassociate}));
                          }
                      }).capture(tensor_shape(extent, extent), tensor_shape(extent, extent),
                                 tensor_shape(extent, extent), tensor_shape(extent, extent));
        auto shader = tile::compile(device, kernel);
        if (!check_native(shader, make_uint3(1u), 4u)) { continue; }
        expect((shader.metadata().source.find("ct::mma(") != luisa::string::npos) == reassociate);
        GuardedBuffer a{device, extent * extent}, b{device, extent * extent},
            initial{device, extent * extent}, output{device, extent * extent};
        auto stream = device.create_stream(StreamTag::COMPUTE);
        for (auto flavor = 0u; flavor < 7u; flavor++) {
            std::fill_n(a.host.begin() + kPad, a.count, 0.0f);
            std::fill_n(b.host.begin() + kPad, b.count, 1.0f);
            std::fill_n(initial.host.begin() + kPad, initial.count, flavor == 5u ? -1.0f : flavor == 3u ? 0x1p-145f : 0.0f);
            for (auto m = 0u; m < extent; m++) {
                switch (flavor) {
                    case 0u: a[m * extent] = 1.0f + 0x1p-23f; break;// low FP32 mantissa bits must survive
                    case 1u: a[m * extent] = 0x1p-146f; break;     // subnormal input, normal output
                    case 2u: a[m * extent] = 0x1p-126f; break;     // normal input, subnormal output
                    case 3u: break;                               // subnormal accumulator survives zero products
                    case 4u:
                        a[m * extent] = 0x1p20f;
                        a[m * extent + 1u] = 1.0f;
                        a[m * extent + 2u] = -0x1p20f;
                        break;// exact cancellation in every permitted grouping
                    case 5u: a[m * extent] = 1.0f + 0x1p-23f; break;// fused versus separately rounded product
                    case 6u: a[m * extent] = 0x1p100f; break;       // wide exponent range without overflow
                }
            }
            for (auto n = 0u; n < extent; n++) {
                b[n] = flavor == 1u ? 0x1p24f : flavor == 2u ? .5f : flavor == 5u ? 1.0f - 0x1p-23f : flavor == 6u ? 0x1p-80f : 1.0f;
            }
            auto original_a = a.host, original_b = b.host, original_initial = initial.host;
            output.poison();
            stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
                   << initial.buffer.copy_from(luisa::span{initial.host}) << output.buffer.copy_from(luisa::span{output.host})
                   << shader(a.view(), b.view(), initial.view(), output.view()).dispatch()
                   << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
                   << initial.buffer.copy_to(luisa::span{initial.host}) << output.buffer.copy_to(luisa::span{output.host}) << synchronize();
            check_readonly(a, original_a);
            check_readonly(b, original_b);
            check_readonly(initial, original_initial);
            output.check_guards();
            constexpr auto operations = 2.0 * (extent + 1u), unit = 0x1p-24;
            constexpr auto gamma = operations * unit / (1.0 - operations * unit);
            for (auto m = 0u; m < extent; m++) {
                for (auto n = 0u; n < extent; n++) {
                    auto expected = static_cast<double>(initial[m * extent + n]);
                    auto magnitude = std::abs(expected);
                    auto ordered = initial[m * extent + n];
                    for (auto k = 0u; k < extent; k++) {
                        auto lhs = a[m * extent + k], rhs = b[k * extent + n];
                        auto product = static_cast<double>(lhs) * static_cast<double>(rhs);
                        expected += product;
                        magnitude += std::abs(product);
                        ordered = std::fma(lhs, rhs, ordered);
                    }
                    auto actual = output[m * extent + n];
                    auto bound = magnitude * (gamma + 8.0 * (extent + 1u) * 0x1p-53);
                    expect(std::isfinite(actual) && std::abs(static_cast<double>(actual) - expected) <= bound)
                        << "reassociate=" << reassociate << " precision flavor=" << flavor << " m=" << m << " n=" << n;
                    if (!reassociate) { expect(bits(actual) == bits(ordered)); }
                    if (flavor != 5u) { expect(bits(actual) == bits(static_cast<float>(expected))) << "exact FP32 precision flavor=" << flavor; }
                }
            }
        }
    }
}

void large_accumulator_dynamic_contraction(Device &device) {
    // 8192 output elements exceed the extended-unroll cap. K64 is otherwise
    // eligible, so this checks the output-size guard independently of K256.
    constexpr auto m = 128u, n = 64u, k = 64u;
    using namespace tile;
    auto kernel = tile_kernel("bounded_ordered_accumulator", [=](TensorView<const float, 2> a,
                                                                TensorView<const float, 2> b,
                                                                TensorView<float, 2> c) {
                      auto rows = axis("m", m), columns = axis("n", n), terms = axis("k", k);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto lhs = a.tile(coord(0, 0), shape(rows, terms)).load();
                          auto rhs = b.tile(coord(0, 0), shape(terms, columns)).load();
                          c(coord(0, 0), shape(rows, columns)).store(
                              mma(lhs, rhs, full<float>(shape(rows, columns), kInitial), MmaPolicy{false}));
                      }
                  }).capture(tensor_shape(m, k), tensor_shape(k, n), tensor_shape(m, n));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 3u)) { return; }
    auto &&source = shader.metadata().source;
    auto first_fma = source.find("ct::fma(");
    expect(first_fma != luisa::string::npos);
    if (first_fma != luisa::string::npos) { expect(source.find("ct::fma(", first_fma + 1u) == luisa::string::npos); }
    expect(source.find("for (unsigned mma") != luisa::string::npos);
    GuardedBuffer a{device, m * k}, b{device, k * n}, c{device, m * n};
    for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>(static_cast<int>(i % 17u) - 8) * .125f; }
    for (auto i = 0u; i < b.count; i++) { b[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) * .25f; }
    auto original_a = a.host, original_b = b.host;
    c.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host}) << c.buffer.copy_from(luisa::span{c.host})
           << shader(a.view(), b.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
           << b.buffer.copy_to(luisa::span{b.host}) << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
    check_readonly(a, original_a);
    check_readonly(b, original_b);
    c.check_guards();
    for (auto row = 0u; row < m; row++) {
        for (auto column = 0u; column < n; column++) {
            auto ordered = kInitial;
            auto exact = static_cast<double>(kInitial);
            for (auto term = 0u; term < k; term++) {
                auto lhs = original_a[kPad + row * k + term], rhs = original_b[kPad + term * n + column];
                ordered = std::fma(lhs, rhs, ordered);
                exact += static_cast<double>(lhs) * static_cast<double>(rhs);
            }
            auto actual = c[row * n + column];
            expect(bits(actual) == bits(ordered));
            // These dyadic products and their bounded sum are exactly FP32.
            expect(static_cast<double>(actual) == exact);
        }
    }
}

// Exercise both sides of the total expanded-work budget with identical K.
// The policy is explicit: an optimized reassociated MMA cannot hide this gate.
void fma_operation_budget_boundary(Device &device) {
    using namespace tile;
    constexpr auto columns = 32u, terms = 128u;
    for (auto rows : {16u, 32u}) {
        auto kernel = tile_kernel("ordered_expansion_budget", [=](TensorView<const float, 2> a,
                                                                  TensorView<const float, 2> b,
                                                                  TensorView<float, 2> c) {
                          auto m = axis("m", rows), n = axis("n", columns), k = axis("k", terms);
                          for (auto &nest : parallel(shape(1))) {
                              static_cast<void>(nest);
                              auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                              auto rhs = b.tile(coord(0, 0), shape(k, n)).load();
                              c(coord(0, 0), shape(m, n)).store(
                                  mma(lhs, rhs, full<float>(shape(m, n), kInitial), MmaPolicy{false}));
                          }
                      }).capture(tensor_shape(rows, terms), tensor_shape(terms, columns), tensor_shape(rows, columns));
        auto shader = tile::compile(device, kernel);
        if (!check_native(shader, make_uint3(1u), 3u)) { continue; }
        auto &&source = shader.metadata().source;
        auto count = size_t{0u}, position = size_t{0u};
        constexpr luisa::string_view call = "ct::fma(";
        while ((position = source.find(call, position)) != luisa::string::npos) {
            count++;
            position += call.size();
        }
        expect(source.find("ct::mma(") == luisa::string::npos);
        expect(count == (rows == 16u ? terms : 1u)) << "total-FMA budget rows=" << rows;
        expect((source.find("for (unsigned mma") != luisa::string::npos) == (rows == 32u));
        GuardedBuffer a{device, rows * terms}, b{device, terms * columns}, c{device, rows * columns};
        for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>(static_cast<int>(i % 17u) - 8) * .125f; }
        for (auto i = 0u; i < b.count; i++) { b[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) * .25f; }
        auto original_a = a.host, original_b = b.host;
        c.poison();
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
               << c.buffer.copy_from(luisa::span{c.host}) << shader(a.view(), b.view(), c.view()).dispatch()
               << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
               << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
        check_readonly(a, original_a);
        check_readonly(b, original_b);
        c.check_guards();
        for (auto m = 0u; m < rows; m++) {
            for (auto n = 0u; n < columns; n++) {
                auto ordered = kInitial;
                auto exact = static_cast<double>(kInitial);
                for (auto k = 0u; k < terms; k++) {
                    auto lhs = a[m * terms + k], rhs = b[k * columns + n];
                    ordered = std::fma(lhs, rhs, ordered);
                    exact += static_cast<double>(lhs) * static_cast<double>(rhs);
                }
                auto actual = c[m * columns + n];
                expect(bits(actual) == bits(ordered));
                // The dyadic products and every partial sum are exactly FP32.
                expect(static_cast<double>(actual) == exact);
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
    expect(shader.metadata().source.find("ct::load_masked(") != luisa::string::npos);
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

void proven_pointer_memory_bounds(Device &device) {
    using namespace tile;
    auto count_source = [](luisa::string_view source, luisa::string_view needle) noexcept {
        auto count = size_t{0u}, position = size_t{0u};
        while ((position = source.find(needle, position)) != luisa::string_view::npos) {
            count++;
            position += needle.size();
        }
        return count;
    };
    constexpr std::array patterns{0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                                 0x007fffffu, 0x00800000u, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0x3f812345u};
    for (auto ragged : {false, true}) {
        auto rows = ragged ? 31u : 32u, columns = ragged ? 29u : 32u;
        auto kernel = tile_kernel("proven_pointer_copy_full_or_ragged", [](TensorView<const float, 2> a, TensorView<float, 2> c) {
                          auto gm = axis("gm", 2), gn = axis("gn", 2), m = axis("m", 16), n = axis("n", 16);
                          for (auto &nest : parallel(shape(gm, gn))) {
                              auto origin = coord(nest.index(gm) * 16, nest.index(gn) * 16);
                              c(origin, shape(m, n)).store(a.tile(origin, shape(m, n)).load());
                          }
                      }).capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
        auto shader = tile::compile(device, kernel);
        if (!check_native(shader, make_uint3(2u, 2u, 1u), 2u)) { continue; }
        auto &&source = shader.metadata().source;
        expect(count_source(source, "ct::tensor_span{") == 0u);
        expect(count_source(source, "ct::partition_view{") == 0u);
        expect(count_source(source, "ct::load_masked(") == (ragged ? 1u : 0u));
        expect(count_source(source, "ct::store_masked(") == (ragged ? 1u : 0u));
        expect(count_source(source, "ct::load(") == (ragged ? 0u : 1u));
        expect(count_source(source, "ct::store(") == (ragged ? 0u : 1u));
        expect((source.find("ct::select(") != luisa::string::npos) == ragged);
        // The view starts 52 bytes into its allocation. Mask elision must
        // not strengthen the runtime pointer-alignment contract.
        GuardedBuffer a{device, rows * columns}, c{device, rows * columns};
        for (auto i = 0u; i < a.count; i++) { a[i] = std::bit_cast<float>(patterns[i % patterns.size()]); }
        auto original = a.host;
        c.poison();
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.buffer.copy_from(luisa::span{a.host}) << c.buffer.copy_from(luisa::span{c.host})
               << shader(a.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
               << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
        check_readonly(a, original);
        c.check_guards();
        for (auto i = 0u; i < c.count; i++) {
            expect(bits(c[i]) == bits(original[kPad + i])) << "pointer copy ragged=" << ragged << " element=" << i;
        }
    }

    auto kernel = tile_kernel("conservative_memory_fallbacks", [](TensorView<const float, 1> a, TensorView<float, 1> c,
                                                                 TensorView<float, 1> touched) {
                      auto n = axis("n", 16);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto domain = shape(n);
                          auto positive_unaligned = a.tile(coord(1), domain).load();
                          c(coord(0), domain).store(positive_unaligned);
                          touched(coord(1), domain).store(positive_unaligned);
                          c(coord(16), domain).store(a.tile(coord(32), domain).load());
                          touched(coord(32), domain).store(full<float>(domain, 123.0f));
                          c(coord(32), domain).store(a.tile(coord(16), domain).load(1.25f));
                          c(coord(48), domain).store(a.tile(coord(16), domain).load(-0.0f));
                          auto unknown_origin = cast<int64_t>(a.tile(coord(0), shape(1)).load().at(coord(0)));
                          c(coord(64), domain).store(a.tile(coord(unknown_origin), domain).load());
                      }
                  }).capture(tensor_shape(17), tensor_shape(80), tensor_shape(17));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 3u)) { return; }
    auto &&source = shader.metadata().source;
    // A positive non-tile-aligned origin can still be proved fully in bounds.
    // Only the out-of-bounds/custom-tail/unproved accesses retain masks.
    expect(count_source(source, "ct::load_masked(") == 4u);
    expect(count_source(source, "ct::store_masked(") == 1u);
    expect(count_source(source, "ct::load(") == 2u);
    expect(count_source(source, "ct::store(") == 6u);
    expect(count_source(source, "ct::tensor_span{") == 0u);
    expect(count_source(source, "ct::partition_view{") == 0u);
    GuardedBuffer a{device, 17u}, c{device, 80u}, touched{device, 17u};
    for (auto i = 1u; i < a.count; i++) { a[i] = std::bit_cast<float>(patterns[(i - 1u) % patterns.size()]); }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    for (auto origin : {1u, 32u}) {
        a[0] = static_cast<float>(origin);
        auto original = a.host;
        c.poison();
        for (auto i = 0u; i < touched.count; i++) { touched[i] = -100.0f - static_cast<float>(i); }
        auto previous_touched = touched.host;
        stream << a.buffer.copy_from(luisa::span{a.host}) << c.buffer.copy_from(luisa::span{c.host})
               << touched.buffer.copy_from(luisa::span{touched.host})
               << shader(a.view(), c.view(), touched.view()).dispatch()
               << a.buffer.copy_to(luisa::span{a.host}) << c.buffer.copy_to(luisa::span{c.host})
               << touched.buffer.copy_to(luisa::span{touched.host}) << synchronize();
        check_readonly(a, original);
        c.check_guards();
        touched.check_guards();
        // In particular, the wholly out-of-bounds store may not write element0.
        expect(bits(touched[0]) == bits(previous_touched[kPad]));
        for (auto i = 0u; i < 16u; i++) {
            expect(bits(touched[i + 1u]) == bits(original[kPad + i + 1u]));
            expect(bits(c[i]) == bits(original[kPad + i + 1u]));
            expect(bits(c[16u + i]) == 0u);
            expect(bits(c[32u + i]) == (i == 0u ? bits(original[kPad + 16u]) : bits(1.25f)));
            expect(bits(c[48u + i]) == (i == 0u ? bits(original[kPad + 16u]) : 0x80000000u));
            expect(bits(c[64u + i]) == (origin == 1u ? bits(original[kPad + i + 1u]) : 0u));
        }
    }

    // The DIV interval must be safe for every head and every column. H6
    // includes fully out-of-range source heads; width5 has a partial tail.
    constexpr std::array divided_shapes{std::array{4u, 8u}, std::array{6u, 8u}, std::array{4u, 5u}};
    for (auto dimensions : divided_shapes) {
        auto heads = dimensions[0], width = dimensions[1];
        auto kernel = tile_kernel("divided_head_pointer_bounds", [=](TensorView<const float, 2> a, TensorView<float, 2> c) {
                          auto h = axis("h", 1), n = axis("n", 8);
                          for (auto &nest : parallel(shape(heads))) {
                              auto head = nest.index();
                              c(coord(head, 0), shape(h, n)).store(a.tile(coord(head / 2, 0), shape(h, n)).load());
                          }
                      }).capture(tensor_shape(2, width), tensor_shape(heads, 8));
        auto shader = tile::compile(device, kernel);
        if (!check_native(shader, make_uint3(heads, 1u, 1u), 2u)) { continue; }
        auto &&source = shader.metadata().source;
        auto fully_in_bounds = heads == 4u && width == 8u;
        expect(count_source(source, "ct::load(") == (fully_in_bounds ? 1u : 0u));
        expect(count_source(source, "ct::load_masked(") == (fully_in_bounds ? 0u : 1u));
        expect(count_source(source, "ct::store(") == 1u);
        expect(count_source(source, "ct::store_masked(") == 0u);
        expect(count_source(source, "ct::partition_view{") == 0u);
        GuardedBuffer a{device, 2u * width}, c{device, heads * 8u};
        for (auto i = 0u; i < a.count; i++) { a[i] = std::bit_cast<float>(patterns[i % patterns.size()]); }
        auto original = a.host;
        c.poison();
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.buffer.copy_from(luisa::span{a.host}) << c.buffer.copy_from(luisa::span{c.host})
               << shader(a.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
               << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
        check_readonly(a, original);
        c.check_guards();
        for (auto head = 0u; head < heads; head++) {
            for (auto column = 0u; column < 8u; column++) {
                auto expected = head / 2u < 2u && column < width ? bits(original[kPad + head / 2u * width + column]) : 0u;
                expect(bits(c[head * 8u + column]) == expected) << "divided head=" << head << " width=" << width << " column=" << column;
            }
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

template<typename T>
void blocked_row_operations(Device &device, uint32_t block_rows, uint32_t rows = 17u, uint32_t width = 65u,
                            bool scan_only = false) {
    using namespace test::tile_workloads;
    constexpr auto scalar_type = std::is_same_v<T, float> ? tile::ScalarType::FLOAT32 :
                                 std::is_same_v<T, half> ? tile::ScalarType::FLOAT16 : tile::ScalarType::BFLOAT16;
    using Word = std::conditional_t<sizeof(T) == 2u, uint16_t, uint32_t>;
    auto bits_of = [](T value) noexcept { return std::bit_cast<Word>(value); };
    auto guard = T{-719.5f};
    for (auto op : {"scan", "reduce_sum", "reduce_max"}) {
        if (scan_only && std::strcmp(op, "scan") != 0) { continue; }
        Options options;
        options.backend = "cuda";
        options.lowering = "native";
        options.operation = op;
        options.precision = std::is_same_v<T, float> ? "fp32" : std::is_same_v<T, half> ? "fp16" : "bf16";
        options.dimensions = {rows, width};
        options.tile = {block_rows, std::bit_ceil(width), 1};
        options.pattern = "cancellation";
        options.seed = 20261001u;
        auto fixture = make_fixture<T>(options);
        expect(fixture.error.empty()) << fixture.error;
        if (!fixture.error.empty() || !fixture.kernel) { return; }
        auto shader = tile::compile(device, *fixture.kernel, {}, {.enable_fast_math = false});
        auto programs = (rows + block_rows - 1u) / block_rows;
        if (!check_native(shader, make_uint3(programs, 1u, 1u), 4u)) { return; }
        expect(shader.metadata().arguments[0].element == scalar_type);
        expect(shader.metadata().arguments[3].element == scalar_type);
        expect(shader.metadata().arguments[0].minimum_size_bytes == rows * width * sizeof(T));
        expect(shader.metadata().arguments[3].minimum_size_bytes == fixture.expected.size() * sizeof(T));
        auto intrinsic = std::strcmp(op, "scan") == 0 ? "ct::partial_sum(" :
                         std::strcmp(op, "reduce_sum") == 0 ? "ct::sum(" : "ct::reduce_max(";
        expect(shader.metadata().source.find(intrinsic) != string::npos);
        if (block_rows > 1u) { expect(shader.metadata().source.find("ct::store_masked(") != string::npos); }
        auto stream = device.create_stream(StreamTag::COMPUTE);
        for (auto generation = 0u; generation < 2u; generation++) {
            options.pattern = generation == 0u ? "cancellation" : "random";
            options.seed = 20261001u + generation;
            auto data = make_fixture<T>(options);
            expect(data.error.empty()) << data.error;
            if (!data.error.empty()) { return; }
            if (std::strcmp(op, "reduce_max") == 0) {
                // A zero-filled column/row tail must never beat valid negatives.
                for (auto &value : data.inputs[0]) { value = static_cast<float>(T{-std::abs(value) - .25f}); }
                row_oracle(data, options);
                // The selected value is exactly representable in T; retain the
                // strict zero bound from the complete FP64 max oracle.
            }
            std::array<luisa::vector<T>, 3u> inputs;
            std::array<Buffer<T>, 3u> buffers;
            for (auto input = 0u; input < 3u; input++) {
                inputs[input].assign(data.inputs[input].size() + 2u * kPad, guard);
                for (auto i = size_t{0u}; i < data.inputs[input].size(); i++) { inputs[input][i + kPad] = T{data.inputs[input][i]}; }
                buffers[input] = device.create_buffer<T>(inputs[input].size());
                stream << buffers[input].copy_from(luisa::span{inputs[input]});
            }
            auto readonly = inputs;
            luisa::vector<T> output(data.expected.size() + 2u * kPad, guard);
            std::fill_n(output.begin() + kPad, data.expected.size(), T{std::numeric_limits<float>::quiet_NaN()});
            auto gpu_output = device.create_buffer<T>(output.size());
            stream << gpu_output.copy_from(luisa::span{output})
                   << shader(buffers[0].view(kPad, data.inputs[0].size()), buffers[1].view(kPad, data.inputs[1].size()),
                             buffers[2].view(kPad, data.inputs[2].size()), gpu_output.view(kPad, data.expected.size())).dispatch();
            for (auto input = 0u; input < 3u; input++) { stream << buffers[input].copy_to(luisa::span{inputs[input]}); }
            stream << gpu_output.copy_to(luisa::span{output}) << synchronize();
            for (auto input = 0u; input < 3u; input++) {
                for (auto i = size_t{0u}; i < inputs[input].size(); i++) {
                    expect(bits_of(inputs[input][i]) == bits_of(readonly[input][i])) << op << " readonly=" << input << " i=" << i;
                }
            }
            for (auto i = size_t{0u}; i < data.expected.size(); i++) {
                auto actual = static_cast<float>(output[kPad + i]);
                auto error = std::abs(static_cast<double>(actual) - data.expected[i]);
                expect(std::isfinite(actual) && error <= data.bound[i])
                    << op << " BR=" << block_rows << " i=" << i << " error=" << error << " bound=" << data.bound[i];
            }
            for (auto i = 0u; i < kPad; i++) {
                expect(bits_of(output[i]) == bits_of(guard));
                expect(bits_of(output[kPad + data.expected.size() + i]) == bits_of(guard));
            }
        }
    }
}

void row_operations(Device &device, bool fast_norm = false) {
    using test::tile_llm::RowOp;
    for (auto op : {RowOp::RMS_NORM, RowOp::LAYER_NORM, RowOp::SWIGLU, RowOp::ROPE, RowOp::MASKED_SOFTMAX, RowOp::GELU_RESIDUAL}) {
        if (fast_norm && op != RowOp::RMS_NORM && op != RowOp::LAYER_NORM) { continue; }
        auto fixture = test::tile_llm::rows(op, 7, 32);
        auto shader = tile::compile(device, fixture.kernel, {}, {.enable_fast_math = fast_norm});
        if (fast_norm) { expect(shader.metadata().source.find("ct::rsqrt(") != string::npos); }
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

struct IntegerSeedFive {
    template<tile::scalar_cpp_type T>
    [[nodiscard]] static constexpr T identity() noexcept { return T{5}; }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return a + b; }
};

template<typename T>
void integer_primitives(Device &device, uint32_t width) {
    using namespace tile;
    using U = std::make_unsigned_t<T>;
    auto tile_width = std::bit_ceil(width);
    auto kernel = tile_kernel("integer_rows_and_prefixes", [=](TensorView<const T, 3> input,
                                                               TensorView<T, 3> tree,
                                                               TensorView<T, 3> ordered,
                                                               TensorView<T, 3> sum,
                                                               TensorView<T, 3> low,
                                                               TensorView<T, 3> high,
                                                               TensorView<T, 3> biased) {
                      auto a = axis("a", 2), n = axis("n", tile_width), c = axis("c", 2), out = axis("out", 1);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto space = shape(a, n, c), reduced = shape(a, out, c);
                          auto x = input.tile(coord(0, 0, 0), space).load();
                          auto valid = iota(n) < width;
                          tree(coord(0, 0, 0), space).store(inclusive_sum(x, n, reduction::unordered_tree));
                          ordered(coord(0, 0, 0), space).store(inclusive_sum(x, n, reduction::fold_left));
                          sum(coord(0, 0, 0), reduced).store(broadcast_to(reduce(x, n, add), reduced));
                          low(coord(0, 0, 0), reduced).store(broadcast_to(reduce(ite(valid, x, std::numeric_limits<T>::max()), n, minimum), reduced));
                          high(coord(0, 0, 0), reduced).store(broadcast_to(reduce(ite(valid, x, std::numeric_limits<T>::lowest()), n, maximum), reduced));
                          biased(coord(0, 0, 0), reduced).store(broadcast_to(reduce(x, n, IntegerSeedFive{}), reduced));
                      }
                  }).capture(tensor_shape(2, width, 2), tensor_shape(2, width, 2), tensor_shape(2, width, 2),
                             tensor_shape(2, 1, 2), tensor_shape(2, 1, 2), tensor_shape(2, 1, 2), tensor_shape(2, 1, 2));
    auto shader = tile::compile(device, kernel, {}, ShaderOption{.enable_fast_math = false});
    if (!check_native(shader, make_uint3(1u), 7u)) { return; }
    auto &&source = shader.metadata().source;
    auto occurrences = [&](luisa::string_view needle) {
        auto count = size_t{0u}, position = size_t{0u};
        while ((position = source.find(needle, position)) != luisa::string::npos) { count++; position += needle.size(); }
        return count;
    };
    // The nonzero seed and ordered scan retain their scalar sequence.
    expect(occurrences("ct::sum(") == 1u);
    expect(occurrences("ct::partial_sum(") == 1u);
    expect(occurrences("ct::reduce_min(") == 1u);
    expect(occurrences("ct::reduce_max(") == 1u);
    if constexpr (std::is_signed_v<T>) {
        auto cast = sizeof(T) == 4u ? "ct::element_bitcast<unsigned>(" : "ct::element_bitcast<unsigned long long>(";
        expect(source.find(cast) != luisa::string::npos);
    }
    auto guard = std::bit_cast<T>(static_cast<U>(0xa6f01ce59b731d42ull));
    auto sign = U{1u} << (std::numeric_limits<U>::digits - 1u);
    std::array<U, 12u> patterns{U{}, U{1u}, std::numeric_limits<U>::max(), sign, static_cast<U>(sign - 1u),
                              static_cast<U>(sign + 1u), U{0x01000001u}, U{0x01000003u}, U{17u}, U{31u},
                              static_cast<U>(0x0020000000000001ull), static_cast<U>(0x0020000000000003ull)};
    std::array<size_t, 7u> counts{4u * width, 4u * width, 4u * width, 4u, 4u, 4u, 4u};
    std::array<luisa::vector<T>, 7u> data, expected;
    std::array<Buffer<T>, 7u> buffers;
    for (auto b = 0u; b < data.size(); b++) {
        data[b].assign(counts[b] + 2u * kPad, guard);
        expected[b] = data[b];
        buffers[b] = device.create_buffer<T>(data[b].size());
    }
    for (auto a = 0u; a < 2u; a++) {
        for (auto c = 0u; c < 2u; c++) {
            auto total = U{};
            auto minimum_value = std::numeric_limits<T>::max(), maximum_value = std::numeric_limits<T>::lowest();
            for (auto n = 0u; n < width; n++) {
                auto p = kPad + (a * width + n) * 2u + c;
                // Rotate each independent row through full-width boundary bits.
                auto value = std::bit_cast<T>(patterns[(n + a * 3u + c * 5u) % patterns.size()]);
                expected[0][p] = value;
                total = static_cast<U>(total + std::bit_cast<U>(value));
                expected[1][p] = expected[2][p] = std::bit_cast<T>(total);
                minimum_value = std::min(minimum_value, value);
                maximum_value = std::max(maximum_value, value);
            }
            auto p = kPad + a * 2u + c;
            expected[3][p] = std::bit_cast<T>(total);
            expected[4][p] = minimum_value;
            expected[5][p] = maximum_value;
            expected[6][p] = std::bit_cast<T>(static_cast<U>(total + U{5u}));
        }
    }
    data[0] = expected[0];
    for (auto b = 1u; b < data.size(); b++) {
        for (auto i = 0u; i < counts[b]; i++) {
            data[b][kPad + i] = std::bit_cast<T>(static_cast<U>(std::bit_cast<U>(expected[b][kPad + i]) ^ std::numeric_limits<U>::max()));
        }
    }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    for (auto b = 0u; b < data.size(); b++) { stream << buffers[b].copy_from(luisa::span{data[b]}); }
    stream << shader(buffers[0].view(kPad, counts[0]), buffers[1].view(kPad, counts[1]), buffers[2].view(kPad, counts[2]),
                     buffers[3].view(kPad, counts[3]), buffers[4].view(kPad, counts[4]), buffers[5].view(kPad, counts[5]),
                     buffers[6].view(kPad, counts[6])).dispatch();
    for (auto b = 0u; b < data.size(); b++) { stream << buffers[b].copy_to(luisa::span{data[b]}); }
    stream << synchronize();
    for (auto b = 0u; b < data.size(); b++) {
        for (auto i = 0u; i < data[b].size(); i++) {
            expect(data[b][i] == expected[b][i]) << "integer bytes=" << sizeof(T) << " signed=" << std::is_signed_v<T>
                                               << " width=" << width << " buffer=" << b << " offset=" << i;
        }
    }
}

void native_scan_and_sort(Device &device) {
    using namespace tile;
    auto kernel = tile_kernel("prefix_and_permutation", [](TensorView<const float, 2> a, TensorView<float, 2> prefix,
                                                           TensorView<float, 2> values, TensorView<int64_t, 2> indices) {
                      auto m = axis("m", 2), n = axis("n", 16);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0, 0), shape(m, n)).load();
                          prefix(coord(0, 0), shape(m, n)).store(inclusive_sum(x, n));
                          auto ranked = topk(x, n, 8u, false);
                          values(coord(0, 0), ranked.values.space()).store(ranked.values);
                          indices(coord(0, 0), ranked.indices.space()).store(ranked.indices);
                      }
                  }).capture(tensor_shape(2, 16), tensor_shape(2, 16), tensor_shape(2, 8), tensor_shape(2, 8));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 4u)) { return; }
    expect(shader.metadata().source.find("ct::partial_sum(") != luisa::string::npos);
    expect(shader.metadata().source.find("ct::cat(") != luisa::string::npos);
    GuardedBuffer a{device, 32u}, prefix{device, 32u}, values{device, 16u};
    std::array<int64_t, 16u + 2u * kPad> indices;
    indices.fill(-731ll);
    auto gpu_indices = device.create_buffer<int64_t>(indices.size());
    for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>((i * 7u) % 9u) * .25f - 1.0f; }
    auto original = a.host;
    prefix.poison(); values.poison();
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << prefix.buffer.copy_from(luisa::span{prefix.host})
           << values.buffer.copy_from(luisa::span{values.host}) << gpu_indices.copy_from(luisa::span{indices})
           << shader(a.view(), prefix.view(), values.view(), gpu_indices.view(kPad, 16u)).dispatch()
           << a.buffer.copy_to(luisa::span{a.host}) << prefix.buffer.copy_to(luisa::span{prefix.host})
           << values.buffer.copy_to(luisa::span{values.host}) << gpu_indices.copy_to(luisa::span{indices}) << synchronize();
    check_readonly(a, original); prefix.check_guards(); values.check_guards();
    for (auto row = 0u; row < 2u; row++) {
        std::array<uint32_t, 16u> order;
        auto sum = 0.0f;
        for (auto i = 0u; i < 16u; i++) { order[i] = i; sum += a[row * 16u + i]; expect(bits(prefix[row * 16u + i]) == bits(sum)); }
        std::sort(order.begin(), order.end(), [&](auto x, auto y) { return a[row * 16u + x] == a[row * 16u + y] ? x < y : a[row * 16u + x] < a[row * 16u + y]; });
        for (auto i = 0u; i < 8u; i++) {
            expect(indices[kPad + row * 8u + i] == order[i]);
            expect(bits(values[row * 8u + i]) == bits(a[row * 16u + order[i]]));
        }
    }
    for (auto i = 0u; i < kPad; i++) { expect(indices[i] == -731ll); expect(indices[kPad + 16u + i] == -731ll); }
}

// These benchmark/test compositions require independent input/output/scratch
// allocations. No in-place or overlapping-view contract is implied.
template<typename T>
void chunked_sort_pipeline(Device &device, int64_t columns, int64_t chunk, size_t expected_stages) {
    namespace pipeline = luisa::test::tile_sort_pipeline;
    constexpr auto rows = int64_t{4};
    auto plan = pipeline::plan(rows, columns, chunk);
    expect(plan.error.empty()) << plan.error;
    expect(plan.stages.size() == expected_stages);
    if (!plan.error.empty() || plan.stages.size() != expected_stages) { return; }
    auto *extension = device.extension<CudaGraphExt>();
    expect(extension != nullptr);
    if (extension == nullptr) { return; }
    luisa::vector<tile::Shader> shaders;
    shaders.reserve(plan.stages.size());
    for (size_t stage = 0u; stage < plan.stages.size(); stage++) {
        auto descriptor = plan.stages[stage];
        auto kernel = descriptor.initial ?
                          (descriptor.final ? pipeline::initialize<T, T, int64_t>(plan) : pipeline::initialize<T>(plan)) :
                          (descriptor.final ? pipeline::merge_whole<T, int64_t>(plan, descriptor) : pipeline::merge_whole<>(plan, descriptor));
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto shader = tile::compile(device, kernel, {}, {.enable_fast_math = false});
        if (!check_native(shader, make_uint3(static_cast<uint32_t>(rows), static_cast<uint32_t>(plan.padded / descriptor.width), 1u),
                          descriptor.initial ? 3u : 4u)) { return; }
        shaders.emplace_back(std::move(shader));
    }
    auto count = static_cast<size_t>(rows * columns);
    auto scratch_count = static_cast<size_t>(rows * plan.padded);
    using Word = std::conditional_t<sizeof(T) == 2u, uint16_t, uint32_t>;
    auto word = [](T value) noexcept { return std::bit_cast<Word>(value); };
    auto guard = T{-719.5f};
    constexpr auto scratch_guard = -933.25f;
    constexpr auto index_guard = std::numeric_limits<int64_t>::min() + 37;
    constexpr auto scratch_index_guard = std::numeric_limits<int32_t>::min() + 51;
    luisa::vector<T> input(count + 2u * kPad, guard), output(count + 2u * kPad, guard);
    luisa::vector<int64_t> indices(count + 2u * kPad, index_guard);
    std::array<luisa::vector<float>, 2u> scratch_values;
    std::array<luisa::vector<int32_t>, 2u> scratch_indices;
    auto gpu_input = device.create_buffer<T>(input.size());
    auto gpu_output = device.create_buffer<T>(output.size());
    auto gpu_indices = device.create_buffer<int64_t>(indices.size());
    std::array<Buffer<float>, 2u> gpu_scratch_values;
    std::array<Buffer<int32_t>, 2u> gpu_scratch_indices;
    for (size_t slot = 0u; slot < 2u; slot++) {
        scratch_values[slot].assign(scratch_count + 2u * kPad, scratch_guard);
        scratch_indices[slot].assign(scratch_count + 2u * kPad, scratch_index_guard);
        gpu_scratch_values[slot] = device.create_buffer<float>(scratch_values[slot].size());
        gpu_scratch_indices[slot] = device.create_buffer<int32_t>(scratch_indices[slot].size());
    }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto make_commands = [&](size_t calls) {
        CommandList commands;
        for (size_t call = 0u; call < calls; call++) {
            if (shaders.size() == 1u) {
                commands << shaders.front()(gpu_input.view(kPad, count), gpu_output.view(kPad, count),
                                             gpu_indices.view(kPad, count)).dispatch();
                continue;
            }
            commands << shaders.front()(gpu_input.view(kPad, count), gpu_scratch_values[0].view(kPad, scratch_count),
                                         gpu_scratch_indices[0].view(kPad, scratch_count)).dispatch();
            for (size_t stage = 1u; stage < shaders.size(); stage++) {
                auto read_slot = (stage - 1u) % 2u;
                if (stage + 1u == shaders.size()) {
                    commands << shaders[stage](gpu_scratch_values[read_slot].view(kPad, scratch_count),
                                                gpu_scratch_indices[read_slot].view(kPad, scratch_count),
                                                gpu_output.view(kPad, count), gpu_indices.view(kPad, count)).dispatch();
                } else {
                    auto write_slot = stage % 2u;
                    expect(write_slot != read_slot);
                    commands << shaders[stage](gpu_scratch_values[read_slot].view(kPad, scratch_count),
                                                gpu_scratch_indices[read_slot].view(kPad, scratch_count),
                                                gpu_scratch_values[write_slot].view(kPad, scratch_count),
                                                gpu_scratch_indices[write_slot].view(kPad, scratch_count)).dispatch();
                }
            }
        }
        return commands;
    };
    // Three complete calls expose scratch RAW/WAR/WAW edges between calls.
    // The public graph extension may overlap independent stages; output order
    // and all required data hazards must still preserve the complete result.
    auto graph = extension->create_graph(make_commands(3u));
    expect(graph.handle().valid());
    if (!graph.handle().valid()) { return; }
    auto executable = extension->instantiate(graph.handle().handle);
    expect(executable.handle().valid());
    if (!executable.handle().valid()) { return; }
    const std::array pattern{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             5.0f, 5.0f, -0.0f, 0.0f, .25f, -.25f};
    for (auto generation = 0u; generation < 2u; generation++) {
        std::fill(input.begin(), input.end(), guard);
        for (auto i = int64_t{0}; i < columns; i++) {
            input[kPad + i] = T{pattern[(static_cast<size_t>(i) + 3u * generation) % pattern.size()]};
            input[kPad + columns + i] = T{(i + generation) % 2 == 0 ? -0.0f : 0.0f};
            input[kPad + 2 * columns + i] = T{generation == 0u ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity()};
            input[kPad + 3 * columns + i] = T{generation == 0u ? 2.5f : -2.5f};
        }
        auto original = input;
        luisa::vector<int64_t> expected(count), order(static_cast<size_t>(columns));
        for (auto row = int64_t{0}; row < rows; row++) {
            std::iota(order.begin(), order.end(), int64_t{0});
            std::sort(order.begin(), order.end(), [&](auto a, auto b) {
                auto x = static_cast<float>(original[kPad + row * columns + a]);
                auto y = static_cast<float>(original[kPad + row * columns + b]);
                return x == y ? a < b : x > y;
            });
            std::copy(order.begin(), order.end(), expected.begin() + row * columns);
        }
        stream << gpu_input.copy_from(luisa::span{input});
        for (size_t slot = 0u; slot < 2u; slot++) {
            std::fill(scratch_values[slot].begin(), scratch_values[slot].end(), scratch_guard);
            std::fill(scratch_indices[slot].begin(), scratch_indices[slot].end(), scratch_index_guard);
            stream << gpu_scratch_values[slot].copy_from(luisa::span{scratch_values[slot]})
                   << gpu_scratch_indices[slot].copy_from(luisa::span{scratch_indices[slot]});
        }
        auto poison_output = [&] {
            std::fill(output.begin(), output.end(), guard);
            std::fill_n(output.begin() + kPad, count, std::numeric_limits<T>::quiet_NaN());
            std::fill(indices.begin(), indices.end(), index_guard);
            stream << gpu_output.copy_from(luisa::span{output}) << gpu_indices.copy_from(luisa::span{indices}) << synchronize();
        };
        auto check = [&] {
            stream << gpu_input.copy_to(luisa::span{input}) << gpu_output.copy_to(luisa::span{output})
                   << gpu_indices.copy_to(luisa::span{indices});
            for (size_t slot = 0u; slot < 2u; slot++) {
                stream << gpu_scratch_values[slot].copy_to(luisa::span{scratch_values[slot]})
                       << gpu_scratch_indices[slot].copy_to(luisa::span{scratch_indices[slot]});
            }
            stream << synchronize();
            for (size_t i = 0u; i < input.size(); i++) { expect(word(input[i]) == word(original[i])) << "chunked readonly=" << i; }
            for (size_t i = 0u; i < kPad; i++) {
                expect(word(output[i]) == word(guard));
                expect(word(output[kPad + count + i]) == word(guard));
                expect(indices[i] == index_guard);
                expect(indices[kPad + count + i] == index_guard);
                for (size_t slot = 0u; slot < 2u; slot++) {
                    expect(bits(scratch_values[slot][i]) == bits(scratch_guard));
                    expect(bits(scratch_values[slot][kPad + scratch_count + i]) == bits(scratch_guard));
                    expect(scratch_indices[slot][i] == scratch_index_guard);
                    expect(scratch_indices[slot][kPad + scratch_count + i] == scratch_index_guard);
                }
            }
            // The direct case touches no scratch; a two-stage pipeline only
            // writes slot 0. Check complete unused allocations as well.
            for (size_t slot = 0u; slot < 2u; slot++) {
                if (slot + 1u < shaders.size()) { continue; }
                for (size_t i = 0u; i < scratch_values[slot].size(); i++) {
                    expect(bits(scratch_values[slot][i]) == bits(scratch_guard));
                    expect(scratch_indices[slot][i] == scratch_index_guard);
                }
            }
            for (auto row = int64_t{0}; row < rows; row++) {
                for (auto rank = int64_t{0}; rank < columns; rank++) {
                    auto position = static_cast<size_t>(row * columns + rank);
                    auto index = expected[position];
                    expect(indices[kPad + position] == index) << "chunked stable row=" << row << " rank=" << rank;
                    expect(word(output[kPad + position]) == word(original[kPad + row * columns + index]))
                        << "chunked value bits row=" << row << " rank=" << rank;
                }
            }
        };
        poison_output();
        stream << make_commands(1u).commit() << synchronize();
        check();
        // Reuse the exact graph after changing the immutable input contents,
        // and poison the output before each replay to detect stale results.
        for (auto replay = 0u; replay < 3u; replay++) {
            poison_output();
            extension->launch(executable.handle().handle, stream.handle());
            check();
        }
    }
}


template<typename T>
void embedding_rows(Device &device, uint32_t width, uint32_t tokens, uint32_t feature_tile) {
    constexpr auto vocabulary = 7u;
    auto kernel = luisa::test::tile_embedding::embedding_rows<T>(vocabulary, width, tokens, feature_tile);
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto shader = tile::compile(device, kernel, {}, ShaderOption{.enable_fast_math = false});
    if (!check_native(shader, make_uint3(tokens, (width + feature_tile - 1u) / feature_tile, 1u), 3u)) { return; }
    constexpr auto element = std::is_same_v<T, float> ? tile::ScalarType::FLOAT32 :
                             std::is_same_v<T, half> ? tile::ScalarType::FLOAT16 : tile::ScalarType::BFLOAT16;
    expect(shader.metadata().arguments[0].element == element);
    expect(shader.metadata().arguments[1].element == tile::ScalarType::INT64);
    expect(shader.metadata().arguments[2].element == element);
    expect(shader.metadata().arguments[0].minimum_size_bytes == vocabulary * width * sizeof(T));
    expect(shader.metadata().arguments[1].minimum_size_bytes == tokens * sizeof(int64_t));
    expect(shader.metadata().arguments[2].minimum_size_bytes == tokens * width * sizeof(T));
    expect(shader.metadata().source.find("ct::load_masked(") != luisa::string::npos);
    using Word = std::conditional_t<sizeof(T) == 2u, uint16_t, uint32_t>;
    auto word = [](T x) noexcept { return std::bit_cast<Word>(x); };
    auto guard = T{-719.5f};
    constexpr int64_t index_guard = std::numeric_limits<int64_t>::min() + 37;
    luisa::vector<T> table(vocabulary * width + 2u * kPad, guard), output(tokens * width + 2u * kPad, guard);
    luisa::vector<int64_t> ids(tokens + 2u * kPad, index_guard);
    for (auto i = 0u; i < vocabulary * width; i++) {
        auto value = static_cast<float>(static_cast<int32_t>(i % 47u) - 23) * 0.125f;
        switch (i % 17u) {
            case 0u: value = -0.0f; break;
            case 1u: value = 0.0f; break;
            case 2u: value = std::numeric_limits<float>::infinity(); break;
            case 3u: value = -std::numeric_limits<float>::infinity(); break;
            default: break;
        }
        table[kPad + i] = T{value};
    }
    for (auto i = 0u; i < tokens; i++) { ids[kPad + i] = i % 3u == 0u ? vocabulary - 1 : i % 3u == 1u ? 0 : vocabulary / 2; }
    auto valid_ids = luisa::span<const int64_t>{ids.data() + kPad, tokens};
    expect(luisa::test::tile_embedding::valid_row_indices(valid_ids, vocabulary));
    for (auto invalid : {int64_t{-1}, int64_t{vocabulary}, int64_t{1} << 53u, std::numeric_limits<int64_t>::max()}) {
        auto one = std::array{invalid};
        expect(!luisa::test::tile_embedding::valid_row_indices(luisa::span<const int64_t>{one}, vocabulary));
    }
    auto readonly_table = table;
    auto readonly_ids = ids;
    auto gpu_table = device.create_buffer<T>(table.size()), gpu_output = device.create_buffer<T>(output.size());
    auto gpu_ids = device.create_buffer<int64_t>(ids.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << gpu_table.copy_from(luisa::span{table}) << gpu_ids.copy_from(luisa::span{ids});
    for (auto repeat = 0u; repeat < 2u; repeat++) {
        std::fill(output.begin() + kPad, output.end() - kPad, T{std::numeric_limits<float>::quiet_NaN()});
        stream << gpu_output.copy_from(luisa::span{output})
               << shader(gpu_table.view(kPad, vocabulary * width), gpu_ids.view(kPad, tokens), gpu_output.view(kPad, tokens * width)).dispatch()
               << gpu_table.copy_to(luisa::span{table}) << gpu_ids.copy_to(luisa::span{ids})
               << gpu_output.copy_to(luisa::span{output}) << synchronize();
        for (auto i = size_t{0u}; i < table.size(); i++) { expect(word(table[i]) == word(readonly_table[i])); }
        for (auto i = size_t{0u}; i < ids.size(); i++) { expect(ids[i] == readonly_ids[i]); }
        for (auto token = 0u; token < tokens; token++) {
            for (auto col = 0u; col < width; col++) {
                auto expected = readonly_table[kPad + static_cast<size_t>(readonly_ids[kPad + token]) * width + col];
                expect(word(output[kPad + token * width + col]) == word(expected)) << "embedding token=" << token << " col=" << col;
            }
        }
        for (auto i = 0u; i < kPad; i++) {
            expect(word(output[i]) == word(guard)); expect(word(output[kPad + tokens * width + i]) == word(guard));
        }
    }
}

template<typename T>
void stable_argmax(Device &device, uint32_t columns) {
    constexpr auto rows = 8u;
    auto kernel = luisa::test::tile_selection::stable_argmax<T>(rows, columns, std::bit_ceil(columns));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto shader = tile::compile(device, kernel, {}, ShaderOption{.enable_fast_math = false});
    if (!check_native(shader, make_uint3(rows, 1u, 1u), 3u)) { return; }
    constexpr auto scalar_type = std::is_same_v<T, float> ? tile::ScalarType::FLOAT32 :
                                 std::is_same_v<T, half> ? tile::ScalarType::FLOAT16 : tile::ScalarType::BFLOAT16;
    expect(shader.metadata().arguments[0].element == scalar_type);
    expect(shader.metadata().arguments[1].element == scalar_type);
    expect(shader.metadata().arguments[0].minimum_size_bytes == rows * columns * sizeof(T));
    expect(shader.metadata().arguments[1].minimum_size_bytes == rows * sizeof(T));
    expect(shader.metadata().arguments[2].element == tile::ScalarType::INT64);
    expect(shader.metadata().arguments[2].minimum_size_bytes == rows * sizeof(int64_t));
    expect(shader.metadata().source.find("ct::reduce_max(") != luisa::string::npos);
    expect(shader.metadata().source.find("ct::reduce_min(") != luisa::string::npos);
    expect(shader.metadata().source.find("for (long long") == luisa::string::npos);
    using Word = std::conditional_t<sizeof(T) == 2u, uint16_t, uint32_t>;
    auto word = [](T value) noexcept { return std::bit_cast<Word>(value); };
    auto guard = T{-719.5f};
    constexpr auto index_guard = std::numeric_limits<int64_t>::min() + 37;
    luisa::vector<T> input(rows * columns + 2u * kPad, guard), output(rows + 2u * kPad, guard);
    luisa::vector<int64_t> indices(rows + 2u * kPad, index_guard);
    std::array<T, rows> expected_values{};
    std::array<int64_t, rows> expected_indices{};
    for (auto row = 0u; row < rows; row++) {
        for (auto col = 0u; col < columns; col++) {
            auto value = -3.0f;
            switch (row) {
                case 0u: value = col % 2u == 0u ? -0.0f : 0.0f; break;
                case 1u: value = col % 2u == 0u ? 0.0f : -0.0f; break;
                case 2u: value = -std::numeric_limits<float>::infinity(); break;
                case 3u: value = std::numeric_limits<float>::infinity(); break;
                case 4u: value = col == columns / 2u || col + 1u == columns ? 5.0f : -3.0f; break;
                case 5u: value = col + 1u == columns ? 7.0f : -4.0f; break;
                case 6u: value = -1.25f; break;
                default: value = col == columns / 2u ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity(); break;
            }
            input[kPad + row * columns + col] = T{value};
        }
        auto winner = 0u;
        for (auto col = 1u; col < columns; col++) {
            if (static_cast<float>(input[kPad + row * columns + col]) > static_cast<float>(input[kPad + row * columns + winner])) { winner = col; }
        }
        expected_indices[row] = static_cast<int64_t>(winner);
        expected_values[row] = input[kPad + row * columns + winner];
    }
    auto readonly = input;
    auto gpu_input = device.create_buffer<T>(input.size()), gpu_output = device.create_buffer<T>(output.size());
    auto gpu_indices = device.create_buffer<int64_t>(indices.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << gpu_input.copy_from(luisa::span{input});
    // Poison outputs independently before two complete calls on the same input.
    for (auto repeat = 0u; repeat < 2u; repeat++) {
        std::fill_n(output.begin() + kPad, rows, T{std::numeric_limits<float>::quiet_NaN()});
        std::fill_n(indices.begin() + kPad, rows, index_guard);
        stream << gpu_output.copy_from(luisa::span{output}) << gpu_indices.copy_from(luisa::span{indices})
               << shader(gpu_input.view(kPad, rows * columns), gpu_output.view(kPad, rows), gpu_indices.view(kPad, rows)).dispatch()
               << gpu_input.copy_to(luisa::span{input}) << gpu_output.copy_to(luisa::span{output})
               << gpu_indices.copy_to(luisa::span{indices}) << synchronize();
        for (auto i = size_t{0u}; i < input.size(); i++) { expect(word(input[i]) == word(readonly[i])) << "argmax readonly N=" << columns << " at=" << i; }
        for (auto row = 0u; row < rows; row++) {
            expect(indices[kPad + row] == expected_indices[row]) << "argmax first index N=" << columns << " row=" << row;
            expect(word(output[kPad + row]) == word(expected_values[row])) << "argmax original value bits N=" << columns << " row=" << row;
        }
        for (auto i = 0u; i < kPad; i++) {
            expect(word(output[i]) == word(guard)); expect(word(output[kPad + rows + i]) == word(guard));
            expect(indices[i] == index_guard); expect(indices[kPad + rows + i] == index_guard);
        }
    }
}

template<typename T>
void repeated_extrema_topk(Device &device, uint32_t columns, uint32_t count) {
    constexpr auto rows = 4u;
    auto kernel = luisa::test::tile_selection::repeated_extrema_topk<T>(rows, columns, count, std::bit_ceil(columns));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    // ValueHandle assignments must create exactly two singleton loop carries,
    // not a row-sized active mask or a one-time host-only SSA reassignment.
    const tile::Operation *serial = nullptr;
    for (auto root : kernel.function().body().block(0u)->operations()) {
        if (root->kind() != tile::OperationKind::PARALLEL) { continue; }
        for (auto operation : root->region(0u)->block(0u)->operations()) {
            if (operation->kind() == tile::OperationKind::SERIAL) { serial = operation; }
        }
    }
    expect(serial != nullptr);
    if (serial == nullptr) { return; }
    expect(serial->operand_count() == 2u);
    expect(serial->result_count() == 2u);
    auto body = serial->region(0u)->block(0u);
    expect(body->argument_count() == 3u);
    if (serial->operand_count() != 2u || serial->result_count() != 2u || body->argument_count() != 3u) { return; }
    auto singleton = [](const tile::Type &type) noexcept {
        auto space = type.index_space();
        return type.is_tile() && space != nullptr && space->static_volume() == 1u;
    };
    auto yield = body->operation(body->operation_count() - 1u);
    expect(yield->kind() == tile::OperationKind::YIELD);
    expect(yield->operand_count() == 2u);
    auto float_carries = 0u, index_carries = 0u, point_loads = 0u;
    for (auto i = 0u; i < 2u; i++) {
        auto carry = body->argument(i + 1u);
        expect(singleton(carry->type()));
        float_carries += carry->type().scalar_type() == tile::ScalarType::FLOAT32;
        index_carries += carry->type().scalar_type() == tile::ScalarType::INT32;
        auto used = false;
        for (auto operation : body->operations()) {
            for (auto j = 0u; j < operation->operand_count(); j++) { used |= operation->operand(j) == carry; }
        }
        expect(used);
        if (yield->operand_count() == 2u) { expect(yield->operand(i)->type() == carry->type()); }
    }
    expect(float_carries == 1u && index_carries == 1u);
    for (auto operation : body->operations()) {
        if (operation->kind() == tile::OperationKind::VIEW_LOAD) {
            point_loads++;
            expect(operation->result_count() == 1u);
            if (operation->result_count() == 1u) { expect(singleton(operation->result(0u)->type())); }
        }
        if (operation->kind() == tile::OperationKind::TILE_EXTRACT) {
            expect(singleton(operation->operand(0u)->type()));
        }
    }
    expect(point_loads == 1u);
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(rows, 1u, 1u), 3u)) { return; }
    expect(shader.metadata().source.find("ct::reduce_max(") != luisa::string::npos);
    expect(shader.metadata().source.find("ct::reduce_min(") != luisa::string::npos);
    expect(shader.metadata().source.find("for (long long") != luisa::string::npos);
    expect(shader.metadata().source.find("ct::cat(") == luisa::string::npos);
    using Word = std::conditional_t<sizeof(T) == 2u, uint16_t, uint32_t>;
    auto word = [](T value) noexcept { return std::bit_cast<Word>(value); };
    auto guard = T{-719.5f};
    auto index_guard = std::numeric_limits<int64_t>::min() + 37;
    luisa::vector<T> input(rows * columns + 2u * kPad, guard), output(rows * count + 2u * kPad, guard);
    luisa::vector<int64_t> indices(rows * count + 2u * kPad, index_guard);
    const std::array mixed{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           5.0f, 5.0f, -0.0f, 0.0f, .25f, -.25f};
    for (auto i = 0u; i < columns; i++) {
        input[kPad + i] = T{mixed[i % mixed.size()]};
        input[kPad + columns + i] = T{i % 2u == 0u ? -0.0f : 0.0f};
        input[kPad + 2u * columns + i] = T{-std::numeric_limits<float>::infinity()};
        input[kPad + 3u * columns + i] = T{2.5f};
    }
    auto original = input;
    auto gpu_input = device.create_buffer<T>(input.size()), gpu_output = device.create_buffer<T>(output.size());
    auto gpu_indices = device.create_buffer<int64_t>(indices.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << gpu_input.copy_from(luisa::span{input}) << gpu_output.copy_from(luisa::span{output})
           << gpu_indices.copy_from(luisa::span{indices})
           << shader(gpu_input.view(kPad, rows * columns), gpu_output.view(kPad, rows * count), gpu_indices.view(kPad, rows * count)).dispatch()
           << gpu_input.copy_to(luisa::span{input}) << gpu_output.copy_to(luisa::span{output})
           << gpu_indices.copy_to(luisa::span{indices}) << synchronize();
    for (auto i = 0u; i < input.size(); i++) { expect(word(input[i]) == word(original[i])) << "readonly topk word=" << i; }
    for (auto i = 0u; i < kPad; i++) {
        expect(word(output[i]) == word(guard));
        expect(word(output[output.size() - 1u - i]) == word(guard));
        expect(indices[i] == index_guard);
        expect(indices[indices.size() - 1u - i] == index_guard);
    }
    luisa::vector<int64_t> order(columns);
    for (auto row = 0u; row < rows; row++) {
        std::iota(order.begin(), order.end(), int64_t{0});
        std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            auto x = static_cast<float>(original[kPad + row * columns + a]);
            auto y = static_cast<float>(original[kPad + row * columns + b]);
            return x == y ? a < b : x > y;
        });
        for (auto rank = 0u; rank < count; rank++) {
            auto p = kPad + row * count + rank;
            expect(indices[p] == order[rank]) << "stable topk row=" << row << " rank=" << rank;
            expect(word(output[p]) == word(original[kPad + row * columns + order[rank]])) << "exact topk row=" << row << " rank=" << rank;
        }
    }
}

void native_attention(Device &device) {
    for (auto queries : {1ll, 3ll}) {
        auto fixture = test::tile_llm::attention(2, 4, 2, queries, 5, 8, 4, 2, 4);
        auto shader = tile::compile(device, fixture.kernel);
        if (!check_native(shader, make_uint3(2u, 4u, static_cast<uint32_t>((queries + 1) / 2)), 4u)) { continue; }
        GuardedBuffer q{device, fixture.inputs[0].size()}, k{device, fixture.inputs[1].size()},
            v{device, fixture.inputs[2].size()}, y{device, fixture.expected.size()};
        for (auto i = 0u; i < q.count; i++) { q[i] = fixture.inputs[0][i]; }
        for (auto i = 0u; i < k.count; i++) { k[i] = fixture.inputs[1][i]; }
        for (auto i = 0u; i < v.count; i++) { v[i] = fixture.inputs[2][i]; }
        auto original_q = q.host, original_k = k.host, original_v = v.host;
        y.poison();
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << q.buffer.copy_from(luisa::span{q.host}) << k.buffer.copy_from(luisa::span{k.host})
               << v.buffer.copy_from(luisa::span{v.host}) << y.buffer.copy_from(luisa::span{y.host})
               << shader(q.view(), k.view(), v.view(), y.view()).dispatch()
               << q.buffer.copy_to(luisa::span{q.host}) << k.buffer.copy_to(luisa::span{k.host})
               << v.buffer.copy_to(luisa::span{v.host}) << y.buffer.copy_to(luisa::span{y.host}) << synchronize();
        check_readonly(q, original_q); check_readonly(k, original_k); check_readonly(v, original_v); y.check_guards();
        for (auto i = 0u; i < y.count; i++) {
            expect(std::isfinite(y[i]));
            expect(std::abs(static_cast<double>(y[i]) - fixture.expected[i]) <= 3e-5 * (1.0 + std::abs(fixture.expected[i]))) << "attention index=" << i;
        }
    }
}

void interleaved_singleton_mma(Device &device, bool reassociate) {
    using namespace tile;
    auto kernel = tile_kernel("interleaved_common_singletons", [=](TensorView<const float, 4> a, TensorView<const float, 4> b, TensorView<float, 4> c) {
                      auto x = axis("x", 1), y = axis("y", 1), m = axis("m", 4), n = axis("n", 8), k = axis("k", 4);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto lhs = a.tile(coord(0, 0, 0, 0), shape(k, x, m, y)).load();
                          auto rhs = b.tile(coord(0, 0, 0, 0), shape(y, n, x, k)).load();
                          auto result = mma(lhs, rhs, full<float>(shape(x, m, y, n), .25f), MmaPolicy{reassociate});
                          c(coord(0, 0, 0, 0), shape(x, m, y, n)).store(result);
                      }
                  }).capture(tensor_shape(4, 1, 4, 1), tensor_shape(1, 8, 1, 4), tensor_shape(1, 4, 1, 8));
    auto shader = tile::compile(device, kernel);
    if (!check_native(shader, make_uint3(1u), 3u)) { return; }
    expect((shader.metadata().source.find("ct::mma(") != luisa::string::npos) == reassociate);
    // These dyadic products and all intermediate sums are exact in FP32.
    GuardedBuffer a{device, 16u}, b{device, 32u}, c{device, 32u};
    for (auto i = 0u; i < a.count; i++) { a[i] = static_cast<float>(i) * .125f; }
    for (auto i = 0u; i < b.count; i++) { b[i] = static_cast<float>(i % 11u) * .25f; }
    c.poison(); auto original_a = a.host, original_b = b.host;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host}) << c.buffer.copy_from(luisa::span{c.host})
           << shader(a.view(), b.view(), c.view()).dispatch() << a.buffer.copy_to(luisa::span{a.host})
           << b.buffer.copy_to(luisa::span{b.host}) << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
    check_readonly(a, original_a); check_readonly(b, original_b); c.check_guards();
    for (auto m = 0u; m < 4u; m++) {
        for (auto n = 0u; n < 8u; n++) {
            auto expected = .25f;
            for (auto k = 0u; k < 4u; k++) { expected = std::fma(a[k * 4u + m], b[n * 4u + k], expected); }
            expect(bits(c[m * 8u + n]) == bits(expected));
        }
    }
}

template<typename T>
void narrow_storage_and_mma(Device &device) {
    using namespace tile;
    for (auto reassociate : {false, true}) {
        auto kernel = tile_kernel("declared_narrow_inputs", [=](TensorView<const T, 2> a, TensorView<const T, 2> b,
                                                               TensorView<float, 2> c, TensorView<T, 2> copy) {
                          auto m = axis("m", 16), n = axis("n", 16), k = axis("k", 16);
                          for (auto &nest : parallel(shape(1))) {
                              static_cast<void>(nest);
                              auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                              auto rhs = b.tile(coord(0, 0), shape(k, n)).load();
                              c(coord(0, 0), shape(m, n)).store(mma(lhs, rhs, full<float>(shape(m, n), .25f), MmaPolicy{reassociate}));
                              auto converted = cast<T>(cast<float>(lhs) + .5f);
                              copy(coord(0, 0), shape(m, k)).store(ite(iota(k) < 8, lhs, converted));
                          }
                      }).capture(tensor_shape(13, 11), tensor_shape(11, 15), tensor_shape(13, 15), tensor_shape(13, 11));
        auto shader = tile::compile(device, kernel);
        if (!check_native(shader, make_uint3(1u), 4u)) { continue; }
        expect(shader.metadata().arguments[0].element == scalar_type_v<T>);
        expect(shader.metadata().arguments[0].minimum_size_bytes == 13u * 11u * sizeof(T));
        expect((shader.metadata().source.find("ct::mma(") != luisa::string::npos) == reassociate);
        constexpr auto guard = uint16_t{0x3555u};
        luisa::vector<T> a(13u * 11u + 2u * kPad, std::bit_cast<T>(guard)), b(11u * 15u + 2u * kPad, std::bit_cast<T>(guard)), copy(a.size(), std::bit_cast<T>(guard));
        for (auto i = 0u; i < 13u * 11u; i++) { a[kPad + i] = T{static_cast<float>(static_cast<int>(i % 17u) - 8) * .125f}; }
        for (auto i = 0u; i < 11u * 15u; i++) { b[kPad + i] = T{static_cast<float>(static_cast<int>(i % 13u) - 6) * .25f}; }
        auto original_a = a, original_b = b;
        auto ga = device.create_buffer<T>(a.size()), gb = device.create_buffer<T>(b.size()), gc = device.create_buffer<T>(copy.size());
        GuardedBuffer c{device, 13u * 15u}; c.poison();
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << ga.copy_from(luisa::span{a}) << gb.copy_from(luisa::span{b}) << gc.copy_from(luisa::span{copy}) << c.buffer.copy_from(luisa::span{c.host})
               << shader(ga.view(kPad, 13u * 11u), gb.view(kPad, 11u * 15u), c.view(), gc.view(kPad, 13u * 11u)).dispatch()
               << ga.copy_to(luisa::span{a}) << gb.copy_to(luisa::span{b}) << gc.copy_to(luisa::span{copy}) << c.buffer.copy_to(luisa::span{c.host}) << synchronize();
        for (auto i = 0u; i < a.size(); i++) { expect(std::bit_cast<uint16_t>(a[i]) == std::bit_cast<uint16_t>(original_a[i])); }
        for (auto i = 0u; i < b.size(); i++) { expect(std::bit_cast<uint16_t>(b[i]) == std::bit_cast<uint16_t>(original_b[i])); }
        c.check_guards();
        for (auto i = 0u; i < kPad; i++) { expect(std::bit_cast<uint16_t>(copy[i]) == guard); expect(std::bit_cast<uint16_t>(copy[copy.size() - 1u - i]) == guard); }
        for (auto m = 0u; m < 13u; m++) {
            for (auto n = 0u; n < 15u; n++) {
                auto expected = .25f;
                for (auto k = 0u; k < 11u; k++) { expected = std::fma(static_cast<float>(a[kPad + m * 11u + k]), static_cast<float>(b[kPad + k * 15u + n]), expected); }
                expect(bits(c[m * 15u + n]) == bits(expected));
            }
            for (auto k = 0u; k < 11u; k++) {
                auto value = a[kPad + m * 11u + k];
                auto expected = k < 8u ? value : T{static_cast<float>(value) + .5f};
                expect(std::bit_cast<uint16_t>(copy[kPad + m * 11u + k]) == std::bit_cast<uint16_t>(expected));
            }
        }
    }
}

// The public ShaderOption selects a bounded FP32 elementwise policy. The
// default source, native tree intrinsics, MMA, casts and NaN min/max stay
// strict; ordered/custom reduction bodies honor the elementwise policy.
void explicit_fast_math(Device &device) {
    using namespace tile;
    constexpr auto count = 32u, operations = 10u;
    auto kernel = tile_kernel("explicit_fast_math_policy", [=](TensorView<const float, 1> a,
                                                             TensorView<const float, 1> b,
                                                             TensorView<float, 1> output) {
                      auto n = axis("n", count);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0), shape(n)).load();
                          auto y = b.tile(coord(0), shape(n)).load();
                          output(coord(0), shape(n)).store(x + y);
                          output(coord(count), shape(n)).store(x - y);
                          output(coord(2u * count), shape(n)).store(x * y);
                          output(coord(3u * count), shape(n)).store(x / y);
                          output(coord(4u * count), shape(n)).store(sqrt(abs(x)));
                          output(coord(5u * count), shape(n)).store(exp(x));
                          output(coord(6u * count), shape(n)).store(tanh(x));
                          output(coord(7u * count), shape(n)).store(min(x, 0.25f));
                          output(coord(8u * count), shape(n)).store(max(x, -0.25f));
                          output(coord(9u * count), shape(n)).store(x);
                      }
                  }).capture(tensor_shape(count), tensor_shape(count), tensor_shape(count * operations));
    auto defaults = tile::compile(device, kernel);
    auto strict = tile::compile(device, kernel, {}, {.enable_fast_math = false});
    auto fast = tile::compile(device, kernel, {}, {.enable_fast_math = true});
    if (!check_native(defaults, make_uint3(1u), 3u) ||
        !check_native(strict, make_uint3(1u), 3u) ||
        !check_native(fast, make_uint3(1u), 3u)) { return; }
    expect(defaults.metadata().source == strict.metadata().source);
    expect(defaults.metadata().source.find("round_approximate_t") == string::npos);
    expect(defaults.metadata().source.find("round_subnormals_to_zero_t") == string::npos);
    expect(fast.metadata().source.find("ct::round_approximate_t{}") != string::npos);
    expect(fast.metadata().source.find("ct::round_subnormals_to_zero_t{}") != string::npos);
    expect(fast.metadata().realization.find("elementwise-fp32-approx-ftz-rsqrt-v2") != string::npos);

    GuardedBuffer a{device, count}, b{device, count}, output{device, count * operations};
    for (auto i = 0u; i < 16u; i++) {
        a[i] = (static_cast<float>(i) - 7.25f) * 0.375f;
        b[i] = (static_cast<float>(i) + 0.375f) * (i % 2u == 0u ? 0.25f : -0.25f);
    }
    constexpr auto inf = std::numeric_limits<float>::infinity();
    constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
    std::array<float, 16u> special_a{0.0f, -0.0f, 0x1p-149f, -0x1p-149f,
                                    0x1p-126f, -0x1p-126f, inf, -inf,
                                    nan, 1.0f, -1.0f, 0.0f, -0.0f,
                                    0x1p-126f, -0x1p-126f, 1.0f};
    std::array<float, 16u> special_b{2.0f, 2.0f, 2.0f, 2.0f,
                                    0.5f, 0.5f, 2.0f, 2.0f,
                                    2.0f, 0.0f, -0.0f, 0.0f, -0.0f,
                                    2.0f, 2.0f, 0x1p127f};
    for (auto i = 0u; i < 16u; i++) { a[i + 16u] = special_a[i]; b[i + 16u] = special_b[i]; }
    auto original_a = a.host, original_b = b.host;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto flush = [](float x) noexcept {
        return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x;
    };
    for (auto use_fast : {false, true}) {
        auto &shader = use_fast ? fast : strict;
        output.poison();
        stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
               << output.buffer.copy_from(luisa::span{output.host})
               << shader(a.view(), b.view(), output.view()).dispatch()
               << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
               << output.buffer.copy_to(luisa::span{output.host}) << synchronize();
        check_readonly(a, original_a);
        check_readonly(b, original_b);
        output.check_guards();
        for (auto i = 0u; i < count; i++) {
            auto x = static_cast<double>(use_fast ? flush(a[i]) : a[i]);
            auto y = static_cast<double>(use_fast ? flush(b[i]) : b[i]);
            std::array<double, operations> expected{x + y, x - y, x * y, x / y, std::sqrt(std::abs(x)),
                                                    std::exp(static_cast<double>(a[i])), std::tanh(static_cast<double>(a[i])),
                                                    std::fmin(static_cast<double>(a[i]), .25), std::fmax(static_cast<double>(a[i]), -.25), a[i]};
            for (auto op = 0u; op < operations; op++) {
                auto actual = output[op * count + i];
                if (op == 9u) { expect(bits(actual) == bits(a[i])) << "copy index=" << i; continue; }
                auto rounded = static_cast<float>(expected[op]);
                if (use_fast && op <= 4u) { rounded = flush(rounded); }
                if (std::isnan(rounded)) { expect(std::isnan(actual)) << "fast=" << use_fast << " op=" << op << " index=" << i; }
                else if (std::isinf(rounded)) { expect(actual == rounded) << "fast=" << use_fast << " op=" << op << " index=" << i; }
                else if (op <= 2u || op == 7u || op == 8u || rounded == 0.0f) {
                    expect(bits(actual) == bits(rounded)) << "fast=" << use_fast << " op=" << op << " index=" << i;
                } else {
                    auto error = std::abs(static_cast<double>(actual) - expected[op]);
                    expect(std::isfinite(actual) && error <= 2e-6 * std::abs(expected[op]) + 1e-44)
                        << "fast=" << use_fast << " op=" << op << " index=" << i << " error=" << error;
                }
            }
        }
    }
    // This kernel has integer address arithmetic and MMA only. Selecting fast
    // elementwise math must not change its generated FMA order/policy at all.
    for (auto reassociate : {false, true}) {
        auto contraction = matrix(false, false, reassociate);
        auto strict_mma = tile::compile(device, contraction);
        auto fast_mma = tile::compile(device, contraction, {}, {.enable_fast_math = true});
        if (check_native(strict_mma, make_uint3(2u, 3u, 1u), 3u) &&
            check_native(fast_mma, make_uint3(2u, 3u, 1u), 3u)) {
            expect(strict_mma.metadata().source == fast_mma.metadata().source);
            expect(strict_mma.metadata().source.find(reassociate ? "ct::mma(" : "ct::fma(") != string::npos);
        }
    }
}


// Exact producer matching, both named-axis alignments, shared SQRT users,
// scalar-valued map bodies, and explicit nonmatches all use real DSL capture.
void fast_div_sqrt(Device &device) {
    using namespace tile;
    constexpr auto rows = 8u, columns = 8u, count = rows * columns, operations = 7u;
    auto kernel = tile_kernel("fast_div_sqrt_contract", [=](TensorView<const float, 2> a,
                                                          TensorView<const float, 2> b,
                                                          TensorView<const float, 1> row,
                                                          TensorView<float, 2> output) {
                      auto m = axis("m", rows), n = axis("n", columns);
                      for (auto &nest : parallel(shape(1))) {
                          static_cast<void>(nest);
                          auto x = a.tile(coord(0, 0), shape(m, n)).load();
                          auto y = b.tile(coord(0, 0), shape(n, m)).load();
                          auto root = sqrt(y);
                          auto aligned_y = broadcast_to(y, shape(m, n));
                          auto aligned_root = broadcast_to(root, shape(m, n));
                          output(coord(0, 0), shape(m, n)).store(x / root);
                          output(coord(rows, 0), shape(m, n)).store(aligned_root);
                          auto row_root = sqrt(row.tile(coord(0), shape(m)).load());
                          output(coord(2u * rows, 0), shape(m, n)).store(x / row_root);
                          auto mapped = map<float>(shape(m, n), [&](const Nest &index) {
                              return x.at(index) / sqrt(aligned_y.at(index));
                          });
                          output(coord(3u * rows, 0), shape(m, n)).store(mapped);
                          // These DIV producers are map-broadcast, ADD, CONSTANT.
                          // They must keep their ordinary fast division path.
                          output(coord(4u * rows, 0), shape(m, n)).store(x / aligned_root);
                          output(coord(5u * rows, 0), shape(m, n)).store(x / (root + 1.0f));
                          output(coord(6u * rows, 0), shape(m, n)).store(x / 2.0f);
                      }
                  }).capture(tensor_shape(rows, columns), tensor_shape(columns, rows), tensor_shape(rows), tensor_shape(operations * rows, columns));
    auto defaults = tile::compile(device, kernel);
    auto strict = tile::compile(device, kernel, {}, {.enable_fast_math = false});
    auto fast = tile::compile(device, kernel, {}, {.enable_fast_math = true});
    if (!check_native(defaults, make_uint3(1u), 4u) || !check_native(strict, make_uint3(1u), 4u) || !check_native(fast, make_uint3(1u), 4u)) { return; }
    expect(defaults.metadata().source == strict.metadata().source);
    expect(strict.metadata().source.find("ct::rsqrt(") == string::npos);
    auto occurrences = [](string_view source, string_view needle) noexcept {
        auto count = size_t{0u}, cursor = size_t{0u};
        while ((cursor = source.find(needle, cursor)) != string_view::npos) { count++; cursor += needle.size(); }
        return count;
    };
    expect(occurrences(fast.metadata().source, "ct::rsqrt(") == 3u);
    expect(occurrences(fast.metadata().source, "ct::div(") == 3u);
    expect(fast.metadata().source.find("ct::sqrt(") != string::npos);
    expect(fast.metadata().source.find("ct::permute(") != string::npos);
    expect(fast.metadata().realization.find("elementwise-fp32-approx-ftz-rsqrt-v2") != string::npos);

    GuardedBuffer a{device, count}, b{device, count}, row{device, rows}, output{device, operations * count};
    for (auto i = 0u; i < 32u; i++) {
        a[i] = (static_cast<float>(i) - 15.25f) * .125f;
        b[(i % columns) * rows + i / columns] = std::ldexp(1.125f + static_cast<float>(i % 3u) * .25f, static_cast<int>(i) - 16);
    }
    constexpr auto inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    constexpr auto tiny = std::numeric_limits<float>::denorm_min(), normal = std::numeric_limits<float>::min(), large = std::numeric_limits<float>::max();
    std::array<float, 32u> xs{0.0f, -0.0f, 1.0f, -1.0f, 0.0f, -0.0f, inf, -inf,
                              nan, 1.0f, 1.0f, 1.0f, -1.0f, normal, large, tiny,
                              -tiny, tiny, -tiny, 1.0f, -1.0f, inf, -inf, 0.0f,
                              -0.0f, large, normal, -normal, large, -large, 1.0f, -1.0f};
    std::array<float, 32u> ys{0.0f, -0.0f, 0.0f, -0.0f, inf, inf, inf, inf,
                              1.0f, -1.0f, -tiny, tiny, -normal, large, normal, 2.0f,
                              2.0f, 0.0f, -0.0f, nan, -inf, 4.0f, 4.0f, -4.0f,
                              -4.0f, large, normal, normal, 0.0f, -0.0f, 0x1p-126f, 0x1.fffffep127f};
    for (auto j = 0u; j < xs.size(); j++) {
        auto i = j + 32u;
        a[i] = xs[j];
        b[(i % columns) * rows + i / columns] = ys[j];
    }
    std::array<float, rows> row_values{.25f, 2.0f, 16.0f, 1e-5f, normal, large, .0625f, 1024.0f};
    for (auto i = 0u; i < rows; i++) { row[i] = row_values[i]; }
    auto original_a = a.host, original_b = b.host, original_row = row.host;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto flush = [](float value) noexcept { return std::fpclassify(value) == FP_SUBNORMAL ? std::copysign(0.0f, value) : value; };
    for (auto use_fast : {false, true}) {
        auto &shader = use_fast ? fast : strict;
        output.poison();
        stream << a.buffer.copy_from(luisa::span{a.host}) << b.buffer.copy_from(luisa::span{b.host})
               << row.buffer.copy_from(luisa::span{row.host}) << output.buffer.copy_from(luisa::span{output.host})
               << shader(a.view(), b.view(), row.view(), output.view()).dispatch()
               << a.buffer.copy_to(luisa::span{a.host}) << b.buffer.copy_to(luisa::span{b.host})
               << row.buffer.copy_to(luisa::span{row.host}) << output.buffer.copy_to(luisa::span{output.host}) << synchronize();
        check_readonly(a, original_a);
        check_readonly(b, original_b);
        check_readonly(row, original_row);
        output.check_guards();
        for (auto i = 0u; i < count; i++) {
            auto numerator = use_fast ? flush(a[i]) : a[i];
            auto radicand = b[(i % columns) * rows + i / columns];
            if (use_fast) { radicand = flush(radicand); }
            auto root = std::sqrt(static_cast<double>(radicand));
            auto rounded_root = static_cast<float>(root);
            if (use_fast) { rounded_root = flush(rounded_root); }
            auto biased_root = static_cast<float>(static_cast<double>(rounded_root) + 1.0);
            if (use_fast) { biased_root = flush(biased_root); }
            auto row_root = std::sqrt(static_cast<double>(row[i / columns]));
            std::array<double, operations> expected{static_cast<double>(numerator) / root, root,
                static_cast<double>(numerator) / row_root, static_cast<double>(numerator) / root,
                static_cast<double>(numerator) / rounded_root, static_cast<double>(numerator) / biased_root,
                static_cast<double>(numerator) / 2.0};
            for (auto op = 0u; op < operations; op++) {
                auto actual = output[op * count + i];
                auto rounded = static_cast<float>(expected[op]);
                if (use_fast) { rounded = flush(rounded); }
                if (std::isnan(rounded)) { expect(std::isnan(actual)) << "fast=" << use_fast << " op=" << op << " index=" << i; }
                else if (std::isinf(rounded)) { expect(actual == rounded) << "fast=" << use_fast << " op=" << op << " index=" << i; }
                else if (rounded == 0.0f) { expect(bits(actual) == bits(rounded)) << "fast=" << use_fast << " op=" << op << " index=" << i; }
                else {
                    auto error = std::abs(static_cast<double>(actual) - expected[op]);
                    expect(std::isfinite(actual) && error <= 2e-6 * std::abs(expected[op]) + 1e-44)
                        << "fast=" << use_fast << " op=" << op << " index=" << i << " error=" << error;
                }
            }
        }
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
    auto narrow_arithmetic = tile_kernel("explicit_narrow_arithmetic", [](TensorView<const half, 1> a, TensorView<half, 1> b) {
                                 auto n = axis("n", 16);
                                 for (auto &nest : parallel(shape(1))) {
                                     static_cast<void>(nest);
                                     auto x = a.tile(coord(0), shape(n)).load();
                                     b(coord(0), shape(n)).store(x + x);
                                 }
                             }).capture(tensor_shape(16), tensor_shape(16));
    expect_rejected(tile::compile(device, narrow_arithmetic, {}, strict), "narrow elementwise");
    auto batched = tile_kernel("non_singleton_batch_mma", [](TensorView<const float, 3> a, TensorView<const float, 3> b, TensorView<float, 3> c) {
                       auto batch = axis("batch", 2), m = axis("m", 4), n = axis("n", 4), k = axis("k", 4);
                       for (auto &nest : parallel(shape(1))) {
                           static_cast<void>(nest);
                           auto x = a.tile(coord(0, 0, 0), shape(batch, m, k)).load();
                           auto y = b.tile(coord(0, 0, 0), shape(batch, k, n)).load();
                           c(coord(0, 0, 0), shape(batch, m, n)).store(mma(x, y, zeros<float>(shape(batch, m, n))));
                       }
                   }).capture(tensor_shape(2, 4, 4), tensor_shape(2, 4, 4), tensor_shape(2, 4, 4));
    expect_rejected(tile::compile(device, batched, {}, strict), "singleton shared batch");
    auto grid4 = tile_kernel("four_dimensional_grid", [](TensorView<float, 1> output) {
                     auto n = axis("n", 16);
                     for (auto &nest : parallel(shape(1, 1, 1, 1))) {
                         static_cast<void>(nest);
                         output(coord(0), shape(n)).store(zeros<float>(shape(n)));
                     }
                 }).capture(tensor_shape(16));
    expect_rejected(tile::compile(device, grid4, {}, strict), "launch grids");
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
        fp32_mma_precision_boundaries(device);
    };
    "tile_cuda_ir_ordered_contraction"_test = [&] { matrix_oracle(device, false, false, false); };
    "tile_cuda_ir_fma_unroll_boundary"_test = [&] {
        for (auto tile_k : {32u, 64u, 128u, 256u}) {
            for (auto transpose_a : {false, true}) {
                for (auto transpose_b : {false, true}) {
                    matrix_oracle(device, transpose_a, transpose_b, false, tile_k, tile_k + 3u);
                }
            }
        }
        large_accumulator_dynamic_contraction(device);
        fma_operation_budget_boundary(device);
    };
    "tile_cuda_ir_pointwise_exact_alias"_test = [&] { pointwise_alias(device); };
    "tile_cuda_ir_read_before_write_snapshot"_test = [&] { snapshot_ordering(device); };
    "tile_cuda_ir_negative_origin_bitwise_copy"_test = [&] { shifted_bitwise_copy(device); };
    "tile_cuda_ir_proven_pointer_memory_bounds"_test = [&] { proven_pointer_memory_bounds(device); };
    "tile_cuda_ir_simultaneous_carry_swap"_test = [&] { swap_oracle(device); };
    "tile_cuda_ir_llm_rows"_test = [&] { row_operations(device); };
    "tile_cuda_ir_named_axis_broadcast"_test = [&] { named_axis_broadcast(device); };
    "tile_cuda_ir_ordered_reductions"_test = [&] { ordered_reductions(device); };
    "tile_cuda_ir_elementary_math"_test = [&] { elementary_math(device); };
    "tile_cuda_ir_native_scan_and_sort"_test = [&] {
        native_scan_and_sort(device);
        blocked_row_operations<float>(device, 1u);
        blocked_row_operations<half>(device, 4u);
        blocked_row_operations<tile::bfloat16>(device, 8u);
        blocked_row_operations<float>(device, 8u, 3u, 1u, true);
        chunked_sort_pipeline<float>(device, 129, 256, 1u);
        chunked_sort_pipeline<float>(device, 257, 256, 2u);
        chunked_sort_pipeline<half>(device, 769, 256, 3u);
        chunked_sort_pipeline<tile::bfloat16>(device, 1537, 256, 4u);
        chunked_sort_pipeline<float>(device, 769, 512, 2u);
        chunked_sort_pipeline<half>(device, 1537, 512, 3u);
    };
    "tile_cuda_ir_repeated_extrema_topk"_test = [&] {
        repeated_extrema_topk<float>(device, 1u, 1u);
        repeated_extrema_topk<float>(device, 33u, 7u);
        repeated_extrema_topk<float>(device, 33u, 33u);
        repeated_extrema_topk<float>(device, 65u, 16u);
        repeated_extrema_topk<half>(device, 33u, 7u);
        repeated_extrema_topk<tile::bfloat16>(device, 33u, 7u);
        for (auto width : {1u, 65u}) {
            stable_argmax<float>(device, width);
            stable_argmax<half>(device, width);
            stable_argmax<tile::bfloat16>(device, width);
        }
        stable_argmax<float>(device, 33u);
        embedding_rows<float>(device, 1u, 1u, 1u);
        embedding_rows<float>(device, 65u, 37u, 32u);
        embedding_rows<half>(device, 65u, 37u, 32u);
        embedding_rows<tile::bfloat16>(device, 65u, 37u, 32u);
    };
    "tile_cuda_ir_integer_primitives"_test = [&] {
        for (auto width : {1u, 32u, 33u}) {
            integer_primitives<int32_t>(device, width);
            integer_primitives<uint32_t>(device, width);
            integer_primitives<int64_t>(device, width);
            integer_primitives<uint64_t>(device, width);
        }
    };
    "tile_cuda_ir_attention_rank4"_test = [&] { native_attention(device); };
    "tile_cuda_ir_interleaved_singleton_mma"_test = [&] {
        for (auto reassociate : {false, true}) { interleaved_singleton_mma(device, reassociate); }
    };
    "tile_cuda_ir_narrow_storage_and_mma"_test = [&] { narrow_storage_and_mma<half>(device); narrow_storage_and_mma<tile::bfloat16>(device); };
    "tile_cuda_ir_typed_buffer_views"_test = [&] {
        typed_buffer_copy<bool>(device);
        typed_buffer_copy<int32_t>(device);
        typed_buffer_copy<uint32_t>(device);
        typed_buffer_copy<int64_t>(device);
        typed_buffer_copy<uint64_t>(device);
    };
    "tile_cuda_ir_explicit_fast_math"_test = [&] { explicit_fast_math(device); };
    "tile_cuda_ir_fast_div_sqrt"_test = [&] { fast_div_sqrt(device); row_operations(device, true); };
    "tile_cuda_ir_rejects_unsupported_options"_test = [&] { rejected_options(device); };
    "tile_cuda_ir_rejects_shapes_constraints_and_types"_test = [&] { rejected_shapes_and_constraints(device); };
    return 0;
}
