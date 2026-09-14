// CUDA native-Tile end-to-end tests: LuisaCompute tile kernels lowered through
// the shared TIRx bridge to a CUDA device artifact, compiled to PTX by the
// standalone NVRTC pipeline, and launched with a direct static cuLaunchKernel.
//
// When built without the TIRx bridge (LUISA_TEST_TILE_CUDA_TIRX undefined),
// the same executable verifies the backend fails closed with a clear error
// instead of silently falling back.

#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_llm_test_utils.h"
#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/stream.h>

#ifdef LUISA_TEST_TILE_CUDA_TIRX
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] bool close_tolerance(float actual, double expected) noexcept {
    return std::isfinite(actual) &&
           std::abs(static_cast<double>(actual) - expected) <=
               2e-4 + 2e-4 * std::abs(expected);
}

void test_elementwise(Device &device, tile::CompileOptions options) {
    constexpr auto n = 1003;
    auto definition = tile::tile_kernel("cuda_tile_axpy", [](tile::TensorView<const float, 1> x,
                                                             tile::TensorView<float, 1> result) {
        auto element = tile::axis("element", n);
        for (auto &item : tile::parallel(tile::shape(element))) {
            auto index = item.index();
            result(index).store(1.25f * x(index).load() + 0.5f);
        }
    });
    auto kernel = definition.capture(tile::tensor_shape("x", n), tile::tensor_shape("result", n));
    luisa::vector<float> x_values(n);
    for (auto i = 0u; i < n; i++) { x_values[i] = static_cast<float>(i % 37u) * 0.125f - 2.0f; }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(n);
    auto r = device.create_buffer<float>(n);
    luisa::vector<float> actual(n);
    stream << x.copy_from(luisa::span{x_values})
           << shader(x, r).dispatch() << r.copy_to(luisa::span{actual}) << synchronize();
    for (auto i = 0u; i < n; i++) {
        expect(close_tolerance(actual[i], 1.25 * x_values[i] + 0.5));
    }
}

void test_row_sum(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 37;
    constexpr auto columns = 19;
    auto definition = tile::tile_kernel("cuda_tile_row_sum", [](tile::TensorView<const float, 2> x,
                                                                tile::TensorView<float, 1> result) {
        auto row = tile::axis("row", rows);
        auto column = tile::axis("column", columns);
        for (auto &row_nest : tile::parallel(tile::shape(row))) {
            auto sum = tile::Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(tile::shape(column))) {
                sum += x(row_nest.index(row), item.index(column)).load();
            }
            result(row_nest.index(row)).store(sum);
        }
    });
    auto kernel = definition.capture(
        tile::tensor_shape("x", rows, columns), tile::tensor_shape("result", rows));
    luisa::vector<float> values(rows * columns);
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 23u) - 11) * 0.0625f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    expect(shader.metadata().realization.find("TIRx -> ") != luisa::string::npos);
    expect(!shader.metadata().source.empty());
    expect(shader.metadata().dispatch_size.x * shader.metadata().dispatch_size.y *
               shader.metadata().dispatch_size.z >
           0u);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto r = device.create_buffer<float>(rows);
    luisa::vector<float> actual(rows);
    stream << x.copy_from(luisa::span{values})
           << shader(x, r).dispatch() << r.copy_to(luisa::span{actual}) << synchronize();
    for (auto row = 0; row < rows; row++) {
        auto expected = 0.0;
        for (auto column = 0; column < columns; column++) {
            expected += values[static_cast<size_t>(row * columns + column)];
        }
        expect(close_tolerance(actual[row], expected));
    }
}

void test_row_extrema(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 17;
    constexpr auto columns = 31;
    auto definition = tile::tile_kernel("cuda_tile_extrema", [](tile::TensorView<const float, 2> input,
                                                                tile::TensorView<float, 1> minima,
                                                                tile::TensorView<float, 1> maxima) {
        auto row = tile::axis("row", rows);
        auto one = tile::axis("one", 1);
        auto column = tile::axis("column", columns);
        for (auto &nest : tile::parallel(tile::shape(row))) {
            auto value = input[coord(nest.index(), 0), tile::shape(one, column)];
            minima(coord(nest.index()), tile::shape(one))
                .store(tile::reduce(value, column, tile::minimum));
            maxima(coord(nest.index()), tile::shape(one))
                .store(tile::reduce(value, column, tile::maximum));
        }
    });
    auto kernel = definition.capture(tile::tensor_shape(rows, columns),
                                     tile::tensor_shape(rows),
                                     tile::tensor_shape(rows));
    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 47u) - 23) * 0.125f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto lo = device.create_buffer<float>(rows);
    auto hi = device.create_buffer<float>(rows);
    luisa::vector<float> actual_lo(rows), actual_hi(rows);
    stream << x.copy_from(luisa::span{values})
           << shader(x, lo, hi).dispatch()
           << lo.copy_to(luisa::span{actual_lo}) << hi.copy_to(luisa::span{actual_hi}) << synchronize();
    for (auto row = 0; row < rows; row++) {
        auto expected_lo = std::numeric_limits<double>::infinity();
        auto expected_hi = -std::numeric_limits<double>::infinity();
        for (auto column = 0; column < columns; column++) {
            auto v = static_cast<double>(values[static_cast<size_t>(row * columns + column)]);
            expected_lo = std::min(expected_lo, v);
            expected_hi = std::max(expected_hi, v);
        }
        expect(close_tolerance(actual_lo[row], expected_lo));
        expect(close_tolerance(actual_hi[row], expected_hi));
    }
}

void test_row_softmax(Device &device, tile::CompileOptions options) {
    constexpr auto rows = 9;
    constexpr auto columns = 13;
    auto definition = tile::tile_kernel("cuda_tile_row_softmax", [](tile::TensorView<const float, 2> input,
                                                                    tile::TensorView<float, 2> output) {
        auto row = tile::axis("row", rows);
        auto one = tile::axis("one", 1);
        auto column = tile::axis("column", columns);
        for (auto &nest : tile::parallel(tile::shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto x = input.tile(origin, tile::shape(one, column)).load();
            auto e = tile::exp(x - tile::reduce(x, column, tile::maximum));
            output(origin, tile::shape(one, column)).store(e / tile::reduce(e, column, tile::add));
        }
    });
    auto kernel = definition.capture(tile::tensor_shape(rows, columns),
                                     tile::tensor_shape(rows, columns));
    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 31u) - 15) * 0.5f;
    }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(values.size());
    auto out = device.create_buffer<float>(values.size());
    luisa::vector<float> actual(values.size());
    stream << x.copy_from(luisa::span{values})
           << shader(x, out).dispatch() << out.copy_to(luisa::span{actual}) << synchronize();
    for (auto row = 0; row < rows; row++) {
        double sum = 0.0;
        for (auto column = 0; column < columns; column++) {
            sum += std::exp(static_cast<double>(values[static_cast<size_t>(row * columns + column)]));
        }
        for (auto column = 0; column < columns; column++) {
            auto expected = std::exp(static_cast<double>(values[static_cast<size_t>(row * columns + column)])) / sum;
            expect(close_tolerance(actual[static_cast<size_t>(row * columns + column)], expected));
        }
    }
}

struct GemmConfig {
    int64_t m{32}, n{32}, k{16}, tile{16};
    bool transpose_a{false}, transpose_b{false};
    bool allow_reassociation{true};
};

[[nodiscard]] tile::Kernel gemm_kernel(GemmConfig cfg) {
    using namespace tile;
    auto definition = tile_kernel("cuda_tile_gemm", [=](TensorView<const float, 2> A,
                                                        TensorView<const float, 2> B,
                                                        TensorView<float, 2> C) {
        auto gm = axis("groups_m", ceil_div(cfg.m, cfg.tile));
        auto gn = axis("groups_n", ceil_div(cfg.n, cfg.tile));
        auto row = axis("row", cfg.tile);
        auto col = axis("column", cfg.tile);
        auto k_axis = axis("k", cfg.k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest.index(gm) * cfg.tile;
            auto n0 = nest.index(gn) * cfg.tile;
            auto a = cfg.transpose_a ? A.tile(coord(0, m0), shape(k_axis, row)).load() :
                                       A.tile(coord(m0, 0), shape(row, k_axis)).load();
            auto b = cfg.transpose_b ? B.tile(coord(n0, 0), shape(col, k_axis)).load() :
                                       B.tile(coord(0, n0), shape(k_axis, col)).load();
            auto acc = zeros<float>(shape(row, col));
            acc = mma(a, b, acc, MmaPolicy{cfg.allow_reassociation});
            C(coord(m0, n0), shape(row, col)).store(acc);
        }
    });
    return definition.capture(
        tensor_shape(cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k),
        tensor_shape(cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n),
        tensor_shape(cfg.m, cfg.n));
}

void test_gemm(Device &device, tile::CompileOptions options, GemmConfig cfg) {
    auto kernel = gemm_kernel(cfg);
    auto a_elements = cfg.m * cfg.k;
    auto b_elements = cfg.k * cfg.n;
    auto c_elements = cfg.m * cfg.n;
    luisa::vector<float> a(static_cast<size_t>(a_elements));
    luisa::vector<float> b(static_cast<size_t>(b_elements));
    luisa::vector<float> actual(static_cast<size_t>(c_elements));
    for (auto i = 0u; i < a.size(); i++) { a[i] = std::sin(static_cast<float>(i) * .371f + .13f) * 1.375f; }
    for (auto i = 0u; i < b.size(); i++) { b[i] = std::cos(static_cast<float>(i) * .213f + .47f) * .875f; }
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto ab = device.create_buffer<float>(a.size());
    auto bb = device.create_buffer<float>(b.size());
    auto cb = device.create_buffer<float>(static_cast<size_t>(c_elements));
    stream << ab.copy_from(luisa::span{a}) << bb.copy_from(luisa::span{b})
           << shader(ab, bb, cb).dispatch() << cb.copy_to(luisa::span{actual}) << synchronize();
    for (auto m = 0; m < cfg.m; m++) {
        for (auto n = 0; n < cfg.n; n++) {
            double expected = 0.0;
            for (auto k = 0; k < cfg.k; k++) {
                auto ai = cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k;
                auto bi = cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n;
                expected += static_cast<double>(a[ai]) * b[bi];
            }
            expect(close_tolerance(actual[m * cfg.n + n], expected));
        }
    }
}

// Ragged GEMM edges plus nonzero BufferView offsets on every tensor. The
// direct-buffer ABI folds each view start into the raw CUdeviceptr, so guard
// bytes around every padded buffer must stay untouched after the launch.
void test_gemm_ragged_views(Device &device, tile::CompileOptions options) {
    constexpr auto pad = 13u;
    constexpr auto guard = -1.25f;
    for (auto ta : {false, true}) {
        for (auto tb : {false, true}) {
            GemmConfig cfg{48, 33, 24, 16, ta, tb, true};
            auto kernel = gemm_kernel(cfg);
            auto a_elements = static_cast<size_t>(cfg.m * cfg.k);
            auto b_elements = static_cast<size_t>(cfg.k * cfg.n);
            auto c_elements = static_cast<size_t>(cfg.m * cfg.n);
            luisa::vector<float> a(a_elements);
            luisa::vector<float> b(b_elements);
            for (auto i = 0u; i < a.size(); i++) {
                a[i] = std::sin(static_cast<float>(i) * .371f + .13f) * 1.375f;
            }
            for (auto i = 0u; i < b.size(); i++) {
                b[i] = std::cos(static_cast<float>(i) * .213f + .47f) * .875f;
            }
            auto full_a = luisa::vector<float>(a.size() + 2u * pad, guard);
            auto full_b = luisa::vector<float>(b.size() + 2u * pad, guard);
            auto full_c = luisa::vector<float>(c_elements + 2u * pad, guard);
            std::copy(a.begin(), a.end(), full_a.begin() + pad);
            std::copy(b.begin(), b.end(), full_b.begin() + pad);
            auto shader = tile::compile(device, kernel, options);
            expect(static_cast<bool>(shader)) << shader.metadata().error;
            if (!shader) { return; }
            auto stream = device.create_stream(StreamTag::COMPUTE);
            auto ab = device.create_buffer<float>(full_a.size());
            auto bb = device.create_buffer<float>(full_b.size());
            auto cb = device.create_buffer<float>(full_c.size());
            stream << ab.copy_from(luisa::span{full_a})
                   << bb.copy_from(luisa::span{full_b})
                   << cb.copy_from(luisa::span{full_c})
                   << shader(ab.view(pad, a_elements), bb.view(pad, b_elements),
                             cb.view(pad, c_elements))
                          .dispatch()
                   << cb.copy_to(luisa::span{full_c}) << synchronize();
            for (auto i = 0u; i < pad; i++) {
                expect(full_c[i] == guard);
                expect(full_c[pad + c_elements + i] == guard);
            }
            for (int64_t m = 0; m < cfg.m; m++) {
                for (int64_t n = 0; n < cfg.n; n++) {
                    double expected = 0.0;
                    for (int64_t k = 0; k < cfg.k; k++) {
                        auto ai = cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k;
                        auto bi = cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n;
                        expected += static_cast<double>(a[ai]) * b[bi];
                    }
                    auto index = pad + static_cast<size_t>(m * cfg.n + n);
                    expect(close_tolerance(full_c[index], expected))
                        << "m=" << m << " n=" << n;
                }
            }
        }
    }
}

struct BatchedGemmConfig {
    int64_t batch{3}, m{48}, n{33}, k{24}, tile{16};
    bool transpose_a{false}, transpose_b{false};
};

// A static batch axis over stacked 4D GEMM views with ragged interior (M/N not
// multiples of the 16-tile). Per-batch oracles are checked independently.
[[nodiscard]] tile::Kernel batched_gemm_kernel(BatchedGemmConfig cfg) {
    using namespace tile;
    auto definition = tile_kernel("cuda_tile_batched_gemm", [=](TensorView<const float, 4> A,
                                                                TensorView<const float, 4> B,
                                                                TensorView<float, 4> C) {
        auto batch = axis("batch", cfg.batch);
        auto head = axis("head", 1);
        auto one_b = axis("b", 1);
        auto one_h = axis("h", 1);
        auto groups_m = axis("groups_m", ceil_div(cfg.m, cfg.tile));
        auto groups_n = axis("groups_n", ceil_div(cfg.n, cfg.tile));
        auto row = axis("row", cfg.tile);
        auto col = axis("column", cfg.tile);
        auto k_axis = axis("k", cfg.k);
        for (auto &nest : parallel(shape(batch, head, groups_m, groups_n))) {
            auto b0 = nest.index(batch);
            auto h0 = nest.index(head);
            auto m0 = nest.index(groups_m) * cfg.tile;
            auto n0 = nest.index(groups_n) * cfg.tile;
            auto a = cfg.transpose_a ?
                         A.tile(coord(b0, h0, 0, m0), shape(one_b, one_h, k_axis, row)).load() :
                         A.tile(coord(b0, h0, m0, 0), shape(one_b, one_h, row, k_axis)).load();
            auto b = cfg.transpose_b ?
                         B.tile(coord(b0, h0, n0, 0), shape(one_b, one_h, col, k_axis)).load() :
                         B.tile(coord(b0, h0, 0, n0), shape(one_b, one_h, k_axis, col)).load();
            auto acc = zeros<float>(shape(one_b, one_h, row, col));
            acc = mma(a, b, acc, MmaPolicy{true});
            C(coord(b0, h0, m0, n0), shape(one_b, one_h, row, col)).store(acc);
        }
    });
    return definition.capture(
        cfg.transpose_a ?
            tensor_shape(cfg.batch, 1, cfg.k, cfg.m) :
            tensor_shape(cfg.batch, 1, cfg.m, cfg.k),
        cfg.transpose_b ?
            tensor_shape(cfg.batch, 1, cfg.n, cfg.k) :
            tensor_shape(cfg.batch, 1, cfg.k, cfg.n),
        tensor_shape(cfg.batch, 1, cfg.m, cfg.n));
}

void test_batched_gemm(Device &device, tile::CompileOptions options) {
    for (auto ta : {false, true}) {
        for (auto tb : {false, true}) {
            BatchedGemmConfig cfg{3, 48, 33, 24, 16, ta, tb};
            auto kernel = batched_gemm_kernel(cfg);
            auto a_size = static_cast<size_t>(cfg.batch * cfg.m * cfg.k);
            auto b_size = static_cast<size_t>(cfg.batch * cfg.k * cfg.n);
            auto c_size = static_cast<size_t>(cfg.batch * cfg.m * cfg.n);
            luisa::vector<float> a(a_size), b(b_size), actual(c_size);
            for (auto i = 0u; i < a.size(); i++) {
                a[i] = std::sin(static_cast<float>(i) * .41f + .07f) * 1.25f;
            }
            for (auto i = 0u; i < b.size(); i++) {
                b[i] = std::cos(static_cast<float>(i) * .29f + .31f) * .9f;
            }
            auto shader = tile::compile(device, kernel, options);
            expect(static_cast<bool>(shader)) << shader.metadata().error;
            if (!shader) { return; }
            auto stream = device.create_stream(StreamTag::COMPUTE);
            auto ab = device.create_buffer<float>(a.size());
            auto bb = device.create_buffer<float>(b.size());
            auto cb = device.create_buffer<float>(c_size);
            stream << ab.copy_from(luisa::span{a}) << bb.copy_from(luisa::span{b})
                   << shader(ab, bb, cb).dispatch() << cb.copy_to(luisa::span{actual}) << synchronize();
            auto a_index = [&](int64_t batch, int64_t m, int64_t k) noexcept {
                return static_cast<size_t>(batch * cfg.m * cfg.k) +
                       static_cast<size_t>(cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k);
            };
            auto b_index = [&](int64_t batch, int64_t k, int64_t n) noexcept {
                return static_cast<size_t>(batch * cfg.k * cfg.n) +
                       static_cast<size_t>(cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n);
            };
            for (auto batch = int64_t{0}; batch < cfg.batch; batch++) {
                for (auto m = int64_t{0}; m < cfg.m; m++) {
                    for (auto n = int64_t{0}; n < cfg.n; n++) {
                        double expected = 0.0;
                        for (auto k = int64_t{0}; k < cfg.k; k++) {
                            expected += static_cast<double>(a[a_index(batch, m, k)]) *
                                        b[b_index(batch, k, n)];
                        }
                        auto index = static_cast<size_t>(batch * cfg.m * cfg.n + m * cfg.n + n);
                        expect(close_tolerance(actual[index], expected))
                            << "batch=" << batch << " m=" << m << " n=" << n;
                    }
                }
            }
        }
    }
}

void run_llm_tirx_case(Device &device, tile::CompileOptions options,
                       const test::tile_llm::Case &fixture) {
    expect(fixture.kernel.valid());
    auto shader = tile::compile(device, fixture.kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto a = device.create_buffer<float>(fixture.inputs[0].size());
    auto b = device.create_buffer<float>(fixture.inputs[1].size());
    auto c = device.create_buffer<float>(fixture.inputs[2].size());
    auto out = device.create_buffer<float>(fixture.expected.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    luisa::vector<float> actual(fixture.expected.size());
    stream << a.copy_from(luisa::span{fixture.inputs[0]})
           << b.copy_from(luisa::span{fixture.inputs[1]})
           << c.copy_from(luisa::span{fixture.inputs[2]})
           << shader(a, b, c, out).dispatch() << out.copy_to(luisa::span{actual}) << synchronize();
    for (auto i = 0u; i < fixture.expected.size(); i++) {
        expect(close_tolerance(actual[i], fixture.expected[i]))
            << "element " << i << " actual " << actual[i] << " expected " << fixture.expected[i];
    }
}

// Attention-style kernels and row programs compiled through the Tile TIRx
// CUDA path instead of the XIR/NATIVE route (see also test_xir_llm.cpp).
void test_attention_and_rows_tirx(Device &device, tile::CompileOptions options) {
    // One attention fixture (bounded, already proven cheap by the LLM suite)
    // plus one masked-softmax row program. Larger shapes belong to benchmarks.
    run_llm_tirx_case(device, options, test::tile_llm::attention(2, 4, 2, 7, 11, 8, 7));
    run_llm_tirx_case(device, options,
                      test::tile_llm::rows(test::tile_llm::RowOp::MASKED_SOFTMAX, 9, 13));
}

void run_fail_closed_without_tirx(Device &device) {
    auto definition = tile::tile_kernel("cuda_tile_no_bridge", [](tile::TensorView<float, 1> output) {
        auto element = tile::axis("element", 32);
        for (auto &item : tile::parallel(tile::shape(element))) {
            output(item.index()).store(1.0f);
        }
    });
    {
        auto kernel = definition.capture(tile::tensor_shape(32));
        {
            tile::CompileOptions options;
            options.lowering = tile::Lowering::TIRX;
            {
                auto shader = tile::compile(device, kernel, options);
                {
                    auto error = shader.metadata().error;
                    expect(!shader);
                    expect(error.find("TIRx") != luisa::string::npos);
                    if (error.find("TIRx") == luisa::string::npos) {
                        LUISA_INFO("Unexpected fail-closed diagnostic: {}", error);
                    }
                }
            }
        }
    }
}

#ifdef LUISA_TEST_TILE_CUDA_TIRX

class MemoryBinaryStream final : public luisa::BinaryStream {
private:
    std::vector<std::byte> _data;
    size_t _position{};

public:
    explicit MemoryBinaryStream(luisa::span<const std::byte> data) noexcept
        : _data{data.begin(), data.end()} {}
    explicit MemoryBinaryStream(std::vector<std::byte> data) noexcept
        : _data{std::move(data)} {}
    [[nodiscard]] size_t length() const noexcept override { return _data.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _position; }
    void read(luisa::span<std::byte> destination) noexcept override {
        if (_position > _data.size() || destination.size() > _data.size() - _position) {
            std::memset(destination.data(), 0, destination.size());
            _position = _data.size();
            return;
        }
        std::memcpy(destination.data(), _data.data() + _position, destination.size());
        _position += destination.size();
    }
};

class MemoryBinaryIO final : public luisa::BinaryIO {
public:
    mutable std::unordered_map<std::string, std::vector<std::byte>> entries;
    mutable size_t bytecode_read_count{};
    mutable size_t bytecode_write_count{};

public:
    void clear_shader_cache() const noexcept override { entries.clear(); }
    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_shader_bytecode(
        luisa::string_view name) const noexcept override {
        bytecode_read_count++;
        auto iterator = entries.find(std::string{name});
        if (iterator == entries.end()) { return nullptr; }
        return luisa::make_unique<MemoryBinaryStream>(iterator->second);
    }
    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_shader_cache(
        luisa::string_view) const noexcept override {
        return nullptr;
    }
    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_internal_shader(
        luisa::string_view) const noexcept override {
        return nullptr;
    }
    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        bytecode_write_count++;
        entries[std::string{name}] = {data.begin(), data.end()};
        return {};
    }
    luisa::filesystem::path write_shader_cache(
        luisa::string_view, luisa::span<const std::byte>) const noexcept override {
        return {};
    }
    luisa::filesystem::path write_internal_shader(
        luisa::string_view, luisa::span<const std::byte>) const noexcept override {
        return {};
    }
    [[nodiscard]] const std::vector<std::byte> *entry(luisa::string_view name) const noexcept {
        auto iterator = entries.find(std::string{name});
        return iterator == entries.end() ? nullptr : &iterator->second;
    }
    // Flips one hex digit of the stored CHECKSUM so the sidecar no longer
    // matches the PTX and the next compile must recompile.
    void corrupt_checksum(luisa::string_view ptx_name) const noexcept {
        auto metadata_name = luisa::format("{}.metadata", ptx_name);
        auto iterator = entries.find(std::string{metadata_name});
        if (iterator == entries.end() || iterator->second.empty()) { return; }
        auto &data = iterator->second;
        luisa::string text{reinterpret_cast<const char *>(data.data()), data.size()};
        auto marker = text.find("CHECKSUM ");
        if (marker != luisa::string::npos) {
            auto digit = marker + luisa::string_view{"CHECKSUM "}.size() + 15u;
            if (digit < text.size()) {
                data[digit] ^= std::byte{0x01u};
            }
        }
    }
};

inline void set_force_patch_env(bool enabled) noexcept {
#ifdef _WIN32
    _putenv_s("LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX", enabled ? "1" : "");
#else
    if (enabled) {
        setenv("LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX", "1", 1);
    } else {
        unsetenv("LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX");
    }
#endif
}

struct DeviceWithIO {
    compute::DeviceConfig config;
          std::optional<test::DeviceContext> owner;

    explicit DeviceWithIO(MemoryBinaryIO &io) {
        config.binary_io = &io;
          auto created = test::create_device_from_ut(
              boost::ut::detail::cfg::largc,
              const_cast<char **>(boost::ut::detail::cfg::largv), &config, false);
        expect(created.has_value());
        if (created) { owner.emplace(std::move(*created)); }
    }
    [[nodiscard]] compute::Device &device() noexcept { return owner->device; }
};

// The elementwise kernel used by the cache and old-driver patch tests. Keep it
// tiny so repeated compiles through the fake BinaryIO stay cheap.
  struct CacheKernelFixture {
      tile::Kernel kernel;
      luisa::vector<float> input;
      size_t elements{1003u};
      [[nodiscard]] static tile::Kernel make_kernel() {
          auto definition = tile::tile_kernel("cache_roundtrip_axpy", [](tile::TensorView<const float, 1> x,
                                                                         tile::TensorView<float, 1> result) {
              auto element = tile::axis("element", 1003);
              for (auto &item : tile::parallel(tile::shape(element))) {
                  auto index = item.index();
                  result(index).store(1.25f * x(index).load() + 0.5f);
              }
          });
          return definition.capture(tile::tensor_shape(1003), tile::tensor_shape(1003));
      }
      CacheKernelFixture() : kernel{make_kernel()} {
          input.resize(elements);
        for (auto i = 0u; i < elements; i++) { input[i] = static_cast<float>(i % 37u) * 0.125f - 2.0f; }
    }
};

[[nodiscard]] tile::Shader compile_cached(DeviceWithIO &test_device, tile::CompileOptions options,
                                          const tile::Kernel &kernel, luisa::string_view name) {
    ShaderOption shader_options{.enable_cache = true, .name = luisa::string{name}};
    return tile::compile(test_device.device(), kernel, options, shader_options);
}

void check_cache_sidecar(const MemoryBinaryIO &io, luisa::string_view ptx_name) {
    auto ptx = io.entry(ptx_name);
    auto metadata = io.entry(luisa::format("{}.metadata", ptx_name));
    expect(ptx != nullptr);
    expect(metadata != nullptr);
    if (!ptx || !metadata) { return; }
    luisa::string metadata_text{
        reinterpret_cast<const char *>(metadata->data()), metadata->size()};
    expect(metadata_text.starts_with("// METADATA: ")) << metadata_text;
    expect(metadata_text.find("KIND TILE") != luisa::string::npos) << metadata_text;
    expect(metadata_text.find("CHECKSUM ") != luisa::string::npos) << metadata_text;
}

void test_tile_cache_round_trip(Device &, tile::CompileOptions options) {
    MemoryBinaryIO io;
    CacheKernelFixture fixture;
    DeviceWithIO first(io);
    auto ptx_name = luisa::string_view{"roundtrip_tile.ptx"};
    auto first_compile = compile_cached(first, options, fixture.kernel, "roundtrip_tile");
    expect(static_cast<bool>(first_compile)) << first_compile.metadata().error;
    if (!first_compile) { return; }
    check_cache_sidecar(io, ptx_name);
    auto first_ptx = io.entry(ptx_name);
    expect(first_ptx != nullptr);
    if (!first_ptx) { return; }
    auto first_ptx_bytes = *first_ptx;

    // A second, fresh device/context on the same fake store must be a pure
    // cache hit: shader valid, read occurred, bytes unchanged, no new write.
    DeviceWithIO second(io);
    auto reads_before = io.bytecode_read_count;
    auto writes_before = io.bytecode_write_count;
    auto second_compile = compile_cached(second, options, fixture.kernel, "roundtrip_tile");
    expect(static_cast<bool>(second_compile)) << second_compile.metadata().error;
    if (!second_compile) { return; }
    expect(io.bytecode_read_count > reads_before);
    expect(io.bytecode_write_count == writes_before);
    auto second_ptx = io.entry(ptx_name);
    expect(second_ptx != nullptr);
    if (second_ptx) { expect(*second_ptx == first_ptx_bytes); }

    // Corrupting the sidecar CHECKSUM forces a recompile and rewrite.
    io.corrupt_checksum(ptx_name);
    DeviceWithIO third(io);
    auto writes_before_corruption = io.bytecode_write_count;
    auto third_compile = compile_cached(third, options, fixture.kernel, "roundtrip_tile");
    expect(static_cast<bool>(third_compile)) << third_compile.metadata().error;
    if (!third_compile) { return; }
    expect(io.bytecode_write_count > writes_before_corruption);
    check_cache_sidecar(io, ptx_name);
    // The rewritten entry must now load from cache on yet another fresh device.
    DeviceWithIO fourth(io);
    auto reads_before_fourth = io.bytecode_read_count;
    auto writes_before_fourth = io.bytecode_write_count;
    auto fourth_compile = compile_cached(fourth, options, fixture.kernel, "roundtrip_tile");
    expect(static_cast<bool>(fourth_compile)) << fourth_compile.metadata().error;
    expect(io.bytecode_read_count > reads_before_fourth);
    expect(io.bytecode_write_count == writes_before_fourth);

    // Deleting the stored PTX also forces a fresh compile and a valid sidecar.
    io.entries.erase(std::string{ptx_name});
    DeviceWithIO fifth(io);
    auto writes_before_delete = io.bytecode_write_count;
    auto fifth_compile = compile_cached(fifth, options, fixture.kernel, "roundtrip_tile");
    expect(static_cast<bool>(fifth_compile)) << fifth_compile.metadata().error;
    if (!fifth_compile) { return; }
    expect(io.bytecode_write_count > writes_before_delete);
    check_cache_sidecar(io, ptx_name);
}

void verify_elementwise_result(Device &device, const tile::Shader &shader,
                               const luisa::vector<float> &input) {
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto x = device.create_buffer<float>(input.size());
    auto r = device.create_buffer<float>(input.size());
    luisa::vector<float> actual(input.size());
    stream << x.copy_from(luisa::span{input})
           << shader(x, r).dispatch() << r.copy_to(luisa::span{actual}) << synchronize();
    for (auto i = 0u; i < input.size(); i++) {
        expect(close_tolerance(actual[i], 1.25 * input[i] + 0.5));
    }
}

void test_tile_patch_retry(Device &, tile::CompileOptions options) {
    set_force_patch_env(true);
    MemoryBinaryIO io;
    CacheKernelFixture fixture;
    DeviceWithIO first(io);
    auto ptx_name = luisa::string_view{"patch_tile.ptx"};
    auto first_compile = compile_cached(first, options, fixture.kernel, "patch_tile");
    expect(static_cast<bool>(first_compile)) << first_compile.metadata().error;
    if (!first_compile) { return; }
    verify_elementwise_result(first.device(), first_compile, fixture.input);
    check_cache_sidecar(io, ptx_name);
    auto stored_ptx = io.entry(ptx_name);
    expect(stored_ptx != nullptr);
    if (!stored_ptx) { return; }
    luisa::string stored_text{
        reinterpret_cast<const char *>(stored_ptx->data()), stored_ptx->size()};
    auto version = stored_text.find(".version ");
    expect(version != luisa::string::npos) << stored_text;
    if (version != luisa::string::npos) {
        auto suffix = luisa::string_view{stored_text}.substr(version + 8u);
        auto version_end = 0ull;
        while (version_end < suffix.size() && isdigit(suffix[version_end])) { version_end++; }
        expect(version_end > 0u && version_end + 1u < suffix.size() && suffix[version_end] == '.');
        // The simulated old-driver patch clamps to "<major>.0".
        expect(version_end > 0u && version_end + 2u <= suffix.size() &&
               suffix[version_end + 1u] == '0');
    }
    auto patched_bytes = *stored_ptx;

    set_force_patch_env(false);
    // A cold process must load the patched bytes from the store without
    // re-patching or re-writing them.
    DeviceWithIO second(io);
    auto reads_before = io.bytecode_read_count;
    auto writes_before = io.bytecode_write_count;
    auto second_compile = compile_cached(second, options, fixture.kernel, "patch_tile");
    expect(static_cast<bool>(second_compile)) << second_compile.metadata().error;
    if (!second_compile) { return; }
    verify_elementwise_result(second.device(), second_compile, fixture.input);
    expect(io.bytecode_read_count > reads_before);
    expect(io.bytecode_write_count == writes_before);
    auto second_ptx = io.entry(ptx_name);
    expect(second_ptx != nullptr);
    if (second_ptx) { expect(*second_ptx == patched_bytes); }
}

#endif// LUISA_TEST_TILE_CUDA_TIRX

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
#ifdef LUISA_TEST_TILE_CUDA_TIRX
    tile::CompileOptions options;
    options.lowering = tile::Lowering::TIRX;
    "tile_cuda_ptx_elementwise"_test = [&] { test_elementwise(device, options); };
    "tile_cuda_ptx_reduce"_test = [&] { test_row_sum(device, options); };
    "tile_cuda_ptx_extrema"_test = [&] { test_row_extrema(device, options); };
    "tile_cuda_ptx_softmax"_test = [&] { test_row_softmax(device, options); };
    "tile_cuda_ptx_gemm"_test = [&] {
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) {
                for (auto reassociation : {true, false}) {
                    test_gemm(device, options, {32, 32, 16, 16, ta, tb, reassociation});
                }
            }
        }
        test_gemm(device, options, {48, 32, 24, 16, false, true, true});
        test_gemm(device, options, {32, 64, 32, 16, true, false, false});
    };
    "tile_cuda_ptx_gemm_ragged_views"_test = [&] { test_gemm_ragged_views(device, options); };
    "tile_cuda_ptx_batched_gemm"_test = [&] { test_batched_gemm(device, options); };
    "tile_cuda_ptx_attention_and_rows"_test = [&] { test_attention_and_rows_tirx(device, options); };
    "tile_cuda_ptx_cache_round_trip"_test = [&] { test_tile_cache_round_trip(device, options); };
    "tile_cuda_ptx_patch_retry"_test = [&] { test_tile_patch_retry(device, options); };
#else
    "tile_cuda_ptx_fail_closed_without_bridge"_test = [&] { run_fail_closed_without_tirx(device); };
#endif
}
