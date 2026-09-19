// =============================================================================
// kernel_dtypes.cpp — COPY / BINARY / NEG across the supported ScalarTypes
// =============================================================================
// Port of the dtype device-pass section of
// backup_old_tile/examples/tensor/main.cpp (+ kernel_dtypes.cpp). Covers the
// dtypes the new TileIR models as ScalarTypes: 1-D copy through a Tile for
// f16 / f32 / i32 / i8, elementwise add for f32 / i32, and neg for f32.
// FP8 / I4 / FP4 have no ScalarType in the new TileIR and are intentionally
// excluded (see the unimplemented-feature report in main.cpp).
//
// Unlike the per-element unary kernels in this directory, the dtype kernels
// stage the whole 64-element vector through a Tile in a single root parallel
// block, mirroring the old fragment-copy structure of kernel_dtypes.cpp.
// Host inputs and tolerances mirror the old device pass.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tensor_example {

namespace {

constexpr uint32_t kN = 64u;

// 1-D copy through a Tile: B = A, one tile covering the whole vector.
template<typename T>
[[nodiscard]] tile::Kernel make_dtype_copy_kernel(luisa::string_view name) {
    auto definition = tile::tile_kernel(name, [](tile::TensorView<const T, 1> A,
                                                 tile::TensorView<T, 1> B) {
        auto g = tile::axis("g", 1u);// single tile block covers all kN elements
        auto n = tile::axis("n", kN);
        for (auto &nest : tile::parallel(tile::shape(g))) {
            auto origin = tile::coord(nest.index() * static_cast<int64_t>(kN));
            auto space = tile::shape(n);
            B(origin, space).store(A.tile(origin, space).load());
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

// 1-D tile add: C = A + B.
template<typename T>
[[nodiscard]] tile::Kernel make_dtype_add_kernel(luisa::string_view name) {
    auto definition = tile::tile_kernel(name, [](tile::TensorView<const T, 1> A,
                                                 tile::TensorView<const T, 1> B,
                                                 tile::TensorView<T, 1> C) {
        auto g = tile::axis("g", 1u);
        auto n = tile::axis("n", kN);
        for (auto &nest : tile::parallel(tile::shape(g))) {
            auto origin = tile::coord(nest.index() * static_cast<int64_t>(kN));
            auto space = tile::shape(n);
            C(origin, space).store(A.tile(origin, space).load() + B.tile(origin, space).load());
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN),
                              tile::tensor_shape("C", kN));
}

// 1-D tile negation: B = -A.
template<typename T>
[[nodiscard]] tile::Kernel make_dtype_neg_kernel(luisa::string_view name) {
    auto definition = tile::tile_kernel(name, [](tile::TensorView<const T, 1> A,
                                                 tile::TensorView<T, 1> B) {
        auto g = tile::axis("g", 1u);
        auto n = tile::axis("n", kN);
        for (auto &nest : tile::parallel(tile::shape(g))) {
            auto origin = tile::coord(nest.index() * static_cast<int64_t>(kN));
            auto space = tile::shape(n);
            B(origin, space).store(-A.tile(origin, space).load());
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_dtype_copy_f16(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_copy_kernel<luisa::half>("dtype_copy_f16"), "dtype_copy_f16");
    if (!shader) { record("dtype_copy_f16", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<luisa::half>(kN);
    auto bufB = device.create_buffer<luisa::half>(kN);
    luisa::vector<luisa::half> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) { hA[i] = luisa::half{static_cast<float>(i) * 0.25f}; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(static_cast<float>(hB[i]) - static_cast<float>(hA[i]))));
    }
    check("dtype_copy_f16", err, 1e-3);
}

void run_dtype_copy_f32(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_copy_kernel<float>("dtype_copy_f32"), "dtype_copy_f32");
    if (!shader) { record("dtype_copy_f32", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) { hA[i] = static_cast<float>(i) * 0.5f - 16.0f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hB[i] - hA[i])));
    }
    check("dtype_copy_f32", err, 1e-5);
}

void run_dtype_copy_i32(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_copy_kernel<int32_t>("dtype_copy_i32"), "dtype_copy_i32");
    if (!shader) { record("dtype_copy_i32", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<int32_t>(kN);
    auto bufB = device.create_buffer<int32_t>(kN);
    luisa::vector<int32_t> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) { hA[i] = static_cast<int32_t>(i) - 32; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0;
    for (auto i = 0u; i < kN; ++i) { err = std::max(err, std::abs(hB[i] - hA[i])); }
    check("dtype_copy_i32", static_cast<double>(err), 0.5);
}

void run_dtype_copy_i8(lc::Device &device, lc::Stream &stream) {
    // DX12 storage buffers require >= 4-byte strides: the i8 buffer ABI
    // fast-fails inside the dx backend (the old example skipped i8 on dx for
    // the same reason). Skip it there; it runs on vk and cuda.
    if (active_backend() == "dx") {
        skip("dtype_copy_i8", "DX12 storage buffers do not support <4-byte strides");
        return;
    }
    auto shader = compile_tile(device, make_dtype_copy_kernel<int8_t>("dtype_copy_i8"), "dtype_copy_i8");
    if (!shader) { record("dtype_copy_i8", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<int8_t>(kN);
    auto bufB = device.create_buffer<int8_t>(kN);
    luisa::vector<int8_t> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) { hA[i] = static_cast<int8_t>((i % 31u) - 15u); }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, std::abs(static_cast<int>(hB[i]) - static_cast<int>(hA[i])));
    }
    check("dtype_copy_i8", static_cast<double>(err), 0.5);
}

void run_dtype_add_f32(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_add_kernel<float>("dtype_add_f32"), "dtype_add_f32");
    if (!shader) { record("dtype_add_f32", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    auto bufC = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN), hC(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<float>(i) * 0.5f;
        hB[i] = static_cast<float>(i) * 0.25f + 1.0f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hC[i] - (hA[i] + hB[i]))));
    }
    check("dtype_add_f32", err, 1e-4);
}

void run_dtype_add_i32(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_add_kernel<int32_t>("dtype_add_i32"), "dtype_add_i32");
    if (!shader) { record("dtype_add_i32", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<int32_t>(kN);
    auto bufB = device.create_buffer<int32_t>(kN);
    auto bufC = device.create_buffer<int32_t>(kN);
    luisa::vector<int32_t> hA(kN), hB(kN), hC(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<int32_t>(i);
        hB[i] = -static_cast<int32_t>(i);
    }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0;
    for (auto i = 0u; i < kN; ++i) { err = std::max(err, std::abs(hC[i] - (hA[i] + hB[i]))); }
    check("dtype_add_i32", static_cast<double>(err), 0.5);
}

void run_dtype_neg_f32(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_dtype_neg_kernel<float>("dtype_neg_f32"), "dtype_neg_f32");
    if (!shader) { record("dtype_neg_f32", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) { hA[i] = static_cast<float>(i) * 0.5f - 16.0f; }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hB[i] - (-hA[i]))));
    }
    check("dtype_neg_f32", err, 1e-4);
}

}// namespace

void run_dtypes(lc::Device &device, lc::Stream &stream) {
    run_dtype_copy_f16(device, stream);
    run_dtype_copy_f32(device, stream);
    run_dtype_copy_i32(device, stream);
    run_dtype_copy_i8(device, stream);
    run_dtype_add_f32(device, stream);
    run_dtype_add_i32(device, stream);
    run_dtype_neg_f32(device, stream);
}

}// namespace tensor_example
