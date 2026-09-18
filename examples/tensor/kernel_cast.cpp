// =============================================================================
// kernel_cast.cpp — CAST
// =============================================================================
// Element-wise widening cast: B[i] = (float)A[i] for i in [0, 64), with A an
// i32 tensor. Port of backup_old_tile/examples/tensor/kernel_cast.cpp to the
// execution-structure-first Tile DSL (one root parallel domain, one element
// per index, replacing the old 32-thread fragment staging).
// Host inputs and tolerance mirror the old main.cpp device pass:
// A[i] = i - 32, max-abs-error <= 1e-6.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_cast_kernel() {
    auto definition = tile::tile_kernel("cast", [](tile::TensorView<const int32_t, 1> A,
                                                   tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            B(index).store(tile::cast<float>(A(index).load()));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_cast(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_cast_kernel(), "cast");
    if (!shader) { record("cast", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<int32_t>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<int32_t> hA(kN);
    luisa::vector<float> hB(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<int32_t>(i) - 32;
    }
    stream << bufA.copy_from(hA.data()) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(hB.data()) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hB[i] - static_cast<float>(hA[i]))));
    }
    check("cast", err, 1e-6);
}

}// namespace tensor_example
