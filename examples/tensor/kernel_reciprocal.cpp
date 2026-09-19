// =============================================================================
// kernel_reciprocal.cpp — RECIPROCAL
// =============================================================================
// Element-wise reciprocal: B[i] = 1 / A[i] for i in [0, 64), inputs away
// from 0. Port of backup_old_tile/examples/tensor/kernel_reciprocal.cpp to
// the execution-structure-first Tile DSL (one root parallel domain, one
// element per index, replacing the old 32-thread fragment staging).
// Host inputs and tolerance mirror the old main.cpp device pass:
// A[i] = (i + 1) * 0.25 (!= 0), max-abs-error <= 1e-3.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_reciprocal_kernel() {
    auto definition = tile::tile_kernel("reciprocal", [](tile::TensorView<const float, 1> A,
                                                         tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            B(index).store(1.0f / A(index).load());
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_reciprocal(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_reciprocal_kernel(), "reciprocal");
    if (!shader) { record("reciprocal", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<float>(i + 1u) * 0.25f;// != 0
    }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hB[i] - 1.0f / hA[i])));
    }
    check("reciprocal", err, 1e-3);
}

}// namespace tensor_example
