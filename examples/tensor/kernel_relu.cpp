// =============================================================================
// kernel_relu.cpp — RELU
// =============================================================================
// Element-wise ReLU: B[i] = max(A[i], 0) for i in [0, 64).
// Port of backup_old_tile/examples/tensor/kernel_relu.cpp to the
// execution-structure-first Tile DSL (one root parallel domain, one element
// per index, replacing the old 32-thread fragment staging).
// Host inputs and tolerance mirror the old main.cpp device pass:
// A[i] = (i - 32) * 0.25, max-abs-error <= 1e-6.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_relu_kernel() {
    auto definition = tile::tile_kernel("relu", [](tile::TensorView<const float, 1> A,
                                                   tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            B(index).store(tile::max(A(index).load(), 0.0f));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_relu(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_relu_kernel(), "relu");
    if (!shader) { record("relu", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<float>(static_cast<int>(i) - 32) * 0.25f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(luisa::span{hB}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hB[i] - std::max(hA[i], 0.0f))));
    }
    check("relu", err, 1e-6);
}

}// namespace tensor_example
