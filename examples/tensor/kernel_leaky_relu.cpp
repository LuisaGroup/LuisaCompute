// =============================================================================
// kernel_leaky_relu.cpp — LEAKY_RELU
// =============================================================================
// Leaky ReLU with alpha = 0.01: B[i] = A[i] >= 0 ? A[i] : 0.01 * A[i] for
// i in [0, 64). The old kernel computed max(alpha*x, x) through
// min/max staging; the new capture uses the value-selecting ite() on the
// loaded scalar, which is the same piecewise function.
// Port of backup_old_tile/examples/tensor/kernel_leaky_relu.cpp. Host inputs
// and tolerance mirror the old main.cpp device pass:
// A[i] = (i - 32) * 0.25, max-abs-error <= 1e-5.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
constexpr float kAlpha = 0.01f;
}

tile::Kernel make_leaky_relu_kernel() {
    auto definition = tile::tile_kernel("leaky_relu", [](tile::TensorView<const float, 1> A,
                                                         tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            auto x = A(index).load();
            B(index).store(tile::ite(x >= 0.0f, x, x * kAlpha));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_leaky_relu(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_leaky_relu_kernel(), "leaky_relu");
    if (!shader) { record("leaky_relu", false, shader.metadata().error); return; }
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
        auto ref = hA[i] >= 0.0f ? hA[i] : kAlpha * hA[i];
        err = std::max(err, static_cast<double>(std::abs(hB[i] - ref)));
    }
    check("leaky_relu", err, 1e-5);
}

}// namespace tensor_example
