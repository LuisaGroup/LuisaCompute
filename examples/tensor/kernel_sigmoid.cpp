// =============================================================================
// kernel_sigmoid.cpp — SIGMOID
// =============================================================================
// Sigmoid: B[i] = 1 / (1 + exp(-A[i])) for i in [0, 64), composed from the
// exp elementwise op (the new TileIR has no dedicated SIGMOID op; the old
// kernel's rsqrt(denom) * rsqrt(denom) is algebraically 1/denom, so a plain
// reciprocal division preserves the semantics).
// Port of backup_old_tile/examples/tensor/kernel_sigmoid.cpp. The old example
// traced this kernel but skipped device dispatch (ExternalFunction issue);
// here it is dispatched and verified. Inputs mirror the other unary kernels:
// A[i] = (i - 32) * 0.25, max-abs-error <= 1e-4.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_sigmoid_kernel() {
    auto definition = tile::tile_kernel("sigmoid", [](tile::TensorView<const float, 1> A,
                                                      tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            auto x = A(index).load();
            B(index).store(1.0f / (tile::exp(-x) + 1.0f));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_sigmoid(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_sigmoid_kernel(), "sigmoid");
    if (!shader) { record("sigmoid", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<float>(static_cast<int>(i) - 32) * 0.25f;
    }
    stream << bufA.copy_from(hA.data()) << lc::synchronize();
    stream << shader(bufA, bufB).dispatch() << bufB.copy_to(hB.data()) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        auto ref = 1.0 / (1.0 + std::exp(-static_cast<double>(hA[i])));
        err = std::max(err, std::abs(static_cast<double>(hB[i]) - ref));
    }
    check("sigmoid", err, 1e-4);
}

}// namespace tensor_example
