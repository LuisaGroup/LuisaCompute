// =============================================================================
// kernel_gelu.cpp — GELU
// =============================================================================
// Element-wise GELU for i in [0, 64).
// Port of backup_old_tile/examples/tensor/kernel_gelu.cpp. SUBSTITUTION: the
// old kernel used the exact erf form 0.5*x*(1+erf(x/sqrt(2))) via an
// ExternalFunction call, which the new TileIR cannot express (no ERF
// ElementwiseOp, and the old example skipped device dispatch because of the
// unresolved erf linking). This port uses the standard tanh approximation
//   B[i] = 0.5*x*(1 + tanh(0.7978845608 * x * (1 + 0.044715*x*x))),
// composed from the TANH elementwise op, and verifies against the SAME
// formula on the host (not against erf), so device and reference agree.
// Inputs: A[i] = (i - 32) * 0.25, max-abs-error <= 1e-4.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
// Host-side reference of the tanh approximation, in float arithmetic to
// mirror the device computation as closely as possible.
[[nodiscard]] float gelu_tanh_ref(float x) noexcept {
    constexpr auto kBeta = 0.7978845608f;// sqrt(2/pi)
    constexpr auto kKappa = 0.044715f;
    auto inner = kBeta * x * (1.0f + kKappa * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}
}// namespace

tile::Kernel make_gelu_kernel() {
    auto definition = tile::tile_kernel("gelu", [](tile::TensorView<const float, 1> A,
                                                   tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            auto x = A(index).load();
            auto inner = x * 0.7978845608f * (1.0f + 0.044715f * (x * x));
            B(index).store(0.5f * x * (1.0f + tile::tanh(inner)));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_gelu(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_gelu_kernel(), "gelu");
    if (!shader) { record("gelu", false, shader.metadata().error); return; }
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
        err = std::max(err, static_cast<double>(std::abs(hB[i] - gelu_tanh_ref(hA[i]))));
    }
    check("gelu", err, 1e-4);
}

}// namespace tensor_example
