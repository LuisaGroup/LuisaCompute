// =============================================================================
// kernel_exp.cpp — EXP
// =============================================================================
// Element-wise unary exp: B[i] = exp(A[i]) for i in [0, 64).
// Port of backup_old_tile/examples/tensor/kernel_exp.cpp to the
// execution-structure-first Tile DSL. The old kernel staged the whole
// 64-element vector through a fragment tile inside a 32-thread T.Kernel; the
// new capture is one root parallel domain with one element per index.
// Host inputs and tolerance mirror the old main.cpp device pass:
// A[i] = (i - 32) * 0.25, max-abs-error <= 1e-3.

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_exp_kernel() {
    auto definition = tile::tile_kernel("exp", [](tile::TensorView<const float, 1> A,
                                                  tile::TensorView<float, 1> B) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            B(index).store(tile::exp(A(index).load()));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN));
}

void run_exp(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_exp_kernel(), "exp");
    if (!shader) { record("exp", false, shader.metadata().error); return; }
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
        err = std::max(err, static_cast<double>(std::abs(hB[i] - std::exp(hA[i]))));
    }
    check("exp", err, 1e-3);
}

}// namespace tensor_example
