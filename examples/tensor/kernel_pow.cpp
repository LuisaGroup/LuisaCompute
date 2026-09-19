// =============================================================================
// kernel_pow.cpp — POW
// =============================================================================
// Element-wise power: C[i] = pow(A[i], B[i]) for i in [0, 64), composed as
// exp(B[i] * log(A[i])) — the new TileIR has no POW ElementwiseOp, so the
// power is built from LOG and EXP exactly as the task contract specifies.
// Port of backup_old_tile/examples/tensor/kernel_pow.cpp (the old kernel used
// a dedicated T.pow intrinsic). Host inputs mirror the old main.cpp device
// pass: A[i] = (i % 16 + 1) * 0.25 (> 0), B[i] = (i % 8 + 1) * 0.5. The
// reference is the host std::pow; tolerance follows the old pass (1e-2)
// because the composed exp/log path accumulates ~1e-4 relative error, which
// is ~0.03 absolute at the largest outputs (~256).

#include "tensor_kernels.h"

#include <algorithm>
#include <cmath>

namespace tensor_example {

namespace {
constexpr uint32_t kN = 64u;
}

tile::Kernel make_pow_kernel() {
    auto definition = tile::tile_kernel("pow", [](tile::TensorView<const float, 1> A,
                                                  tile::TensorView<const float, 1> B,
                                                  tile::TensorView<float, 1> C) {
        auto i = tile::axis("i", kN);
        for (auto &nest : tile::parallel(tile::shape(i))) {
            auto index = nest.index();
            C(index).store(tile::exp(B(index).load() * tile::log(A(index).load())));
        }
    });
    return definition.capture(tile::tensor_shape("A", kN), tile::tensor_shape("B", kN),
                              tile::tensor_shape("C", kN));
}

void run_pow(lc::Device &device, lc::Stream &stream) {
    auto shader = compile_tile(device, make_pow_kernel(), "pow");
    if (!shader) { record("pow", false, shader.metadata().error); return; }
    auto bufA = device.create_buffer<float>(kN);
    auto bufB = device.create_buffer<float>(kN);
    auto bufC = device.create_buffer<float>(kN);
    luisa::vector<float> hA(kN), hB(kN), hC(kN);
    for (auto i = 0u; i < kN; ++i) {
        hA[i] = static_cast<float>(i % 16u + 1u) * 0.25f;// > 0
        hB[i] = static_cast<float>(i % 8u + 1u) * 0.5f;
    }
    stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << lc::synchronize();
    stream << shader(bufA, bufB, bufC).dispatch() << bufC.copy_to(luisa::span{hC}) << lc::synchronize();
    auto err = 0.0;
    for (auto i = 0u; i < kN; ++i) {
        err = std::max(err, static_cast<double>(std::abs(hC[i] - std::pow(hA[i], hB[i]))));
    }
    check("pow", err, 1e-2);
}

}// namespace tensor_example
