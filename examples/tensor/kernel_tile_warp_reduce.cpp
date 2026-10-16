// =============================================================================
// kernel_tile_warp_reduce.cpp — whole-tile reduction (legacy warp_reduce)
// =============================================================================
// Ported op  : fill with 7.0, reduce sum and max (old
//              kernel_tile_warp_reduce.cpp: register-level warp_reduce_sum /
//              warp_reduce_max over a 32-thread warp).
// Provenance : backup_old_tile/examples/tensor/kernel_tile_warp_reduce.cpp
// Semantics  : semantic mapping — the old per-warp intrinsic over a
//              per-thread 1-element fragment becomes a whole-tile reduction
//              over a 64-lane Tile of 7.0 in ONE parallel nest:
//                * sum = 7.0 * 64 = 448   (old: 7.0 * 32 per warp)
//                * max = 7.0              (both old and new)
//              The old kernel applied BOTH reductions to the same 1-element
//              output W(0) (every thread wrote it; the old host check only
//              verified W[0] == 7.0, which the max satisfies). The new ABI
//              makes both results independently observable: W[0] = sum and
//              W[1] = max, recorded separately against 448 and 7.0.
// =============================================================================

#include "tensor_kernels.h"

namespace tensor_example {

[[nodiscard]] tile::Kernel make_tile_warp_reduce_kernel() {
    constexpr int64_t lanes = 64;
    auto definition = tile::tile_kernel("tile_warp_reduce", [=](tile::TensorView<float, 1> W) {
        using namespace tile;
        auto g = axis("g", 1), n = axis("lane", lanes);
        for (auto &nest : parallel(shape(g))) {
            auto v = full<float>(shape(n), 7.0f);
            // Scalar-carried whole-tile reduction (legacy warp_reduce_sum/max).
            auto sum = Scalar<float>{0.0f};
            auto mx = Scalar<float>{-std::numeric_limits<float>::infinity()};
            for (auto &lane : nest.reduce(shape(n))) {
                auto element = v.at(coord(lane.index(n)));
                sum += element;
                mx = max(mx, element);
            }
            W(Scalar<int64_t>{0}).store(sum);
            W(Scalar<int64_t>{1}).store(mx);
        }
    });
    return definition.capture(tile::tensor_shape(2));
}

void run_tile_warp_reduce(lc::Device &device, lc::Stream &stream) {
    auto kernel = make_tile_warp_reduce_kernel();
    auto shader = compile_tile(device, kernel, "tile_warp_reduce");
    if (!shader) {
        record("tile_warp_reduce", false, shader.metadata().error);
        return;
    }
    auto bufW = device.create_buffer<float>(2);
    luisa::vector<float> hW(2);
    stream << shader(bufW).dispatch() << bufW.copy_to(luisa::span{hW}) << lc::synchronize();
    // 64 lanes of 7.0: sum = 448, max = 7 (see the banner for the mapping to
    // the old per-warp intrinsic and the old W[0] == 7.0 host check).
    check("tile_warp_reduce.sum", luisa::abs(static_cast<double>(hW[0]) - 448.0), 1e-5);
    check("tile_warp_reduce.max", luisa::abs(static_cast<double>(hW[1]) - 7.0), 1e-5);
}

}// namespace tensor_example
