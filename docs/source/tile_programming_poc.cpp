// Canonical executable Tile DSL example. Captured by the C++20/C++23 frontend
// tests; the same syntax uses a typed comma adapter or native operator[].
#include <luisa/tile/dsl.h>
#include <cstdint>

namespace luisa::compute::tile::poc {

struct GemmConfig {
    uint32_t block_m{8u};
    uint32_t block_n{8u};
    uint32_t block_k{8u};
    uint32_t stages{2u};
};

// Host configuration is captured normally. Each capture/JIT with a different
// configuration records a new TileIR candidate, with no symbolic meta-DSL.
auto make_gemm(GemmConfig cfg) {
    return tile_kernel("gemm", [=](TensorView<const float, 2> A,
                                   TensorView<const float, 2> B,
                                   TensorView<float, 2> C) {
        auto gm = axis("block_m", (A.extent<0>() + cfg.block_m - 1) / cfg.block_m);
        auto gn = axis("block_n", (B.extent<1>() + cfg.block_n - 1) / cfg.block_n);
        auto kt = axis("k_tiles", (A.extent<1>() + cfg.block_k - 1) / cfg.block_k);
        auto m = axis("m", cfg.block_m);
        auto n = axis("n", cfg.block_n);
        auto k = axis("k", cfg.block_k);

        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest[gm] * cfg.block_m;
            auto n0 = nest[gn] * cfg.block_n;
            auto acc = zeros<float>(shape(m, n));

            for (auto &step : nest.pipeline(shape(kt), {.stages = cfg.stages,
                                                        .initiation_interval = 1u})) {
                auto k0 = step.index() * cfg.block_k;
                step.stage("load");
                auto a = A[coord(m0, k0), shape(m, k)];
                auto b = B[coord(k0, n0), shape(k, n)];

                step.stage("compute");
                acc = mma(a, b, acc);
            }

            C(coord(m0, n0), shape(m, n)).store(acc);
        }
    });
}

// Read equivalence, including the default zero-fill/tail-store bounds policy:
//
//   A[origin, shape] == A(origin, shape).load()
//                    == A.tile(origin, shape).load()
//
// "=" updates a named Tile SSA value only. Neither MemoryRef assignment nor
// assigning to the temporary produced by A[origin, shape] is allowed.

}// namespace luisa::compute::tile::poc
