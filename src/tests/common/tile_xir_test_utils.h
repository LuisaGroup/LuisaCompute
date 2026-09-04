#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/tile/dsl.h>

namespace luisa::test::tile_xir {

struct Gemm {
    int64_t m, n, k;
    int64_t bm{1}, bn{1}, bk{8};
    bool transpose_a{false}, transpose_b{false};
    float initial{0.0f};
    uint32_t window{2u};
};

[[nodiscard]] inline compute::tile::Kernel gemm(Gemm cfg) {
    using namespace compute::tile;
    auto definition = tile_kernel("xir_gemm", [=](TensorView<const float, 2> A,
                                                  TensorView<const float, 2> B,
                                                  TensorView<float, 2> C) {
        auto gm = axis("gm", ceil_div(cfg.m, cfg.bm));
        auto gn = axis("gn", ceil_div(cfg.n, cfg.bn));
        auto m = axis("m", cfg.bm), n = axis("n", cfg.bn), k = axis("k", cfg.bk);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest.index(gm) * cfg.bm, n0 = nest.index(gn) * cfg.bn;
            auto acc = full<float>(shape(m, n), cfg.initial);
            for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, cfg.bk)), {.stages = cfg.window, .initiation_interval = 1u})) {
                step.stage("load");
                auto k0 = step.index() * cfg.bk;
                auto a = cfg.transpose_a ? A.tile(coord(k0, m0), shape(k, m)).load() : A.tile(coord(m0, k0), shape(m, k)).load();
                auto b = cfg.transpose_b ? B.tile(coord(n0, k0), shape(n, k)).load() : B.tile(coord(k0, n0), shape(k, n)).load();
                step.stage("compute");
                acc = mma(a, b, acc);
            }
            C(coord(m0, n0), shape(m, n)).store(acc);
        }
    });
    return definition.capture(tensor_shape(cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k),
                              tensor_shape(cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n),
                              tensor_shape(cfg.m, cfg.n));
}

}// namespace luisa::test::tile_xir
