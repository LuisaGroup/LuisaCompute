#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/tile/dsl.h>

namespace luisa::test::tile_native {

struct Gemm {
    int64_t m, n, k;
    int64_t tile_m{32}, tile_n{32};
    // Zero denotes a whole-group operation. Otherwise the child domain is
    // (subgroups_m, subgroups_n), with one independent MMA per subgroup.
    int64_t subgroups_m{0}, subgroups_n{1};
    bool transpose_a{false}, transpose_b{false};
    compute::tile::MmaPolicy math;
};

[[nodiscard]] inline compute::tile::Kernel gemm(Gemm cfg) {
    using namespace compute::tile;
    auto definition = tile_kernel("native_gemm", [=](TensorView<const float, 2> A,
                                                     TensorView<const float, 2> B,
                                                     TensorView<float, 2> C) {
        auto cm = cfg.subgroups_m == 0 ? 1 : cfg.subgroups_m;
        auto cn = cfg.subgroups_m == 0 ? 1 : cfg.subgroups_n;
        auto gm = axis("groups_m", ceil_div(cfg.m, cfg.tile_m * cm));
        auto gn = axis("groups_n", ceil_div(cfg.n, cfg.tile_n * cn));
        auto m = axis("m", cfg.tile_m);
        auto n = axis("n", cfg.tile_n);
        auto k = axis("k", cfg.k);
        auto body = [&](auto m0, auto n0) {
            auto a = cfg.transpose_a ? A.tile(coord(0, m0), shape(k, m)).load() : A.tile(coord(m0, 0), shape(m, k)).load();
            auto b = cfg.transpose_b ? B.tile(coord(n0, 0), shape(n, k)).load() : B.tile(coord(0, n0), shape(k, n)).load();
            auto acc = zeros<float>(shape(m, n));
            acc = mma(a, b, acc, cfg.math);
            C(coord(m0, n0), shape(m, n)).store(acc);
        };
        for (auto &group : parallel(shape(gm, gn), exec::Scope::GROUP)) {
            if (cfg.subgroups_m == 0) {
                body(group.index(gm) * cfg.tile_m, group.index(gn) * cfg.tile_n);
            } else {
                auto sm = axis("subgroups_m", cm);
                auto sn = axis("subgroups_n", cn);
                for (auto &subnest : group.parallel(shape(sm, sn), exec::Scope::SUBGROUP)) {
                    body((group.index(gm) * cm + subnest.index(sm)) * cfg.tile_m,
                         (group.index(gn) * cn + subnest.index(sn)) * cfg.tile_n);
                }
            }
        }
    });
    return definition.capture(tensor_shape(cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k),
                              tensor_shape(cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n),
                              tensor_shape(cfg.m, cfg.n));
}

}// namespace luisa::test::tile_native
