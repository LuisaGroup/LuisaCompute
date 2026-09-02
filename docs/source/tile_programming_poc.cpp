// From-scratch Luisa Tile DSL syntax PoC.
//
// This file is an executable-design sketch, not an implemented API test. Its
// purpose is to keep the proposed C++ surface honest: no explicit IR builder,
// no fixed block/warp/thread hierarchy, and no loop result plumbing.

#include <cstdint>

namespace luisa::compute::tile::poc {

struct GemmConfig {
    int32_t block_m;
    int32_t block_n;
    int32_t block_k;
    int32_t groups;
    int32_t max_in_flight;
};

// `cfg` is an ordinary host value captured while this kernel variant is built.
// Calling make_gemm with another config simply builds and JITs another variant.
auto make_gemm(GemmConfig cfg) {
    return tile_kernel([=](TensorView<bf16, 2> A,
                           TensorView<bf16, 2> B,
                           TensorView<bf16, 2> C) {
        // Dimension symbols are local identities. Their strings are only
        // diagnostic labels; the language has no built-in M/K/N axes.
        auto [m, n, reduction] = dims("m", "n", "reduction");
        auto A_mk = A.with_dims(m, reduction);
        auto B_kn = B.with_dims(reduction, n);
        auto C_mn = C.with_dims(m, n);

        auto M = A_mk.extent(m);
        auto N = B_kn.extent(n);
        auto K = A_mk.extent(reduction);

        // These range-for loops execute their C++ bodies once while capturing
        // typed execution regions. They do not enumerate nests on the host.
        for (auto &nest : parallel(
                 shape(ceil_div(M, cfg.block_m),
                       ceil_div(N, cfg.block_n)))) {
            auto [tile_m, tile_n] = nest.index();
            auto m0 = tile_m * cfg.block_m;
            auto n0 = tile_n * cfg.block_n;

            // A data Tile is a staged variable. Its declaration in this scope
            // gives it the outer tile anchor; it is not addressable storage.
            auto acc = zeros<f32>(
                shape(m(cfg.block_m), n(cfg.block_n)));

            for (auto &subnest : nest.parallel(shape(cfg.groups))) {
                for (auto &k : subnest.pipeline(
                         range(0, K, cfg.block_k),
                         pipeline_policy{
                             .max_in_flight = cfg.max_in_flight,
                             .initiation_interval = 1})) {
                    auto k0 = k.index();

                    k.stage("load");
                    auto a = A_mk.tile(
                                     coord(m0, k0),
                                     shape(cfg.block_m, cfg.block_k),
                                     bounds::zero)
                                 .load();

                    auto b = B_kn.tile(
                                     coord(k0, n0),
                                     shape(cfg.block_k, cfg.block_n),
                                     bounds::zero)
                                 .load();

                    k.stage("compute");
                    // Cross-stage Tile SSA is the normal staging surface. The
                    // planner materializes and versions a/b only if required.
                    acc = mma(a, b, acc);
                }
            }

            for (auto &leaf : nest.parallel(exec::infer)) {
                auto out = cast<bf16>(maximum(acc, 0.0f));

                C_mn.tile(
                        coord(m0, n0),
                        shape(cfg.block_m, cfg.block_n),
                        bounds::predicate)
                    .store(out);
            }
        }
    });
}

// An expert may request stable addressable temporaries without changing the
// execution hierarchy. This is an escape hatch for explicit aliasing, pinned
// layouts, mailboxes, or protocols; ordinary pipeline staging uses Tile SSA.
auto make_explicit_memory_gemm(GemmConfig cfg) {
    return tile_kernel([=](TensorView<bf16, 2> A,
                           TensorView<bf16, 2> B,
                           TensorView<bf16, 2> C) {
        auto [m, n, reduction] = dims("m", "n", "reduction");
        auto A_mk = A.with_dims(m, reduction);
        auto B_kn = B.with_dims(reduction, n);
        auto C_mn = C.with_dims(m, n);

        auto M = A_mk.extent(m);
        auto N = B_kn.extent(n);
        auto K = A_mk.extent(reduction);

        for (auto &nest : parallel(
                 shape(ceil_div(M, cfg.block_m),
                       ceil_div(N, cfg.block_n)))) {
            auto [tile_m, tile_n] = nest.index();
            auto m0 = tile_m * cfg.block_m;
            auto n0 = tile_n * cfg.block_n;

            auto a_layout = layout(
                shape(m(cfg.block_m), reduction(cfg.block_k)),
                stride(cfg.block_k, 1));
            auto b_layout = layout(
                shape(reduction(cfg.block_k), n(cfg.block_n)),
                stride(cfg.block_n, 1));

            auto acc = zeros<f32>(
                shape(m(cfg.block_m), n(cfg.block_n)));

            for (auto &subnest : nest.parallel(
                     shape(cfg.groups), exec::warp)) {
                // The lexical declaration site implies ownership by subnest.
                auto As = memory<bf16>(a_layout, mem::shared);
                auto Bs = memory<bf16>(b_layout, mem::shared);

                for (auto &k : subnest.pipeline(
                         range(0, K, cfg.block_k),
                         pipeline_policy{
                             .max_in_flight = cfg.max_in_flight,
                             .initiation_interval = 1})) {
                    auto k0 = k.index();

                    k.stage("load");
                    // Every addressable read and write is explicit. A target
                    // may later fuse each load/store pair into an async copy.
                    As.store(A_mk.tile(
                                     coord(m0, k0),
                                     shape(cfg.block_m, cfg.block_k),
                                     bounds::zero)
                                 .load());
                    Bs.store(B_kn.tile(
                                     coord(k0, n0),
                                     shape(cfg.block_k, cfg.block_n),
                                     bounds::zero)
                                 .load());

                    k.stage("compute");
                    // The stage ordinal is local to this pipeline. It is a
                    // logical consumer phase, not a warp or buffer slot.
                    acc = mma(As.load(), Bs.load(), acc);
                }
            }

            for (auto &leaf : nest.parallel(exec::infer)) {
                C_mn.tile(
                        coord(m0, n0),
                        shape(cfg.block_m, cfg.block_n),
                        bounds::predicate)
                    .store(cast<bf16>(acc));
            }
        }
    });
}

// Typical autotuning is deliberately simple:
//
//   for (auto cfg : candidates) {
//       auto executable = device.jit(make_gemm(cfg));
//       measure(executable, A, B, C);
//   }
//
// The JIT cache key contains cfg, target features, and requested argument
// specializations. A symbolic one-capture search graph can be added later as
// an optimization, not as a frontend semantic requirement.

}// namespace luisa::compute::tile::poc
