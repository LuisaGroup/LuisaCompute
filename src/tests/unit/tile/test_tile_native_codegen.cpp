#include "ut/ut.hpp"
#include "tile_native_test_utils.h"
#include "metal_tile_codegen.h"

using namespace luisa::compute;
using namespace luisa::compute::tile;
using namespace boost::ut;
using luisa::test::tile_native::gemm;
using luisa::test::tile_native::Gemm;

namespace {

void test_mapping() {
    for (auto cfg : {Gemm{127, 193, 61, 64, 64}, Gemm{513, 257, 129, 32, 32, 4, 1},
                     Gemm{37, 29, 23, 16, 32, 2, 2, true, true}}) {
        auto kernel = gemm(cfg);
        expect(kernel.valid());
        auto code = metal::lower_tile_to_mpp(kernel.function(), {}, 1024u);
        expect(code.ok()) << code.metadata.error;
        if (!code.ok()) { continue; }
        auto subgroups = cfg.subgroups_m == 0 ? 4u : static_cast<uint32_t>(cfg.subgroups_m * cfg.subgroups_n);
        auto cm = cfg.subgroups_m == 0 ? 1 : cfg.subgroups_m;
        auto cn = cfg.subgroups_m == 0 ? 1 : cfg.subgroups_n;
        auto groups = luisa::ceil_div(cfg.m, cm * cfg.tile_m) * luisa::ceil_div(cfg.n, cn * cfg.tile_n);
        expect(eq(code.block_size.x, subgroups * 32u));
        expect(eq(code.metadata.dispatch_size.x, static_cast<uint32_t>(groups) * code.block_size.x));
        expect(eq(code.metadata.arguments.size(), size_t{3}));
        expect(code.metadata.arguments[0].usage == Usage::READ);
        expect(code.metadata.arguments[1].usage == Usage::READ);
        expect(code.metadata.arguments[2].usage == Usage::WRITE);
        expect(code.metadata.disjoint_writes);
        auto &src = code.metadata.source;
        expect(src.find("matmul2d_descriptor(") != luisa::string::npos);
        expect(src.find(cfg.subgroups_m == 0 ? "execution_simdgroups<4>" : "execution_simdgroups<1>") != luisa::string::npos);
        expect(src.find("kernel_main_indirect") != luisa::string::npos);
        // Lowering is read-only; it must not relabel the semantic input as machine IR.
        expect(kernel.function().form() == IRForm::CANDIDATE);
    }
}

void test_rejections() {
    Module module;
    luisa::compute::tile::Function orphan{nullptr, 0u, "orphan", IRForm::CANDIDATE};
    expect(!metal::lower_tile_to_mpp(orphan, {}, 1024u).ok());
    auto detached = module.create_function("detached", IRForm::CANDIDATE)->remove_self();
    auto unattached = metal::lower_tile_to_mpp(*detached, {}, 1024u);
    expect(!unattached.ok());
    expect(unattached.metadata.error.find("owning module") != luisa::string::npos);
    auto kernel = gemm({32, 32, 32});
    for (auto threads : {1u, 31u, 33u, 2048u}) {
        auto code = metal::lower_tile_to_mpp(kernel.function(), {threads}, 1024u);
        expect(!code.ok());
        expect(!code.metadata.error.empty());
        expect(code.metadata.source.empty());
    }
    auto cohort = gemm({64, 64, 32, 16, 16, 2, 2});
    expect(!metal::lower_tile_to_mpp(cohort.function(), {32u}, 1024u).ok());
    expect(!metal::lower_tile_to_mpp(cohort.function(), {}, 64u).ok());
    auto invalid_atom = gemm({16, 16, 32, 8, 8});
    expect(invalid_atom.valid());
    expect(!metal::lower_tile_to_mpp(invalid_atom.function(), {}, 1024u).ok());
    Gemm ordered{32, 32, 32};
    ordered.math.allow_reassociation = false;
    auto ordered_kernel = gemm(ordered);
    expect(!metal::lower_tile_to_mpp(ordered_kernel.function(), {}, 1024u).ok());
    auto serial = tile_kernel("not_silently_erased", [] {
                      for (auto &group : parallel(shape(1), exec::Scope::GROUP)) {
                          for (auto &step : group.pipeline(shape(2), {.stages = 2})) { step.stage("unsupported"); }
                      }
                  }).capture();
    expect(serial.valid());
    auto unsupported = metal::lower_tile_to_mpp(serial.function(), {}, 1024u);
    expect(!unsupported.ok());
    expect(unsupported.metadata.error.find("unsupported") != luisa::string::npos);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_native_execution_mapping"_test = test_mapping;
    "tile_native_fail_closed"_test = test_rejections;
}
