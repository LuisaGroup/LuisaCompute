#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_native_test_utils.h"
#include <luisa/tile/runtime.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef LUISA_TEST_TILE_NATIVE_TIRX
#include "tile_tirx_test_utils.h"
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using luisa::test::tile_native::Gemm;
using luisa::test::tile_native::gemm;

namespace {

bool check(Gemm cfg, span<const float> a, span<const float> b, span<const float> c) {
    for (auto m = int64_t{0}; m < cfg.m; m++) {
        for (auto n = int64_t{0}; n < cfg.n; n++) {
            auto expected = 0.0;
            for (auto k = int64_t{0}; k < cfg.k; k++) {
                auto ai = cfg.transpose_a ? k * cfg.m + m : m * cfg.k + k;
                auto bi = cfg.transpose_b ? n * cfg.k + k : k * cfg.n + n;
                expected += static_cast<double>(a[ai]) * b[bi];
            }
            auto actual = c[m * cfg.n + n];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-4 + 1e-4 * std::abs(expected)) { return false; }
        }
    }
    return true;
}

void run(Device &device, Gemm cfg, tile::CompileOptions options = {}) {
    auto kernel = gemm(cfg);
    expect(kernel.valid());
    auto shader = tile::compile(device, kernel, options);
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    // Nonzero, differently aligned view offsets exercise Runtime argument ABI.
    constexpr auto pad = size_t{19};
    constexpr auto guard = -731.25f;
    vector<float> a(cfg.m * cfg.k), b(cfg.k * cfg.n), c(cfg.m * cfg.n + 2 * pad, guard);
    auto ab = device.create_buffer<float>(a.size() + pad);
    auto bb = device.create_buffer<float>(b.size() + pad + 1);
    auto cb = device.create_buffer<float>(c.size());
    auto av = ab.view(pad, a.size());
    auto bv = bb.view(pad + 1, b.size());
    auto cv = cb.view(pad, cfg.m * cfg.n);
    for (auto repeat = 0; repeat < 2; repeat++) {
        for (auto i = size_t{0}; i < a.size(); i++) { a[i] = std::sin(static_cast<float>(i) * .371f + .13f + repeat) * 1.375f; }
        for (auto i = size_t{0}; i < b.size(); i++) { b[i] = std::cos(static_cast<float>(i) * .213f + .47f - repeat) * .875f; }
        std::fill(c.begin() + pad, c.end() - pad, std::numeric_limits<float>::quiet_NaN());
        stream << av.copy_from(a.data()) << bv.copy_from(b.data()) << cb.copy_from(c.data())
               << shader(av, bv, cv).dispatch() << cb.copy_to(c.data()) << synchronize();
        expect(check(cfg, a, b, span{c}.subspan(pad, cfg.m * cfg.n)))
            << "transpose A/B=" << cfg.transpose_a << "/" << cfg.transpose_b;
        expect(std::all_of(c.begin(), c.begin() + pad, [](float x) { return x == guard; }));
        expect(std::all_of(c.end() - pad, c.end(), [](float x) { return x == guard; }));
    }
    // The ordinary Resource move/destruction path owns the shader handle.
    auto moved = std::move(shader);
    expect(!shader && static_cast<bool>(moved));
    shader = std::move(moved);
    stream << shader(av, bv, cv).dispatch() << cb.copy_to(c.data()) << synchronize();
    expect(check(cfg, a, b, span{c}.subspan(pad, cfg.m * cfg.n)));

    // Aliasing is range-based, not a blanket ban on sharing a Buffer. This
    // also exercises the validator's combined read/write resource tracking.
    auto storage = device.create_buffer<float>(a.size() + b.size() + cfg.m * cfg.n);
    auto sa = storage.view(0, a.size());
    auto sb = storage.view(a.size(), b.size());
    auto sc = storage.view(a.size() + b.size(), cfg.m * cfg.n);
    stream << sa.copy_from(a.data()) << sb.copy_from(b.data())
           << shader(sa, sb, sc).dispatch() << sc.copy_to(c.data() + pad) << synchronize();
    expect(check(cfg, a, b, span{c}.subspan(pad, cfg.m * cfg.n)));

#ifdef LUISA_TEST_TILE_NATIVE_TIRX
    // Exact same captured TileIR, not a separate handwritten reference kernel.
    // The bridge currently realizes group MMA; explicit nested subgroups have
    // their own native checks above and are not silently flattened for TVM.
    if (cfg.subgroups_m == 0) {
        test::tile_tirx::Runtime runtime{"metal"};
        auto executable = runtime.build(kernel, true, true);
        if (cfg.tile_m == 64 && cfg.tile_n == 64) {
            // This full-K fixture exceeds the current bridge's shared-memory
            // budget. Native forwarding succeeds without those materialized
            // Tiles; keep the bridge rejection explicit, never use a fallback.
            expect(!executable.ok());
            expect(executable.error.find("shared-memory capacity") != string::npos);
            return;
        }
        expect(executable.ok()) << executable.error;
        if (executable.ok()) {
            auto ta = runtime.upload<float>({cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k}, a);
            auto tb = runtime.upload<float>({cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n}, b);
            auto tc = runtime.allocate<float>({cfg.m, cfg.n});
            (*executable.entry)(ta, tb, tc);
            auto actual = runtime.download<float>(tc, cfg.m * cfg.n);
            expect(check(cfg, a, b, actual));
            auto agree = true;
            for (auto i = size_t{0}; i < actual.size(); i++) {
                agree &= std::abs(actual[i] - c[i + pad]) <= 2e-4f + 2e-4f * std::abs(actual[i]);
            }
            expect(agree);
        }
    }
#endif
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    "tile_native_runtime_group_and_ragged"_test = [&] {
        for (auto cfg : {Gemm{32, 32, 32}, Gemm{127, 193, 61, 64, 64}, Gemm{16, 17, 1, 16, 32}}) { run(device, cfg); }
    };
    "tile_native_runtime_subgroup_cohorts"_test = [&] {
        for (auto cfg : {Gemm{128, 128, 128, 32, 32, 4, 1}, Gemm{37, 29, 23, 16, 32, 2, 2},
                         Gemm{129, 67, 61, 32, 16, 1, 8}}) { run(device, cfg); }
    };
    "tile_native_runtime_transposed"_test = [&] {
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) { run(device, {37, 53, 23, 32, 16, 0, 1, ta, tb}); }
        }
    };
    "tile_native_runtime_rejects_ordered_mma"_test = [&] {
        Gemm cfg{32, 32, 32};
        cfg.math.allow_reassociation = false;
        auto shader = tile::compile(device, gemm(cfg));
        expect(!shader);
        expect(!shader.metadata().error.empty());
    };
#ifdef LUISA_TEST_TILE_NATIVE_TIRX
    "tile_tirx_mpp_runtime_same_ir_offsets_and_transposes"_test = [&] {
        if (!tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version")) { return; }
        tile::bridge::tirx::CompileOptions options;
        options.cooperative_matrix = true;
        options.metal_mpp = true;
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) {
                run(device, {37, 53, 32, 32, 16, 0, 1, ta, tb}, {.lowering = tile::Lowering::TIRX, .tirx = &options});
            }
        }
    };
    "tile_tirx_runtime_same_ir_offsets_and_transposes"_test = [&] {
        for (auto ta : {false, true}) {
            for (auto tb : {false, true}) { run(device, {37, 53, 23, 32, 16, 0, 1, ta, tb}, {.lowering = tile::Lowering::TIRX}); }
        }
        run(device, {32, 32, 32}, {.lowering = tile::Lowering::TIRX});
    };
    "tile_tirx_runtime_parameter_permutation_and_unused_argument"_test = [&] {
        using namespace tile;
        auto definition = tile_kernel("permuted_arguments", [](TensorView<const float, 1> z,
                                                               TensorView<const float, 1> unused,
                                                               TensorView<float, 1> m,
                                                               TensorView<const float, 1> a) {
            for (auto &nest : parallel(shape(axis("elements", 97)), exec::Scope::WORKER)) {
                auto one = shape(axis("one", 1));
                auto origin = coord(nest.index());
                m(origin, one).store(z[origin, one] + a[origin, one] * 2.0f);
            }
        });
        auto kernel = definition.capture(tensor_shape(97), tensor_shape(97), tensor_shape(97), tensor_shape(97));
        // Kernel arguments are positional. Deliberately force the Metal entry
        // to sort differently and drop an unused parameter during splitting.
        auto root = kernel.function().body().block(0u);
        root->argument(0u)->set_name("z");
        root->argument(1u)->set_name("unused");
        root->argument(2u)->set_name("m");
        root->argument(3u)->set_name("a");
        auto native = bridge::tirx::lower(kernel.function());
        bridge::tirx::CompileOptions compiler_options;
        compiler_options.target = "metal";
        auto artifact = bridge::tirx::compile_device(native.value, kernel.function().name(), compiler_options);
        expect(static_cast<bool>(artifact)) << artifact.error;
        if (!artifact) { return; }
        expect(artifact.artifact.buffer_arguments == vector<uint32_t>{3u, 2u, 0u});
        auto shader = tile::compile(device, kernel, {.lowering = Lowering::TIRX});
        expect(static_cast<bool>(shader)) << shader.metadata().error;
        if (!shader) { return; }
        expect(shader.metadata().arguments[0].usage == Usage::READ);
        expect(shader.metadata().arguments[1].usage == Usage::NONE);
        expect(shader.metadata().arguments[2].usage == Usage::WRITE);
        auto a = device.create_buffer<float>(97);
        auto z = device.create_buffer<float>(97);
        auto c = device.create_buffer<float>(97);
        vector<float> ah(97), zh(97), ch(97);
        for (auto i = 0u; i < 97u; i++) {
            ah[i] = std::sin(i * .31f);
            zh[i] = std::cos(i * .73f);
        }
        auto stream = device.create_stream(StreamTag::COMPUTE);
        stream << a.copy_from(ah.data()) << z.copy_from(zh.data())
               << shader(z, a, c, a).dispatch() << c.copy_to(ch.data()) << synchronize();
        for (auto i = 0u; i < 97u; i++) { expect(std::abs(ch[i] - (zh[i] + 2.0f * ah[i])) < 1e-6f); }
    };
    "tile_tirx_runtime_rejects_multiple_launches"_test = [&] {
        using namespace tile;
        auto definition = tile_kernel("two_launches", [](TensorView<float, 1> c) {
            auto one = shape(axis("one", 1));
            for (auto &nest : parallel(shape(axis("first", 32)))) { c(coord(nest.index()), one).store(full<float>(one, 1.0f)); }
            for (auto &nest : parallel(shape(axis("second", 32)))) { c(coord(nest.index()), one).store(full<float>(one, 2.0f)); }
        });
        auto kernel = definition.capture(tensor_shape(32));
        expect(kernel.valid());
        auto shader = tile::compile(device, kernel, {.lowering = Lowering::TIRX});
        expect(!shader);
        expect(shader.metadata().error.find("exactly one") != string::npos) << shader.metadata().error;
    };
#endif
}
