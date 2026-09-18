// =============================================================================
// main.cpp — New Tile DSL tensor example runner
// =============================================================================
// Rewrite of backup_old_tile/examples/tensor/main.cpp. Every kernel is
// captured with the execution-structure-first Tile DSL
// (tile_kernel(...).capture(...)), compiled with tile::compile(), dispatched
// on real device buffers, and verified against a host-side reference.
//
// Usage:
//   example_tensor <backend>                    -- run the whole kernel suite
//   example_tensor <backend> --kernel <name>    -- run one kernel (repeatable)
//   example_tensor <backend> --guard            -- two-root-parallel guard test
//   example_tensor <backend> --report           -- print the unimplemented list
//   example_tensor <backend> --cnn [cnn_input.bin]
//   example_tensor <backend> --poly-fit [--steps N]
//   example_tensor <backend> --linear-regression [--steps N]
//   example_tensor <backend> --mlp [--epochs N]
//   example_tensor <backend> --mnist [--epochs N]
//   example_tensor <backend> --rnn [--epochs N]
//   example_tensor <backend> --basics
// =============================================================================

#include "tensor_kernels.h"

#include <luisa/core/stl/algorithm.h>

#include <algorithm>

using namespace tensor_example;

namespace {

struct KernelCase {
    luisa::string_view name;
    void (*run)(luisa::compute::Device &, luisa::compute::Stream &);
};

constexpr KernelCase cases[]{
    {"elementwise_add", run_elementwise_add},
    {"pipelined_matmul", run_pipelined_matmul},
    {"rms_norm", run_rms_norm},
    {"tile_fill", run_tile_fill},
    {"tile_transpose", run_tile_transpose},
    {"tile_clamp", run_tile_clamp},
    {"tile_reduce", run_tile_reduce},
    {"tile_scan", run_tile_scan},
    {"tile_min_abs", run_tile_min_abs},
    {"tile_sync", run_tile_sync},
    {"tile_warp_reduce", run_tile_warp_reduce},
    {"softmax", run_softmax},
    {"exp", run_exp},
    {"log", run_log},
    {"sqrt", run_sqrt},
    {"tanh", run_tanh},
    {"sigmoid", run_sigmoid},
    {"relu", run_relu},
    {"leaky_relu", run_leaky_relu},
    {"gelu", run_gelu},
    {"identity", run_identity},
    {"reciprocal", run_reciprocal},
    {"neg", run_neg},
    {"cast", run_cast},
    {"pow", run_pow},
    {"dtypes", run_dtypes},
};

// ---- unimplemented-feature report ------------------------------------------
// Old tile-language features with no counterpart in the new TileIR / native
// lowerings. Printed by --report and at the end of every full run.
void print_unimplemented_report() {
    LUISA_INFO("=== tensor: unimplemented features in the new Tile DSL ===");
    LUISA_INFO("  (these kernels from backup_old_tile/examples/tensor are not ported;)");
    LUISA_INFO("  (they are absent from the registry above, not silently dropped.)");
    LUISA_INFO("  * tile_atomic      — TileIR has no atomic operations");
    LUISA_INFO("  * tile_vote_shuffle— TileIR has no warp shuffle/vote intrinsics");
    LUISA_INFO("  * loop_break       — TileIR nests have no break/early-exit");
    LUISA_INFO("  * sin/cos/tan/erf  — no SIN/COS/TAN/ERF ElementwiseOp");
    LUISA_INFO("  * ceil/floor/round — no CEIL/FLOOR/ROUND ElementwiseOp");
    LUISA_INFO("  * isinf/isnan      — no ISINF/ISNAN ElementwiseOp");
    LUISA_INFO("  * fp8/i4/fp4 dtype kernels — no FP8 lowering in the XIR bridge;");
    LUISA_INFO("                       I4/FP4 are not ScalarTypes in the new TileIR");
    LUISA_INFO("  * T.print          — no device print in the Tile runtime");
    LUISA_INFO("  * explicit shared  — manual mem::shared/Memory has no native XIR");
    LUISA_INFO("                       realization (the planner owns staging)");
    LUISA_INFO("  * torch2 import    — the torch.export graph importer targets the");
    LUISA_INFO("                       removed AST tile DSL; needs a TileIR rewrite");
    LUISA_INFO("===========================================================");
}

// ---- two-root-parallel guard (legacy two_kernels) --------------------------
void run_guard_test(luisa::compute::Device &device) {
    LUISA_INFO("=== tensor: two-root-parallel guard test ===");
    // The legacy example's two_kernels() traced two T.Kernel blocks and
    // aborted. The new TileIR contract is exactly one root parallel execution
    // domain per kernel; capturing two must fail verification.
    auto definition = tile::tile_kernel("two_parallel_roots", [] {
        auto i = tile::axis("i", 8u);
        for (auto &a : tile::parallel(tile::shape(i))) {
            static_cast<void>(a.index());
        }
        for (auto &b : tile::parallel(tile::shape(i))) {
            static_cast<void>(b.index());
        }
    });
    auto kernel = definition.capture();
    auto verified = tile::verify(kernel.module());
    // Unlike the old AST tile DSL (which aborted at trace time on a second
    // T.Kernel), the new TileIR captures multiple root parallel domains fine;
    // the rejection happens at lowering time — the XIR bridge requires
    // exactly one root parallel execution domain. Both the capture and the
    // rejection are part of the contract, so verify both directions.
    record("two_parallel_roots_captured", kernel.valid() && verified.ok(),
           "TileIR capture/verify accepts the structure");
    auto shader = compile_tile(device, kernel, "two_parallel_roots");
    record("two_parallel_roots_device_rejected", !static_cast<bool>(shader),
           shader ? "unexpectedly compiled" : shader.metadata().error);
}

}// namespace

int main(int argc, char *argv[]) {
    auto has_flag = [&](luisa::string_view flag) {
        for (auto i = 1; i < argc; ++i) {
            if (argv[i] != nullptr && luisa::string_view{argv[i]} == flag) { return true; }
        }
        return false;
    };
    auto flag_value = [&](luisa::string_view flag) -> luisa::string_view {
        for (auto i = 1; i + 1 < argc; ++i) {
            if (argv[i] != nullptr && luisa::string_view{argv[i]} == flag) {
                return argv[i + 1] == nullptr ? luisa::string_view{} : luisa::string_view{argv[i + 1]};
            }
        }
        return {};
    };
    auto backend = argc > 1 && argv[1] != nullptr && !luisa::string_view{argv[1]}.starts_with("--")
                       ? luisa::string_view{argv[1]}
                       : luisa::string_view{};

    // ---- demo subcommands (own drivers, own device) -------------------------
    if (has_flag("--cnn")) { return cnn::run_cnn_inference(argc, argv); }
    if (has_flag("--poly-fit")) { return polyfit::run_poly_fit(argc, argv); }
    if (has_flag("--linear-regression")) { return lreg::run_linear_regression(argc, argv); }
    if (has_flag("--mlp")) { return mlptrain::run_mlp(argc, argv); }
    if (has_flag("--mnist")) { return mnisttrain::run_mnist(argc, argv); }
    if (has_flag("--rnn")) { return rnntrain::run_rnn(argc, argv); }
    if (has_flag("--basics")) { return basics::run_basics(argc, argv); }

    if (backend.empty()) {
        LUISA_WARNING("Usage: example_tensor <backend> [--kernel <name>]... [--guard] [--report]");
        print_unimplemented_report();
        return 2;
    }

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);
    set_active_backend(backend);

    if (has_flag("--report")) {
        print_unimplemented_report();
        return 0;
    }

    if (has_flag("--guard")) {
        run_guard_test(device);
        LUISA_INFO("=== tensor: {} passed, {} failed ===", pass_count(), failure_count());
        return failure_count() == 0 ? 0 : 1;
    }

    // ---- kernel suite -------------------------------------------------------
    luisa::vector<luisa::string_view> filters;
    for (auto i = 1; i + 1 < argc; ++i) {
        if (argv[i] != nullptr && luisa::string_view{argv[i]} == "--kernel") {
            filters.emplace_back(argv[i + 1] == nullptr ? "" : argv[i + 1]);
        }
    }
    LUISA_INFO("=== tensor: running {} kernel case(s) on '{}' ===",
               filters.empty() ? std::size(cases) : filters.size(), backend);
    for (auto &&c : cases) {
        if (!filters.empty() &&
            std::find(filters.begin(), filters.end(), c.name) == filters.end()) {
            continue;
        }
        LUISA_INFO("--- tensor: case '{}' ---", c.name);
        c.run(device, stream);
    }
    print_unimplemented_report();
    LUISA_INFO("=== tensor: {} passed, {} failed ===", pass_count(), failure_count());
    return failure_count() == 0 ? 0 : 1;
}
