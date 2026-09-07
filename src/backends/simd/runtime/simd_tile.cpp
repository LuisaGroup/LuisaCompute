#include <exception>

#include <luisa/core/stl/format.h>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/verifier.h>

#include "simd_device.h"
#include "simd_shader.h"
#include "simd_thread_pool.h"
#include "../../common/env_flag.h"

namespace luisa::compute::simd {

ShaderCreationInfo SIMDDevice::create_tile_kernel(
    const ShaderOption &option, const tile::Function &kernel,
    const tile::CompileOptions &tile_options, tile::KernelMetadata &metadata) noexcept {
    metadata = {};
    try {
        if (tile_options.lowering != tile::Lowering::NATIVE || tile_options.tirx != nullptr || option.compile_only) {
            metadata.error = "SIMD Tile factory requires native XIR JIT; use the independent TIRx bridge for TVM compilation";
            return ShaderCreationInfo::make_invalid();
        }
        auto planner_options = tile_options.xir ? *tile_options.xir : tile::bridge::xir::PlannerOptions{};
        if (tile_options.threads_per_group != 0u) {
            if (planner_options.block_size != 0u && planner_options.block_size != tile_options.threads_per_group) {
                metadata.error = "Conflicting XIR and Runtime block width constraints";
                return ShaderCreationInfo::make_invalid();
            }
            planner_options.block_size = tile_options.threads_per_group;
        }
        auto planned = tile::bridge::xir::plan(kernel, {_warp_width, _thread_pool->worker_count()}, planner_options);
        if (!planned) {
            metadata.error = std::move(planned.error);
            return ShaderCreationInfo::make_invalid();
        }
        auto &plan = planned.selected;
        auto threads = plan.block_size;
        auto lowered = tile::bridge::xir::lower(kernel, {.block_size = threads, .root_axis_order = plan.root_axis_order});
        if (!lowered) {
            metadata.error = std::move(lowered.error);
            return ShaderCreationInfo::make_invalid();
        }
        // The bridge has already produced plain CFG/SSA. Reuse the shared
        // SSA factory; do not rerun AST destructuring/inlining or invent a
        // different pass list. Resource reads are not declared noalias.
        if (!detail::env_flag("LUISA_SIMD_DISABLE_TILE_XIR_CLEANUP")) {
            auto cleanup = xir::create_ssa_optimization_pipeline({.enable_fast_math = option.enable_fast_math});
            if (!cleanup.run(lowered.module.get()).succeeded()) {
                metadata.error = "Tile XIR SSA cleanup failed";
                return ShaderCreationInfo::make_invalid();
            }
            static_cast<void>(xir::simplify_cfg_pass_run_on_module(lowered.module.get()));
            if (!xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true}).succeeded()) {
                metadata.error = "Invalid Tile XIR after SSA/CFG cleanup";
                return ShaderCreationInfo::make_invalid();
            }
        }
        if (detail::env_flag("LUISA_SIMD_REPORT_XIR")) {
            luisa::string text;
            xir::XIRDebugPrinter printer;
            printer.emit_function(text, lowered.function);
            LUISA_INFO("Tile XIR before scheduling [{}]:\n{}", kernel.name(), text);
        }
        auto packet_batch = _warp_width != 1u && threads > _warp_width && !detail::env_flag("LUISA_SIMD_DISABLE_PACKET_BATCH_ENTRY");
        auto block_batch = packet_batch && !detail::env_flag("LUISA_SIMD_DISABLE_BLOCK_BATCH_ENTRY");
        auto compiled = compile_simd_kernel(lowered.function, _warp_width, kernel.name(), option.enable_fast_math,
                                            !detail::env_flag("LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST"),
                                            !detail::env_flag("LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER"),
                                            std::getenv("LUISA_SIMD_DUMP_ASSEMBLY_DIR") != nullptr,
                                            _thread_pool->worker_count(), packet_batch, block_batch, true);
        if (!compiled.succeeded()) {
            for (auto &error : compiled.diagnostics) { metadata.error.append(error).append("\n"); }
            return ShaderCreationInfo::make_invalid();
        }
        metadata.dispatch_size = make_uint3(lowered.dispatch_size, 1u, 1u);
        metadata.source = compiled.llvm_ir;
        metadata.realization = luisa::format(
            "TileIR -> XIR SSA -> SIMD Schedule -> LLVM; W{}, {} workers/block, {} CPU workers; "
            "ordered CPU pipeline; Schedule blocks={}, direct CFG={}, contiguous reads={}, broadcasts={}",
            _warp_width, threads, _thread_pool->worker_count(), compiled.schedule_block_count,
            compiled.direct_control_flow, compiled.contiguous_buffer_read_count, compiled.uniform_buffer_broadcast_count);
        metadata.realization.append(luisa::format(
            "; exact search {} candidates, uncalibrated cost {:.3f} (arithmetic {:.3f}, memory {:.3f}, dispatch {:.3f}, imbalance {:.3f}), root order [",
            planned.candidates.size(), plan.cost.score, plan.cost.arithmetic_work, plan.cost.memory_work,
            plan.cost.dispatch_work, plan.cost.imbalance_work));
        for (size_t i = 0u; i < plan.root_axis_order.size(); i++) {
            if (i != 0u) { metadata.realization.append(","); }
            metadata.realization.append(luisa::format("{}", plan.root_axis_order[i]));
        }
        metadata.realization.append("]");
        auto &arguments = kernel.body().block(0u)->arguments();
        for (size_t i = 0u; i < arguments.size(); i++) {
            metadata.arguments.emplace_back(tile::KernelArgument{arguments[i]->type().scalar_type(), lowered.argument_sizes_bytes[i], lowered.argument_usages[i]});
        }
        auto block_size = make_uint3(threads, 1u, 1u);
        auto shader = luisa::new_with_allocator<SIMDShader>(std::move(compiled), block_size, std::move(lowered.argument_usages));
        ShaderCreationInfo info;
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->native_handle();
        info.block_size = block_size;
        return info;
    } catch (const std::exception &error) {
        metadata.error = error.what();
    } catch (...) {
        metadata.error = "unknown failure creating an XIR/SIMD Tile kernel";
    }
    return ShaderCreationInfo::make_invalid();
}

}// namespace luisa::compute::simd
