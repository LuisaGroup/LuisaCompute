#include <charconv>
#include <cstdlib>
#include <exception>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/analysis.h>
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

namespace {

class SIMDTileTargetInfo final : public tile::bridge::xir::ThreadPoolExecutionTargetInfo {
public:
    explicit SIMDTileTargetInfo(tile::bridge::xir::ExecutionTarget target) noexcept
        : ThreadPoolExecutionTargetInfo{target} {}
    [[nodiscard]] tile::bridge::xir::ExecutionResourceLimits resource_limits(const tile::bridge::xir::ExecutionPlan &) const noexcept override {
        // The workspace ABI is packet-wide, including complete-program lanes.
        // Native codegen still checks alignment and its final workspace size.
        return {simd_max_private_workspace_bytes / target().packet_width};
    }
};

// Diagnostic fixed constraint, not a search heuristic. Do not silently repair
// malformed metadata or override a conflicting explicit Runtime constraint.
[[nodiscard]] bool root_axis_tiles_from_environment(
    tile::bridge::xir::PlannerOptions &options, luisa::string &error) {
    auto text = std::getenv("LUISA_SIMD_ROOT_AXIS_TILES");
    if (text == nullptr) { return true; }
    auto remaining = std::string_view{text};
    luisa::vector<uint32_t> tiles;
    while (true) {
        auto delimiter = remaining.find(',');
        auto token = remaining.substr(0u, delimiter);
        auto tile = uint32_t{0u};
        auto parsed = std::from_chars(token.data(), token.data() + token.size(), tile);
        if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || tile == 0u) {
            error = "LUISA_SIMD_ROOT_AXIS_TILES requires comma-separated positive uint32 factors";
            return false;
        }
        tiles.emplace_back(tile);
        if (delimiter == std::string_view::npos) { break; }
        remaining.remove_prefix(delimiter + 1u);
    }
    if (!options.root_axis_tiles.empty() && options.root_axis_tiles != tiles) {
        error = "Conflicting XIR and LUISA_SIMD_ROOT_AXIS_TILES constraints";
        return false;
    }
    options.root_axis_tiles = std::move(tiles);
    return true;
}

}// namespace

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
        planner_options.enable_load_reduction_fusion |= detail::env_flag("LUISA_SIMD_ENABLE_LOAD_REDUCTION_FUSION");
        planner_options.enable_load_reduction_fusion &= !detail::env_flag("LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION");
        planner_options.enable_pointwise_fusion |= detail::env_flag("LUISA_SIMD_ENABLE_POINTWISE_FUSION");
        planner_options.enable_pointwise_fusion &= !detail::env_flag("LUISA_SIMD_DISABLE_POINTWISE_FUSION");
        planner_options.enable_expression_reduction_fusion |= detail::env_flag("LUISA_SIMD_ENABLE_EXPRESSION_REDUCTION_FUSION");
        planner_options.enable_expression_reduction_fusion &= !detail::env_flag("LUISA_SIMD_DISABLE_EXPRESSION_REDUCTION_FUSION");
        planner_options.enable_map_fusion |= detail::env_flag("LUISA_SIMD_ENABLE_MAP_FUSION");
        planner_options.enable_map_fusion &= !detail::env_flag("LUISA_SIMD_DISABLE_MAP_FUSION");
        if (!root_axis_tiles_from_environment(planner_options, metadata.error)) {
            return ShaderCreationInfo::make_invalid();
        }
        if (tile_options.threads_per_group != 0u) {
            if (planner_options.block_size != 0u && planner_options.block_size != tile_options.threads_per_group) {
                metadata.error = "Conflicting XIR and Runtime block width constraints";
                return ShaderCreationInfo::make_invalid();
            }
            planner_options.block_size = tile_options.threads_per_group;
        }
        const auto target_info = SIMDTileTargetInfo{{_warp_width, _thread_pool->worker_count()}};
        auto planned = tile::bridge::xir::plan(kernel, target_info, planner_options);
        if (!planned) {
            metadata.error = std::move(planned.error);
            return ShaderCreationInfo::make_invalid();
        }
        auto &plan = planned.selected;
        auto threads = plan.block_size;
        auto lowered = tile::bridge::xir::lower(kernel, {.block_size = threads,
                                                         .root_axis_order = plan.root_axis_order,
                                                         .max_local_bytes = plan.resource_limits.max_snapshot_bytes_per_worker,
                                                         .max_unrolled_tile_elements = planner_options.max_unrolled_tile_elements,
                                                         .reduction_partitions = planner_options.reduction_partitions,
                                                         .local_lanes = plan.local_lanes,
                                                         .enable_load_reduction_fusion = planner_options.enable_load_reduction_fusion,
                                                         .enable_pointwise_fusion = planner_options.enable_pointwise_fusion,
                                                         .enable_expression_reduction_fusion = planner_options.enable_expression_reduction_fusion,
                                                         .enable_map_fusion = planner_options.enable_map_fusion,
                                                         .root_axis_tiles = plan.root_axis_tiles});
        if (!lowered) {
            metadata.error = std::move(lowered.error);
            return ShaderCreationInfo::make_invalid();
        }
        if (lowered.resources.snapshot_bytes_per_worker != plan.resources.snapshot_bytes_per_worker ||
            lowered.resources.snapshot_allocations != plan.resources.snapshot_allocations) {
            metadata.error = "SIMD Tile planner/lowering static snapshot analysis mismatch";
            return ShaderCreationInfo::make_invalid();
        }
        if (lowered.required_packet_width != 0u && lowered.required_packet_width != _warp_width) {
            metadata.error = "Tile XIR packet-width contract differs from the SIMD target";
            return ShaderCreationInfo::make_invalid();
        }
        auto ordered_reduction = tile::OrderedReductionAnalysis::run(kernel);
        auto enable_fast_math = option.enable_fast_math && !ordered_reduction;
        // The bridge has already produced plain CFG/SSA. Reuse the shared
        // SSA factory; do not rerun AST destructuring/inlining or invent a
        // different pass list. Resource reads are not declared noalias.
        if (!detail::env_flag("LUISA_SIMD_DISABLE_TILE_XIR_CLEANUP")) {
            auto cleanup = xir::create_ssa_optimization_pipeline({.enable_fast_math = enable_fast_math});
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
        auto compiled = compile_simd_kernel(lowered.function, _warp_width, kernel.name(), enable_fast_math,
                                            !detail::env_flag("LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST"),
                                            !detail::env_flag("LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER"),
                                            std::getenv("LUISA_SIMD_DUMP_ASSEMBLY_DIR") != nullptr,
                                            _thread_pool->worker_count(), packet_batch, block_batch, true,
                                            64u * 1024u, !detail::env_flag("LUISA_SIMD_DISABLE_INTERLEAVED_PRIVATE_ARRAYS"),
                                            !detail::env_flag("LUISA_SIMD_DISABLE_CONTIGUOUS_PRIVATE_ACCESS"));
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
        if (!plan.root_axis_tiles.empty()) {
            metadata.realization.append("; fixed_root_axis_tiles=[");
            for (size_t i = 0u; i < plan.root_axis_tiles.size(); i++) {
                if (i != 0u) { metadata.realization.append(","); }
                metadata.realization.append(luisa::format("{}", plan.root_axis_tiles[i]));
            }
            metadata.realization.append("]; root_temporal_cache_cost=unmodeled");
        }
        metadata.realization.append(luisa::format("; local_lanes={}", plan.local_lanes));
        metadata.realization.append(luisa::format("; static_snapshot_bytes_per_worker={}; static_snapshot_allocations={}; snapshot_budget={}; rejected_candidates={}",
                                                  plan.resources.snapshot_bytes_per_worker, plan.resources.snapshot_allocations,
                                                  plan.resource_limits.max_snapshot_bytes_per_worker, planned.rejected.size()));
        metadata.realization.append(luisa::format("; blocks_per_task={}; task_dispatch_cost={:.3f}; worker_activation_cost={:.3f}; custom_cost_policy={}",
                                                  plan.blocks_per_task, plan.cost.task_dispatch_work, plan.cost.activation_work, planner_options.cost_policy != nullptr));
        metadata.realization.append(luisa::format("; max_unrolled_tile_elements={}", planner_options.max_unrolled_tile_elements));
        metadata.realization.append(luisa::format("; unordered_reduction_partitions={}", planner_options.reduction_partitions));
        metadata.realization.append(luisa::format("; load_reduction_fusion={}; fused_reduction_loads={}; elided_load_snapshots={}",
                                                  planner_options.enable_load_reduction_fusion, lowered.fused_reduction_loads, lowered.elided_load_snapshots));
        metadata.realization.append(luisa::format("; interleaved_private_arrays={}", compiled.interleaved_private_arrays));
        metadata.realization.append(luisa::format("; map_fusion={}; deferred_maps={}", planner_options.enable_map_fusion, lowered.deferred_maps));
        metadata.realization.append(luisa::format("; expression_reduction_fusion={}; fused_reduction_expressions={}; elided_expression_snapshots={}",
                                                  planner_options.enable_expression_reduction_fusion, lowered.fused_reduction_expressions, lowered.elided_expression_snapshots));
        metadata.realization.append(luisa::format("; pointwise_fusion={}; fused_pointwise_regions={}; fused_pointwise_loads={}; fused_pointwise_stores={}; pointwise_alias_checks={}",
                                                  planner_options.enable_pointwise_fusion, lowered.fused_pointwise_regions, lowered.fused_pointwise_loads,
                                                  lowered.fused_pointwise_stores, lowered.pointwise_alias_checks));
        metadata.realization.append(luisa::format("; contiguous_private_reads={}; contiguous_private_writes={}",
                                                  compiled.contiguous_private_read_count, compiled.contiguous_private_write_count));
        metadata.realization.append(luisa::format("; private_workspace_bytes={}", compiled.private_workspace_size));
        metadata.realization.append(luisa::format("; full_packet_specializations={}; full_packet_cloned_instructions={}",
                                                  compiled.full_packet_specialization_count, compiled.full_packet_cloned_instruction_count));
        metadata.realization.append(luisa::format("; fast_math={}; ordered_reduction={}", enable_fast_math, ordered_reduction));
        auto &arguments = kernel.body().block(0u)->arguments();
        for (size_t i = 0u; i < arguments.size(); i++) {
            metadata.arguments.emplace_back(tile::KernelArgument{arguments[i]->type().scalar_type(), lowered.argument_sizes_bytes[i], lowered.argument_usages[i]});
        }
        auto block_size = make_uint3(threads, 1u, 1u);
        auto shader = luisa::new_with_allocator<SIMDShader>(std::move(compiled), block_size, std::move(lowered.argument_usages), plan.blocks_per_task);
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
