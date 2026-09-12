#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/tile/analysis.h>
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/translators/xir2ast.h>

#include "env_flag.h"

// Shared Tile fallback for buffer-only GPU backends (DX/VK) without a native
// Tile compiler: TileIR -> XIR bridge -> xir2ast -> the backend's ordinary
// create_shader entry. Header-only, mirroring xir_autodiff.h; each backend
// includes it from exactly one translation unit.
namespace luisa::compute::backend_detail {

// Backend-supplied device facts for the GPU Tile XIR->AST fallback.
struct GPUTileTargetConfig {
    // Physical warp/subgroup width; also the XIR packet width.
    uint32_t warp_size;
    // Maximum compute block width (threads/group) the device admits.
    uint32_t max_block_size;
    // Compiler-owned snapshot budget per physical lane (hard bound).
    uint64_t max_local_bytes;
    // Route label used in KernelMetadata::realization.
    luisa::string_view backend_label;
};

// GPU execution target: complete-program lanes only (v1 disables local
// distribution, which excludes warp_lane_id/WARP_READ_LANE emission and the
// unvetted collective ABI through xir2ast), warp-multiple power-of-two block
// widths, and a single-owner launch schedule with no CPU home chunks, work
// stealing, or caller-thread activation costs.
class GPUTileExecutionTargetInfo final : public tile::bridge::xir::ExecutionTargetInfo {
private:
    GPUTileTargetConfig _config;

public:
    explicit GPUTileExecutionTargetInfo(GPUTileTargetConfig config) noexcept
        : _config{config} {}

    [[nodiscard]] tile::bridge::xir::ExecutionTarget target() const noexcept override {
        return {.packet_width = _config.warp_size,
                .worker_count = 1u,
                .task_chunks_per_worker = 1u};
    }

    [[nodiscard]] luisa::vector<uint32_t> block_sizes() const noexcept override {
        luisa::vector<uint32_t> sizes;
        for (auto width = _config.warp_size; width <= _config.max_block_size;) {
            sizes.emplace_back(width);
            if (width > std::numeric_limits<uint32_t>::max() / 2u) { break; }
            width *= 2u;
        }
        if (sizes.empty()) { sizes.emplace_back(_config.max_block_size); }
        return sizes;
    }

    [[nodiscard]] bool supports_local_distribution() const noexcept override { return false; }

    [[nodiscard]] bool accepts(const tile::bridge::xir::ExecutionPlan &candidate) const noexcept override {
        return candidate.block_size <= _config.max_block_size &&
               candidate.dispatch_size != 0u;
    }

    [[nodiscard]] tile::bridge::xir::ExecutionResourceLimits resource_limits(
        const tile::bridge::xir::ExecutionPlan &) const noexcept override {
        return {.max_snapshot_bytes_per_worker = _config.max_local_bytes};
    }

    [[nodiscard]] tile::bridge::xir::ExecutionWork schedule(
        const tile::bridge::xir::ExecutionPlan &, tile::bridge::xir::ExecutionWork work) const noexcept override {
        // The launch grid owns every block: one task per block, one owner,
        // and critical paths that span the whole launch.
        work.task_count = work.block_count;
        work.blocks_per_task = 1u;
        work.active_workers = 1u;
        work.critical_packets = work.packet_count;
        work.critical_blocks = work.block_count;
        work.critical_tasks = work.task_count;
        return work;
    }

    [[nodiscard]] const tile::bridge::xir::ExecutionCostPolicy &cost_policy() const noexcept override {
        static const tile::bridge::xir::AnalyticExecutionCostPolicy policy;
        return policy;
    }
};

// Diagnostic override for the compiler-owned snapshot budget. Malformed or
// zero values keep the backend-supplied default.
[[nodiscard]] inline uint64_t gpu_tile_max_local_bytes(uint64_t fallback) noexcept {
    auto *text = std::getenv("LUISA_TILE_XIR_MAX_LOCAL_BYTES");
    if (text == nullptr) { return fallback; }
    auto view = std::string_view{text};
    auto value = uint64_t{0u};
    auto parsed = std::from_chars(view.data(), view.data() + view.size(), value);
    if (parsed.ec == std::errc{} && parsed.ptr == view.data() + view.size() && value != 0u) {
        return value;
    }
    return fallback;
}

// Fail-closed Tile realization for backends without a native Tile compiler.
// Every error is reported through metadata.error; no exception escapes.
[[nodiscard]] inline ShaderCreationInfo create_tile_kernel_via_ast(
    DeviceInterface *device, const ShaderOption &option, const tile::Function &kernel,
    const tile::CompileOptions &tile_options, tile::KernelMetadata &metadata,
    const GPUTileTargetConfig &target) noexcept {
    metadata = {};
    auto fail = [&metadata](luisa::string message) noexcept {
        metadata.error = std::move(message);
        return ShaderCreationInfo::make_invalid();
    };
    try {
        if (option.compile_only) {
            return fail("Tile compile-only archives are not supported on this backend");
        }
        if (tile_options.lowering == tile::Lowering::TIRX || tile_options.tirx != nullptr) {
            return fail("TIRx lowering is unavailable on this backend; the XIR fallback requires native lowering");
        }
        if (detail::env_flag("LUISA_TILE_XIR2AST_DISABLE")) {
            return fail("Tile XIR->AST fallback disabled by LUISA_TILE_XIR2AST_DISABLE");
        }
        auto config = target;
        config.max_local_bytes = gpu_tile_max_local_bytes(config.max_local_bytes);
        if (config.warp_size == 0u || (config.warp_size & (config.warp_size - 1u)) != 0u) {
            return fail("Tile XIR target requires a power-of-two warp size");
        }
        auto planner_options = tile_options.xir ? *tile_options.xir : tile::bridge::xir::PlannerOptions{};
        // The GPU fallback never reads the SIMD backend's LUISA_SIMD_* test
        // environment constraints; the Runtime/XIR conflict check below is the
        // only exact block-width reconciliation.
        if (tile_options.threads_per_group != 0u) {
            if (planner_options.block_size != 0u && planner_options.block_size != tile_options.threads_per_group) {
                return fail("Conflicting XIR and Runtime block width constraints");
            }
            planner_options.block_size = tile_options.threads_per_group;
        }
        const GPUTileExecutionTargetInfo target_info{config};
        auto planned = tile::bridge::xir::plan(kernel, target_info, planner_options);
        if (!planned) {
            return fail(std::move(planned.error));
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
            return fail(std::move(lowered.error));
        }
        if (lowered.resources.snapshot_bytes_per_worker != plan.resources.snapshot_bytes_per_worker ||
            lowered.resources.snapshot_allocations != plan.resources.snapshot_allocations) {
            return fail("Tile XIR planner/lowering static snapshot analysis mismatch");
        }
        if (lowered.required_packet_width != 0u) {
            return fail("Tile XIR packet-width contract requires warp-distributed lanes, "
                        "which this backend realization does not support");
        }
        auto ordered_reduction = tile::OrderedReductionAnalysis::run(kernel);
        auto adjusted_option = option;
        adjusted_option.enable_fast_math = option.enable_fast_math && !ordered_reduction;
        if (detail::env_flag("LUISA_TILE_XIR2AST_REPORT_XIR")) {
            luisa::string text;
            xir::XIRDebugPrinter printer;
            printer.emit_function(text, lowered.function);
            LUISA_INFO("Tile XIR before xir2ast [{}]:\n{}", kernel.name(), text);
        }
        // The bridge has already produced plain CFG/SSA; normalization runs the
        // shared restructure/reg2mem pipeline and guarantees no-PHI output.
        xir::xir_to_ast_normalize_module(lowered.module.get());
        const xir::FunctionDefinition *kernel_def = nullptr;
        for (auto *function : lowered.module->function_list()) {
            if (function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
                kernel_def = static_cast<const xir::FunctionDefinition *>(function);
                break;
            }
        }
        if (kernel_def == nullptr) {
            return fail("Tile XIR lowering did not produce a kernel definition");
        }
        // Tile kernels are buffer-only; unbound buffer arguments are the
        // standard DSL case handled by the backend shader serializer.
        auto builder = xir::xir_to_ast_translate(*kernel_def, {.strict = true});
        if (builder == nullptr) {
            return fail("Tile XIR->AST translation failed");
        }
        auto info = device->create_shader(adjusted_option, builder->function());
        if (!info.valid()) {
            if (metadata.error.empty()) {
                metadata.error = "Tile XIR->AST kernel failed backend shader compilation";
            }
            return ShaderCreationInfo::make_invalid();
        }
        if (info.block_size.x != threads || info.block_size.y != 1u || info.block_size.z != 1u) {
            LUISA_WARNING(
                "Tile XIR->AST kernel block size mismatch: planner selected {}, "
                "backend returned {}; trusting the backend value.",
                threads, info.block_size.x);
        }
        metadata.dispatch_size = make_uint3(lowered.dispatch_size, 1u, 1u);
        auto &arguments = kernel.body().block(0u)->arguments();
        for (size_t i = 0u; i < arguments.size(); i++) {
            metadata.arguments.emplace_back(tile::KernelArgument{arguments[i]->type().scalar_type(),
                                                                 lowered.argument_sizes_bytes[i],
                                                                 lowered.argument_usages[i]});
        }
        // The XIR bridge realization performs no view-forwarding that would
        // require the disjoint-writes launch contract.
        metadata.disjoint_writes = false;
        metadata.realization = luisa::format(
            "TileIR -> XIR SSA -> AST -> {}; {} threads/group; fast_math={}; ordered_reduction={}",
            config.backend_label, threads, adjusted_option.enable_fast_math, ordered_reduction);
        metadata.realization.append(luisa::format(
            "; exact search {} candidates, uncalibrated cost {:.3f} (arithmetic {:.3f}, memory {:.3f}, "
            "dispatch {:.3f}, imbalance {:.3f}), root order [",
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
        metadata.realization.append(luisa::format(
            "; static_snapshot_bytes_per_worker={}; static_snapshot_allocations={}; snapshot_budget={}; rejected_candidates={}",
            plan.resources.snapshot_bytes_per_worker, plan.resources.snapshot_allocations,
            plan.resource_limits.max_snapshot_bytes_per_worker, planned.rejected.size()));
        metadata.realization.append(luisa::format(
            "; blocks_per_task={}; task_dispatch_cost={:.3f}; worker_activation_cost={:.3f}; custom_cost_policy={}",
            plan.blocks_per_task, plan.cost.task_dispatch_work, plan.cost.activation_work,
            planner_options.cost_policy != nullptr));
        metadata.realization.append(luisa::format("; max_unrolled_tile_elements={}", planner_options.max_unrolled_tile_elements));
        metadata.realization.append(luisa::format("; unordered_reduction_partitions={}", planner_options.reduction_partitions));
        metadata.realization.append(luisa::format(
            "; load_reduction_fusion={}; fused_reduction_loads={}; elided_load_snapshots={}",
            planner_options.enable_load_reduction_fusion, lowered.fused_reduction_loads, lowered.elided_load_snapshots));
        metadata.realization.append(luisa::format(
            "; map_fusion={}; deferred_maps={}", planner_options.enable_map_fusion, lowered.deferred_maps));
        metadata.realization.append(luisa::format(
            "; expression_reduction_fusion={}; fused_reduction_expressions={}; elided_expression_snapshots={}",
            planner_options.enable_expression_reduction_fusion, lowered.fused_reduction_expressions,
            lowered.elided_expression_snapshots));
        metadata.realization.append(luisa::format(
            "; pointwise_fusion={}; fused_pointwise_regions={}; fused_pointwise_loads={}; fused_pointwise_stores={}; pointwise_alias_checks={}",
            planner_options.enable_pointwise_fusion, lowered.fused_pointwise_regions, lowered.fused_pointwise_loads,
            lowered.fused_pointwise_stores, lowered.pointwise_alias_checks));
        return info;
    } catch (const std::exception &error) {
        metadata.error = error.what();
    } catch (...) {
        metadata.error = "unknown failure creating a Tile XIR->AST kernel";
    }
    return ShaderCreationInfo::make_invalid();
}

}// namespace luisa::compute::backend_detail
