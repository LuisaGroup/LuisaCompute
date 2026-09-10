#include <algorithm>

#include <luisa/ast/type.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/tile/analysis.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/function.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/verifier.h>

#include "../metal_air_pipeline.h"
#include "../metal_compiler.h"
#include "../metal_device.h"
#include "../metal_shader.h"
#include "../../common/env_flag.h"

namespace luisa::compute::metal {

namespace {

namespace tx = tile::bridge::xir;

// These are compiler budgets, not queried register/private-memory capacities.
// The AIR compiler and resulting pipeline still enforce physical constraints.
constexpr auto tile_private_snapshot_budget = 64u * 1024u;
constexpr auto tile_subgroup_width = 32u;

class MetalTileCostPolicy final : public tx::ExecutionCostPolicy {
public:
    [[nodiscard]] tx::ExecutionCost evaluate(
        tx::ExecutionTarget, const tx::ExecutionPlan &,
        const tx::ExecutionWork &work, const tx::ExecutionCostModel &model) const noexcept override {
        tx::ExecutionCost cost;
        // Total SIMD-group work is a relative prior, not a latency prediction.
        // No CPU workers, task chunks, or work-stealing critical path are used.
        cost.arithmetic_work = work.arithmetic_per_packet * static_cast<double>(work.packet_count);
        cost.memory_work = work.memory_per_packet * static_cast<double>(work.packet_count);
        cost.dispatch_work = model.block_dispatch * static_cast<double>(work.block_count);
        cost.score = cost.arithmetic_work + cost.memory_work + cost.dispatch_work;
        return cost;
    }
};

class MetalTileTargetInfo final : public tx::ExecutionTargetInfo {
private:
    uint32_t _max_threads;
    MetalTileCostPolicy _cost;

public:
    explicit MetalTileTargetInfo(uint32_t max_threads) noexcept : _max_threads{max_threads} {}
    [[nodiscard]] tx::ExecutionTarget target() const noexcept override { return {tile_subgroup_width, 1u, 1u}; }
    [[nodiscard]] luisa::vector<uint32_t> block_sizes() const noexcept override {
        luisa::vector<uint32_t> widths;
        // Bounded bootstrap search, not a calibrated occupancy optimum. An
        // explicit wider block remains legal when the device can execute it.
        for (auto width : {32u, 64u, 128u, 256u}) {
            if (width <= _max_threads) { widths.emplace_back(width); }
        }
        return widths;
    }
    [[nodiscard]] bool accepts(const tx::ExecutionPlan &candidate) const noexcept override {
        return candidate.block_size <= _max_threads &&
               candidate.block_size % tile_subgroup_width == 0u &&
               candidate.blocks_per_task == 0u &&
               (candidate.local_lanes == 1u || candidate.local_lanes == tile_subgroup_width);
    }
    [[nodiscard]] tx::ExecutionResourceLimits resource_limits(const tx::ExecutionPlan &) const noexcept override {
        return {tile_private_snapshot_budget};
    }
    [[nodiscard]] tx::ExecutionWork schedule(const tx::ExecutionPlan &, tx::ExecutionWork work) const noexcept override {
        // Keep generic packet/block counts. GPU occupancy and residency are
        // deliberately unmodeled rather than represented as CPU home chunks.
        return work;
    }
    [[nodiscard]] const tx::ExecutionCostPolicy &cost_policy() const noexcept override { return _cost; }
};

void append_axes(luisa::string &text, luisa::span<const uint32_t> axes) {
    text.append("[");
    for (size_t i = 0u; i < axes.size(); i++) {
        if (i != 0u) { text.append(","); }
        text.append(luisa::format("{}", axes[i]));
    }
    text.append("]");
}

}// namespace

ShaderCreationInfo MetalDevice::create_tile_kernel(
    const ShaderOption &requested_option, const tile::Function &kernel,
    const tile::CompileOptions &tile_options, tile::KernelMetadata &metadata) noexcept {
    return with_autorelease_pool([&] {
        metadata = {};
        auto fail = [&](luisa::string_view message) {
            metadata.error = message;
            return ShaderCreationInfo::make_invalid();
        };
        if (tile_options.lowering != tile::Lowering::NATIVE || tile_options.tirx != nullptr) {
            return fail("Metal4 Tile kernels require native XIR -> AIR lowering; TIRx/MPP source compilation belongs to the metal backend");
        }
        if (requested_option.compile_only) { return fail("Metal4 Tile compile-only archives are not supported yet"); }
        auto planner_options = tile_options.xir ? *tile_options.xir : tx::PlannerOptions{};
        if (planner_options.blocks_per_task != 0u || planner_options.search_task_grain) {
            return fail("Metal4 Tile planning does not support CPU task-grain constraints");
        }
        if (tile_options.threads_per_group != 0u) {
            if (planner_options.block_size != 0u && planner_options.block_size != tile_options.threads_per_group) {
                return fail("Conflicting XIR and Runtime threadgroup width constraints");
            }
            planner_options.block_size = tile_options.threads_per_group;
        }
        auto max_threads = static_cast<uint32_t>(std::min<NS::UInteger>(
            _handle->maxThreadsPerThreadgroup().width, UINT32_MAX));
        MetalTileTargetInfo target_info{max_threads};
        Clock codegen_clock;
        auto planned = tx::plan(kernel, target_info, planner_options);
        if (!planned) { return fail(planned.error); }
        const auto &plan = planned.selected;
        auto lowered = tx::lower(kernel, {.block_size = plan.block_size,
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
        if (!lowered) { return fail(lowered.error); }
        if (lowered.resources.snapshot_bytes_per_worker != plan.resources.snapshot_bytes_per_worker ||
            lowered.resources.snapshot_allocations != plan.resources.snapshot_allocations) {
            return fail("Metal4 Tile planner/lowering static snapshot analysis mismatch");
        }
        if (lowered.required_packet_width != 0u && lowered.required_packet_width != tile_subgroup_width) {
            return fail("Tile XIR subgroup-width contract differs from the Metal4 target");
        }
        if (lowered.dispatch_size != plan.dispatch_size ||
            (lowered.required_packet_width != 0u && lowered.dispatch_size % tile_subgroup_width != 0u)) {
            return fail("Tile XIR launch does not preserve complete Metal4 SIMD-group participation");
        }
        auto option = requested_option;
        auto ordered_reduction = tile::OrderedReductionAnalysis::run(kernel);
        option.enable_fast_math &= !ordered_reduction;
        // Tile lowering already produces plain CFG/SSA. Do not run the AST
        // destructuring/inlining pipeline again, or invent a separate pass list.
        auto cleanup = xir::create_ssa_optimization_pipeline({.enable_fast_math = option.enable_fast_math});
        if (!cleanup.run(lowered.module.get()).succeeded()) { return fail("Metal4 Tile XIR SSA cleanup failed"); }
        static_cast<void>(xir::simplify_cfg_pass_run_on_module(lowered.module.get()));
        auto verification = xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true});
        if (!verification.succeeded()) { return fail(verification.errors.front().message); }

        xir::XIRDebugPrinter printer;
        printer.emit_module(metadata.source, lowered.module.get());
        if (compute::detail::env_flag("LUISA_DUMP_XIR")) {
            LUISA_INFO("Metal4 Tile XIR after SSA cleanup [{}]:\n{}", kernel.name(), metadata.source);
        }
        auto air_target = metal_air_target_for_current_device();
        auto config = metal_air_codegen_config(air_target, option.name.empty() ? luisa::string{kernel.name()} : option.name);
        config.native_include = option.native_include;
        config.enable_fast_math = option.enable_fast_math;
        config.enable_extended_accel_limits = option.enable_extended_accel_limits;
        if (!luisa_compute_metal_codegen_llvm_supported(*lowered.module, config, &metadata.error)) {
            return ShaderCreationInfo::make_invalid();
        }
        auto block_size = make_uint3(plan.block_size, 1u, 1u);
        MetalShaderMetadata shader_metadata{};
        shader_metadata.block_size = block_size;
        const auto &arguments = kernel.body().block(0u)->arguments();
        auto argument_index = size_t{0u};
        for (auto argument : lowered.function->arguments()) {
            auto usage = lowered.argument_usages[argument_index];
            shader_metadata.argument_types.emplace_back(argument->type()->description());
            shader_metadata.argument_usages.emplace_back(usage);
            shader_metadata.argument_sampled.emplace_back(0u);
            metadata.arguments.emplace_back(tile::KernelArgument{
                arguments[argument_index]->type().scalar_type(), lowered.argument_sizes_bytes[argument_index], usage});
            argument_index++;
        }
        auto air = metal_codegen_air(*lowered.module, option, air_target);
        // This factory has only ordinary buffer root arguments: each uses the
        // existing 16-byte {device pointer, byte size} MetalShader binding.
        auto expected_root_size = std::max(size_t{16u}, arguments.size() * 16u);
        if (air.root_argument_size != expected_root_size || air.root_argument_size > 65536u || !air.intersection_functions.empty()) {
            return fail("Metal4 Tile AIR root-argument layout exceeds or differs from the buffer Runtime ABI");
        }
        shader_metadata.format_types = std::move(air.format_types);
        // Artifact-based cache identity includes the selected PSO block size:
        // AIR receives block_size as a runtime builtin, so identical libraries
        // can require different pipeline descriptors. No pre-AIR cache yet.
        shader_metadata.checksum = luisa::hash_combine({luisa::hash64(air.library.data(), air.library.size(), luisa::hash64_default_seed),
                                                        luisa::hash_value(option), plan.block_size, 0x54494c455f414952ull});
        auto codegen_ms = codegen_clock.toc();
        Clock compile_clock;
        auto pipeline = _compiler->compile(air.library, option, shader_metadata);
        auto compile_ms = compile_clock.toc();
        if (!pipeline.entry || !pipeline.indirect_entry) { return fail("Metal4 Tile AIR pipeline creation failed"); }
        for (auto pso : {pipeline.entry.get(), pipeline.indirect_entry.get()}) {
            if (pso->threadExecutionWidth() != tile_subgroup_width ||
                pso->maxTotalThreadsPerThreadgroup() < plan.block_size ||
                pso->staticThreadgroupMemoryLength() > _handle->maxThreadgroupMemoryLength()) {
                return fail("Compiled Metal4 Tile pipeline violates the selected execution/resource limits");
            }
        }
        metadata.dispatch_size = make_uint3(lowered.dispatch_size, 1u, 1u);
        metadata.realization = luisa::format(
            "TileIR -> XIR SSA -> LLVM AIR -> Metal4 Runtime; W{}; {} threads/group; local_lanes={}; "
            "source_format=XIR; exact search {} candidates; uncalibrated GPU work cost {:.3f}; "
            "occupancy/cache/communication_cost=unmodeled; CPU_task_scheduling=none; root order ",
            tile_subgroup_width, plan.block_size, plan.local_lanes, planned.candidates.size(), plan.cost.score);
        append_axes(metadata.realization, plan.root_axis_order);
        if (!plan.root_axis_tiles.empty()) {
            metadata.realization.append("; fixed_root_axis_tiles=");
            append_axes(metadata.realization, plan.root_axis_tiles);
            metadata.realization.append("; root_temporal_cache_cost=unmodeled");
        }
        metadata.realization.append(luisa::format(
            "; private_snapshot_budget={}; max_unrolled_tile_elements={}; unordered_reduction_partitions={}; "
            "fused_reduction_loads={}; fused_reduction_expressions={}; fused_pointwise_regions={}; deferred_maps={}; "
            "custom_cost_policy={}; fast_math={}; ordered_reduction={}",
            plan.resource_limits.max_snapshot_bytes_per_worker, planner_options.max_unrolled_tile_elements, planner_options.reduction_partitions,
            lowered.fused_reduction_loads, lowered.fused_reduction_expressions, lowered.fused_pointwise_regions,
            lowered.deferred_maps, planner_options.cost_policy != nullptr, option.enable_fast_math, ordered_reduction));
        metadata.realization.append(luisa::format("; static_snapshot_bytes_per_worker={}; static_snapshot_allocations={}; rejected_candidates={}",
                                                  plan.resources.snapshot_bytes_per_worker, plan.resources.snapshot_allocations, planned.rejected.size()));
        auto shader = luisa::new_with_allocator<MetalShader>(
            this, std::move(pipeline), std::move(shader_metadata.argument_usages),
            std::move(shader_metadata.argument_sampled), luisa::vector<MetalShader::Argument>{},
            shader_metadata.format_types, block_size, shader_metadata.checksum,
            air.library.size(), 0u, codegen_ms, compile_ms);
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pso();
        info.block_size = block_size;
        return info;
    });
}

}// namespace luisa::compute::metal
