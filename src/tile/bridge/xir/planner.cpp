#include <algorithm>
#include <cmath>
#include <numeric>

#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/verifier.h>
#include "representation.h"
#include "root_mapping.h"

namespace luisa::compute::tile::bridge::xir {
namespace {

[[noreturn]] void fail(const char *message) { LUISA_ERROR("{}", message); }

[[nodiscard]] uint64_t volume(const IndexSpace &space) {
    uint64_t count = 1u;
    for (auto &axis : space.axes()) {
        if (!axis.extent.is_constant()) { fail("XIR planner requires static domains"); }
        auto extent = axis.extent.constant_value();
        if (extent > UINT32_MAX || (extent && count > UINT32_MAX / extent)) { fail("XIR planner domain exceeds uint32 range"); }
        count *= extent;
    }
    return count;
}

[[nodiscard]] luisa::optional<double> literal(const Value *value) {
    auto op = value->defining_operation();
    if (op == nullptr || op->kind() != OperationKind::CONSTANT) { return {}; }
    auto attribute = op->attribute("value");
    if (attribute == nullptr) { return {}; }
    if (auto n = luisa::get_if<int64_t>(&attribute->value())) { return static_cast<double>(*n); }
    if (auto n = luisa::get_if<uint64_t>(&attribute->value())) { return static_cast<double>(*n); }
    return {};
}

// A cost estimate only, never a bounds/dependence proof. Unknown and nonlinear
// addressing pays the gather prior rather than acquiring a false contiguity
// guarantee. Root block-argument identity, not dimension names, identifies axes.
[[nodiscard]] luisa::optional<double> slope(const Value *value, const Value *axis,
                                            const luisa::vector<const Value *> &indices, uint32_t depth = 0u) {
    if (value == axis) { return 1.0; }
    if (std::find(indices.begin(), indices.end(), value) != indices.end() || literal(value)) { return 0.0; }
    auto op = value->defining_operation();
    if (depth > 64u || op == nullptr || op->kind() != OperationKind::ELEMENTWISE) { return {}; }
    auto a = slope(op->operand(0u), axis, indices, depth + 1u);
    if (!a) { return {}; }
    if (op->elementwise_op() == ElementwiseOp::NEG) { return -*a; }
    if (op->operand_count() != 2u) { return {}; }
    auto b = slope(op->operand(1u), axis, indices, depth + 1u);
    if (!b) { return {}; }
    switch (op->elementwise_op()) {
        case ElementwiseOp::ADD: return *a + *b;
        case ElementwiseOp::SUB: return *a - *b;
        case ElementwiseOp::MUL:
            if (auto c = literal(op->operand(1u))) { return *a * *c; }
            if (auto c = literal(op->operand(0u))) { return *b * *c; }
            break;
        default: break;
    }
    return {};
}

struct Work {
    double arithmetic{0.0};
    double memory{0.0};
    ExecutionMmaWork mma;
};

[[nodiscard]] LowerOptions representation_options(const PlannerOptions &options, uint32_t lanes) {
    return {.max_unrolled_tile_elements = options.max_unrolled_tile_elements,
            .max_unrolled_region_work = options.max_unrolled_region_work,
            .mma_output_block = options.mma_output_block,
            .enable_mma_2d_blocking = options.enable_mma_2d_blocking,
            .max_unrolled_mma_terms = options.max_unrolled_mma_terms,
            .native_mma_vector_width = options.native_mma_vector_width,
            .reduction_partitions = options.reduction_partitions,
            .local_lanes = lanes,
            .enable_load_reduction_fusion = options.enable_load_reduction_fusion,
            .enable_pointwise_fusion = options.enable_pointwise_fusion,
            .enable_expression_reduction_fusion = options.enable_expression_reduction_fusion,
            .enable_map_fusion = options.enable_map_fusion};
}

struct SpatialAxis {
    const Value *value;
    double scale{1.0};
    // False when a packet can cross a root traversal digit boundary. This
    // must not become a null axis passed to slope(), which would incorrectly
    // classify every root coordinate as uniform.
    bool coherent{true};
};

[[nodiscard]] double local_iterations(uint64_t count, uint32_t lanes) {
    return static_cast<double>(count >= lanes ? ceil_div(count, static_cast<uint64_t>(lanes)) : count);
}
[[nodiscard]] bool materialized(const Value *value, uint32_t limit, uint32_t lanes, uint32_t region_budget) {
    auto op = value->defining_operation();
    return detail::bounded_tile(value, limit) || (lanes > 1u && detail::bounded_tile(value, lanes - 1u)) ||
           (op && op->kind() == OperationKind::TILE_MAP && detail::map_runtime_loop(*op, limit, region_budget));
}

[[nodiscard]] ExecutionWork distribute_thread_pool_work(ExecutionWork work, const ExecutionPlan &candidate, ExecutionTarget target) {
    auto packets = work.packet_count;
    auto blocks = work.block_count;
    auto grain = candidate.blocks_per_task ? static_cast<uint64_t>(candidate.blocks_per_task) :
                                             ceil_div(blocks, static_cast<uint64_t>(target.worker_count) * target.task_chunks_per_worker);
    grain = std::min(grain, blocks);
    auto tasks = ceil_div(blocks, grain);
    auto workers = std::min<uint64_t>(tasks, target.worker_count);
    // SIMDThreadPool executes one whole-range callback on the caller when
    // only one worker can run, regardless of the requested subdivision.
    if (workers == 1u) {
        return {work.arithmetic_per_packet, work.memory_per_packet, packets, blocks, 1u, 1u,
                static_cast<uint32_t>(blocks), packets, blocks, 1u, work.mma_per_packet};
    }
    // Round-robin home chunks: every chunk but the last is full. Compute the
    // maximum load without iterating over the launch or the worker count.
    auto critical = [&](uint64_t count, uint64_t capacity) {
        auto full_tasks = tasks - 1u;
        auto last = count - full_tasks * capacity;
        return std::max(ceil_div(full_tasks, workers) * capacity,
                        (full_tasks / workers) * capacity + last);
    };
    return {work.arithmetic_per_packet, work.memory_per_packet, packets, blocks, tasks,
            static_cast<uint32_t>(workers), static_cast<uint32_t>(grain),
            critical(packets, grain * candidate.block_size / target.packet_width),
            critical(blocks, grain), ceil_div(tasks, workers), work.mma_per_packet};
}

void read_work(const Value *value, double repetitions, bool dynamic,
               ExecutionTarget target, const ExecutionCostModel &cost, uint32_t limit, uint32_t lanes, Work &work,
               const PlannerOptions &options, const Operation *consumer, uint32_t depth = 0u) {
    if (!value->type().is_tile()) { return; }
    auto producer = value->defining_operation();
    auto realization = representation_options(options, lanes);
    if (detail::native_mma_snapshot(value, realization)) {
        if (dynamic || materialized(value, limit, lanes, options.max_unrolled_region_work) || detail::native_mma_plan(*consumer, realization)) {
            work.memory += repetitions * cost.gathered_lane * target.packet_width;
        }
        return;
    }
    if (detail::deferred_map(value, options.enable_map_fusion, lanes)) {
        if (depth >= 64u) { fail("XIR deferred recipe exceeds the depth budget"); }
        // Charge the scalar recipe at every actual consumer read. Broadcasting
        // and indirect reads may recompute it; do not price them as a free view.
        for (auto child : producer->region(0u)->block(0u)->operations()) {
            if (child->kind() == OperationKind::ELEMENTWISE) {
                work.arithmetic += repetitions * cost.arithmetic;
            } else if (child->kind() == OperationKind::TILE_EXTRACT) {
                read_work(child->operand(0u), repetitions, dynamic || !detail::expanded_extract(*child, limit, options.max_unrolled_region_work),
                          target, cost, limit, lanes, work, options, child, depth + 1u);
                work.arithmetic += repetitions * (4u + 3u * child->operand(0u)->type().index_space()->rank()) * cost.arithmetic;
            }
        }
        return;
    }
    if (detail::deferred_expression(value, limit, lanes, options.enable_map_fusion)) {
        if (depth >= 64u) { fail("XIR deferred recipe exceeds the depth budget"); }
        work.arithmetic += repetitions * cost.arithmetic;
        for (size_t i = 0u; i < producer->operand_count(); i++) { read_work(producer->operand(i), repetitions, dynamic, target, cost, limit, lanes, work, options, consumer, depth + 1u); }
        return;
    }
    if (auto fusion = detail::reduction_producer_fusion(value, limit, lanes, options.reduction_partitions,
                                                        options.enable_load_reduction_fusion, options.enable_expression_reduction_fusion);
        fusion && consumer->parent_block() == fusion->reduction->region(0u)->block(0u)) {
        // Production is charged once at its definition; the first reduction
        // uses that scalar directly, including repeated x*x. No rematerialization.
        return;
    }
    if (materialized(value, limit, lanes, options.max_unrolled_region_work)) {
        auto op = value->defining_operation();
        if (op && op->kind() == OperationKind::CONSTANT) { return; }
        work.memory += repetitions * cost.gathered_lane * target.packet_width;
    } else if (dynamic) {
        auto count = volume(*value->type().index_space());
        if (detail::definition_snapshot(value, count, representation_options(options, lanes))) {
            work.memory += repetitions * cost.gathered_lane * target.packet_width;
        } else {
            work.arithmetic += repetitions * count * 2.0 * cost.arithmetic;
        }
    }
}

void measure(const Block &block, SpatialAxis axis, double repetitions,
             ExecutionTarget target, const ExecutionCostModel &cost,
             luisa::vector<const Value *> indices, uint32_t limit, uint32_t lanes, Work &work, const PlannerOptions &options) {
    auto snapshot = [&](const Value *value) {
        if (!value->type().is_tile()) { return; }
        if (detail::native_mma_snapshot(value, representation_options(options, lanes))) {
            auto op = value->defining_operation();
            // Buffered carries use their already charged parallel copies.
            if (materialized(value, limit, lanes, options.max_unrolled_region_work) &&
                (!op || op->kind() == OperationKind::SERIAL || op->kind() == OperationKind::PIPELINE || op->kind() == OperationKind::REDUCE)) { return; }
            work.memory += repetitions * local_iterations(volume(*value->type().index_space()), lanes) * cost.gathered_lane * target.packet_width;
            if (op && detail::native_mma_plan(*op, representation_options(options, lanes)) &&
                !detail::traversal_snapshot(volume(*value->type().index_space()), representation_options(options, lanes))) {
                work.memory += repetitions * static_cast<double>(volume(*value->type().index_space())) * cost.gathered_lane * target.packet_width;
            }
            return;
        }
        if (detail::deferred_map(value, options.enable_map_fusion, lanes) ||
            detail::deferred_expression(value, limit, lanes, options.enable_map_fusion)) { return; }
        if (auto fusion = detail::reduction_producer_fusion(value, limit, lanes, options.reduction_partitions,
                                                            options.enable_load_reduction_fusion, options.enable_expression_reduction_fusion);
            fusion && !fusion->retain_snapshot) { return; }
        if (materialized(value, limit, lanes, options.max_unrolled_region_work)) {
            auto op = value->defining_operation();
            // Large carries are parallel copies, charged at their loop below.
            if (!op || op->kind() == OperationKind::CONSTANT || op->kind() == OperationKind::SERIAL ||
                op->kind() == OperationKind::PIPELINE || op->kind() == OperationKind::REDUCE ||
                detail::deferred_elementwise(value, limit, lanes)) { return; }
            work.memory += repetitions * local_iterations(volume(*value->type().index_space()), lanes) * cost.gathered_lane * target.packet_width;
        } else if (detail::definition_snapshot(value, volume(*value->type().index_space()), representation_options(options, lanes))) {
            auto count = volume(*value->type().index_space());
            if (count > 1u) { work.memory += repetitions * count * cost.gathered_lane * target.packet_width; }
        }
    };
    for (auto &argument : block.arguments()) { snapshot(argument.get()); }
    for (auto op : block.operations()) {
        if (auto binding = op->execution_scope_constraint(); binding && *binding != "worker" && *binding != "auto") {
            fail("XIR planner cannot satisfy this explicit execution binding");
        }
        if (op->memory_layout() || op->resource_class_constraint()) { fail("XIR planner cannot realize manual Memory"); }
        auto kind = op->kind();
        if (kind == OperationKind::PARALLEL || kind == OperationKind::SERIAL || kind == OperationKind::REDUCE || kind == OperationKind::PIPELINE) {
            auto body = op->region(0u)->block(0u);
            auto child_indices = indices;
            for (size_t i = 0u; i < op->domain()->rank(); i++) { child_indices.emplace_back(body->argument(i)); }
            auto iterations = repetitions * local_iterations(volume(*op->domain()), lanes);
            for (size_t i = 0u; i < op->result_count(); i++) {
                if (detail::bounded_tile(op->result(i), limit)) {
                    auto count = volume(*op->result(i)->type().index_space());
                    read_work(op->operand(i), repetitions * count, true, target, cost, limit, lanes, work, options, op);
                    // Initialization plus staging the next value, loading it,
                    // and updating current only after all staged copies exist.
                    work.memory += (repetitions + 3.0 * iterations) * count * cost.gathered_lane * target.packet_width;
                    for (auto term : body->operations()) {
                        if (term->kind() == OperationKind::YIELD) { read_work(term->operand(i), iterations * count, true, target, cost, limit, lanes, work, options, term); }
                    }
                }
            }
            measure(*body, axis, iterations, target, cost, std::move(child_indices), limit, lanes, work, options);
            if (lanes > 1u && kind == OperationKind::REDUCE && volume(*op->domain()) >= lanes) {
                // Local partials converge through a fixed tree and one root
                // broadcast. This is a relative prior, not measured cycles.
                work.arithmetic += repetitions * (2.0 * std::log2(lanes) + 2.0) * cost.arithmetic;
            }
        } else if (kind == OperationKind::TILE_MAP) {
            if (detail::value_allocation_plan(op->result(0u), volume(*op->result(0u)->type().index_space()), representation_options(options, lanes)).representation != detail::ValueRepresentation::DEFERRED_MAP) {
                measure(*op->region(0u)->block(0u), axis, repetitions * local_iterations(volume(*op->domain()), lanes), target, cost, indices, limit, lanes, work, options);
            }
        } else if (kind == OperationKind::VIEW_LOAD || kind == OperationKind::VIEW_STORE) {
            auto &space = *op->operand(0u)->type().index_space();
            luisa::optional<double> stride{0.0};
            for (size_t i = 0u; i < space.rank(); i++) {
                auto coefficient = lanes == 1u ? (axis.coherent ? slope(op->operand(i + 1u), axis.value, indices) : luisa::optional<double>{}) :
                                                 luisa::optional<double>{op->domain() && op->domain()->axis(i).extent.constant_value() > 1u ? 1.0 : 0.0};
                if (!coefficient || !stride || !space.axis(i).extent.is_constant()) {
                    stride.reset();
                    break;
                }
                *stride = *stride * static_cast<double>(space.axis(i).extent.constant_value()) + *coefficient * (lanes == 1u ? axis.scale : 1.0);
            }
            auto weight = cost.gathered_lane * target.packet_width;
            if (stride && std::isfinite(*stride)) {
                if (*stride == 0.0 && kind == OperationKind::VIEW_LOAD) { weight = cost.broadcast_load; }
                if (std::abs(*stride) == 1.0) { weight = cost.contiguous_memory; }
            }
            work.memory += repetitions * (op->domain() ? local_iterations(volume(*op->domain()), lanes) : 1.0) * weight;
            if (kind == OperationKind::VIEW_STORE) {
                auto count = op->domain() ? local_iterations(volume(*op->domain()), lanes) : 1.0;
                read_work(op->operand(space.rank() + 1u), repetitions * count,
                          op->domain() && detail::bounded_domain(*op->domain(), limit), target, cost, limit, lanes, work, options, op);
            }
        } else if (kind == OperationKind::MMA) {
            auto &output = *op->result(0u)->type().index_space();
            auto realization = representation_options(options, lanes);
            auto emission = detail::mma_emission_plan(*op, realization);
            auto native = detail::native_mma_plan(*op, realization);
            if (native) {
                emission = {};
                emission.contraction_runtime_loop = true;
                work.mma.native_calls += repetitions;
                if (native->vectorization == compute::xir::StridedMmaVectorization::OUTPUT) {
                    auto inner = native->output_extents.size();
                    while (inner > 1u && native->output_extents[inner - 1u] == 1u) { inner--; }
                    emission.columns = native->output_extents[inner - 1u];
                    emission.output_block = native->vector_width;
                    emission.broadcast_lhs = native->lhs_output_strides[inner - 1u] == 0u;
                    work.mma.native_output_vector_updates += repetitions * static_cast<double>(volume(output) / emission.columns * (emission.columns / native->vector_width)) * static_cast<double>(native->contraction_extent);
                } else {
                    work.mma.native_contraction_vector_updates += repetitions * static_cast<double>(volume(output)) * static_cast<double>(native->contraction_extent / native->vector_width);
                }
            }
            auto contraction = static_cast<double>(volume(detail::mma_contraction_plan(*op, realization).domain));
            auto outputs = volume(output);
            auto groups = emission.output_block == 1u ? outputs :
                                                        outputs / emission.columns * ceil_div(emission.columns, static_cast<uint64_t>(emission.output_block));
            if (native && native->vectorization == compute::xir::StridedMmaVectorization::OUTPUT) {
                // The helper's ragged output tail is scalar, not a partially
                // active vector update with one shared broadcast projection.
                groups = outputs / emission.columns * (emission.columns / native->vector_width + emission.columns % native->vector_width);
            }
            auto updates = repetitions * static_cast<double>(outputs) * contraction;
            auto common_reads = repetitions * static_cast<double>(groups) * contraction;
            auto lhs_reads = emission.broadcast_lhs ? common_reads : updates;
            auto rhs_reads = emission.broadcast_lhs ? updates : common_reads;
            if (emission.output_rows > 1u) {
                auto batches = outputs / emission.columns / emission.row_extent;
                auto row_groups = ceil_div(emission.row_extent, static_cast<uint64_t>(emission.output_rows));
                auto column_groups = ceil_div(emission.columns, static_cast<uint64_t>(emission.output_block));
                groups = batches * row_groups * column_groups;
                auto row_reads = repetitions * static_cast<double>(batches * emission.row_extent * column_groups) * contraction;
                auto column_reads = repetitions * static_cast<double>(batches * emission.columns * row_groups) * contraction;
                lhs_reads = emission.broadcast_lhs ? row_reads : column_reads;
                rhs_reads = emission.broadcast_lhs ? column_reads : row_reads;
            }
            work.mma.multiply_adds += updates;
            work.mma.lhs_reads += lhs_reads;
            work.mma.rhs_reads += rhs_reads;
            work.mma.seed_reads += repetitions * static_cast<double>(outputs);
            if (emission.contraction_runtime_loop && contraction != 0.0) {
                auto invocations = 1.0;
                auto iterations = contraction;
                if (native && native->vectorization == compute::xir::StridedMmaVectorization::CONTRACTION) {
                    auto chunks = native->contraction_extent / native->vector_width;
                    auto tail = native->contraction_extent % native->vector_width;
                    // First vector products initialize partials without an
                    // inserted +0. Only subsequent chunks enter the fold.
                    invocations = static_cast<double>((chunks > 1u) + (tail != 0u));
                    iterations = static_cast<double>((chunks == 0u ? 0u : chunks - 1u) + tail);
                }
                work.mma.loop_invocations += repetitions * static_cast<double>(groups) * invocations;
                work.mma.loop_iterations += repetitions * static_cast<double>(groups) * iterations;
            }
            work.arithmetic += updates * 2.0 * cost.arithmetic;
            auto dynamic_output = detail::bounded_domain(output, limit);
            auto dynamic_projection = [&](uint32_t operand) {
                for (const auto &dimension : op->operand(operand)->type().index_space()->axes()) {
                    if (dimension.extent.constant_value() > 1u &&
                        (output.contains(dimension.dimension) ? dynamic_output : emission.contraction_runtime_loop)) { return true; }
                }
                return false;
            };
            read_work(op->operand(0u), lhs_reads, dynamic_projection(0u), target, cost, limit, lanes, work, options, op);
            read_work(op->operand(1u), rhs_reads, dynamic_projection(1u), target, cost, limit, lanes, work, options, op);
            read_work(op->operand(2u), repetitions * static_cast<double>(outputs), dynamic_output, target, cost, limit, lanes, work, options, op);
        } else if (kind == OperationKind::TILE_EXTRACT) {
            auto dynamic = !detail::expanded_extract(*op, limit, options.max_unrolled_region_work);
            read_work(op->operand(0u), repetitions, dynamic, target, cost, limit, lanes, work, options, op);
            if (dynamic) {
                work.arithmetic += repetitions * (4u + 3u * op->operand(0u)->type().index_space()->rank()) * cost.arithmetic;
            }
        } else if (kind == OperationKind::ELEMENTWISE) {
            auto &type = op->result(0u)->type();
            auto count = type.is_tile() ? local_iterations(volume(*type.index_space()), lanes) : 1.0;
            if (detail::native_mma_snapshot(op->result(0u), representation_options(options, lanes)) ||
                !detail::deferred_expression(op->result(0u), limit, lanes, options.enable_map_fusion)) {
                work.arithmetic += repetitions * count * cost.arithmetic;
                for (size_t i = 0u; i < op->operand_count(); i++) {
                    read_work(op->operand(i), repetitions * count, materialized(op->result(0u), limit, lanes, options.max_unrolled_region_work), target, cost, limit, lanes, work, options, op);
                }
            }
        } else if (kind != OperationKind::CONSTANT && kind != OperationKind::YIELD && kind != OperationKind::STAGE) {
            fail("unsupported operation in XIR execution planning");
        }
        for (size_t i = 0u; i < op->result_count(); i++) { snapshot(op->result(i)); }
    }
}

[[nodiscard]] PlanningResult solve(const Function &function, const ExecutionTargetInfo &info, const PlannerOptions &options) {
    auto target = info.target();
    auto reject = [](luisa::string_view message) { return PlanningResult{.error = luisa::string{message}}; };
    if (!target.packet_width || (target.packet_width & (target.packet_width - 1u)) || !options.max_candidates ||
        !options.reduction_partitions || options.reduction_partitions > 16u) {
        return reject("invalid XIR target or search budget");
    }
    if (!info.supports_task_grain() && (options.blocks_per_task || options.search_task_grain)) {
        return reject("XIR target does not support CPU task-grain constraints");
    }
    if (options.native_mma_vector_width != 0u && !info.supports_native_mma_vector_width(options.native_mma_vector_width)) {
        return reject("XIR target does not support the requested native MMA vector width");
    }
    if (info.supports_task_grain() && (!target.worker_count || !target.task_chunks_per_worker)) {
        return reject("invalid XIR thread-pool scheduling parameters");
    }
    auto &policy = options.cost_policy ? *options.cost_policy : info.cost_policy();
    auto model = policy.coefficients(target, options.cost);
    for (auto coefficient : {model.arithmetic, model.broadcast_load, model.contiguous_memory, model.gathered_lane, model.block_dispatch, model.task_dispatch, model.worker_activation}) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) { return reject("XIR cost coefficients must be finite and nonnegative"); }
    }
    if (!function.parent_module() || !verify(*function.parent_module()) || function.body().block_count() != 1u) { return reject("invalid TileIR before XIR planning"); }
    const Operation *root = nullptr;
    for (auto op : function.body().block(0u)->operations()) {
        if (op->kind() == OperationKind::PARALLEL) {
            if (root) { return reject("XIR planner requires a single root parallel"); }
            root = op;
        } else if (op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE) {
            return reject("XIR planner cannot schedule root effects outside parallel");
        }
    }
    if (!root || !root->domain() || root->domain()->empty() || root->result_count()) { return reject("XIR planner requires a nonempty independent root parallel"); }
    if (auto binding = root->execution_scope_constraint(); binding && *binding != "worker" && *binding != "auto") { return reject("XIR planner cannot satisfy this explicit execution binding"); }
    if (auto error = detail::root_mapping_error(*root->domain(), options.root_axis_order, options.root_axis_tiles); !error.empty()) {
        return reject(error);
    }
    auto count = volume(*root->domain());
    if (!count) { return reject("XIR planner requires a nonempty launch"); }
    if (options.local_lanes != 0u && options.local_lanes != 1u && options.local_lanes != target.packet_width) {
        return reject("XIR local-axis distribution must span exactly one target packet");
    }
    luisa::vector<uint32_t> local_widths{1u};
    auto local_legal = info.supports_local_distribution() && target.packet_width > 1u && count <= UINT32_MAX / target.packet_width && detail::packet_local_program(function, target.packet_width);
    if (options.local_lanes > 1u) {
        if (!local_legal) { return reject("XIR local-axis distribution cannot realize this target/program access/reduction contract"); }
        local_widths = {target.packet_width};
    } else if (options.local_lanes == 0u && local_legal) {
        local_widths.emplace_back(target.packet_width);
    }
    auto rank = root->domain()->rank();
    auto widths = info.block_sizes();
    if (options.block_size) { widths = {options.block_size}; }
    auto valid_width = [&](uint32_t width) {
        return compute::xir::KernelFunction::is_valid_block_size(luisa::make_uint3(width, 1u, 1u)) && width % target.packet_width == 0u;
    };
    if (options.block_size && !valid_width(options.block_size)) { return reject("invalid XIR block width constraint"); }
    // Backend candidates are proposals, not permission to violate XIR/warp
    // invariants. Filter unsupported proposals; never relax a pinned width.
    widths.erase(std::remove_if(widths.begin(), widths.end(), [&](auto width) { return !valid_width(width); }), widths.end());
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    if (widths.empty()) {
        return reject("XIR target provided no legal block widths");
    }
    auto order = options.root_axis_order;
    auto fixed_order = !order.empty();
    if (fixed_order) {
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() != rank) { return reject("XIR axis order must be a complete permutation"); }
        for (size_t i = 0u; i < rank; i++) {
            if (sorted[i] != i) { return reject("XIR axis order must be a complete permutation"); }
        }
    } else {
        order.resize(rank);
        std::iota(order.begin(), order.end(), 0u);
    }
    uint64_t candidates = widths.size() * local_widths.size();
    if (!fixed_order) {
        for (size_t i = 2u; i <= rank; i++) {
            if (candidates > options.max_candidates / i) { return reject("XIR exact search exceeds its candidate budget; constrain the execution order"); }
            candidates *= i;
        }
    }
    if (candidates > options.max_candidates) { return reject("XIR exact search exceeds its candidate budget"); }
    luisa::vector<const Value *> indices;
    auto body = root->region(0u)->block(0u);
    for (size_t i = 0u; i < rank; i++) { indices.emplace_back(body->argument(i)); }
    PlanningResult result;
    // Root traversal, block size and CPU task grain change coordinates and
    // scheduling, not the static snapshot sites within a logical program.
    // Representation options are fixed for this search; only local_lanes
    // changes storage distribution. Analyze each such realization once.
    luisa::vector<std::pair<uint32_t, ResourceAnalysis>> resource_cache;
    auto resources = [&](uint32_t lanes) -> const ResourceAnalysis & {
        for (auto &entry : resource_cache) {
            if (entry.first == lanes) { return entry.second; }
        }
        auto realization = representation_options(options, lanes);
        realization.block_size = widths.front();
        auto analysis = analyze_resources(function, realization);
        resource_cache.emplace_back(lanes, std::move(analysis));
        return resource_cache.back().second;
    };
    uint32_t considered = 0u;
    do {
        auto mapping = detail::root_mapping(*root->domain(), order, options.root_axis_tiles);
        for (auto lanes : local_widths) {
            Work work;
            bool work_ready = false;
            auto physical_count = count * lanes;
            for (auto width : widths) {
                auto blocks = ceil_div(physical_count, static_cast<uint64_t>(width));
                luisa::vector<uint32_t> grains{options.blocks_per_task};
                if (options.search_task_grain && !options.blocks_per_task) {
                    grains = {static_cast<uint32_t>(ceil_div(blocks, static_cast<uint64_t>(target.worker_count) * target.task_chunks_per_worker)),
                              static_cast<uint32_t>(blocks)};
                    for (auto grain = uint64_t{1u}; grain < blocks; grain *= 2u) { grains.emplace_back(static_cast<uint32_t>(grain)); }
                    std::sort(grains.begin(), grains.end());
                    grains.erase(std::unique(grains.begin(), grains.end()), grains.end());
                }
                for (auto grain : grains) {
                    if (considered++ >= options.max_candidates) {
                        result.error = "XIR exact search exceeds its candidate budget";
                        return result;
                    }
                    ExecutionPlan candidate{width, order, static_cast<uint32_t>(physical_count), {}, lanes, grain, options.root_axis_tiles};
                    candidate.mma_output_block = options.mma_output_block;
                    candidate.enable_mma_2d_blocking = options.enable_mma_2d_blocking;
                    candidate.max_unrolled_mma_terms = options.max_unrolled_mma_terms;
                    candidate.native_mma_vector_width = options.native_mma_vector_width;
                    if (!info.accepts(candidate)) {
                        result.rejected.emplace_back(ExecutionRejection{std::move(candidate), "XIR target rejected execution geometry"});
                        continue;
                    }
                    const auto &analysis = resources(lanes);
                    if (!analysis) {
                        result.rejected.emplace_back(ExecutionRejection{std::move(candidate), analysis.error});
                        continue;
                    }
                    candidate.resources = analysis.resources;
                    candidate.resource_limits = info.resource_limits(candidate);
                    if (candidate.resources.snapshot_bytes_per_worker > candidate.resource_limits.max_snapshot_bytes_per_worker) {
                        auto reason = luisa::format("XIR candidate local_lanes={} requires {} static snapshot bytes per worker; backend budget is {}",
                                                    lanes, candidate.resources.snapshot_bytes_per_worker, candidate.resource_limits.max_snapshot_bytes_per_worker);
                        result.rejected.emplace_back(ExecutionRejection{std::move(candidate), std::move(reason)});
                        continue;
                    }
                    if (!work_ready) {
                        // Extract dynamic work only for resource-admissible
                        // representations, then reuse it across block/task
                        // geometries with the same root order and local lanes.
                        auto spatial = SpatialAxis{indices[order.back()]};
                        auto extent = root->domain()->axis(order.back()).extent.constant_value();
                        if (!mapping.identity && lanes == 1u) {
                            auto digit = mapping.digits.back();
                            spatial = {indices[digit.axis], static_cast<double>(digit.scale), digit.extent % target.packet_width == 0u};
                        }
                        measure(*body, spatial, 1.0, target, model, indices, options.max_unrolled_tile_elements, lanes, work, options);
                        // Identity preserves the historical estimate. A blocked
                        // traversal uses the fastest digit or the gather prior
                        // when packets span digits; neither is a codegen proof.
                        if (mapping.identity && lanes == 1u && extent % target.packet_width != 0u) { work.memory *= 2.0; }
                        // Charge root decoding once, not per nested K/fold.
                        // Temporal cache reuse is deliberately still unmodeled.
                        work.arithmetic += mapping.decode_arithmetic * model.arithmetic;
                        if (!std::isfinite(work.arithmetic) || !std::isfinite(work.memory)) {
                            result.error = "XIR work estimate overflow";
                            return result;
                        }
                        work_ready = true;
                    }
                    ExecutionWork execution_work{work.arithmetic, work.memory,
                                                 ceil_div(physical_count, static_cast<uint64_t>(target.packet_width)), blocks};
                    execution_work.mma_per_packet = work.mma;
                    auto cost = policy.evaluate(target, candidate, info.schedule(candidate, execution_work), model);
                    for (auto component : {cost.arithmetic_work, cost.memory_work, cost.dispatch_work, cost.imbalance_work,
                                           cost.score, cost.task_dispatch_work, cost.activation_work}) {
                        if (!std::isfinite(component) || component < 0.0) { return reject("XIR cost policy returned a nonfinite or negative cost"); }
                    }
                    candidate.cost = cost;
                    result.candidates.emplace_back(std::move(candidate));
                }
            }
        }
    } while (!fixed_order && std::next_permutation(order.begin(), order.end()));
    if (result.candidates.empty()) {
        result.error = "XIR target rejected every execution candidate";
        if (!result.rejected.empty()) { result.error.append(": ").append(result.rejected.front().reason); }
        return result;
    }
    result.selected = *std::min_element(result.candidates.begin(), result.candidates.end(), [](auto &a, auto &b) { return a.cost.score < b.cost.score; });
    return result;
}

}// namespace

ExecutionCost AnalyticExecutionCostPolicy::evaluate(
    ExecutionTarget, const ExecutionPlan &, const ExecutionWork &work, const ExecutionCostModel &model) const noexcept {
    auto average_packets = static_cast<double>(work.packet_count) / work.active_workers;
    ExecutionCost cost;
    cost.arithmetic_work = work.arithmetic_per_packet * average_packets;
    cost.memory_work = work.memory_per_packet * average_packets;
    cost.dispatch_work = model.block_dispatch * work.critical_blocks;
    cost.imbalance_work = std::max(0.0, static_cast<double>(work.critical_packets) - average_packets) * (work.arithmetic_per_packet + work.memory_per_packet);
    cost.task_dispatch_work = model.task_dispatch * work.critical_tasks;
    cost.activation_work = work.active_workers > 1u ? model.worker_activation : 0.0;
    cost.score = cost.arithmetic_work + cost.memory_work + cost.dispatch_work + cost.imbalance_work + cost.task_dispatch_work + cost.activation_work;
    return cost;
}

PlanningResult plan(const Function &function, ExecutionTarget target, const PlannerOptions &options) noexcept {
    return solve(function, ThreadPoolExecutionTargetInfo{target}, options);
}

PlanningResult plan_with_target_info(const Function &function, const ExecutionTargetInfo &info, const PlannerOptions &options) noexcept {
    return solve(function, info, options);
}

luisa::vector<uint32_t> ThreadPoolExecutionTargetInfo::block_sizes() const noexcept {
    return {32u, 64u, 128u, 256u, 512u, 1024u};
}

bool ThreadPoolExecutionTargetInfo::accepts(const ExecutionPlan &) const noexcept {
    return _target.worker_count != 0u && _target.task_chunks_per_worker != 0u;
}

ExecutionWork ThreadPoolExecutionTargetInfo::schedule(const ExecutionPlan &candidate, ExecutionWork work) const noexcept {
    return distribute_thread_pool_work(work, candidate, _target);
}

const ExecutionCostPolicy &ThreadPoolExecutionTargetInfo::cost_policy() const noexcept {
    static const AnalyticExecutionCostPolicy policy;
    return policy;
}

}// namespace luisa::compute::tile::bridge::xir
