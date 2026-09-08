#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <luisa/core/mathematics.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/verifier.h>
#include "representation.h"

namespace luisa::compute::tile::bridge::xir {
namespace {

[[noreturn]] void fail(const char *message) { throw std::invalid_argument{message}; }

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
};

[[nodiscard]] double local_iterations(uint64_t count, uint32_t lanes) {
    return static_cast<double>(count >= lanes ? ceil_div(count, static_cast<uint64_t>(lanes)) : count);
}
[[nodiscard]] bool materialized(const Value *value, uint32_t limit, uint32_t lanes) {
    return detail::bounded_tile(value, limit) || (lanes > 1u && detail::bounded_tile(value, lanes - 1u));
}

void read_work(const Value *value, double repetitions, bool dynamic,
               ExecutionTarget target, const ExecutionCostModel &cost, uint32_t limit, uint32_t lanes, Work &work) {
    if (!value->type().is_tile()) { return; }
    if (materialized(value, limit, lanes)) {
        auto op = value->defining_operation();
        if (op && op->kind() == OperationKind::CONSTANT) { return; }
        if (detail::deferred_elementwise(value, limit, lanes)) {
            work.arithmetic += repetitions * cost.arithmetic;
            for (size_t i = 0u; i < op->operand_count(); i++) { read_work(op->operand(i), repetitions, dynamic, target, cost, limit, lanes, work); }
            return;
        }
        work.memory += repetitions * cost.gathered_lane * target.packet_width;
    } else if (dynamic) {
        auto count = volume(*value->type().index_space());
        if (count > 1u && detail::needs_indexable_snapshot(value, limit)) {
            work.memory += repetitions * cost.gathered_lane * target.packet_width;
        } else {
            work.arithmetic += repetitions * count * 2.0 * cost.arithmetic;
        }
    }
}

void measure(const Block &block, const Value *axis, double repetitions,
             ExecutionTarget target, const ExecutionCostModel &cost,
             luisa::vector<const Value *> indices, uint32_t limit, uint32_t lanes, Work &work) {
    auto snapshot = [&](const Value *value) {
        if (materialized(value, limit, lanes)) {
            auto op = value->defining_operation();
            // Large carries are parallel copies, charged at their loop below.
            if (!op || op->kind() == OperationKind::CONSTANT || op->kind() == OperationKind::SERIAL ||
                op->kind() == OperationKind::PIPELINE || op->kind() == OperationKind::REDUCE ||
                detail::deferred_elementwise(value, limit, lanes)) { return; }
            work.memory += repetitions * local_iterations(volume(*value->type().index_space()), lanes) * cost.gathered_lane * target.packet_width;
        } else if (detail::needs_indexable_snapshot(value, limit)) {
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
                    read_work(op->operand(i), repetitions * count, true, target, cost, limit, lanes, work);
                    // Initialization plus staging the next value, loading it,
                    // and updating current only after all staged copies exist.
                    work.memory += (repetitions + 3.0 * iterations) * count * cost.gathered_lane * target.packet_width;
                    for (auto term : body->operations()) {
                        if (term->kind() == OperationKind::YIELD) { read_work(term->operand(i), iterations * count, true, target, cost, limit, lanes, work); }
                    }
                }
            }
            measure(*body, axis, iterations, target, cost, std::move(child_indices), limit, lanes, work);
            if (lanes > 1u && kind == OperationKind::REDUCE && volume(*op->domain()) >= lanes) {
                // Local partials converge through a fixed tree and one root
                // broadcast. This is a relative prior, not measured cycles.
                work.arithmetic += repetitions * (2.0 * std::log2(lanes) + 2.0) * cost.arithmetic;
            }
        } else if (kind == OperationKind::TILE_MAP) {
            measure(*op->region(0u)->block(0u), axis, repetitions * local_iterations(volume(*op->domain()), lanes), target, cost, indices, limit, lanes, work);
        } else if (kind == OperationKind::VIEW_LOAD || kind == OperationKind::VIEW_STORE) {
            auto &space = *op->operand(0u)->type().index_space();
            luisa::optional<double> stride{0.0};
            for (size_t i = 0u; i < space.rank(); i++) {
                auto coefficient = lanes == 1u ? slope(op->operand(i + 1u), axis, indices) :
                                                 luisa::optional<double>{op->domain() && op->domain()->axis(i).extent.constant_value() > 1u ? 1.0 : 0.0};
                if (!coefficient || !stride || !space.axis(i).extent.is_constant()) {
                    stride.reset();
                    break;
                }
                *stride = *stride * static_cast<double>(space.axis(i).extent.constant_value()) + *coefficient;
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
                          op->domain() && detail::bounded_domain(*op->domain(), limit), target, cost, limit, lanes, work);
            }
        } else if (kind == OperationKind::MMA) {
            auto &output = *op->result(0u)->type().index_space();
            double contraction = 1.0;
            for (auto &dimension : op->operand(0u)->type().index_space()->axes()) {
                if (!output.contains(dimension.dimension)) { contraction *= dimension.extent.constant_value(); }
            }
            work.arithmetic += repetitions * volume(output) * contraction * 2.0 * cost.arithmetic;
            auto dynamic = detail::bounded_domain(output, limit) || (limit && contraction > limit);
            read_work(op->operand(0u), repetitions * volume(output) * contraction, dynamic, target, cost, limit, lanes, work);
            read_work(op->operand(1u), repetitions * volume(output) * contraction, dynamic, target, cost, limit, lanes, work);
            read_work(op->operand(2u), repetitions * volume(output), detail::bounded_domain(output, limit), target, cost, limit, lanes, work);
        } else if (kind == OperationKind::TILE_EXTRACT) {
            auto dynamic = !detail::expanded_extract(*op, limit);
            read_work(op->operand(0u), repetitions, dynamic, target, cost, limit, lanes, work);
            if (dynamic) {
                work.arithmetic += repetitions * (4u + 3u * op->operand(0u)->type().index_space()->rank()) * cost.arithmetic;
            }
        } else if (kind == OperationKind::ELEMENTWISE) {
            auto &type = op->result(0u)->type();
            auto count = type.is_tile() ? local_iterations(volume(*type.index_space()), lanes) : 1.0;
            if (!detail::deferred_elementwise(op->result(0u), limit, lanes)) {
                work.arithmetic += repetitions * count * cost.arithmetic;
                for (size_t i = 0u; i < op->operand_count(); i++) {
                    read_work(op->operand(i), repetitions * count, materialized(op->result(0u), limit, lanes), target, cost, limit, lanes, work);
                }
            }
        } else if (kind != OperationKind::CONSTANT && kind != OperationKind::YIELD && kind != OperationKind::STAGE) {
            fail("unsupported operation in XIR execution planning");
        }
        for (size_t i = 0u; i < op->result_count(); i++) { snapshot(op->result(i)); }
    }
}

[[nodiscard]] PlanningResult solve(const Function &function, ExecutionTarget target, const PlannerOptions &options) {
    if (!target.packet_width || target.packet_width > 16u || (target.packet_width & (target.packet_width - 1u)) || !target.worker_count || !options.max_candidates ||
        !options.reduction_partitions || options.reduction_partitions > 16u) {
        fail("invalid XIR target or search budget");
    }
    for (auto coefficient : {options.cost.arithmetic, options.cost.broadcast_load, options.cost.contiguous_memory, options.cost.gathered_lane, options.cost.block_dispatch}) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) { fail("XIR cost coefficients must be finite and nonnegative"); }
    }
    if (!function.parent_module() || !verify(*function.parent_module()) || function.body().block_count() != 1u) { fail("invalid TileIR before XIR planning"); }
    const Operation *root = nullptr;
    for (auto op : function.body().block(0u)->operations()) {
        if (op->kind() == OperationKind::PARALLEL) {
            if (root) { fail("XIR planner requires a single root parallel"); }
            root = op;
        } else if (op->kind() != OperationKind::CONSTANT && op->kind() != OperationKind::ELEMENTWISE) {
            fail("XIR planner cannot schedule root effects outside parallel");
        }
    }
    if (!root || !root->domain() || root->domain()->empty() || root->result_count()) { fail("XIR planner requires a nonempty independent root parallel"); }
    if (auto binding = root->execution_scope_constraint(); binding && *binding != "worker" && *binding != "auto") { fail("XIR planner cannot satisfy this explicit execution binding"); }
    auto count = volume(*root->domain());
    if (!count) { fail("XIR planner requires a nonempty launch"); }
    if (options.local_lanes != 0u && options.local_lanes != 1u && options.local_lanes != target.packet_width) {
        fail("XIR local-axis distribution must span exactly one target packet");
    }
    luisa::vector<uint32_t> local_widths{1u};
    auto local_legal = target.packet_width > 1u && count <= UINT32_MAX / target.packet_width && detail::packet_local_program(function, target.packet_width);
    if (options.local_lanes > 1u) {
        if (!local_legal) { fail("XIR local-axis distribution cannot realize this program's access/reduction contract"); }
        local_widths = {target.packet_width};
    } else if (options.local_lanes == 0u && local_legal) {
        local_widths.emplace_back(target.packet_width);
    }
    auto rank = root->domain()->rank();
    luisa::vector<uint32_t> widths{32u, 64u, 128u, 256u, 512u, 1024u};
    if (options.block_size) { widths = {options.block_size}; }
    for (auto width : widths) {
        if (!compute::xir::KernelFunction::is_valid_block_size(luisa::make_uint3(width, 1u, 1u)) || width % target.packet_width) { fail("invalid XIR block width constraint"); }
    }
    auto order = options.root_axis_order;
    auto fixed_order = !order.empty();
    if (fixed_order) {
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() != rank) { fail("XIR axis order must be a complete permutation"); }
        for (size_t i = 0u; i < rank; i++) {
            if (sorted[i] != i) { fail("XIR axis order must be a complete permutation"); }
        }
    } else {
        order.resize(rank);
        std::iota(order.begin(), order.end(), 0u);
    }
    uint64_t candidates = widths.size() * local_widths.size();
    if (!fixed_order) {
        for (size_t i = 2u; i <= rank; i++) {
            if (candidates > options.max_candidates / i) { fail("XIR exact search exceeds its candidate budget; constrain the execution order"); }
            candidates *= i;
        }
    }
    if (candidates > options.max_candidates) { fail("XIR exact search exceeds its candidate budget"); }
    luisa::vector<const Value *> indices;
    auto body = root->region(0u)->block(0u);
    for (size_t i = 0u; i < rank; i++) { indices.emplace_back(body->argument(i)); }
    PlanningResult result;
    do {
        for (auto lanes : local_widths) {
            Work work;
            measure(*body, indices[order.back()], 1.0, target, options.cost, indices, options.max_unrolled_tile_elements, lanes, work);
            // If a packet crosses the chosen innermost axis, its memory estimate
            // is conservatively penalized. No lane-coherence fact reaches codegen.
            auto extent = root->domain()->axis(order.back()).extent.constant_value();
            if (lanes == 1u && extent % target.packet_width != 0u) { work.memory *= 2.0; }
            auto physical_count = count * lanes;
            for (auto width : widths) {
                auto blocks = ceil_div(physical_count, static_cast<uint64_t>(width));
                auto workers = std::min<uint64_t>(blocks, target.worker_count);
                auto packets = ceil_div(physical_count, static_cast<uint64_t>(target.packet_width));
                auto waves = ceil_div(blocks, workers);
                ExecutionCost cost;
                cost.arithmetic_work = work.arithmetic * packets / workers;
                cost.memory_work = work.memory * packets / workers;
                cost.dispatch_work = options.cost.block_dispatch * waves;
                auto issued = static_cast<double>(waves * ceil_div(std::min<uint64_t>(physical_count, width), static_cast<uint64_t>(target.packet_width)));
                cost.imbalance_work = std::max(0.0, issued - static_cast<double>(packets) / workers) * (work.arithmetic + work.memory);
                cost.score = cost.arithmetic_work + cost.memory_work + cost.dispatch_work + cost.imbalance_work;
                if (!std::isfinite(cost.score)) { fail("XIR cost estimate overflow"); }
                result.candidates.emplace_back(ExecutionPlan{width, order, static_cast<uint32_t>(physical_count), cost, lanes});
            }
        }
    } while (!fixed_order && std::next_permutation(order.begin(), order.end()));
    result.selected = *std::min_element(result.candidates.begin(), result.candidates.end(), [](auto &a, auto &b) { return a.cost.score < b.cost.score; });
    return result;
}

}// namespace

PlanningResult plan(const Function &function, ExecutionTarget target, const PlannerOptions &options) noexcept {
    try {
        return solve(function, target, options);
    } catch (const std::exception &error) {
        PlanningResult result;
        result.error = error.what();
        return result;
    } catch (...) {
        PlanningResult result;
        result.error = "unknown XIR planning failure";
        return result;
    }
}

}// namespace luisa::compute::tile::bridge::xir
