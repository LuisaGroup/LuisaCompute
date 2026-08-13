#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] std::optional<ScheduleEmitter::PredicatedMemoryDiamond>
ScheduleEmitter::_find_predicated_memory_diamond(
    const schedule::BasicBlock &block) const noexcept {
    if (_width == 1u || luisa::compute::detail::env_flag(
                            "LUISA_SIMD_DISABLE_PREDICATED_IF")) {
        return std::nullopt;
    }
    auto *split = std::get_if<schedule::SplitTerminator>(
        &block.terminator);
    if (split == nullptr || !split->convergence) {
        return std::nullopt;
    }
    auto *condition = _source.value(split->condition);
    if (condition == nullptr ||
        condition->value_class != schedule::ValueClass::varying ||
        !split->true_edge.assignments.empty() ||
        !split->false_edge.assignments.empty() ||
        !split->true_edge.joins.empty() ||
        !split->false_edge.joins.empty() ||
        split->true_edge.loop_back || split->false_edge.loop_back ||
        split->true_edge.target == split->false_edge.target) {
        return std::nullopt;
    }
    auto *true_block = _source.block(split->true_edge.target);
    auto *false_block = _source.block(split->false_edge.target);
    if (true_block == nullptr || false_block == nullptr) {
        return std::nullopt;
    }
    auto *true_branch = std::get_if<schedule::BranchTerminator>(
        &true_block->terminator);
    auto *false_branch = std::get_if<schedule::BranchTerminator>(
        &false_block->terminator);
    if (true_branch == nullptr || false_branch == nullptr ||
        true_branch->edge.target != false_branch->edge.target ||
        true_branch->edge.loop_back || false_branch->edge.loop_back ||
        true_branch->edge.joins.size() != 1u ||
        false_branch->edge.joins.size() != 1u ||
        true_branch->edge.joins.front() != *split->convergence ||
        false_branch->edge.joins.front() != *split->convergence) {
        return std::nullopt;
    }
    auto *point = _source.convergence(*split->convergence);
    auto merge = true_branch->edge.target;
    if (point == nullptr || point->target != merge ||
        merge == block.id || merge == true_block->id ||
        merge == false_block->id) {
        return std::nullopt;
    }
    auto assignments_are_lane_masked = [&](const auto &assignments) noexcept {
        for (auto assignment : assignments) {
            auto *destination = _source.value(assignment.destination);
            if (destination == nullptr ||
                (destination->value_class !=
                     schedule::ValueClass::varying &&
                 destination->value_class != schedule::ValueClass::mask)) {
                return false;
            }
        }
        return true;
    };
    if (!assignments_are_lane_masked(true_branch->edge.assignments) ||
        !assignments_are_lane_masked(false_branch->edge.assignments)) {
        return std::nullopt;
    }

    auto predecessor_count = [&](schedule::BlockId target) noexcept {
        auto count = size_t{0u};
        auto add_edge = [&](const schedule::ControlEdge &edge) noexcept {
            count += edge.target == target;
        };
        for (auto &&candidate : _source.blocks()) {
            std::visit(
                [&](const auto &control) noexcept {
                    using T = std::decay_t<decltype(control)>;
                    if constexpr (std::is_same_v<
                                      T, schedule::BranchTerminator>) {
                        add_edge(control.edge);
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::SplitTerminator>) {
                        add_edge(control.true_edge);
                        add_edge(control.false_edge);
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::SwitchTerminator>) {
                        for (auto &&item : control.cases) {
                            add_edge(item.edge);
                        }
                        add_edge(control.default_edge);
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::JoinTerminator>) {
                        auto *join = _source.convergence(
                            control.convergence);
                        count += join != nullptr &&
                                 join->target == target;
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::LoopBackTerminator>) {
                        auto *loop = _source.loop(control.loop);
                        count += loop != nullptr &&
                                 loop->header == target;
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::BlockBarrierTerminator>) {
                        add_edge(control.resume_edge);
                    }
                },
                candidate.terminator);
        }
        return count;
    };
    if (predecessor_count(true_block->id) != 1u ||
        predecessor_count(false_block->id) != 1u) {
        return std::nullopt;
    }

    auto safe_arithmetic = [](xir::ArithmeticOp op) noexcept {
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
            case xir::ArithmeticOp::UNARY_BIT_NOT:
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::BINARY_BIT_AND:
            case xir::ArithmeticOp::BINARY_BIT_OR:
            case xir::ArithmeticOp::BINARY_BIT_XOR:
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            case xir::ArithmeticOp::ALL:
            case xir::ArithmeticOp::ANY:
            case xir::ArithmeticOp::SELECT:
            case xir::ArithmeticOp::ABS:
            case xir::ArithmeticOp::MIN:
            case xir::ArithmeticOp::MAX:
            case xir::ArithmeticOp::ISINF:
            case xir::ArithmeticOp::ISNAN:
            case xir::ArithmeticOp::COPYSIGN: return true;
            default: return false;
        }
    };
    auto safe_instruction = [&](const schedule::Instruction &instruction,
                                bool &has_memory) noexcept {
        if (instruction.opcode == schedule::Opcode::resource_read) {
            if (!instruction.source_op ||
                *instruction.source_op != static_cast<uint32_t>(
                                              xir::ResourceReadOp::BUFFER_READ) ||
                instruction.operands.size() != 2u ||
                instruction.cohort_uniform_operand_index) {
                return false;
            }
            auto *index = _source.value(instruction.operands[1u]);
            auto *result = instruction.result ?
                               _source.value(*instruction.result) :
                               nullptr;
            if (index == nullptr || result == nullptr ||
                index->value_class != schedule::ValueClass::varying ||
                result->value_class != schedule::ValueClass::varying) {
                return false;
            }
            has_memory = true;
            return true;
        }
        if (instruction.opcode == schedule::Opcode::arithmetic &&
            instruction.source_op) {
            return safe_arithmetic(static_cast<xir::ArithmeticOp>(
                *instruction.source_op));
        }
        if (instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op && instruction.result &&
            instruction.operands.size() == 1u) {
            auto op = static_cast<xir::CastOp>(*instruction.source_op);
            if (op == xir::CastOp::BITWISE_CAST) { return true; }
            if (op != xir::CastOp::STATIC_CAST) { return false; }
            auto *source = _source.value(instruction.operands.front());
            auto *target = _source.value(*instruction.result);
            if (source == nullptr || target == nullptr ||
                source->type == nullptr || target->type == nullptr) {
                return false;
            }
            auto source_is_float =
                source->type->is_float_or_float_vector();
            auto target_is_integer =
                target->type->is_int_or_int_vector() ||
                target->type->is_uint_or_uint_vector();
            return !(source_is_float && target_is_integer);
        }
        return false;
    };
    static constexpr auto max_instruction_count = size_t{8u};
    auto instruction_count = true_block->instructions.size() +
                             false_block->instructions.size();
    if (instruction_count > max_instruction_count) {
        return std::nullopt;
    }
    auto has_memory = false;
    for (auto *arm : {true_block, false_block}) {
        for (auto &&instruction : arm->instructions) {
            if (!safe_instruction(instruction, has_memory)) {
                return std::nullopt;
            }
        }
    }
    if (!has_memory) { return std::nullopt; }
    return PredicatedMemoryDiamond{
        .true_block = true_block,
        .false_block = false_block,
        .merge = merge,
        .instruction_count = instruction_count,
    };
}

void ScheduleEmitter::_emit_predicated_memory_diamond(
    const schedule::BasicBlock &block,
    const schedule::SplitTerminator &control,
    const PredicatedMemoryDiamond &diamond,
    const std::vector<::llvm::BasicBlock *> *direct_blocks) {
    static_cast<void>(block);
    auto *condition = _load_value(control.condition);
    if (condition == nullptr) { return; }
    auto *outer_mask = _active_mask;
    auto *outer_seed = _seed_lane;
    auto *true_mask = _builder.CreateAnd(outer_mask, condition);
    auto *false_mask = _builder.CreateAnd(
        outer_mask, _builder.CreateNot(condition));
    auto emit_arm = [&](const schedule::BasicBlock &arm,
                        ::llvm::Value *mask) noexcept {
        _active_mask = mask;
        _seed_lane = _safe_first_lane(mask);
        for (auto &&instruction : arm.instructions) {
            auto *lane_affine_seed = static_cast<::llvm::Value *>(nullptr);
            if (instruction.lane_consecutive_operand_index) {
                auto operand_index =
                    *instruction.lane_consecutive_operand_index;
                if (operand_index < instruction.operands.size()) {
                    auto *index = _source.value(
                        instruction.operands[operand_index]);
                    if (index != nullptr &&
                        (!index->defining_block ||
                         *index->defining_block != arm.id)) {
                        // The lane-consecutive proof describes the full
                        // physical packet. If the index was already produced
                        // under the outer mask, extract it at the outer seed
                        // instead of dynamically extracting the submask's
                        // first lane. This keeps the affine base scalar and
                        // avoids a vector spill/reload solely for horizontal
                        // extraction. Arm-local indices still use the submask
                        // seed because inactive operands may be poison.
                        lane_affine_seed = outer_seed;
                    }
                }
            }
            _emit_instruction(
                instruction, lane_affine_seed, mask);
            if (_failed()) { return; }
        }
        auto *branch = std::get_if<schedule::BranchTerminator>(
            &arm.terminator);
        _apply_assignments(branch->edge.assignments, mask);
    };
    emit_arm(*diamond.true_block, true_mask);
    if (!_failed()) { emit_arm(*diamond.false_block, false_mask); }
    _active_mask = outer_mask;
    _seed_lane = outer_seed;
    if (_failed()) { return; }
    _result.predicated_memory_diamond_count++;
    _result.predicated_memory_instruction_count +=
        diamond.instruction_count;
    if (direct_blocks != nullptr) {
        _builder.CreateBr((*direct_blocks)[diamond.merge.value]);
    } else {
        _continue_at(diamond.merge, outer_mask);
    }
}

}// namespace luisa::compute::simd::detail
