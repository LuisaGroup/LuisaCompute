#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

#include <algorithm>

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

[[nodiscard]] const schedule::Loop *
ScheduleEmitter::_innermost_loop_containing(
    schedule::BlockId block) const noexcept {
    auto *innermost = static_cast<const schedule::Loop *>(nullptr);
    auto innermost_depth = size_t{0u};
    for (auto &&loop : _source.loops()) {
        if (std::find(loop.blocks.cbegin(), loop.blocks.cend(), block) ==
            loop.blocks.cend()) {
            continue;
        }
        auto depth = size_t{1u};
        auto parent = loop.parent;
        while (parent) {
            auto *parent_loop = _source.loop(*parent);
            if (parent_loop == nullptr) { break; }
            depth++;
            parent = parent_loop->parent;
        }
        if (innermost == nullptr || depth > innermost_depth) {
            innermost = &loop;
            innermost_depth = depth;
        }
    }
    return innermost;
}

[[nodiscard]] std::optional<ScheduleEmitter::GuardedPredicatedMathDiamond>
ScheduleEmitter::_find_guarded_predicated_math_diamond(
    const schedule::BasicBlock &block,
    bool allow_tiny_speculation) const noexcept {
    if (_width == 1u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS")) {
        return std::nullopt;
    }
    auto *innermost_loop = _innermost_loop_containing(block.id);
    if (innermost_loop == nullptr) { return std::nullopt; }
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
    auto *point = _source.convergence(*split->convergence);
    if (point == nullptr || point->target == block.id) {
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

    static constexpr auto max_chain_block_count = size_t{3u};
    auto collect_chain = [&](const schedule::ControlEdge &entry)
        -> std::optional<std::vector<const schedule::BasicBlock *>> {
        std::vector<const schedule::BasicBlock *> chain;
        auto target = entry.target;
        while (target != point->target &&
               chain.size() < max_chain_block_count) {
            auto *arm = _source.block(target);
            if (arm == nullptr || arm->id == block.id ||
                predecessor_count(target) != 1u) {
                return std::nullopt;
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &arm->terminator);
            if (branch == nullptr || branch->edge.loop_back) {
                return std::nullopt;
            }
            chain.emplace_back(arm);
            auto reaches_merge = branch->edge.target == point->target;
            if (reaches_merge) {
                if (branch->edge.joins.size() != 1u ||
                    branch->edge.joins.front() !=
                        *split->convergence) {
                    return std::nullopt;
                }
            } else if (!branch->edge.joins.empty()) {
                return std::nullopt;
            }
            target = branch->edge.target;
        }
        return target == point->target && !chain.empty() ?
                   std::optional{std::move(chain)} :
                   std::nullopt;
    };
    auto true_blocks = collect_chain(split->true_edge);
    auto false_blocks = collect_chain(split->false_edge);
    if (!true_blocks || !false_blocks) { return std::nullopt; }
    auto belongs_to_innermost_loop = [&](const auto *candidate) noexcept {
        return candidate != nullptr &&
               std::find(
                   innermost_loop->blocks.cbegin(),
                   innermost_loop->blocks.cend(), candidate->id) !=
                   innermost_loop->blocks.cend();
    };
    if (!belongs_to_innermost_loop(_source.block(point->target)) ||
        !std::all_of(
            true_blocks->cbegin(), true_blocks->cend(),
            belongs_to_innermost_loop) ||
        !std::all_of(
            false_blocks->cbegin(), false_blocks->cend(),
            belongs_to_innermost_loop)) {
        return std::nullopt;
    }

    auto assignments_are_lane_masked = [&](const auto &assignments) noexcept {
        for (auto assignment : assignments) {
            auto *destination = _source.value(assignment.destination);
            if (destination == nullptr ||
                (destination->value_class !=
                     schedule::ValueClass::varying &&
                 destination->value_class !=
                     schedule::ValueClass::mask)) {
                return false;
            }
        }
        return true;
    };
    auto true_has_instructions = std::any_of(
        true_blocks->cbegin(), true_blocks->cend(),
        [](const auto *arm) noexcept {
            return !arm->instructions.empty();
        });
    auto false_has_instructions = std::any_of(
        false_blocks->cbegin(), false_blocks->cend(),
        [](const auto *arm) noexcept {
            return !arm->instructions.empty();
        });
    auto potentially_two_sided =
        true_has_instructions && false_has_instructions;
    auto safe_instruction = [&](const schedule::Instruction &instruction,
                                bool &has_expensive_math) noexcept {
        if (potentially_two_sided &&
            instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op && instruction.result &&
            instruction.operands.size() == 1u &&
            !instruction.participant_mask) {
            auto op = static_cast<xir::CastOp>(
                *instruction.source_op);
            return op == xir::CastOp::STATIC_CAST ||
                   op == xir::CastOp::BITWISE_CAST;
        }
        if (instruction.opcode != schedule::Opcode::arithmetic ||
            !instruction.source_op || !instruction.result ||
            instruction.participant_mask) {
            return false;
        }
        auto op = static_cast<xir::ArithmeticOp>(
            *instruction.source_op);
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::BINARY_BIT_AND:
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            case xir::ArithmeticOp::SELECT:
            case xir::ArithmeticOp::MIN:
            case xir::ArithmeticOp::MAX:
            case xir::ArithmeticOp::AGGREGATE: return true;
            case xir::ArithmeticOp::SQRT:
            case xir::ArithmeticOp::RSQRT:
                has_expensive_math = true;
                return true;
            case xir::ArithmeticOp::BINARY_DIV: {
                auto *result = _source.value(*instruction.result);
                auto safe = result != nullptr && result->type != nullptr &&
                            result->type->is_float_or_float_vector();
                has_expensive_math = has_expensive_math || safe;
                return safe;
            }
            default: return false;
        }
    };
    auto true_instruction_count = size_t{0u};
    auto false_instruction_count = size_t{0u};
    auto has_expensive_math = false;
    auto validate_chain = [&](const auto &chain,
                              size_t &instruction_count) noexcept {
        for (auto *arm : chain) {
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &arm->terminator);
            if (branch == nullptr ||
                !assignments_are_lane_masked(
                    branch->edge.assignments)) {
                return false;
            }
            for (auto &&instruction : arm->instructions) {
                if (!safe_instruction(
                        instruction, has_expensive_math)) {
                    return false;
                }
                instruction_count++;
            }
        }
        return true;
    };
    if (!validate_chain(*true_blocks, true_instruction_count) ||
        !validate_chain(*false_blocks, false_instruction_count)) {
        return std::nullopt;
    }
    auto instruction_count =
        true_instruction_count + false_instruction_count;
    auto two_sided =
        std::min(true_instruction_count, false_instruction_count) != 0u;
    auto assignment_only = instruction_count == 0u;
    auto guarded_math = !two_sided && has_expensive_math &&
                        instruction_count >= 4u &&
                        instruction_count <= 24u;
    auto tiny_speculation_safe =
        allow_tiny_speculation &&
        !two_sided && !has_expensive_math &&
        instruction_count != 0u && instruction_count <= 3u;
    auto force_two_sided = luisa::compute::detail::env_flag(
        "LUISA_SIMD_FORCE_TWO_SIDED_LOCAL_PREDICATION");
    auto enable_two_sided =
        (force_two_sided || _width == 2u || _width == 4u ||
         _width == 8u) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_TWO_SIDED_LOCAL_PREDICATION");
    auto bounded_two_sided = two_sided && enable_two_sided &&
                             instruction_count >= 4u &&
                             instruction_count <= 24u;
    if (!assignment_only && !guarded_math && !bounded_two_sided &&
        !tiny_speculation_safe) {
        return std::nullopt;
    }
    return GuardedPredicatedMathDiamond{
        .true_blocks = std::move(*true_blocks),
        .false_blocks = std::move(*false_blocks),
        .merge = point->target,
        .two_sided = two_sided,
        .instruction_count = instruction_count,
    };
}

[[nodiscard]] std::optional<ScheduleEmitter::NestedPredicatedRegion>
ScheduleEmitter::_find_nested_predicated_region(
    const schedule::BasicBlock &block,
    bool allow_tiny_speculation) const noexcept {
    if (_width == 1u ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS") ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_NESTED_PREDICATED_REGION")) {
        return std::nullopt;
    }
    auto *innermost_loop = _innermost_loop_containing(block.id);
    if (innermost_loop == nullptr) { return std::nullopt; }
    auto *split = std::get_if<schedule::SplitTerminator>(
        &block.terminator);
    if (split == nullptr || !split->convergence) {
        return std::nullopt;
    }
    auto *condition = _source.value(split->condition);
    auto *outer_point = _source.convergence(*split->convergence);
    if (condition == nullptr || outer_point == nullptr ||
        condition->value_class != schedule::ValueClass::varying ||
        !split->true_edge.assignments.empty() ||
        !split->false_edge.assignments.empty() ||
        !split->true_edge.joins.empty() ||
        !split->false_edge.joins.empty() ||
        split->true_edge.loop_back || split->false_edge.loop_back ||
        split->true_edge.target == split->false_edge.target ||
        outer_point->target == block.id) {
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
    auto assignments_are_lane_masked = [&](const auto &assignments) noexcept {
        for (auto assignment : assignments) {
            auto *destination = _source.value(assignment.destination);
            if (destination == nullptr ||
                (destination->value_class !=
                     schedule::ValueClass::varying &&
                 destination->value_class !=
                     schedule::ValueClass::mask)) {
                return false;
            }
        }
        return true;
    };
    auto closes_outer = [&](const schedule::ControlEdge &edge) noexcept {
        return edge.target == outer_point->target &&
               !edge.loop_back && edge.joins.size() == 1u &&
               edge.joins.front() == *split->convergence &&
               assignments_are_lane_masked(edge.assignments);
    };
    auto safe_instruction = [&](const schedule::Instruction &instruction) {
        if (!instruction.result || instruction.participant_mask) {
            return false;
        }
        if (instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op &&
            instruction.operands.size() == 1u) {
            auto op = static_cast<xir::CastOp>(
                *instruction.source_op);
            return op == xir::CastOp::STATIC_CAST ||
                   op == xir::CastOp::BITWISE_CAST;
        }
        if (instruction.opcode != schedule::Opcode::arithmetic ||
            !instruction.source_op) {
            return false;
        }
        auto op = static_cast<xir::ArithmeticOp>(
            *instruction.source_op);
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::BINARY_BIT_AND:
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            case xir::ArithmeticOp::SELECT:
            case xir::ArithmeticOp::AGGREGATE:
            case xir::ArithmeticOp::EXTRACT: return true;
            case xir::ArithmeticOp::BINARY_DIV: {
                auto *result = _source.value(*instruction.result);
                return result != nullptr && result->type != nullptr &&
                       result->type->is_float_or_float_vector();
            }
            default: return false;
        }
    };

    auto try_side = [&](const schedule::ControlEdge &nested_edge,
                        const schedule::ControlEdge &other_edge,
                        bool nested_on_true)
        -> std::optional<NestedPredicatedRegion> {
        auto *nested_split_block = _source.block(nested_edge.target);
        auto *other_block = _source.block(other_edge.target);
        if (nested_split_block == nullptr || other_block == nullptr ||
            predecessor_count(nested_split_block->id) != 1u ||
            predecessor_count(other_block->id) != 1u ||
            !other_block->instructions.empty()) {
            return std::nullopt;
        }
        auto nested_diamond =
            _find_guarded_predicated_math_diamond(
                *nested_split_block, allow_tiny_speculation);
        if (!nested_diamond ||
            nested_diamond->instruction_count > 3u) {
            return std::nullopt;
        }
        auto *nested_control =
            std::get_if<schedule::SplitTerminator>(
                &nested_split_block->terminator);
        auto *nested_point = nested_control &&
                                     nested_control->convergence ?
                                 _source.convergence(
                                     *nested_control->convergence) :
                                 nullptr;
        auto *nested_merge_block = nested_point == nullptr ?
                                       nullptr :
                                       _source.block(
                                           nested_point->target);
        auto *nested_merge_branch = nested_merge_block == nullptr ?
                                        nullptr :
                                        std::get_if<
                                            schedule::BranchTerminator>(
                                            &nested_merge_block
                                                 ->terminator);
        auto *other_branch = std::get_if<
            schedule::BranchTerminator>(&other_block->terminator);
        if (nested_point == nullptr ||
            nested_point->parent != split->convergence ||
            nested_merge_block == nullptr ||
            !nested_merge_block->instructions.empty() ||
            predecessor_count(nested_merge_block->id) != 2u ||
            nested_merge_branch == nullptr || other_branch == nullptr ||
            !closes_outer(nested_merge_branch->edge) ||
            !closes_outer(other_branch->edge)) {
            return std::nullopt;
        }
        auto belongs_to_innermost_loop = [&](const auto *candidate) noexcept {
            return candidate != nullptr &&
                   std::find(
                       innermost_loop->blocks.cbegin(),
                       innermost_loop->blocks.cend(), candidate->id) !=
                       innermost_loop->blocks.cend();
        };
        if (!belongs_to_innermost_loop(nested_split_block) ||
            !belongs_to_innermost_loop(nested_merge_block) ||
            !belongs_to_innermost_loop(other_block) ||
            !belongs_to_innermost_loop(
                _source.block(outer_point->target)) ||
            !std::all_of(
                nested_diamond->true_blocks.cbegin(),
                nested_diamond->true_blocks.cend(),
                belongs_to_innermost_loop) ||
            !std::all_of(
                nested_diamond->false_blocks.cbegin(),
                nested_diamond->false_blocks.cend(),
                belongs_to_innermost_loop)) {
            return std::nullopt;
        }
        if (nested_split_block->instructions.empty() ||
            nested_split_block->instructions.size() > 12u ||
            !std::all_of(
                nested_split_block->instructions.cbegin(),
                nested_split_block->instructions.cend(),
                safe_instruction)) {
            return std::nullopt;
        }
        return NestedPredicatedRegion{
            .nested_split_block = nested_split_block,
            .nested_diamond = std::move(*nested_diamond),
            .nested_merge_block = nested_merge_block,
            .other_block = other_block,
            .merge = outer_point->target,
            .nested_on_true = nested_on_true,
            .instruction_count =
                nested_split_block->instructions.size() +
                nested_diamond->instruction_count,
        };
    };
    if (auto region = try_side(
            split->true_edge, split->false_edge, true)) {
        return region;
    }
    return try_side(split->false_edge, split->true_edge, false);
}

void ScheduleEmitter::_emit_guarded_predicated_math_diamond(
    const schedule::SplitTerminator &control,
    const GuardedPredicatedMathDiamond &diamond,
    bool continue_at_merge) {
    auto *condition = _load_value(control.condition);
    if (condition == nullptr) { return; }
    auto *outer_mask = _active_mask;
    auto *outer_seed = _seed_lane;
    auto outer_locals = _locals;
    auto *true_mask = _builder.CreateAnd(outer_mask, condition);
    auto *false_mask = _builder.CreateAnd(
        outer_mask, _builder.CreateNot(condition));
    auto emit_chain = [&](const auto &chain,
                          ::llvm::Value *mask) noexcept {
        auto instruction_count = size_t{0u};
        for (auto *arm : chain) {
            instruction_count += arm->instructions.size();
        }
        auto *resume = static_cast<::llvm::BasicBlock *>(nullptr);
        if (instruction_count != 0u) {
            auto *execute = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "guarded.math.execute", _entry);
            resume = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "guarded.math.resume", _entry);
            _builder.CreateCondBr(
                _builder.CreateOrReduce(mask), execute, resume);
            _builder.SetInsertPoint(execute);
        }
        _active_mask = mask;
        // The candidate contains only pure arithmetic/casts. It has no
        // operation whose address or scalarization depends on the narrowed
        // arm's first active lane, while cohort-uniform inputs remain equal
        // at the enclosing cohort's seed. Reuse that seed and avoid a second
        // horizontal mask reduction after the nonempty guard.
        _seed_lane = outer_seed;
        _locals = outer_locals;
        for (auto *arm : chain) {
            for (auto &&instruction : arm->instructions) {
                _emit_instruction(instruction, nullptr, mask);
                if (_failed()) { return; }
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &arm->terminator);
            _apply_assignments(branch->edge.assignments, mask);
            if (_failed()) { return; }
        }
        if (instruction_count != 0u) {
            _builder.CreateBr(resume);
            _builder.SetInsertPoint(resume);
        }
    };
    emit_chain(diamond.true_blocks, true_mask);
    if (!_failed()) { emit_chain(diamond.false_blocks, false_mask); }
    _active_mask = outer_mask;
    _seed_lane = outer_seed;
    _locals = std::move(outer_locals);
    if (_failed()) { return; }
    _result.local_predicated_diamond_count++;
    _result.local_predicated_two_sided_diamond_count +=
        diamond.two_sided;
    _result.local_predicated_assignment_diamond_count +=
        diamond.instruction_count == 0u;
    _result.local_predicated_block_count +=
        diamond.true_blocks.size() + diamond.false_blocks.size();
    _result.local_predicated_instruction_count +=
        diamond.instruction_count;
    if (continue_at_merge) {
        _continue_at(diamond.merge, outer_mask);
    }
}

void ScheduleEmitter::_emit_nested_predicated_region(
    const schedule::SplitTerminator &control,
    const NestedPredicatedRegion &region,
    bool continue_at_merge) {
    auto *condition = _load_value(control.condition);
    if (condition == nullptr) { return; }
    auto *outer_mask = _active_mask;
    auto *outer_seed = _seed_lane;
    auto outer_locals = _locals;
    auto *true_mask = _builder.CreateAnd(outer_mask, condition);
    auto *false_mask = _builder.CreateAnd(
        outer_mask, _builder.CreateNot(condition));
    auto *nested_mask = region.nested_on_true ?
                            true_mask :
                            false_mask;
    auto *other_mask = region.nested_on_true ?
                           false_mask :
                           true_mask;

    auto *execute = ::llvm::BasicBlock::Create(
        _module.getContext(), "nested.region.execute", _entry);
    auto *resume = ::llvm::BasicBlock::Create(
        _module.getContext(), "nested.region.resume", _entry);
    _builder.CreateCondBr(
        _builder.CreateOrReduce(nested_mask), execute, resume);
    _builder.SetInsertPoint(execute);
    _active_mask = nested_mask;
    _seed_lane = outer_seed;
    _locals = outer_locals;
    for (auto &&instruction :
         region.nested_split_block->instructions) {
        _emit_instruction(instruction, nullptr, nested_mask);
        if (_failed()) { return; }
    }
    auto *nested_control =
        std::get_if<schedule::SplitTerminator>(
            &region.nested_split_block->terminator);
    _emit_guarded_predicated_math_diamond(
        *nested_control, region.nested_diamond, false);
    if (_failed()) { return; }
    auto *nested_merge_branch =
        std::get_if<schedule::BranchTerminator>(
            &region.nested_merge_block->terminator);
    _apply_assignments(
        nested_merge_branch->edge.assignments, nested_mask);
    if (_failed()) { return; }
    _builder.CreateBr(resume);

    _builder.SetInsertPoint(resume);
    _active_mask = other_mask;
    _seed_lane = outer_seed;
    _locals = outer_locals;
    auto *other_branch = std::get_if<schedule::BranchTerminator>(
        &region.other_block->terminator);
    _apply_assignments(other_branch->edge.assignments, other_mask);
    _active_mask = outer_mask;
    _seed_lane = outer_seed;
    _locals = std::move(outer_locals);
    if (_failed()) { return; }
    _result.nested_predicated_region_count++;
    _result.nested_predicated_block_count +=
        region.nested_diamond.true_blocks.size() +
        region.nested_diamond.false_blocks.size() + 3u;
    _result.nested_predicated_instruction_count +=
        region.instruction_count;
    if (continue_at_merge) {
        _continue_at(region.merge, outer_mask);
    }
}

[[nodiscard]] std::optional<ScheduleEmitter::PredicatedLoop>
ScheduleEmitter::_find_predicated_loop(
    const schedule::BasicBlock &header) const noexcept {
    static constexpr auto max_block_count = size_t{24u};
    static constexpr auto max_instruction_count = size_t{96u};
    static constexpr auto min_default_block_count = size_t{6u};
    auto force = luisa::compute::detail::env_flag(
        "LUISA_SIMD_FORCE_PREDICATED_LOOP");
    // W16 wins even with one worker because sixteen independent lanes hide
    // the dependent gather chain. W8 crosses over only when enough workers
    // make the old scheduler's front-end/branch pressure dominant. Keep the
    // narrower widths on the generic scheduler; their real Voxel gates are
    // neutral or negative. The worker count is fixed by the device before JIT
    // compilation, so this does not introduce a dispatch-time code version.
    auto enabled_width = _enable_native_predicated_loop &&
                         (_width == 16u ||
                          (_width == 8u &&
                           _dispatch_worker_count >= 24u));
    if (_result.predicated_loop_count != 0u ||
        (!enabled_width && !force) ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_PREDICATED_LOOP")) {
        return std::nullopt;
    }

    const schedule::Loop *loop = nullptr;
    for (auto &&candidate : _source.loops()) {
        if (candidate.header == header.id) {
            if (loop != nullptr) { return std::nullopt; }
            loop = &candidate;
        }
    }
    if (loop == nullptr) { return std::nullopt; }
    // Only the innermost loop is eligible. Flattening a child loop would
    // otherwise turn its dynamic epoch into an unbounded predicated region.
    for (auto &&candidate : _source.loops()) {
        if (candidate.parent == loop->id) {
            return std::nullopt;
        }
    }

    auto *header_split = std::get_if<schedule::SplitTerminator>(
        &header.terminator);
    if (header_split == nullptr || !header_split->convergence) {
        return std::nullopt;
    }
    auto *header_condition = _source.value(header_split->condition);
    auto *loop_gate = _source.convergence(
        *header_split->convergence);
    if (header_condition == nullptr ||
        header_condition->value_class !=
            schedule::ValueClass::varying ||
        loop_gate == nullptr ||
        loop_gate->target == header.id) {
        return std::nullopt;
    }
    auto convergence_target = loop_gate->target;
    if (std::find(loop->exits.cbegin(), loop->exits.cend(),
                  convergence_target) == loop->exits.cend()) {
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
    auto is_predicated_value = [&](schedule::ValueId id) noexcept {
        auto *value = _source.value(id);
        return value != nullptr &&
               (value->value_class ==
                    schedule::ValueClass::varying ||
                value->value_class == schedule::ValueClass::mask);
    };
    auto safe_instruction = [&](const schedule::Instruction &instruction) {
        if (!instruction.result ||
            !is_predicated_value(*instruction.result) ||
            instruction.participant_mask) {
            return false;
        }
        if (instruction.opcode == schedule::Opcode::arithmetic &&
            instruction.source_op) {
            return safe_arithmetic(static_cast<xir::ArithmeticOp>(
                *instruction.source_op));
        }
        if (instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op &&
            instruction.operands.size() == 1u &&
            is_predicated_value(instruction.operands.front())) {
            auto op = static_cast<xir::CastOp>(
                *instruction.source_op);
            return op == xir::CastOp::STATIC_CAST ||
                   op == xir::CastOp::BITWISE_CAST;
        }
        if (instruction.opcode == schedule::Opcode::resource_read &&
            instruction.source_op &&
            *instruction.source_op == static_cast<uint32_t>(
                                          xir::ResourceReadOp::BUFFER_READ) &&
            instruction.operands.size() == 2u &&
            !instruction.cohort_uniform_operand_index) {
            return is_predicated_value(
                instruction.operands[1u]);
        }
        return false;
    };
    auto safe_assignments = [&](const auto &assignments) noexcept {
        return std::all_of(
            assignments.cbegin(), assignments.cend(),
            [&](schedule::EdgeAssignment assignment) noexcept {
                return is_predicated_value(
                    assignment.destination);
            });
    };
    auto edges = [&](const schedule::BasicBlock &block)
        -> std::optional<std::vector<schedule::ControlEdge>> {
        return std::visit(
            [&](const auto &control)
                -> std::optional<std::vector<schedule::ControlEdge>> {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    return std::vector<schedule::ControlEdge>{
                        control.edge};
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SplitTerminator>) {
                    auto *condition = _source.value(
                        control.condition);
                    if (condition == nullptr ||
                        (condition->value_class !=
                             schedule::ValueClass::varying &&
                         condition->value_class !=
                             schedule::ValueClass::warp_uniform)) {
                        return std::nullopt;
                    }
                    return std::vector<schedule::ControlEdge>{
                        control.true_edge, control.false_edge};
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        control.convergence);
                    if (point == nullptr) { return std::nullopt; }
                    schedule::ControlEdge edge{point->target};
                    edge.joins.emplace_back(control.convergence);
                    edge.assignments = control.assignments;
                    return std::vector<schedule::ControlEdge>{
                        std::move(edge)};
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    auto *target_loop = _source.loop(control.loop);
                    if (target_loop == nullptr) {
                        return std::nullopt;
                    }
                    schedule::ControlEdge edge{target_loop->header};
                    edge.loop_back = control.loop;
                    edge.assignments = control.assignments;
                    return std::vector<schedule::ControlEdge>{
                        std::move(edge)};
                } else {
                    return std::nullopt;
                }
            },
            block.terminator);
    };

    PredicatedLoop result{
        .loop = loop,
        .convergence = *header_split->convergence,
        .exits = loop->exits,
    };
    static constexpr auto max_complete_batch_iterations = uint64_t{4096u};
    if (loop->max_trip_count && *loop->max_trip_count != 0u &&
        *loop->max_trip_count <= max_complete_batch_iterations) {
        // A trip-count bound describes successful loop-body iterations. The
        // top-tested header executes once more to observe the final false
        // condition, so include that check in the batch as well. Otherwise a
        // lane that reaches the bound would re-enter the generic scheduler
        // solely to take the already-proven exit edge.
        result.batch_iteration_count = static_cast<uint32_t>(
            *loop->max_trip_count + 1u);
    } else if (!force) {
        return std::nullopt;
    }
    std::vector<uint8_t> is_loop_block(
        _source.blocks().size(), uint8_t{0u});
    for (auto id : loop->blocks) {
        if (id.value >= is_loop_block.size()) {
            return std::nullopt;
        }
        is_loop_block[id.value] = 1u;
    }
    if (is_loop_block[header.id.value] == 0u ||
        result.exits.empty()) {
        return std::nullopt;
    }
    std::vector<uint8_t> in_region(
        _source.blocks().size(), uint8_t{0u});
    std::vector<schedule::BlockId> pending{header.id};
    in_region[header.id.value] = 1u;
    auto saw_back_edge = false;
    auto saw_exit = false;
    for (auto cursor = size_t{0u}; cursor < pending.size(); cursor++) {
        if (pending.size() > max_block_count) {
            return std::nullopt;
        }
        auto id = pending[cursor];
        auto *block = _source.block(id);
        if (block == nullptr ||
            is_loop_block[id.value] == 0u) {
            return std::nullopt;
        }
        for (auto &&instruction : block->instructions) {
            if (!safe_instruction(instruction) ||
                ++result.instruction_count > max_instruction_count) {
                return std::nullopt;
            }
        }
        auto outgoing = edges(*block);
        if (!outgoing) { return std::nullopt; }
        for (auto &&edge : *outgoing) {
            if (!safe_assignments(edge.assignments)) {
                return std::nullopt;
            }
            if (edge.loop_back) {
                if (*edge.loop_back != loop->id ||
                    edge.target != header.id) {
                    return std::nullopt;
                }
                saw_back_edge = true;
                continue;
            }
            auto exit_iter = std::find(
                result.exits.cbegin(), result.exits.cend(),
                edge.target);
            if (exit_iter != result.exits.cend()) {
                saw_exit = true;
                continue;
            }
            if (edge.target == header.id ||
                edge.target.value >= in_region.size() ||
                is_loop_block[edge.target.value] == 0u) {
                return std::nullopt;
            }
            if (in_region[edge.target.value] == 0u) {
                in_region[edge.target.value] = 1u;
                pending.emplace_back(edge.target);
            }
        }
    }
    if (!saw_back_edge || !saw_exit ||
        (!force && pending.size() < min_default_block_count)) {
        return std::nullopt;
    }
    for (auto id : loop->blocks) {
        if (id.value >= in_region.size() ||
            in_region[id.value] == 0u) {
            return std::nullopt;
        }
    }

    // Every convergence suppressed by the flattened region must have been
    // declared by a split in that same region. This excludes outer dynamic
    // gates: the header gate is recreated lazily at each batch boundary, and
    // all remaining outer tokens are left to the ordinary destination-side
    // cascade.
    std::vector<uint8_t> declared(
        _source.convergence_points().size(), uint8_t{0u});
    for (auto id : pending) {
        auto *block = _source.block(id);
        if (auto *split = std::get_if<schedule::SplitTerminator>(
                &block->terminator);
            split != nullptr && split->convergence) {
            auto convergence = *split->convergence;
            auto *point = _source.convergence(convergence);
            if (convergence.value >= declared.size() ||
                point == nullptr ||
                (std::find(
                     result.exits.cbegin(), result.exits.cend(),
                     point->target) == result.exits.cend() &&
                 (point->target.value >= in_region.size() ||
                  in_region[point->target.value] == 0u))) {
                return std::nullopt;
            }
            declared[convergence.value] = 1u;
        }
    }
    for (auto id : pending) {
        auto *block = _source.block(id);
        auto outgoing = edges(*block);
        for (auto &&edge : *outgoing) {
            for (auto convergence : edge.joins) {
                if (convergence.value >= declared.size() ||
                    declared[convergence.value] == 0u) {
                    return std::nullopt;
                }
            }
        }
    }

    // A reducible natural loop has no external entry into a non-header block.
    // Recheck that property on Schedule IR and include every terminator kind
    // in the predecessor scan, even though those kinds are rejected inside
    // the candidate itself.
    std::vector<size_t> external_predecessors(
        _source.blocks().size(), 0u);
    for (auto &&source : _source.blocks()) {
        if (in_region[source.id.value] != 0u) { continue; }
        auto record = [&](schedule::BlockId target) noexcept {
            if (target.value < in_region.size() &&
                in_region[target.value] != 0u) {
                external_predecessors[target.value]++;
            }
        };
        std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    record(control.edge.target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SplitTerminator>) {
                    record(control.true_edge.target);
                    record(control.false_edge.target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SwitchTerminator>) {
                    for (auto &&item : control.cases) {
                        record(item.edge.target);
                    }
                    record(control.default_edge.target);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        control.convergence);
                    if (point != nullptr) { record(point->target); }
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    auto *target_loop = _source.loop(control.loop);
                    if (target_loop != nullptr) {
                        record(target_loop->header);
                    }
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    record(control.resume_edge.target);
                }
            },
            source.terminator);
    }
    for (auto id : pending) {
        if (id != header.id &&
            external_predecessors[id.value] != 0u) {
            return std::nullopt;
        }
    }

    // Removing the annotated loop back-edges must leave one acyclic region.
    // Its topological order lets codegen form masks with straight-line ORs
    // rather than introducing another program-counter state machine.
    std::vector<size_t> indegree(
        _source.blocks().size(), 0u);
    for (auto id : pending) {
        auto *block = _source.block(id);
        auto outgoing = edges(*block);
        for (auto &&edge : *outgoing) {
            if (!edge.loop_back &&
                std::find(result.exits.cbegin(), result.exits.cend(),
                          edge.target) == result.exits.cend()) {
                indegree[edge.target.value]++;
            }
        }
    }
    std::vector<schedule::BlockId> ready;
    ready.reserve(pending.size());
    for (auto id : pending) {
        if (indegree[id.value] == 0u) {
            ready.emplace_back(id);
        }
    }
    if (ready.size() != 1u || ready.front() != header.id) {
        return std::nullopt;
    }
    for (auto cursor = size_t{0u}; cursor < ready.size(); cursor++) {
        auto id = ready[cursor];
        auto *block = _source.block(id);
        result.order.emplace_back(block);
        auto outgoing = edges(*block);
        for (auto &&edge : *outgoing) {
            if (edge.loop_back ||
                std::find(result.exits.cbegin(), result.exits.cend(),
                          edge.target) != result.exits.cend()) {
                continue;
            }
            auto &degree = indegree[edge.target.value];
            if (--degree == 0u) { ready.emplace_back(edge.target); }
        }
    }
    if (result.order.size() != pending.size()) {
        return std::nullopt;
    }
    return result;
}

void ScheduleEmitter::_emit_predicated_loop(
    const PredicatedLoop &loop) {
    auto &context = _module.getContext();
    auto *preheader = _builder.GetInsertBlock();
    auto *iteration = ::llvm::BasicBlock::Create(
        context, "predicated.loop.iteration", _entry);
    auto *batch_exit = ::llvm::BasicBlock::Create(
        context, "predicated.loop.batch.exit", _entry);
    _builder.CreateBr(iteration);

    _builder.SetInsertPoint(iteration);
    auto *iteration_mask = _builder.CreatePHI(
        _layout.mask_type(), 2u,
        "predicated.loop.iteration.mask");
    auto *iteration_index = _builder.CreatePHI(
        _builder.getInt32Ty(), 2u,
        "predicated.loop.iteration.index");
    iteration_mask->addIncoming(_active_mask, preheader);
    iteration_index->addIncoming(
        _builder.getInt32(0u), preheader);
    std::vector<::llvm::PHINode *> exited_masks;
    exited_masks.reserve(loop.exits.size());
    for (auto i = size_t{0u}; i < loop.exits.size(); i++) {
        auto *mask = _builder.CreatePHI(
            _layout.mask_type(), 2u,
            "predicated.loop.exited.mask." + std::to_string(i));
        mask->addIncoming(_zero_mask(), preheader);
        exited_masks.emplace_back(mask);
    }

    std::vector<uint8_t> in_region(
        _source.blocks().size(), uint8_t{0u});
    for (auto *block : loop.order) {
        in_region[block->id.value] = 1u;
    }
    std::vector<::llvm::Value *> incoming(
        _source.blocks().size(), nullptr);
    incoming[loop.loop->header.value] = iteration_mask;
    auto *next_mask = _zero_mask();
    std::vector<::llvm::Value *> iteration_exits(
        loop.exits.size(), nullptr);
    auto add_mask = [&](::llvm::Value *&destination,
                        ::llvm::Value *mask) noexcept {
        destination = destination == nullptr ?
                          mask :
                          _builder.CreateOr(destination, mask);
    };
    auto route = [&](const schedule::ControlEdge &edge,
                     ::llvm::Value *mask) noexcept {
        auto *flow = _route_edge(edge, mask);
        if (flow == nullptr) { return; }
        if (edge.loop_back) {
            if (*edge.loop_back != loop.loop->id ||
                edge.target != loop.loop->header) {
                _fail("predicated loop encountered a foreign back-edge");
                return;
            }
            add_mask(next_mask, flow);
        } else if (auto iter = std::find(
                       loop.exits.cbegin(), loop.exits.cend(),
                       edge.target);
                   iter != loop.exits.cend()) {
            auto index = static_cast<size_t>(
                std::distance(loop.exits.cbegin(), iter));
            add_mask(iteration_exits[index], flow);
        } else if (edge.target.value < in_region.size() &&
                   in_region[edge.target.value] != 0u) {
            add_mask(incoming[edge.target.value], flow);
        } else {
            _fail("predicated loop escaped its audited region");
        }
    };

    for (auto *block : loop.order) {
        auto *block_mask = incoming[block->id.value];
        if (block_mask == nullptr) {
            _fail("predicated loop has an unavailable block mask");
            return;
        }
        _active_mask = block_mask;
        _seed_lane = _safe_first_lane(block_mask);
        _locals.clear();
        for (auto &&instruction : block->instructions) {
            _emit_instruction(instruction, nullptr, block_mask);
            if (_failed()) { return; }
        }
        std::visit(
            [&](const auto &control) {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    route(control.edge, block_mask);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SplitTerminator>) {
                    auto *condition_value = _source.value(
                        control.condition);
                    auto *condition = _load_value(
                        control.condition);
                    if (condition_value == nullptr ||
                        condition == nullptr) {
                        return;
                    }
                    ::llvm::Value *true_mask = nullptr;
                    ::llvm::Value *false_mask = nullptr;
                    if (condition_value->value_class ==
                        schedule::ValueClass::varying) {
                        auto *safe_condition = _builder.CreateSelect(
                            block_mask, condition, _zero_mask());
                        true_mask = _builder.CreateAnd(
                            block_mask, safe_condition);
                        false_mask = _builder.CreateAnd(
                            block_mask,
                            _builder.CreateNot(safe_condition));
                    } else {
                        true_mask = _builder.CreateSelect(
                            condition, block_mask, _zero_mask());
                        false_mask = _builder.CreateSelect(
                            condition, _zero_mask(), block_mask);
                    }
                    route(control.true_edge, true_mask);
                    if (!_failed()) {
                        route(control.false_edge, false_mask);
                    }
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        control.convergence);
                    if (point == nullptr) {
                        _fail("predicated loop references an invalid join");
                        return;
                    }
                    schedule::ControlEdge edge{point->target};
                    edge.joins.emplace_back(control.convergence);
                    edge.assignments = control.assignments;
                    route(edge, block_mask);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    auto *target_loop = _source.loop(control.loop);
                    if (target_loop == nullptr) {
                        _fail("predicated loop references an invalid back-edge");
                        return;
                    }
                    schedule::ControlEdge edge{target_loop->header};
                    edge.loop_back = control.loop;
                    edge.assignments = control.assignments;
                    route(edge, block_mask);
                } else {
                    _fail("predicated loop encountered an unaudited terminator");
                }
            },
            block->terminator);
        if (_failed()) { return; }
    }

    std::vector<::llvm::Value *> new_exited_masks;
    new_exited_masks.reserve(loop.exits.size());
    auto *any_exited_mask = _zero_mask();
    for (auto i = size_t{0u}; i < loop.exits.size(); i++) {
        auto *iteration_exit = iteration_exits[i] == nullptr ?
                                   _zero_mask() :
                                   iteration_exits[i];
        auto *new_exited = _builder.CreateOr(
            exited_masks[i], iteration_exit);
        new_exited_masks.emplace_back(new_exited);
        any_exited_mask = _builder.CreateOr(
            any_exited_mask, new_exited);
    }
    auto *next_iteration_index = _builder.CreateAdd(
        iteration_index, _builder.getInt32(1u));
    auto *continue_batch = _builder.CreateAnd(
        _builder.CreateOrReduce(next_mask),
        _builder.CreateICmpULT(
            next_iteration_index,
            _builder.getInt32(loop.batch_iteration_count)));
    auto *latch = _builder.GetInsertBlock();
    _builder.CreateCondBr(continue_batch, iteration, batch_exit);
    iteration_mask->addIncoming(next_mask, latch);
    iteration_index->addIncoming(next_iteration_index, latch);
    for (auto i = size_t{0u}; i < exited_masks.size(); i++) {
        exited_masks[i]->addIncoming(
            new_exited_masks[i], latch);
    }

    _builder.SetInsertPoint(batch_exit);
    auto *next_nonempty = _builder.CreateOrReduce(next_mask);
    auto *exit_nonempty = _builder.CreateOrReduce(any_exited_mask);
    ::llvm::Value *has_destination = next_nonempty;
    ::llvm::Value *divergent_destinations = _builder.getFalse();
    for (auto *mask : new_exited_masks) {
        auto *nonempty = _builder.CreateOrReduce(mask);
        divergent_destinations = _builder.CreateOr(
            divergent_destinations,
            _builder.CreateAnd(has_destination, nonempty));
        has_destination = _builder.CreateOr(
            has_destination, nonempty);
    }
    _active_mask = _builder.CreateOr(
        next_mask, any_exited_mask);
    _seed_lane = _safe_first_lane(_active_mask);
    auto *release_exit = ::llvm::BasicBlock::Create(
        context, "predicated.loop.release.exit", _entry);
    auto *continue_only = ::llvm::BasicBlock::Create(
        context, "predicated.loop.continue.only", _entry);
    _builder.CreateCondBr(
        exit_nonempty, release_exit, continue_only);

    _builder.SetInsertPoint(release_exit);
    // Recreate only the loop-exit gate, and only when this batch actually
    // has an exit. If a prior batch already owns the top frame,
    // _declare_convergence reuses it and preserves its original expected
    // mask. A batch in which all lanes choose one exit target allocates no
    // frame; multiple exit targets still need the gate even when no lane
    // continues around the back-edge. Keeping this call out of the no-exit
    // path also avoids the otherwise branchless frame search every batch.
    _declare_convergence(
        loop.convergence, divergent_destinations);
    _resume(loop.loop->header, next_mask);
    for (auto i = size_t{0u}; i < loop.exits.size(); i++) {
        _resume(loop.exits[i], new_exited_masks[i]);
    }
    _builder.CreateBr(_scheduler_loop);

    _builder.SetInsertPoint(continue_only);
    _continue_at(loop.loop->header, next_mask);

    _result.predicated_loop_count++;
    _result.predicated_loop_block_count += loop.order.size();
    _result.predicated_loop_instruction_count +=
        loop.instruction_count;
    _result.predicated_loop_batch_iteration_count = std::max(
        _result.predicated_loop_batch_iteration_count,
        static_cast<size_t>(loop.batch_iteration_count));
}

[[nodiscard]] std::optional<ScheduleEmitter::CoherentAllOnRegion>
ScheduleEmitter::_find_coherent_all_on_region(
    const schedule::SplitTerminator &control,
    const schedule::ControlEdge &entry_edge) const noexcept {
    static constexpr auto max_block_count = size_t{4u};
    static constexpr auto max_weighted_cost = size_t{24u};
    static constexpr auto max_region_count = size_t{1u};
    static constexpr auto min_w8_block_count = size_t{3u};
    auto enabled_width = _width == 2u || _width == 8u;
    if (_result.all_on_region_version_count >= max_region_count ||
        !enabled_width || !control.convergence ||
        entry_edge.loop_back || !entry_edge.joins.empty() ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_ALL_ON_REGION_VERSIONING")) {
        return std::nullopt;
    }
    auto *condition = _source.value(control.condition);
    auto *point = _source.convergence(*control.convergence);
    if (condition == nullptr ||
        condition->value_class != schedule::ValueClass::varying ||
        point == nullptr ||
        point->target.value >= _target_convergence_depths.size() ||
        _target_convergence_depths[point->target.value] != 1u) {
        return std::nullopt;
    }

    auto cheap_arithmetic = [](xir::ArithmeticOp op) noexcept {
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
            case xir::ArithmeticOp::UNARY_BIT_NOT:
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::BINARY_BIT_AND:
            case xir::ArithmeticOp::BINARY_BIT_OR:
            case xir::ArithmeticOp::BINARY_BIT_XOR:
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
            case xir::ArithmeticOp::BINARY_ROTATE_LEFT:
            case xir::ArithmeticOp::BINARY_ROTATE_RIGHT:
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
            case xir::ArithmeticOp::ALL:
            case xir::ArithmeticOp::ANY:
            case xir::ArithmeticOp::SELECT:
            case xir::ArithmeticOp::CLAMP:
            case xir::ArithmeticOp::SATURATE:
            case xir::ArithmeticOp::LERP:
            case xir::ArithmeticOp::STEP:
            case xir::ArithmeticOp::ABS:
            case xir::ArithmeticOp::MIN:
            case xir::ArithmeticOp::MAX:
            case xir::ArithmeticOp::CLZ:
            case xir::ArithmeticOp::CTZ:
            case xir::ArithmeticOp::POPCOUNT:
            case xir::ArithmeticOp::REVERSE:
            case xir::ArithmeticOp::ISINF:
            case xir::ArithmeticOp::ISNAN:
            case xir::ArithmeticOp::FMA:
            case xir::ArithmeticOp::COPYSIGN:
            case xir::ArithmeticOp::CROSS:
            case xir::ArithmeticOp::DOT:
            case xir::ArithmeticOp::LENGTH_SQUARED:
            case xir::ArithmeticOp::FACEFORWARD:
            case xir::ArithmeticOp::REFLECT:
            case xir::ArithmeticOp::REDUCE_SUM:
            case xir::ArithmeticOp::REDUCE_PRODUCT:
            case xir::ArithmeticOp::REDUCE_MIN:
            case xir::ArithmeticOp::REDUCE_MAX:
            case xir::ArithmeticOp::AGGREGATE:
            case xir::ArithmeticOp::SHUFFLE:
            case xir::ArithmeticOp::INSERT:
            case xir::ArithmeticOp::EXTRACT: return true;
            default: return false;
        }
    };
    auto instruction_cost = [&](const schedule::Instruction &instruction) {
        if (instruction.opcode == schedule::Opcode::arithmetic &&
            instruction.source_op &&
            cheap_arithmetic(static_cast<xir::ArithmeticOp>(
                *instruction.source_op))) {
            auto *result = instruction.result ?
                               _source.value(*instruction.result) :
                               nullptr;
            auto units = result == nullptr || result->type == nullptr ?
                             size_t{1u} :
                             std::max(
                                 size_t{1u},
                                 (_abi_size(result->type) + 3u) / 4u);
            auto op = static_cast<xir::ArithmeticOp>(
                *instruction.source_op);
            if ((op == xir::ArithmeticOp::DOT ||
                 op == xir::ArithmeticOp::CROSS ||
                 op == xir::ArithmeticOp::LENGTH_SQUARED) &&
                !instruction.operands.empty()) {
                auto *operand = _source.value(
                    instruction.operands.front());
                if (operand != nullptr && operand->type != nullptr) {
                    units = std::max(
                        units,
                        std::max(
                            size_t{1u},
                            (_abi_size(operand->type) + 3u) / 4u));
                }
            }
            return std::optional<size_t>{units};
        }
        if (instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op) {
            auto op = static_cast<xir::CastOp>(*instruction.source_op);
            if (op == xir::CastOp::STATIC_CAST ||
                op == xir::CastOp::BITWISE_CAST) {
                auto *result = instruction.result ?
                                   _source.value(*instruction.result) :
                                   nullptr;
                return std::optional<size_t>{
                    result == nullptr || result->type == nullptr ?
                        size_t{1u} :
                        std::max(
                            size_t{1u},
                            (_abi_size(result->type) + 3u) / 4u)};
            }
        }
        return std::optional<size_t>{};
    };

    CoherentAllOnRegion region;
    std::vector<bool> visited(_source.blocks().size(), false);
    auto current = entry_edge.target;
    auto joined = false;
    while (region.blocks.size() < max_block_count) {
        if (current.value >= visited.size() || visited[current.value]) {
            return std::nullopt;
        }
        visited[current.value] = true;
        auto *candidate = _source.block(current);
        if (candidate == nullptr) { return std::nullopt; }
        if (current == point->target) {
            if (!joined) { return std::nullopt; }
        } else if (current.value < _target_convergence_depths.size() &&
                   _target_convergence_depths[current.value] != 0u) {
            return std::nullopt;
        }
        for (auto &&instruction : candidate->instructions) {
            auto cost = instruction_cost(instruction);
            if (!cost || region.weighted_cost + *cost >
                             max_weighted_cost) {
                return std::nullopt;
            }
            region.weighted_cost += *cost;
            region.instruction_count++;
        }
        region.blocks.emplace_back(candidate);

        if (auto *split = std::get_if<schedule::SplitTerminator>(
                &candidate->terminator)) {
            auto *next_condition = _source.value(split->condition);
            if (!joined || next_condition == nullptr ||
                next_condition->value_class !=
                    schedule::ValueClass::varying) {
                return std::nullopt;
            }
            // The cloned region ends at this split. Do not let its generic
            // terminator lowering absorb a following memory diamond beyond
            // the block/cost budget recorded here.
            if (_find_predicated_memory_diamond(*candidate)) {
                return std::nullopt;
            }
            // The all-on test and the cloned CFG have a fixed cost. Paired
            // measurements show that W8 does not amortize that cost over a
            // two-block region, while W2 does. Keep this structural and
            // fail-closed: W8 needs at least one additional scheduler edge.
            if (_width == 8u &&
                region.blocks.size() < min_w8_block_count) {
                return std::nullopt;
            }
            return region;
        }
        auto *branch = std::get_if<schedule::BranchTerminator>(
            &candidate->terminator);
        if (branch == nullptr || branch->edge.loop_back) {
            return std::nullopt;
        }
        if (!branch->edge.joins.empty()) {
            if (joined || branch->edge.joins.size() != 1u ||
                branch->edge.joins.front() != *control.convergence ||
                branch->edge.target != point->target) {
                return std::nullopt;
            }
            joined = true;
        }
        current = branch->edge.target;
    }
    return std::nullopt;
}

void ScheduleEmitter::_emit_coherent_all_on_region(
    const schedule::ControlEdge &entry_edge,
    const CoherentAllOnRegion &region) {
    auto *all_on = ::llvm::Constant::getAllOnesValue(
        _layout.mask_type());
    _active_mask = all_on;
    _seed_lane = _builder.getInt32(0u);
    if (_route_edge(entry_edge, all_on) == nullptr) { return; }
    _result.all_on_region_version_count++;
    _result.all_on_region_block_count += region.blocks.size();
    _result.all_on_region_instruction_count += region.instruction_count;
    for (auto i = size_t{0u}; i < region.blocks.size(); i++) {
        auto *block = region.blocks[i];
        for (auto &&instruction : block->instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        if (i + 1u == region.blocks.size()) {
            _emit_terminator(*block, false);
            return;
        }
        auto *branch = std::get_if<schedule::BranchTerminator>(
            &block->terminator);
        if (branch == nullptr ||
            _route_edge(branch->edge, all_on) == nullptr) {
            _fail("invalid coherent all-on region during LLVM emission");
            return;
        }
    }
}

}// namespace luisa::compute::simd::detail
