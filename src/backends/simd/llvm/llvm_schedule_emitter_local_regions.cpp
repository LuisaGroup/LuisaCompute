#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] std::optional<ScheduleEmitter::ChainedPredicatedRegion>
ScheduleEmitter::_find_chained_predicated_region(
    const schedule::BasicBlock &block) const noexcept {
    static constexpr auto max_diamond_count = size_t{4u};
    static constexpr auto max_bridge_block_count = size_t{4u};
    static constexpr auto max_bridge_instruction_count = size_t{12u};
    static constexpr auto max_instruction_count = size_t{128u};
    static constexpr auto max_terminal_block_count = size_t{4u};
    static constexpr auto max_terminal_instruction_count = size_t{96u};
    auto enable_chaining =
        _width >= 4u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_CHAINING");
    auto force_terminal_bridge =
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_FORCE_LOCAL_PREDICATED_TERMINAL_BRIDGE");
    auto enable_terminal_bridge =
        (_width >= 4u || force_terminal_bridge) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_TERMINAL_BRIDGE");
    if (!enable_chaining && !enable_terminal_bridge) {
        return std::nullopt;
    }
    auto first = _find_guarded_predicated_math_diamond(block);
    auto *first_control = std::get_if<schedule::SplitTerminator>(
        &block.terminator);
    auto *innermost_loop = _innermost_loop_containing(block.id);
    if (!first || first->instruction_count == 0u ||
        first_control == nullptr || !first_control->convergence ||
        innermost_loop == nullptr) {
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
    auto convergence_is_exclusive = [&](schedule::BlockId target,
                                        schedule::ConvergenceId expected) noexcept {
        auto count = size_t{0u};
        auto matches = false;
        for (auto &&point : _source.convergence_points()) {
            if (point.target == target) {
                count++;
                matches = matches || point.id == expected;
            }
        }
        return count == 1u && matches;
    };
    auto is_convergence_target = [&](schedule::BlockId target) noexcept {
        return std::any_of(
            _source.convergence_points().cbegin(),
            _source.convergence_points().cend(),
            [&](const auto &point) noexcept {
                return point.target == target;
            });
    };
    auto belongs_to_loop = [&](const schedule::BasicBlock *candidate) noexcept {
        return candidate != nullptr &&
               std::find(
                   innermost_loop->blocks.cbegin(),
                   innermost_loop->blocks.cend(), candidate->id) !=
                   innermost_loop->blocks.cend();
    };
    auto safe_bridge_instruction = [&](const schedule::Instruction &instruction) {
        if (!instruction.result || instruction.participant_mask) {
            return false;
        }
        if (instruction.opcode == schedule::Opcode::cast &&
            instruction.source_op && instruction.operands.size() == 1u) {
            auto op = static_cast<xir::CastOp>(*instruction.source_op);
            return op == xir::CastOp::STATIC_CAST ||
                   op == xir::CastOp::BITWISE_CAST;
        }
        if (instruction.opcode != schedule::Opcode::arithmetic ||
            !instruction.source_op) {
            return false;
        }
        auto op = static_cast<xir::ArithmeticOp>(*instruction.source_op);
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
            case xir::ArithmeticOp::ABS:
            case xir::ArithmeticOp::AGGREGATE:
            case xir::ArithmeticOp::EXTRACT:
            case xir::ArithmeticOp::DOT:
            case xir::ArithmeticOp::LENGTH_SQUARED:
            case xir::ArithmeticOp::SQRT:
            case xir::ArithmeticOp::RSQRT: return true;
            case xir::ArithmeticOp::BINARY_DIV: {
                auto *result = _source.value(*instruction.result);
                return result != nullptr && result->type != nullptr &&
                       result->type->is_float_or_float_vector();
            }
            default: return false;
        }
    };

    auto first_merge = first->merge;
    auto first_instruction_count = first->instruction_count;
    ChainedPredicatedRegion region{
        .first_diamond = std::move(*first),
        .merge = first_merge,
        .instruction_count = first_instruction_count,
    };
    auto append_block = [&](const schedule::BasicBlock *candidate) noexcept {
        if (std::find(
                region.inlined_blocks.cbegin(),
                region.inlined_blocks.cend(), candidate) ==
            region.inlined_blocks.cend()) {
            region.inlined_blocks.emplace_back(candidate);
        }
    };
    auto append_diamond = [&](const auto &diamond) noexcept {
        for (auto *arm : diamond.true_blocks) { append_block(arm); }
        for (auto *arm : diamond.false_blocks) { append_block(arm); }
    };
    append_diamond(region.first_diamond);

    auto current_convergence = *first_control->convergence;
    while (enable_chaining &&
           region.continuations.size() + 1u < max_diamond_count) {
        auto target = region.merge;
        std::vector<const schedule::BasicBlock *> bridge;
        auto bridge_instruction_count = size_t{0u};
        auto found = false;
        for (auto i = size_t{0u}; i < max_bridge_block_count; i++) {
            auto *candidate = _source.block(target);
            if (!belongs_to_loop(candidate) || candidate->id == block.id ||
                predecessor_count(candidate->id) != (i == 0u ? 2u : 1u) ||
                (i == 0u ?
                     !convergence_is_exclusive(
                         candidate->id, current_convergence) :
                     is_convergence_target(candidate->id))) {
                break;
            }
            if (!std::all_of(
                    candidate->instructions.cbegin(),
                    candidate->instructions.cend(),
                    safe_bridge_instruction)) {
                break;
            }
            bridge_instruction_count += candidate->instructions.size();
            if (bridge_instruction_count >
                max_bridge_instruction_count) {
                break;
            }
            bridge.emplace_back(candidate);
            if (auto next =
                    _find_guarded_predicated_math_diamond(*candidate);
                next && next->instruction_count != 0u) {
                auto next_instruction_count =
                    region.instruction_count + bridge_instruction_count +
                    next->instruction_count;
                if (next_instruction_count > max_instruction_count) {
                    break;
                }
                auto *next_control =
                    std::get_if<schedule::SplitTerminator>(
                        &candidate->terminator);
                if (next_control == nullptr ||
                    !next_control->convergence ||
                    _innermost_loop_containing(candidate->id) !=
                        innermost_loop) {
                    break;
                }
                region.continuations.emplace_back(
                    ChainedPredicatedRegion::Continuation{
                        .blocks = std::move(bridge),
                        .diamond = std::move(*next),
                    });
                auto &continuation = region.continuations.back();
                for (auto *bridge_block : continuation.blocks) {
                    append_block(bridge_block);
                }
                append_diamond(continuation.diamond);
                region.merge = continuation.diamond.merge;
                region.instruction_count = next_instruction_count;
                current_convergence = *next_control->convergence;
                found = true;
                break;
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &candidate->terminator);
            if (branch == nullptr || branch->edge.loop_back ||
                !branch->edge.joins.empty() ||
                branch->edge.target == candidate->id) {
                break;
            }
            target = branch->edge.target;
        }
        if (!found) { break; }
    }
    if (enable_chaining && _width == 8u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_CHAINED_NESTED_TAIL")) {
        auto target = region.merge;
        std::vector<const schedule::BasicBlock *> bridge;
        auto bridge_instruction_count = size_t{0u};
        for (auto i = size_t{0u}; i < max_bridge_block_count; i++) {
            auto *candidate = _source.block(target);
            if (!belongs_to_loop(candidate) || candidate->id == block.id ||
                predecessor_count(candidate->id) != (i == 0u ? 2u : 1u) ||
                (i == 0u ?
                     !convergence_is_exclusive(
                         candidate->id, current_convergence) :
                     is_convergence_target(candidate->id))) {
                break;
            }
            if (!std::all_of(
                    candidate->instructions.cbegin(),
                    candidate->instructions.cend(),
                    safe_bridge_instruction)) {
                break;
            }
            bridge_instruction_count += candidate->instructions.size();
            if (bridge_instruction_count >
                max_bridge_instruction_count) {
                break;
            }
            bridge.emplace_back(candidate);
            if (auto nested =
                    _find_nested_predicated_region(*candidate)) {
                auto *outer_control = std::get_if<
                    schedule::SplitTerminator>(
                    &candidate->terminator);
                auto next_instruction_count =
                    region.instruction_count + bridge_instruction_count +
                    nested->instruction_count;
                if (next_instruction_count > max_instruction_count ||
                    _innermost_loop_containing(candidate->id) !=
                        innermost_loop ||
                    outer_control == nullptr ||
                    !outer_control->convergence) {
                    break;
                }
                region.nested_continuation =
                    ChainedPredicatedRegion::NestedContinuation{
                        .blocks = std::move(bridge),
                        .region = std::move(*nested),
                    };
                auto &continuation = *region.nested_continuation;
                for (auto *bridge_block : continuation.blocks) {
                    append_block(bridge_block);
                }
                append_block(continuation.region.nested_split_block);
                append_diamond(
                    continuation.region.nested_diamond);
                append_block(
                    continuation.region.nested_merge_block);
                append_block(continuation.region.other_block);
                region.merge = continuation.region.merge;
                region.instruction_count = next_instruction_count;
                current_convergence = *outer_control->convergence;
                break;
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &candidate->terminator);
            if (branch == nullptr || branch->edge.loop_back ||
                !branch->edge.joins.empty() ||
                branch->edge.target == candidate->id) {
                break;
            }
            target = branch->edge.target;
        }
    }
    if (enable_terminal_bridge) {
        // Both predicated arms have arrived at this exclusive convergence, so
        // the original outer cohort is restored before the merge block. A
        // bounded single-entry tail can therefore remain in the same LLVM
        // emission region without speculation or code cloning. Stop before a
        // separately recognizable local region or another convergence target;
        // the final terminator returns to the complete scheduler semantics.
        auto target = region.merge;
        auto terminal_instruction_count = size_t{0u};
        for (auto i = size_t{0u}; i < max_terminal_block_count; i++) {
            auto *candidate = _source.block(target);
            if (!belongs_to_loop(candidate) || candidate->id == block.id ||
                predecessor_count(candidate->id) != (i == 0u ? 2u : 1u) ||
                (i == 0u ?
                     !convergence_is_exclusive(
                         candidate->id, current_convergence) :
                     is_convergence_target(candidate->id))) {
                break;
            }
            if (_find_guarded_predicated_math_diamond(*candidate) ||
                _find_nested_predicated_region(*candidate) ||
                _find_predicated_memory_diamond(*candidate)) {
                break;
            }
            auto next_instruction_count = terminal_instruction_count +
                                          candidate->instructions.size();
            if (next_instruction_count > max_terminal_instruction_count) {
                break;
            }
            region.terminal_blocks.emplace_back(candidate);
            append_block(candidate);
            terminal_instruction_count = next_instruction_count;

            auto *branch = std::get_if<schedule::BranchTerminator>(
                &candidate->terminator);
            if (branch == nullptr || branch->edge.loop_back ||
                !branch->edge.joins.empty() ||
                branch->edge.target == candidate->id) {
                break;
            }
            auto *next = _source.block(branch->edge.target);
            if (!belongs_to_loop(next) ||
                _find_guarded_predicated_math_diamond(*next) ||
                _find_nested_predicated_region(*next) ||
                _find_predicated_memory_diamond(*next)) {
                break;
            }
            target = branch->edge.target;
        }
        region.terminal_instruction_count = terminal_instruction_count;
        // Short W4 tails consistently lose to the established scheduler on
        // the analytic path kernel, while its 81-instruction renderer tail is
        // profitable. Wider vectors amortize the boundary even for short
        // tails. Keep W2 diagnostic-only until it demonstrates a stable win.
        static constexpr auto min_w4_terminal_instruction_count = size_t{32u};
        auto profitable =
            force_terminal_bridge || _width >= 8u ||
            terminal_instruction_count >=
                min_w4_terminal_instruction_count;
        if (!profitable) {
            for (auto *terminal : region.terminal_blocks) {
                std::erase(region.inlined_blocks, terminal);
            }
            region.terminal_blocks.clear();
            region.terminal_instruction_count = 0u;
        }
    }
    return region.continuations.empty() &&
                   !region.nested_continuation &&
                   region.terminal_blocks.empty() ?
               std::nullopt :
               std::optional{std::move(region)};
}

void ScheduleEmitter::_emit_chained_predicated_region(
    const schedule::SplitTerminator &control,
    const ChainedPredicatedRegion &region) {
    auto *outer_mask = _active_mask;
    auto *outer_seed = _seed_lane;
    _emit_guarded_predicated_math_diamond(
        control, region.first_diamond, false);
    if (_failed()) { return; }
    for (auto &&continuation : region.continuations) {
        for (auto i = size_t{0u}; i < continuation.blocks.size(); i++) {
            auto *block = continuation.blocks[i];
            _active_mask = outer_mask;
            _seed_lane = outer_seed;
            for (auto &&instruction : block->instructions) {
                _emit_instruction(instruction, nullptr, outer_mask);
                if (_failed()) { return; }
            }
            if (i + 1u == continuation.blocks.size()) {
                auto *split = std::get_if<schedule::SplitTerminator>(
                    &block->terminator);
                if (split == nullptr) {
                    _fail("chained predicated region lost its split terminator");
                    return;
                }
                _emit_guarded_predicated_math_diamond(
                    *split, continuation.diamond, false);
            } else {
                auto *branch = std::get_if<schedule::BranchTerminator>(
                    &block->terminator);
                if (branch == nullptr) {
                    _fail("chained predicated region lost its bridge terminator");
                    return;
                }
                _apply_assignments(branch->edge.assignments, outer_mask);
            }
            if (_failed()) { return; }
        }
    }
    if (region.nested_continuation) {
        auto &&continuation = *region.nested_continuation;
        for (auto i = size_t{0u}; i < continuation.blocks.size(); i++) {
            auto *block = continuation.blocks[i];
            _active_mask = outer_mask;
            _seed_lane = outer_seed;
            for (auto &&instruction : block->instructions) {
                _emit_instruction(instruction, nullptr, outer_mask);
                if (_failed()) { return; }
            }
            if (i + 1u == continuation.blocks.size()) {
                auto *split = std::get_if<schedule::SplitTerminator>(
                    &block->terminator);
                if (split == nullptr) {
                    _fail("chained nested tail lost its split terminator");
                    return;
                }
                _emit_nested_predicated_region(
                    *split, continuation.region, false);
            } else {
                auto *branch = std::get_if<schedule::BranchTerminator>(
                    &block->terminator);
                if (branch == nullptr) {
                    _fail("chained nested tail lost its bridge terminator");
                    return;
                }
                _apply_assignments(branch->edge.assignments, outer_mask);
            }
            if (_failed()) { return; }
        }
    }
    for (auto i = size_t{0u}; i < region.terminal_blocks.size(); i++) {
        auto *block = region.terminal_blocks[i];
        _active_mask = outer_mask;
        _seed_lane = outer_seed;
        for (auto &&instruction : block->instructions) {
            _emit_instruction(instruction, nullptr, outer_mask);
            if (_failed()) { return; }
        }
        if (i + 1u == region.terminal_blocks.size()) {
            _emit_terminator(*block);
        } else {
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &block->terminator);
            if (branch == nullptr) {
                _fail("local predicated terminal bridge lost its branch terminator");
                return;
            }
            _apply_assignments(branch->edge.assignments, outer_mask);
        }
        if (_failed()) { return; }
    }
    _active_mask = outer_mask;
    _seed_lane = outer_seed;
    _result.chained_predicated_region_count++;
    _result.chained_predicated_transition_count +=
        region.continuations.size() +
        static_cast<size_t>(region.nested_continuation.has_value());
    _result.chained_predicated_block_count +=
        region.inlined_blocks.size();
    _result.chained_predicated_nested_tail_count +=
        region.nested_continuation.has_value();
    _result.chained_predicated_terminal_block_count +=
        region.terminal_blocks.size();
    _result.chained_predicated_terminal_instruction_count +=
        region.terminal_instruction_count;
    if (region.terminal_blocks.empty()) {
        _continue_at(region.merge, outer_mask);
    }
}

}// namespace luisa::compute::simd::detail
