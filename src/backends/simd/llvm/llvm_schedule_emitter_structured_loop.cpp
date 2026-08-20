#include "llvm_schedule_emitter.h"

#include <algorithm>

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

[[nodiscard]] std::optional<ScheduleEmitter::StructuredEarlyExitLoop>
ScheduleEmitter::_find_structured_early_exit_loop(
    const schedule::BasicBlock &header) const noexcept {
    auto force = luisa::compute::detail::env_flag(
        "LUISA_SIMD_FORCE_STRUCTURED_EARLY_EXIT_LOOP");
    if ((!force && _width != 8u) ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_STRUCTURED_EARLY_EXIT_LOOP")) {
        return std::nullopt;
    }

    const schedule::Loop *loop = nullptr;
    for (auto &&candidate : _source.loops()) {
        if (candidate.header == header.id) {
            if (loop != nullptr) { return std::nullopt; }
            loop = &candidate;
        }
    }
    if (loop == nullptr || !loop->max_trip_count ||
        *loop->max_trip_count == 0u ||
        *loop->max_trip_count > 16u ||
        (!force && loop->blocks.size() < 25u) ||
        (force && loop->blocks.size() < 4u) ||
        loop->blocks.size() > 64u) {
        return std::nullopt;
    }
    for (auto &&candidate : _source.loops()) {
        if (candidate.parent == loop->id) {
            return std::nullopt;
        }
    }

    auto *header_split = std::get_if<schedule::SplitTerminator>(
        &header.terminator);
    if (header_split == nullptr ||
        !header_split->cohort_uniform_condition ||
        !header_split->convergence) {
        return std::nullopt;
    }
    auto *loop_gate = _source.convergence(
        *header_split->convergence);
    if (loop_gate == nullptr || loop_gate->target == header.id ||
        std::find(loop->exits.cbegin(), loop->exits.cend(),
                  loop_gate->target) == loop->exits.cend()) {
        return std::nullopt;
    }

    auto value_count = _source.values().size();
    auto block_count = _source.blocks().size();
    std::vector<uint8_t> in_loop(block_count, uint8_t{0u});
    for (auto id : loop->blocks) {
        if (id.value >= block_count) { return std::nullopt; }
        in_loop[id.value] = 1u;
    }
    if (header.id.value >= in_loop.size() ||
        in_loop[header.id.value] == 0u) {
        return std::nullopt;
    }

    std::vector<std::vector<schedule::BlockId>> predecessors(block_count);
    auto valid_cfg_targets = true;
    auto add_predecessor = [&](schedule::BlockId source,
                               schedule::BlockId target) noexcept {
        if (target.value >= predecessors.size()) {
            valid_cfg_targets = false;
            return;
        }
        predecessors[target.value].emplace_back(source);
    };
    for (auto &&block : _source.blocks()) {
        std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    add_predecessor(block.id, control.edge.target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SplitTerminator>) {
                    add_predecessor(
                        block.id, control.true_edge.target);
                    add_predecessor(
                        block.id, control.false_edge.target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SwitchTerminator>) {
                    for (auto &&item : control.cases) {
                        add_predecessor(block.id, item.edge.target);
                    }
                    add_predecessor(
                        block.id, control.default_edge.target);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        control.convergence);
                    if (point == nullptr) {
                        valid_cfg_targets = false;
                    } else {
                        add_predecessor(block.id, point->target);
                    }
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    auto *target_loop = _source.loop(control.loop);
                    if (target_loop == nullptr) {
                        valid_cfg_targets = false;
                    } else {
                        add_predecessor(
                            block.id, target_loop->header);
                    }
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    add_predecessor(
                        block.id, control.resume_edge.target);
                }
            },
            block.terminator);
    }
    if (!valid_cfg_targets) { return std::nullopt; }
    for (auto id : loop->blocks) {
        if (id == header.id) { continue; }
        if (predecessors[id.value].empty()) { return std::nullopt; }
        if (std::any_of(
                predecessors[id.value].cbegin(),
                predecessors[id.value].cend(),
                [&](schedule::BlockId predecessor) noexcept {
                    return predecessor.value >= in_loop.size() ||
                           in_loop[predecessor.value] == 0u;
                })) {
            return std::nullopt;
        }
    }

    auto safe_instruction = [&](const schedule::Instruction &instruction) {
        if (!instruction.result || instruction.participant_mask ||
            !instruction.source_op) {
            return false;
        }
        if (instruction.opcode == schedule::Opcode::cast) {
            auto op = static_cast<xir::CastOp>(
                *instruction.source_op);
            return instruction.operands.size() == 1u &&
                   (op == xir::CastOp::STATIC_CAST ||
                    op == xir::CastOp::BITWISE_CAST);
        }
        if (instruction.opcode != schedule::Opcode::arithmetic) {
            return false;
        }
        auto op = static_cast<xir::ArithmeticOp>(
            *instruction.source_op);
        switch (op) {
            case xir::ArithmeticOp::BINARY_DIV: {
                auto *result = _source.value(*instruction.result);
                return result != nullptr && result->type != nullptr &&
                       result->type->is_float_or_float_vector();
            }
            case xir::ArithmeticOp::BINARY_MOD:
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: return false;
            default: return true;
        }
    };

    auto instruction_count = size_t{0u};
    for (auto id : loop->blocks) {
        auto *block = _source.block(id);
        if (block == nullptr) { return std::nullopt; }
        for (auto &&instruction : block->instructions) {
            if (!safe_instruction(instruction) ||
                ++instruction_count > 256u) {
                return std::nullopt;
            }
        }
    }

    const schedule::Instruction *header_condition = nullptr;
    for (auto &&instruction : header.instructions) {
        if (instruction.result &&
            *instruction.result == header_split->condition) {
            header_condition = &instruction;
            break;
        }
    }
    if (header_condition == nullptr ||
        header_condition->opcode != schedule::Opcode::arithmetic) {
        return std::nullopt;
    }
    auto induction = schedule::ValueId{};
    auto varying_operand_count = size_t{0u};
    for (auto operand : header_condition->operands) {
        auto *value = _source.value(operand);
        if (value == nullptr) { return std::nullopt; }
        if (value->value_class == schedule::ValueClass::varying) {
            if (value->origin != schedule::ValueOrigin::state_slot) {
                return std::nullopt;
            }
            induction = operand;
            varying_operand_count++;
        } else if (!schedule::is_uniform(value->value_class)) {
            return std::nullopt;
        }
    }
    if (varying_operand_count != 1u) { return std::nullopt; }

    std::vector<uint8_t> cohort_uniform(value_count, uint8_t{0u});
    for (auto &&value : _source.values()) {
        if (schedule::is_uniform(value.value_class)) {
            cohort_uniform[value.id.value] = 1u;
        }
    }
    cohort_uniform[induction.value] = 1u;
    auto changed = true;
    while (changed) {
        changed = false;
        for (auto id : loop->blocks) {
            auto *block = _source.block(id);
            for (auto &&instruction : block->instructions) {
                if (!instruction.result ||
                    instruction.result->value >= cohort_uniform.size() ||
                    cohort_uniform[instruction.result->value] != 0u ||
                    (instruction.opcode != schedule::Opcode::arithmetic &&
                     instruction.opcode != schedule::Opcode::cast)) {
                    continue;
                }
                auto all_uniform = std::all_of(
                    instruction.operands.cbegin(),
                    instruction.operands.cend(),
                    [&](schedule::ValueId operand) noexcept {
                        return operand.value < cohort_uniform.size() &&
                               cohort_uniform[operand.value] != 0u;
                    });
                if (all_uniform) {
                    cohort_uniform[instruction.result->value] = 1u;
                    changed = true;
                }
            }
        }
    }

    StructuredEarlyExitLoop result{
        .loop = loop,
        .header = &header,
        .common_exit = loop_gate->target,
        .induction = induction,
        .instruction_count = instruction_count,
    };
    for (auto i = size_t{0u}; i < cohort_uniform.size(); i++) {
        if (cohort_uniform[i] != 0u) {
            result.cohort_uniform_values.emplace_back(
                schedule::ValueId{static_cast<uint32_t>(i)});
        }
    }

    auto find_exit_tail = [&](schedule::BlockId entry)
        -> std::optional<StructuredEarlyExitLoop::ExitTail> {
        StructuredEarlyExitLoop::ExitTail tail{.entry = entry};
        auto target = entry;
        auto instruction_count = size_t{0u};
        std::vector<uint8_t> visited(block_count, uint8_t{0u});
        for (auto depth = size_t{0u}; target != result.common_exit;
             depth++) {
            if (depth >= 4u || target.value >= block_count ||
                in_loop[target.value] != 0u ||
                visited[target.value] != 0u) {
                return std::nullopt;
            }
            visited[target.value] = 1u;
            auto *block = _source.block(target);
            if (block == nullptr) { return std::nullopt; }
            for (auto &&instruction : block->instructions) {
                if (!safe_instruction(instruction) ||
                    ++instruction_count > 64u) {
                    return std::nullopt;
                }
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &block->terminator);
            if (branch == nullptr || branch->edge.loop_back ||
                branch->edge.target == block->id) {
                return std::nullopt;
            }
            tail.blocks.emplace_back(block);
            target = branch->edge.target;
        }
        return tail;
    };
    std::vector<uint8_t> absorbed(block_count, uint8_t{0u});
    for (auto exit : loop->exits) {
        auto tail = find_exit_tail(exit);
        if (!tail) { return std::nullopt; }
        for (auto *block : tail->blocks) {
            if (absorbed[block->id.value] != 0u) {
                return std::nullopt;
            }
            absorbed[block->id.value] = 1u;
            result.absorbed_blocks.emplace_back(block);
        }
        result.exit_tails.emplace_back(std::move(*tail));
    }
    for (auto &&tail : result.exit_tails) {
        if (tail.blocks.empty()) { continue; }
        auto &&entry_predecessors =
            predecessors[tail.blocks.front()->id.value];
        if (entry_predecessors.size() != 1u ||
            std::any_of(
                entry_predecessors.cbegin(),
                entry_predecessors.cend(),
                [&](schedule::BlockId predecessor) noexcept {
                    return predecessor.value >= in_loop.size() ||
                           in_loop[predecessor.value] == 0u;
                })) {
            return std::nullopt;
        }
        for (auto i = size_t{1u}; i < tail.blocks.size(); i++) {
            auto &&block_predecessors =
                predecessors[tail.blocks[i]->id.value];
            if (block_predecessors.size() != 1u ||
                block_predecessors.front() !=
                    tail.blocks[i - 1u]->id) {
                return std::nullopt;
            }
        }
    }

    auto is_loop_target = [&](schedule::BlockId target) noexcept {
        return target.value < in_loop.size() &&
               in_loop[target.value] != 0u;
    };
    auto has_exit_tail = [&](schedule::BlockId target) noexcept {
        return std::any_of(
            result.exit_tails.cbegin(), result.exit_tails.cend(),
            [&](const auto &tail) noexcept {
                return tail.entry == target;
            });
    };
    auto is_cohort_uniform = [&](schedule::ValueId value) noexcept {
        return value.value < cohort_uniform.size() &&
               cohort_uniform[value.value] != 0u;
    };
    auto valid_control = [&](const schedule::BasicBlock &block) noexcept {
        return std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    if (control.edge.loop_back) {
                        return *control.edge.loop_back == loop->id &&
                               control.edge.target == loop->header;
                    }
                    return is_loop_target(control.edge.target) ||
                           has_exit_tail(control.edge.target);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    return control.loop == loop->id;
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    auto true_inside = is_loop_target(
                        control.true_edge.target);
                    auto false_inside = is_loop_target(
                        control.false_edge.target);
                    auto true_exit = has_exit_tail(
                        control.true_edge.target);
                    auto false_exit = has_exit_tail(
                        control.false_edge.target);
                    auto *point = control.convergence ?
                                      _source.convergence(
                                          *control.convergence) :
                                      nullptr;
                    if (point == nullptr) { return false; }
                    if (true_inside && false_inside) {
                        return is_cohort_uniform(control.condition);
                    }
                    return point->target == result.common_exit &&
                           ((true_inside && false_exit) ||
                            (false_inside && true_exit));
                } else {
                    return false;
                }
            },
            block.terminator);
    };

    std::vector<uint8_t> locally_inlined(block_count, uint8_t{0u});
    for (auto id : loop->blocks) {
        if (locally_inlined[id.value] != 0u) { continue; }
        auto *block = _source.block(id);
        if (auto region = _find_chained_predicated_region(*block)) {
            for (auto *inlined : region->inlined_blocks) {
                if (!is_loop_target(inlined->id)) {
                    return std::nullopt;
                }
                locally_inlined[inlined->id.value] = 1u;
            }
            if (region->terminal_blocks.empty()) {
                if (!is_loop_target(region->merge)) {
                    return std::nullopt;
                }
            } else if (!valid_control(
                           *region->terminal_blocks.back())) {
                return std::nullopt;
            }
            continue;
        }
        if (auto region = _find_nested_predicated_region(*block)) {
            auto mark = [&](const schedule::BasicBlock *inlined) noexcept {
                if (inlined != nullptr &&
                    is_loop_target(inlined->id)) {
                    locally_inlined[inlined->id.value] = 1u;
                    return true;
                }
                return false;
            };
            if (!mark(region->nested_split_block) ||
                !mark(region->nested_merge_block) ||
                !mark(region->other_block) ||
                !std::all_of(
                    region->nested_diamond.true_blocks.cbegin(),
                    region->nested_diamond.true_blocks.cend(), mark) ||
                !std::all_of(
                    region->nested_diamond.false_blocks.cbegin(),
                    region->nested_diamond.false_blocks.cend(), mark) ||
                !is_loop_target(region->merge)) {
                return std::nullopt;
            }
            continue;
        }
        if (auto diamond =
                _find_guarded_predicated_math_diamond(*block)) {
            auto mark = [&](const schedule::BasicBlock *inlined) noexcept {
                if (inlined != nullptr &&
                    is_loop_target(inlined->id)) {
                    locally_inlined[inlined->id.value] = 1u;
                    return true;
                }
                return false;
            };
            if (!std::all_of(
                    diamond->true_blocks.cbegin(),
                    diamond->true_blocks.cend(), mark) ||
                !std::all_of(
                    diamond->false_blocks.cbegin(),
                    diamond->false_blocks.cend(), mark) ||
                !is_loop_target(diamond->merge)) {
                return std::nullopt;
            }
        }
    }

    for (auto id : loop->blocks) {
        if (id == header.id || locally_inlined[id.value] != 0u) {
            continue;
        }
        auto *block = _source.block(id);
        if (!valid_control(*block) &&
            !_find_chained_predicated_region(*block) &&
            !_find_nested_predicated_region(*block) &&
            !_find_guarded_predicated_math_diamond(*block)) {
            return std::nullopt;
        }
        result.emitted_blocks.emplace_back(block);
    }
    if (!valid_control(header)) { return std::nullopt; }
    return result;
}

void ScheduleEmitter::_emit_structured_early_exit_loop(
    const StructuredEarlyExitLoop &loop) {
    auto &context = _module.getContext();
    auto *initial_mask = _active_mask;
    auto *iteration_header = ::llvm::BasicBlock::Create(
        context, "structured.loop.header", _entry);
    _builder.CreateStore(initial_mask, _current_mask);
    _builder.CreateBr(iteration_header);

    auto in_loop = [&](schedule::BlockId target) noexcept {
        return std::find(
                   loop.loop->blocks.cbegin(),
                   loop.loop->blocks.cend(), target) !=
               loop.loop->blocks.cend();
    };
    auto tail_for = [&](schedule::BlockId entry) noexcept
        -> const StructuredEarlyExitLoop::ExitTail * {
        auto iter = std::find_if(
            loop.exit_tails.cbegin(), loop.exit_tails.cend(),
            [&](const auto &tail) noexcept {
                return tail.entry == entry;
            });
        return iter == loop.exit_tails.cend() ? nullptr : &*iter;
    };
    auto cohort_uniform = [&](schedule::ValueId value) noexcept {
        return std::find(
                   loop.cohort_uniform_values.cbegin(),
                   loop.cohort_uniform_values.cend(), value) !=
               loop.cohort_uniform_values.cend();
    };

    auto emit_exit_tail = [&](const schedule::ControlEdge &edge,
                              ::llvm::Value *mask) {
        auto *tail = tail_for(edge.target);
        if (tail == nullptr) {
            _fail("structured loop lost an audited exit tail");
            return;
        }
        _active_mask = mask;
        _seed_lane = _safe_first_lane(mask);
        if (_route_edge(edge, mask) == nullptr) { return; }
        for (auto *block : tail->blocks) {
            _locals.clear();
            _active_mask = mask;
            _seed_lane = _safe_first_lane(mask);
            for (auto &&instruction : block->instructions) {
                _emit_instruction(instruction, nullptr, mask);
                if (_failed()) { return; }
            }
            auto *branch = std::get_if<schedule::BranchTerminator>(
                &block->terminator);
            if (branch == nullptr ||
                _route_edge(branch->edge, mask) == nullptr) {
                _fail("structured loop exit tail lost its branch");
                return;
            }
        }
    };

    auto finish = [&]() {
        _active_mask = initial_mask;
        _seed_lane = _safe_first_lane(initial_mask);
        _continue_at(loop.common_exit, initial_mask);
    };

    std::function<void(const schedule::BasicBlock &)> emit_control;
    emit_control = [&](const schedule::BasicBlock &block) {
        auto *mask = _active_mask;
        auto outer_locals = _locals;
        std::visit(
            [&](const auto &control) {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    if (control.edge.loop_back) {
                        auto *flow = _route_edge(control.edge, mask);
                        if (flow != nullptr) {
                            _builder.CreateStore(flow, _current_mask);
                            _builder.CreateBr(iteration_header);
                        }
                    } else if (in_loop(control.edge.target)) {
                        auto *flow = _route_edge(control.edge, mask);
                        if (flow != nullptr) {
                            _continue_at(control.edge.target, flow);
                        }
                    } else {
                        emit_exit_tail(control.edge, mask);
                        if (!_failed()) { finish(); }
                    }
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::LoopBackTerminator>) {
                    schedule::ControlEdge edge{loop.loop->header};
                    edge.loop_back = control.loop;
                    edge.assignments = control.assignments;
                    auto *flow = _route_edge(edge, mask);
                    if (flow != nullptr) {
                        _builder.CreateStore(flow, _current_mask);
                        _builder.CreateBr(iteration_header);
                    }
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    auto *condition_value = _source.value(
                        control.condition);
                    auto *condition = _load_value(control.condition);
                    if (condition_value == nullptr || condition == nullptr) {
                        return;
                    }
                    ::llvm::Value *true_mask = nullptr;
                    ::llvm::Value *false_mask = nullptr;
                    if (condition_value->value_class ==
                        schedule::ValueClass::varying) {
                        auto *safe_condition = _builder.CreateSelect(
                            mask, condition, _zero_mask());
                        true_mask = _builder.CreateAnd(
                            mask, safe_condition);
                        false_mask = _builder.CreateAnd(
                            mask, _builder.CreateNot(safe_condition));
                    } else {
                        true_mask = _builder.CreateSelect(
                            condition, mask, _zero_mask());
                        false_mask = _builder.CreateSelect(
                            condition, _zero_mask(), mask);
                    }
                    auto true_inside = in_loop(
                        control.true_edge.target);
                    auto false_inside = in_loop(
                        control.false_edge.target);
                    if (true_inside && false_inside) {
                        if (!cohort_uniform(control.condition)) {
                            _fail("structured loop encountered a varying internal fork");
                            return;
                        }
                        auto *true_path = ::llvm::BasicBlock::Create(
                            context, "structured.loop.uniform.true", _entry);
                        auto *false_path = ::llvm::BasicBlock::Create(
                            context, "structured.loop.uniform.false", _entry);
                        auto *take_true =
                            condition_value->value_class ==
                                    schedule::ValueClass::varying ?
                                _builder.CreateOrReduce(true_mask) :
                                condition;
                        _builder.CreateCondBr(
                            take_true, true_path, false_path);
                        _builder.SetInsertPoint(true_path);
                        _locals = outer_locals;
                        _active_mask = mask;
                        auto *true_flow = _route_edge(
                            control.true_edge, mask);
                        if (true_flow != nullptr) {
                            _continue_at(
                                control.true_edge.target, true_flow);
                        }
                        _builder.SetInsertPoint(false_path);
                        _locals = std::move(outer_locals);
                        _active_mask = mask;
                        auto *false_flow = _route_edge(
                            control.false_edge, mask);
                        if (false_flow != nullptr) {
                            _continue_at(
                                control.false_edge.target, false_flow);
                        }
                        return;
                    }

                    auto *continue_mask = true_inside ?
                                              true_mask :
                                              false_mask;
                    auto *exit_mask = true_inside ?
                                          false_mask :
                                          true_mask;
                    auto &continue_edge = true_inside ?
                                              control.true_edge :
                                              control.false_edge;
                    auto &exit_edge = true_inside ?
                                          control.false_edge :
                                          control.true_edge;
                    auto *execute_exit = ::llvm::BasicBlock::Create(
                        context, "structured.loop.exit.execute", _entry);
                    auto *after_exit = ::llvm::BasicBlock::Create(
                        context, "structured.loop.exit.resume", _entry);
                    _builder.CreateCondBr(
                        _builder.CreateOrReduce(exit_mask),
                        execute_exit, after_exit);

                    _builder.SetInsertPoint(execute_exit);
                    _locals = outer_locals;
                    emit_exit_tail(exit_edge, exit_mask);
                    if (_failed()) { return; }
                    _builder.CreateBr(after_exit);

                    _builder.SetInsertPoint(after_exit);
                    _locals = std::move(outer_locals);
                    auto *continue_path = ::llvm::BasicBlock::Create(
                        context, "structured.loop.continue", _entry);
                    auto *finished_path = ::llvm::BasicBlock::Create(
                        context, "structured.loop.finished", _entry);
                    _builder.CreateCondBr(
                        _builder.CreateOrReduce(continue_mask),
                        continue_path, finished_path);

                    _builder.SetInsertPoint(continue_path);
                    _active_mask = continue_mask;
                    auto *flow = _route_edge(
                        continue_edge, continue_mask);
                    if (flow != nullptr) {
                        _continue_at(continue_edge.target, flow);
                    }

                    _builder.SetInsertPoint(finished_path);
                    finish();
                } else {
                    _fail("structured loop encountered an unaudited terminator");
                }
            },
            block.terminator);
    };

    auto emit_block = [&](const schedule::BasicBlock &block) {
        _locals.clear();
        _active_mask = _builder.CreateLoad(
            _layout.mask_type(), _current_mask);
        _seed_lane = _safe_first_lane(_active_mask);
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction, nullptr, _active_mask);
            if (_failed()) { return; }
        }
        if (auto region = _find_chained_predicated_region(block)) {
            auto *split = std::get_if<schedule::SplitTerminator>(
                &block.terminator);
            _emit_chained_predicated_region(
                *split, *region, false);
            if (_failed()) { return; }
            if (region->terminal_blocks.empty()) {
                _continue_at(region->merge, _active_mask);
            } else {
                emit_control(*region->terminal_blocks.back());
            }
            return;
        }
        if (auto region = _find_nested_predicated_region(block)) {
            auto *split = std::get_if<schedule::SplitTerminator>(
                &block.terminator);
            _emit_nested_predicated_region(*split, *region, false);
            if (!_failed()) {
                _continue_at(region->merge, _active_mask);
            }
            return;
        }
        if (auto diamond =
                _find_guarded_predicated_math_diamond(block)) {
            auto *split = std::get_if<schedule::SplitTerminator>(
                &block.terminator);
            _emit_guarded_predicated_math_diamond(
                *split, *diamond, false);
            if (!_failed()) {
                _continue_at(diamond->merge, _active_mask);
            }
            return;
        }
        emit_control(block);
    };

    _builder.SetInsertPoint(iteration_header);
    emit_block(*loop.header);
    if (_failed()) { return; }
    for (auto *block : loop.emitted_blocks) {
        _builder.SetInsertPoint(_schedule_blocks[block->id.value]);
        emit_block(*block);
        if (_failed()) { return; }
    }
    _result.structured_early_exit_loop_count++;
    _result.structured_early_exit_loop_block_count +=
        loop.loop->blocks.size();
    _result.structured_early_exit_loop_instruction_count +=
        loop.instruction_count;
    _result.structured_early_exit_loop_absorbed_block_count +=
        loop.absorbed_blocks.size();
}

}// namespace luisa::compute::simd::detail
