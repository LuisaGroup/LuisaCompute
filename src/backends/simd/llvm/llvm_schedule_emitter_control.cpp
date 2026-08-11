#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_emit_instruction(const schedule::Instruction &instruction) {
    ::llvm::Value *value = nullptr;
    switch (instruction.opcode) {
        case schedule::Opcode::alloca:
            value = _local_alloca(instruction);
            break;
        case schedule::Opcode::load:
            value = _local_load(instruction);
            break;
        case schedule::Opcode::store:
            _local_store(instruction);
            break;
        case schedule::Opcode::gep:
            value = _local_gep(instruction);
            break;
        case schedule::Opcode::arithmetic:
            value = _arithmetic(instruction);
            break;
        case schedule::Opcode::cast:
            value = _cast(instruction);
            break;
        case schedule::Opcode::resource_query:
            value = _resource_query(instruction);
            break;
        case schedule::Opcode::resource_read:
            value = _resource_read(instruction);
            break;
        case schedule::Opcode::resource_write:
            _resource_write(instruction);
            break;
        case schedule::Opcode::atomic:
            value = _atomic(instruction);
            break;
        case schedule::Opcode::warp_collective:
            value = _collective(instruction);
            break;
        default:
            _fail("unsupported Schedule IR instruction reached LLVM emission");
            return;
    }
    if (instruction.result && value != nullptr) {
        _locals.insert_or_assign(instruction.result->value, value);
        auto id = instruction.result->value;
        if (id < _spilled_instruction_values.size() &&
            _spilled_instruction_values[id] != 0u) {
            auto *schedule_value = _source.value(*instruction.result);
            auto *slot = _state_slots[id];
            if (_is_local_lvalue(*instruction.result)) {
                auto *old = _builder.CreateLoad(
                    slot->getAllocatedType(), slot);
                _builder.CreateStore(
                    _merge_local_handles(
                        value, old, _active_mask),
                    slot);
            } else if (schedule_value->value_class ==
                schedule::ValueClass::warp_uniform) {
                _builder.CreateStore(value, slot);
            } else {
                auto *lanes = schedule_value->value_class ==
                        schedule::ValueClass::token ?
                    _splat(value) :
                    _as_lane_vector(value, *schedule_value);
                if (lanes == nullptr) { return; }
                auto *old = _builder.CreateLoad(
                    slot->getAllocatedType(), slot);
                auto *merged = schedule_value->type == nullptr ?
                    _builder.CreateSelect(_active_mask, lanes, old) :
                    _masked_merge(
                        lanes, old, schedule_value->type,
                        _active_mask);
                if (merged != nullptr) {
                    _builder.CreateStore(merged, slot);
                }
            }
        }
    }
    if (!_collectives.succeeded()) { _fail(_collectives.error()); }
}

void ScheduleEmitter::_assign(schedule::EdgeAssignment assignment,
             ::llvm::Value *mask) {
    auto *destination = _source.value(assignment.destination);
    auto *source = _source.value(assignment.source);
    auto *value = _load_value(assignment.source);
    if (destination == nullptr || source == nullptr || value == nullptr) {
        return;
    }
    auto *slot = _state_slots[assignment.destination.value];
    if (_is_local_lvalue(assignment.destination)) {
        auto *old = _builder.CreateLoad(
            slot->getAllocatedType(), slot);
        _builder.CreateStore(
            _merge_local_handles(value, old, mask), slot);
        return;
    }
    if (destination->value_class == schedule::ValueClass::warp_uniform) {
        _builder.CreateStore(value, slot);
        return;
    }
    auto *lanes = _as_lane_vector(value, *source);
    if (lanes == nullptr) { return; }
    auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
    auto *merged = _masked_merge(lanes, old, destination->type, mask);
    if (merged != nullptr) { _builder.CreateStore(merged, slot); }
}

void ScheduleEmitter::_apply_assignments(
    const std::vector<schedule::EdgeAssignment> &assignments,
    ::llvm::Value *mask) {
    for (auto assignment : assignments) {
        _assign(assignment, mask);
        if (_failed()) { return; }
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_zero_mask() noexcept {
    return ::llvm::Constant::getNullValue(_layout.mask_type());
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_splat(::llvm::Value *scalar) {
    return scalar->getType()->isVectorTy() ? scalar :
                                            _builder.CreateVectorSplat(
                                                _width, scalar);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_safe_first_lane(::llvm::Value *mask) {
    auto *any = _builder.CreateOrReduce(mask);
    auto *first = _collectives.first_active_lane(_builder, mask);
    return _builder.CreateSelect(any, first, _builder.getInt32(0u));
}

void ScheduleEmitter::_masked_write(::llvm::AllocaInst *slot, ::llvm::Value *value,
                   ::llvm::Value *mask) {
    auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
    auto *lanes = _splat(value);
    _builder.CreateStore(_builder.CreateSelect(mask, lanes, old), slot);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_frame_mask_pointer(
    ::llvm::AllocaInst *frames, ::llvm::Value *index) {
    auto *array = ::llvm::cast<::llvm::ArrayType>(
        frames->getAllocatedType());
    return _builder.CreateInBoundsGEP(
        array, frames, {_builder.getInt32(0u), index});
}

void ScheduleEmitter::_trap_if(::llvm::Value *condition, std::string_view label) {
    auto *trap = ::llvm::BasicBlock::Create(
        _module.getContext(), std::string{label} + ".trap", _entry);
    auto *resume = ::llvm::BasicBlock::Create(
        _module.getContext(), std::string{label} + ".resume", _entry);
    _builder.CreateCondBr(condition, trap, resume);
    _builder.SetInsertPoint(trap);
#if LLVM_VERSION_MAJOR >= 22
    auto *intrinsic = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
    auto *intrinsic = ::llvm::Intrinsic::getDeclaration(
#endif
        &_module, ::llvm::Intrinsic::trap);
    _builder.CreateCall(intrinsic);
    _builder.CreateUnreachable();
    _builder.SetInsertPoint(resume);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_current_token(::llvm::Value *mask) {
    auto *tokens = _builder.CreateLoad(
        _token_state->getAllocatedType(), _token_state);
    return _builder.CreateExtractElement(
        tokens, _safe_first_lane(mask));
}

void ScheduleEmitter::_declare_convergence(schedule::ConvergenceId convergence,
                          ::llvm::Value *true_mask,
                          ::llvm::Value *false_mask) {
    auto *has_true = _builder.CreateOrReduce(true_mask);
    auto *has_false = _builder.CreateOrReduce(false_mask);
    auto *divergent = _builder.CreateAnd(has_true, has_false);
    auto *current_token = _current_token(_active_mask);
    auto *has_current = _builder.CreateICmpNE(
        current_token, _builder.getInt32(0u));
    auto *current_index = _builder.CreateSelect(
        has_current,
        _builder.CreateSub(current_token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active->getAllocatedType(), _frame_active);
    auto *current_active = _builder.CreateExtractElement(
        active_frames, current_index);
    auto *static_ids = _builder.CreateLoad(
        _frame_static_id->getAllocatedType(), _frame_static_id);
    auto *current_static = _builder.CreateExtractElement(
        static_ids, current_index);
    auto *reuse = _builder.CreateAnd(
        has_current,
        _builder.CreateAnd(
            current_active,
            _builder.CreateICmpEQ(
                current_static,
                _builder.getInt32(convergence.value))));
    auto *allocate = _builder.CreateAnd(divergent,
                                        _builder.CreateNot(reuse));
    auto *free_frames = _builder.CreateNot(active_frames);
    auto *has_free = _builder.CreateOrReduce(free_frames);
    _trap_if(_builder.CreateAnd(allocate, _builder.CreateNot(has_free)),
             "convergence.overflow");
    auto *free_index = _safe_first_lane(free_frames);

    auto *old_free_active = _builder.CreateExtractElement(
        active_frames, free_index);
    auto *new_free_active = _builder.CreateSelect(
        allocate, _builder.getTrue(), old_free_active);
    _builder.CreateStore(
        _builder.CreateInsertElement(
            active_frames, new_free_active, free_index),
        _frame_active);

    auto *old_static = _builder.CreateExtractElement(
        static_ids, free_index);
    auto *new_static = _builder.CreateSelect(
        allocate, _builder.getInt32(convergence.value), old_static);
    _builder.CreateStore(
        _builder.CreateInsertElement(
            static_ids, new_static, free_index),
        _frame_static_id);

    auto *parents = _builder.CreateLoad(
        _frame_parent_token->getAllocatedType(), _frame_parent_token);
    auto *old_parent = _builder.CreateExtractElement(parents, free_index);
    auto *new_parent = _builder.CreateSelect(
        allocate, current_token, old_parent);
    _builder.CreateStore(
        _builder.CreateInsertElement(parents, new_parent, free_index),
        _frame_parent_token);

    auto *expected_ptr = _frame_mask_pointer(
        _frame_expected, free_index);
    auto *arrived_ptr = _frame_mask_pointer(
        _frame_arrived, free_index);
    auto *old_expected = _builder.CreateLoad(
        _layout.mask_type(), expected_ptr);
    auto *old_arrived = _builder.CreateLoad(
        _layout.mask_type(), arrived_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(allocate, _active_mask, old_expected),
        expected_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(allocate, _zero_mask(), old_arrived),
        arrived_ptr);

    auto *allocated_token = _builder.CreateAdd(
        free_index, _builder.getInt32(1u));
    auto *gate_token = _builder.CreateSelect(
        reuse, current_token, allocated_token);
    auto *next_token = _builder.CreateSelect(
        divergent, gate_token, current_token);
    _masked_write(_token_state, next_token, _active_mask);
}

void ScheduleEmitter::_advance_loop_epoch(schedule::LoopId loop, ::llvm::Value *mask) {
    auto *slot = _loop_epochs[loop.value];
    auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
    auto *one = _builder.CreateVectorSplat(
        _width, _builder.getInt32(1u));
    _builder.CreateStore(
        _builder.CreateSelect(mask, _builder.CreateAdd(old, one), old),
        slot);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_arrive_at_convergence(
    schedule::ConvergenceId convergence, ::llvm::Value *flow) {
    auto *any = _builder.CreateOrReduce(flow);
    auto *token = _current_token(flow);
    auto *has_token = _builder.CreateAnd(
        any, _builder.CreateICmpNE(token, _builder.getInt32(0u)));
    auto *index = _builder.CreateSelect(
        has_token,
        _builder.CreateSub(token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active->getAllocatedType(), _frame_active);
    auto *frame_active = _builder.CreateExtractElement(
        active_frames, index);
    auto *static_ids = _builder.CreateLoad(
        _frame_static_id->getAllocatedType(), _frame_static_id);
    auto *static_id = _builder.CreateExtractElement(static_ids, index);
    auto *matches = _builder.CreateAnd(
        has_token,
        _builder.CreateAnd(
            frame_active,
            _builder.CreateICmpEQ(
                static_id,
                _builder.getInt32(convergence.value))));

    auto *expected_ptr = _frame_mask_pointer(_frame_expected, index);
    auto *arrived_ptr = _frame_mask_pointer(_frame_arrived, index);
    auto *expected = _builder.CreateLoad(
        _layout.mask_type(), expected_ptr);
    auto *arrived = _builder.CreateLoad(
        _layout.mask_type(), arrived_ptr);
    auto *new_arrived = _builder.CreateOr(arrived, flow);
    auto *live = _builder.CreateLoad(
        _live_mask->getAllocatedType(), _live_mask);
    auto *expected_live = _builder.CreateAnd(expected, live);
    auto *complete = _builder.CreateAnd(
        matches,
        _builder.CreateAndReduce(
            _builder.CreateICmpEQ(new_arrived, expected_live)));
    auto *stored_arrived = _builder.CreateSelect(
        matches,
        _builder.CreateSelect(complete, _zero_mask(), new_arrived),
        arrived);
    _builder.CreateStore(stored_arrived, arrived_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(complete, _zero_mask(), expected),
        expected_ptr);

    auto *old_frame_active = _builder.CreateExtractElement(
        active_frames, index);
    _builder.CreateStore(
        _builder.CreateInsertElement(
            active_frames,
            _builder.CreateSelect(
                complete, _builder.getFalse(), old_frame_active),
            index),
        _frame_active);

    auto *matching_lanes = _builder.CreateAnd(flow, _splat(matches));
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask);
    _builder.CreateStore(
        _builder.CreateAnd(runnable,
                           _builder.CreateNot(matching_lanes)),
        _runnable_mask);

    auto *released = _builder.CreateSelect(
        _splat(complete), new_arrived, _zero_mask());
    auto *parents = _builder.CreateLoad(
        _frame_parent_token->getAllocatedType(), _frame_parent_token);
    auto *parent_token = _builder.CreateExtractElement(parents, index);
    _masked_write(_token_state, parent_token, released);
    return _builder.CreateSelect(_splat(matches), released, flow);
}

void ScheduleEmitter::_resume(schedule::BlockId target, ::llvm::Value *mask) {
    _masked_write(_pc_state, _builder.getInt32(target.value), mask);
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask);
    _builder.CreateStore(_builder.CreateOr(runnable, mask),
                         _runnable_mask);
}

void ScheduleEmitter::_route_edge(const schedule::ControlEdge &edge,
                 ::llvm::Value *mask) {
    _apply_assignments(edge.assignments, mask);
    if (_failed()) { return; }
    if (edge.loop_back) {
        _advance_loop_epoch(*edge.loop_back, mask);
    }
    auto *flow = mask;
    for (auto convergence : edge.joins) {
        flow = _arrive_at_convergence(convergence, flow);
    }
    _resume(edge.target, flow);
}

void ScheduleEmitter::_emit_arrival(const schedule::ControlEdge &edge,
                   ::llvm::Value *mask) {
    _route_edge(edge, mask);
    _builder.CreateBr(_scheduler_loop);
}

void ScheduleEmitter::_emit_terminator(const schedule::Terminator &terminator) {
    std::visit(
        [&](const auto &control) {
            using T = std::decay_t<decltype(control)>;
            if constexpr (std::is_same_v<T, schedule::BranchTerminator>) {
                _emit_arrival(control.edge, _active_mask);
            } else if constexpr (
                std::is_same_v<T, schedule::SplitTerminator>) {
                auto *condition_value = _source.value(control.condition);
                auto *condition = _load_value(control.condition);
                if (condition == nullptr || condition_value == nullptr) {
                    return;
                }
                if (condition_value->value_class ==
                    schedule::ValueClass::varying) {
                    auto *true_mask = _builder.CreateAnd(
                        _active_mask, condition);
                    auto *false_mask = _builder.CreateAnd(
                        _active_mask, _builder.CreateNot(condition));
                    if (control.convergence) {
                        _declare_convergence(
                            *control.convergence,
                            true_mask, false_mask);
                    }
                    _route_edge(control.true_edge, true_mask);
                    _route_edge(control.false_edge, false_mask);
                    if (_failed()) { return; }
                    _builder.CreateBr(_scheduler_loop);
                } else {
                    auto *true_path = ::llvm::BasicBlock::Create(
                        _module.getContext(), "uniform.true", _entry);
                    auto *false_path = ::llvm::BasicBlock::Create(
                        _module.getContext(), "uniform.false", _entry);
                    _builder.CreateCondBr(
                        condition, true_path, false_path);
                    _builder.SetInsertPoint(true_path);
                    _emit_arrival(control.true_edge, _active_mask);
                    _builder.SetInsertPoint(false_path);
                    _emit_arrival(control.false_edge, _active_mask);
                }
            } else if constexpr (
                std::is_same_v<T, schedule::JoinTerminator>) {
                auto *point = _source.convergence(control.convergence);
                schedule::ControlEdge edge{point->target};
                edge.joins.emplace_back(control.convergence);
                edge.assignments = control.assignments;
                _emit_arrival(edge, _active_mask);
            } else if constexpr (
                std::is_same_v<T, schedule::ReturnTerminator>) {
                if (control.value) {
                    auto *schedule_value = _source.value(*control.value);
                    auto *value = _as_lane_vector(
                        _load_value(*control.value), *schedule_value);
                    if (value != nullptr) {
                        _builder.CreateMaskedStore(
                            value, _return_buffer,
                            ::llvm::Align{schedule_value->type->alignment()},
                            _active_mask);
                    }
                }
                auto *live = _builder.CreateLoad(
                    _live_mask->getAllocatedType(), _live_mask);
                auto *new_live = _builder.CreateAnd(
                    live, _builder.CreateNot(_active_mask));
                _builder.CreateStore(new_live, _live_mask);
                auto *runnable = _builder.CreateLoad(
                    _runnable_mask->getAllocatedType(),
                    _runnable_mask);
                _builder.CreateStore(
                    _builder.CreateAnd(
                        runnable, _builder.CreateNot(_active_mask)),
                    _runnable_mask);

                // A lane that terminates is removed from every frame's
                // expected mask. Frame storage is bounded by W rather
                // than the number of static CFG convergence points.
                std::vector<::llvm::Constant *> targets;
                targets.reserve(_source.convergence_points().size());
                for (auto &&point : _source.convergence_points()) {
                    targets.emplace_back(
                        _builder.getInt32(point.target.value));
                }
                auto *target_table = targets.empty() ? nullptr :
                    ::llvm::ConstantVector::get(targets);
                for (auto frame = uint32_t{0u}; frame < _width; frame++) {
                    auto *index = _builder.getInt32(frame);
                    auto *active_frames = _builder.CreateLoad(
                        _frame_active->getAllocatedType(),
                        _frame_active);
                    auto *frame_active = _builder.CreateExtractElement(
                        active_frames, index);
                    auto *expected_ptr = _frame_mask_pointer(
                        _frame_expected, index);
                    auto *arrived_ptr = _frame_mask_pointer(
                        _frame_arrived, index);
                    auto *expected = _builder.CreateLoad(
                        _layout.mask_type(), expected_ptr);
                    auto *arrived = _builder.CreateLoad(
                        _layout.mask_type(), arrived_ptr);
                    auto *expected_live = _builder.CreateAnd(
                        expected, new_live);
                    auto *complete = _builder.CreateAnd(
                        frame_active,
                        _builder.CreateAndReduce(
                            _builder.CreateICmpEQ(
                                arrived, expected_live)));
                    auto *released = _builder.CreateSelect(
                        _splat(complete), arrived, _zero_mask());
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, _zero_mask(), expected_live),
                        expected_ptr);
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, _zero_mask(), arrived),
                        arrived_ptr);
                    _builder.CreateStore(
                        _builder.CreateInsertElement(
                            active_frames,
                            _builder.CreateSelect(
                                complete, _builder.getFalse(),
                                frame_active),
                            index),
                        _frame_active);

                    auto *parents = _builder.CreateLoad(
                        _frame_parent_token->getAllocatedType(),
                        _frame_parent_token);
                    auto *parent = _builder.CreateExtractElement(
                        parents, index);
                    _masked_write(_token_state, parent, released);
                    auto *static_ids = _builder.CreateLoad(
                        _frame_static_id->getAllocatedType(),
                        _frame_static_id);
                    auto *static_id = _builder.CreateExtractElement(
                        static_ids, index);
                    if (target_table != nullptr) {
                        auto *target = _builder.CreateExtractElement(
                            target_table, static_id);
                        _masked_write(_pc_state, target, released);
                    }
                    auto *current_runnable = _builder.CreateLoad(
                        _runnable_mask->getAllocatedType(),
                        _runnable_mask);
                    _builder.CreateStore(
                        _builder.CreateOr(
                            current_runnable, released),
                        _runnable_mask);
                }
                _builder.CreateBr(_scheduler_loop);
            } else if constexpr (
                std::is_same_v<T, schedule::UnreachableTerminator>) {
                _builder.CreateUnreachable();
            } else {
                _fail("unsupported Schedule IR terminator reached LLVM emission");
            }
        },
        terminator);
}

void ScheduleEmitter::_find_instruction_spills() {
    _spilled_instruction_values.assign(
        _source.values().size(), uint8_t{0u});
    auto mark = [&](schedule::ValueId id,
                    schedule::BlockId use_block) noexcept {
        auto *value = _source.value(id);
        if (value != nullptr &&
            value->origin == schedule::ValueOrigin::instruction &&
            value->defining_block &&
            *value->defining_block != use_block) {
            _spilled_instruction_values[id.value] = 1u;
        }
    };
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            for (auto operand : instruction.operands) {
                mark(operand, block.id);
            }
            if (instruction.participant_mask) {
                mark(*instruction.participant_mask, block.id);
            }
        }
        auto mark_assignments = [&](const auto &assignments) noexcept {
            for (auto assignment : assignments) {
                mark(assignment.source, block.id);
            }
        };
        auto mark_edge = [&](const schedule::ControlEdge &edge) noexcept {
            mark_assignments(edge.assignments);
        };
        std::visit(
            [&](const auto &terminator) noexcept {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<
                                  T, schedule::BranchTerminator>) {
                    mark_edge(terminator.edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SplitTerminator>) {
                    mark(terminator.condition, block.id);
                    mark_edge(terminator.true_edge);
                    mark_edge(terminator.false_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::SwitchTerminator>) {
                    mark(terminator.selector, block.id);
                    for (auto &&item : terminator.cases) {
                        mark_edge(item.edge);
                    }
                    mark_edge(terminator.default_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::JoinTerminator> ||
                                     std::is_same_v<
                                         T, schedule::LoopBackTerminator>) {
                    mark_assignments(terminator.assignments);
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::BlockBarrierTerminator>) {
                    mark_edge(terminator.resume_edge);
                } else if constexpr (std::is_same_v<
                                         T, schedule::ReturnTerminator>) {
                    if (terminator.value) {
                        mark(*terminator.value, block.id);
                    }
                }
            },
            block.terminator);
    }
}

void ScheduleEmitter::_allocate_state() {
    auto *mask_type = _layout.mask_type();
    auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *zero_i32_lanes = ::llvm::Constant::getNullValue(i32_lanes);
    _live_mask = _builder.CreateAlloca(
        mask_type, nullptr, "live.mask");
    _runnable_mask = _builder.CreateAlloca(
        mask_type, nullptr, "runnable.mask");
    _pc_state = _builder.CreateAlloca(
        i32_lanes, nullptr, "lane.pc");
    _token_state = _builder.CreateAlloca(
        i32_lanes, nullptr, "lane.convergence.token");
    _frame_active = _builder.CreateAlloca(
        mask_type, nullptr, "frame.active");
    _frame_static_id = _builder.CreateAlloca(
        i32_lanes, nullptr, "frame.static.id");
    _frame_parent_token = _builder.CreateAlloca(
        i32_lanes, nullptr, "frame.parent.token");
    auto *frame_masks = ::llvm::ArrayType::get(mask_type, _width);
    _frame_expected = _builder.CreateAlloca(
        frame_masks, nullptr, "frame.expected");
    _frame_arrived = _builder.CreateAlloca(
        frame_masks, nullptr, "frame.arrived");
    _builder.CreateStore(zero_mask, _live_mask);
    _builder.CreateStore(zero_mask, _runnable_mask);
    _builder.CreateStore(zero_i32_lanes, _pc_state);
    _builder.CreateStore(zero_i32_lanes, _token_state);
    _builder.CreateStore(zero_mask, _frame_active);
    _builder.CreateStore(zero_i32_lanes, _frame_static_id);
    _builder.CreateStore(zero_i32_lanes, _frame_parent_token);
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(frame_masks), _frame_expected);
    _builder.CreateStore(
        ::llvm::Constant::getNullValue(frame_masks), _frame_arrived);

    _loop_epochs.resize(_source.loops().size());
    _block_loops.resize(_source.blocks().size());
    for (auto &&loop : _source.loops()) {
        auto *epoch = _builder.CreateAlloca(
            i32_lanes, nullptr,
            "loop.epoch." + std::to_string(loop.id.value));
        _builder.CreateStore(zero_i32_lanes, epoch);
        _loop_epochs[loop.id.value] = epoch;
        for (auto block : loop.blocks) {
            _block_loops[block.value].emplace_back(loop.id);
        }
    }
    _find_instruction_spills();
    _local_allocations.resize(_source.values().size(), nullptr);
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::alloca ||
                !instruction.result) {
                continue;
            }
            auto *value = _source.value(*instruction.result);
            if (value == nullptr || value->type == nullptr ||
                !_is_local_lvalue(*instruction.result) ||
                value->type->size() == 0u) {
                _fail("thread-local allocation has an invalid data type");
                return;
            }
            auto byte_count = static_cast<uint64_t>(_width) *
                              value->type->size();
            auto *storage_type = ::llvm::ArrayType::get(
                _builder.getInt8Ty(), byte_count);
            auto *storage = _builder.CreateAlloca(
                storage_type, nullptr, value->name + ".local");
            storage->setAlignment(
                ::llvm::Align{value->type->alignment()});
            auto *offsets = _lane_offsets(
                _lane_ids(), value->type->size());
            _local_allocations[instruction.result->value] =
                _local_handle(
                    _builder.CreateVectorSplat(_width, storage),
                    offsets);
        }
    }
    _state_slots.resize(_source.values().size(), nullptr);
    for (auto &&value : _source.values()) {
        auto spill = value.id.value <
                         _spilled_instruction_values.size() &&
                     _spilled_instruction_values[value.id.value] != 0u;
        if (value.origin != schedule::ValueOrigin::state_slot &&
            !spill) {
            continue;
        }
        auto *type = _is_local_lvalue(value.id) ?
            static_cast<::llvm::Type *>(_local_handle_type()) :
            _layout.state_type(value);
        if (type == nullptr) {
            _fail(_layout.error());
            return;
        }
        auto *slot = _builder.CreateAlloca(
            type, nullptr,
            value.name + (spill ? ".spill" : ".slot"));
        _builder.CreateStore(::llvm::Constant::getNullValue(type), slot);
        _state_slots[value.id.value] = slot;
    }
}

void ScheduleEmitter::_build() {
    auto &context = _module.getContext();
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context),
        {::llvm::PointerType::getUnqual(context),
         ::llvm::PointerType::getUnqual(context),
         ::llvm::PointerType::getUnqual(context),
         ::llvm::Type::getInt32Ty(context)},
        false);
    if (_entry_name.empty()) {
        _entry_name = _source.name().empty() ? "simd_kernel" :
                                              _source.name();
        _entry_name += ".simd_w" + std::to_string(_width);
    }
    if (_module.getFunction(_entry_name) != nullptr) {
        _fail("LLVM module already contains the requested SIMD entry name");
        return;
    }
    _entry = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::ExternalLinkage,
        _entry_name, _module);
    auto argument = _entry->arg_begin();
    _argument_buffer = &*argument++;
    _argument_buffer->setName("argument_buffer");
    _return_buffer = &*argument++;
    _return_buffer->setName("return_lanes");
    _launch_config = &*argument++;
    _launch_config->setName("launch_config");
    _active_lane_count = &*argument;
    _active_lane_count->setName("active_lane_count");

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "prologue", _entry);
    _builder.SetInsertPoint(prologue);
    _allocate_state();
    if (_failed()) { return; }
    _create_external_values();
    if (_failed()) { return; }

    auto *lane_ids = _lane_ids();
    auto *count = _builder.CreateVectorSplat(
        _width, _active_lane_count);
    auto *initial_mask = _builder.CreateICmpULT(lane_ids, count);
    _ensure_launch_vectors();
    for (auto i = uint32_t{0u}; i < 3u; i++) {
        initial_mask = _builder.CreateAnd(
            initial_mask,
            _builder.CreateICmpULT(
                _dispatch_id[i],
                _builder.CreateVectorSplat(
                    _width, _dispatch_size[i])));
    }
    _builder.CreateStore(initial_mask, _live_mask);
    _builder.CreateStore(initial_mask, _runnable_mask);
    _builder.CreateStore(
        _builder.CreateVectorSplat(
            _width, _builder.getInt32(_source.entry().value)),
        _pc_state);

    _scheduler_loop = ::llvm::BasicBlock::Create(
        context, "scheduler.loop", _entry);
    auto *dispatch = ::llvm::BasicBlock::Create(
        context, "scheduler.dispatch", _entry);
    auto *done = ::llvm::BasicBlock::Create(
        context, "scheduler.done", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "scheduler.exit", _entry);
    auto *stalled = ::llvm::BasicBlock::Create(
        context, "scheduler.stalled", _entry);
    auto *invalid = ::llvm::BasicBlock::Create(
        context, "scheduler.invalid", _entry);
    std::vector<::llvm::BasicBlock *> cases;
    cases.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        cases.emplace_back(::llvm::BasicBlock::Create(
            context,
            "schedule." + std::to_string(block.id.value), _entry));
    }
    _builder.CreateBr(_scheduler_loop);

    _builder.SetInsertPoint(_scheduler_loop);
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask);
    _builder.CreateCondBr(
        _builder.CreateOrReduce(runnable),
        dispatch, done);

    _builder.SetInsertPoint(dispatch);
    _seed_lane = _safe_first_lane(runnable);
    auto *pcs = _builder.CreateLoad(
        _pc_state->getAllocatedType(), _pc_state);
    auto *pc = _builder.CreateExtractElement(pcs, _seed_lane);
    auto *dispatch_switch = _builder.CreateSwitch(
        pc,
        invalid, static_cast<unsigned>(cases.size()));
    for (auto &&block : _source.blocks()) {
        dispatch_switch->addCase(
            _builder.getInt32(block.id.value), cases[block.id.value]);
    }

    _builder.SetInsertPoint(invalid);
    auto *invalid_trap =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::trap);
    _builder.CreateCall(invalid_trap);
    _builder.CreateUnreachable();

    _builder.SetInsertPoint(done);
    auto *live = _builder.CreateLoad(
        _live_mask->getAllocatedType(), _live_mask);
    _builder.CreateCondBr(
        _builder.CreateOrReduce(live), stalled, exit);

    _builder.SetInsertPoint(stalled);
    auto *stalled_trap =
#if LLVM_VERSION_MAJOR >= 22
        ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::trap);
    _builder.CreateCall(stalled_trap);
    _builder.CreateUnreachable();

    _builder.SetInsertPoint(exit);
    _builder.CreateRetVoid();

    for (auto &&block : _source.blocks()) {
        _builder.SetInsertPoint(cases[block.id.value]);
        _locals.clear();
        auto *current_runnable = _builder.CreateLoad(
            _runnable_mask->getAllocatedType(), _runnable_mask);
        auto *current_pcs = _builder.CreateLoad(
            _pc_state->getAllocatedType(), _pc_state);
        auto *current_tokens = _builder.CreateLoad(
            _token_state->getAllocatedType(), _token_state);
        auto *seed_token = _builder.CreateExtractElement(
            current_tokens, _seed_lane);
        _active_mask = _builder.CreateAnd(
            current_runnable,
            _builder.CreateAnd(
                _builder.CreateICmpEQ(
                    current_pcs,
                    _builder.CreateVectorSplat(
                        _width,
                        _builder.getInt32(block.id.value))),
                _builder.CreateICmpEQ(
                    current_tokens,
                    _builder.CreateVectorSplat(
                        _width, seed_token))));
        for (auto loop : _block_loops[block.id.value]) {
            auto *epochs = _builder.CreateLoad(
                _loop_epochs[loop.value]->getAllocatedType(),
                _loop_epochs[loop.value]);
            auto *seed_epoch = _builder.CreateExtractElement(
                epochs, _seed_lane);
            _active_mask = _builder.CreateAnd(
                _active_mask,
                _builder.CreateICmpEQ(
                    epochs,
                    _builder.CreateVectorSplat(
                        _width, seed_epoch)));
        }
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        _emit_terminator(block.terminator);
        if (_failed()) { return; }
    }
}

}// namespace luisa::compute::simd::detail
