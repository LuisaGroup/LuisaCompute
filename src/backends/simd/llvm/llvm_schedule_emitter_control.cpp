#include "llvm_schedule_emitter.h"

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_emit_instruction(
    const schedule::Instruction &instruction,
    ::llvm::Value *lane_affine_seed,
    ::llvm::Value *operand_sanitization_mask) {
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
        case schedule::Opcode::ray_query_read:
            value = _ray_query_read(instruction);
            break;
        case schedule::Opcode::ray_query_write:
            _ray_query_write(instruction);
            break;
        case schedule::Opcode::resource_read:
            value = _resource_read(
                instruction, lane_affine_seed,
                operand_sanitization_mask);
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
                           schedule::ValueClass::warp_uniform ||
                       (_direct_control_flow &&
                        schedule_value->value_class ==
                            schedule::ValueClass::cohort_uniform)) {
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
    if (destination->value_class == schedule::ValueClass::warp_uniform ||
        (_direct_control_flow &&
         destination->value_class ==
             schedule::ValueClass::cohort_uniform)) {
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

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_frame_bit(
    ::llvm::Value *index) {
    auto *bits_type = ::llvm::cast<::llvm::IntegerType>(
        _frame_active->getAllocatedType());
    auto *bit_index = _builder.CreateZExtOrTrunc(index, bits_type);
    return _builder.CreateShl(
        ::llvm::ConstantInt::get(bits_type, 1u), bit_index);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_frame_is_active(
    ::llvm::Value *active_bits, ::llvm::Value *index) {
    return _builder.CreateICmpNE(
        _builder.CreateAnd(active_bits, _frame_bit(index)),
        ::llvm::Constant::getNullValue(active_bits->getType()));
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_frame_mask_pointer(
    ::llvm::AllocaInst *frames, ::llvm::Value *index) {
    auto *array = ::llvm::cast<::llvm::ArrayType>(
        frames->getAllocatedType());
    return _builder.CreateInBoundsGEP(
        array, frames, {_builder.getInt32(0u), index});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_convergence_target(
    ::llvm::Value *static_id) {
    if (auto *global = ::llvm::dyn_cast<::llvm::GlobalVariable>(
            _convergence_targets)) {
        auto *array = ::llvm::cast<::llvm::ArrayType>(
            global->getValueType());
        auto *pointer = _builder.CreateInBoundsGEP(
            array, global, {_builder.getInt32(0u), static_id},
            "convergence.target.pointer");
        return _builder.CreateLoad(
            _builder.getInt32Ty(), pointer,
            "convergence.dynamic.target");
    }
    return _builder.CreateExtractElement(
        _convergence_targets, static_id,
        "convergence.dynamic.target");
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_ready_element_pointer(
    ::llvm::AllocaInst *array, ::llvm::Value *index) {
    auto *array_type = ::llvm::cast<::llvm::ArrayType>(
        array->getAllocatedType());
    return _builder.CreateInBoundsGEP(
        array_type, array,
        {_builder.getInt32(0u), index});
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

void ScheduleEmitter::_declare_convergence(
    schedule::ConvergenceId convergence, ::llvm::Value *divergent) {
    auto *current_token = _builder.CreateLoad(
        _current_token->getAllocatedType(), _current_token);
    auto *has_current = _builder.CreateICmpNE(
        current_token, _builder.getInt32(0u));
    auto *current_index = _builder.CreateSelect(
        has_current,
        _builder.CreateSub(current_token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active->getAllocatedType(), _frame_active);
    auto *current_active = _frame_is_active(
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
    auto *has_free = _builder.CreateICmpNE(
        free_frames,
        ::llvm::Constant::getNullValue(free_frames->getType()));
    _trap_if(_builder.CreateAnd(allocate, _builder.CreateNot(has_free)),
             "convergence.overflow");
    auto *raw_free_index = _builder.CreateBinaryIntrinsic(
        ::llvm::Intrinsic::cttz, free_frames, _builder.getFalse());
    auto *free_index = _builder.CreateSelect(
        has_free,
        _builder.CreateZExtOrTrunc(
            raw_free_index, _builder.getInt32Ty()),
        _builder.getInt32(0u));
    auto *allocated_frames = _builder.CreateOr(
        active_frames, _frame_bit(free_index));
    _builder.CreateStore(
        _builder.CreateSelect(
            allocate, allocated_frames, active_frames),
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
    _builder.CreateStore(next_token, _current_token);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_arrive_at_convergence_target(
    ::llvm::Value *target, ::llvm::Value *flow,
    ::llvm::Value **matched) {
    if (_convergence_targets == nullptr) { return flow; }
    auto *any = _builder.CreateOrReduce(flow);
    auto *token = _builder.CreateLoad(
        _current_token->getAllocatedType(), _current_token);
    auto *has_token = _builder.CreateAnd(
        any, _builder.CreateICmpNE(token, _builder.getInt32(0u)));
    auto *index = _builder.CreateSelect(
        has_token,
        _builder.CreateSub(token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active->getAllocatedType(), _frame_active);
    auto *frame_active = _frame_is_active(active_frames, index);
    auto *static_ids = _builder.CreateLoad(
        _frame_static_id->getAllocatedType(), _frame_static_id);
    auto *static_id = _builder.CreateExtractElement(static_ids, index);
    auto *dynamic_target = _load_convergence_target(static_id);
    auto *matches = _builder.CreateAnd(
        has_token,
        _builder.CreateAnd(
            frame_active,
            _builder.CreateICmpEQ(dynamic_target, target)));
    *matched = matches;

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

    auto *released_frames = _builder.CreateAnd(
        active_frames, _builder.CreateNot(_frame_bit(index)));
    _builder.CreateStore(
        _builder.CreateSelect(
            complete, released_frames, active_frames),
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
    _builder.CreateStore(
        _builder.CreateSelect(complete, parent_token, token),
        _current_token);
    return _builder.CreateSelect(_splat(matches), released, flow);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_cascade_at_convergence_target(
    ::llvm::Value *target, ::llvm::Value *flow) {
    if (_convergence_targets == nullptr) { return flow; }
    auto *preheader = _builder.GetInsertBlock();
    auto *loop = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade.exit", _entry);
    _builder.CreateBr(loop);
    _builder.SetInsertPoint(loop);
    auto *current_flow = _builder.CreatePHI(
        _layout.mask_type(), 2u, "convergence.cascade.flow");
    auto *depth = _builder.CreatePHI(
        _builder.getInt32Ty(), 2u, "convergence.cascade.depth");
    current_flow->addIncoming(flow, preheader);
    depth->addIncoming(_builder.getInt32(0u), preheader);
    ::llvm::Value *matched = nullptr;
    auto *next_flow = _arrive_at_convergence_target(
        target, current_flow, &matched);
    auto *next_depth = _builder.CreateAdd(
        depth, _builder.getInt32(1u));
    auto *more = _builder.CreateAnd(
        matched,
        _builder.CreateAnd(
            _builder.CreateOrReduce(next_flow),
            _builder.CreateICmpULT(
                next_depth, _builder.getInt32(_width))));
    auto *latch = _builder.GetInsertBlock();
    _builder.CreateCondBr(more, loop, exit);
    current_flow->addIncoming(next_flow, latch);
    depth->addIncoming(next_depth, latch);
    _builder.SetInsertPoint(exit);
    return next_flow;
}

void ScheduleEmitter::_resume(
    ::llvm::Value *target, ::llvm::Value *mask,
    ::llvm::Value *token) {
    auto *nonempty = _builder.CreateOrReduce(mask);
    auto *count = _builder.CreateLoad(
        _ready_count->getAllocatedType(), _ready_count);
    _trap_if(
        _builder.CreateAnd(
            nonempty,
            _builder.CreateICmpUGE(
                count, _builder.getInt32(_width))),
        "ready.overflow");
    auto *index = _builder.CreateSelect(
        nonempty, count, _builder.getInt32(0u));
    auto *mask_ptr = _ready_element_pointer(
        _ready_masks, index);
    auto *old_mask = _builder.CreateLoad(
        _layout.mask_type(), mask_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(nonempty, mask, old_mask),
        mask_ptr);
    auto *target_ptr = _ready_element_pointer(
        _ready_targets, index);
    auto *old_target = _builder.CreateLoad(
        _builder.getInt32Ty(), target_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(nonempty, target, old_target),
        target_ptr);
    auto *token_ptr = _ready_element_pointer(
        _ready_tokens, index);
    auto *old_token = _builder.CreateLoad(
        _builder.getInt32Ty(), token_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(nonempty, token, old_token),
        token_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(
            nonempty,
            _builder.CreateAdd(count, _builder.getInt32(1u)),
            count),
        _ready_count);
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask);
    _builder.CreateStore(_builder.CreateOr(runnable, mask),
                         _runnable_mask);
}

void ScheduleEmitter::_resume(
    ::llvm::Value *target, ::llvm::Value *mask) {
    auto *token = _builder.CreateLoad(
        _current_token->getAllocatedType(), _current_token);
    _resume(target, mask, token);
}

void ScheduleEmitter::_resume(
    schedule::BlockId target, ::llvm::Value *mask) {
    _resume(_builder.getInt32(target.value), mask);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_route_edge(
    const schedule::ControlEdge &edge, ::llvm::Value *mask) {
    _apply_assignments(edge.assignments, mask);
    if (_failed()) { return nullptr; }
    // Convergence arrival is emitted once at the target block's entry rather
    // than duplicated on every incoming edge. Edge assignments still happen
    // before the arrival, so parked lanes retain their per-lane PHI state.
    return mask;
}

void ScheduleEmitter::_continue_at(
    schedule::BlockId target, ::llvm::Value *mask) {
    _builder.CreateStore(mask, _current_mask);
    _builder.CreateCondBr(
        _builder.CreateOrReduce(mask),
        _schedule_blocks[target.value], _scheduler_loop);
}

void ScheduleEmitter::_emit_arrival(const schedule::ControlEdge &edge,
                                    ::llvm::Value *mask) {
    auto *flow = _route_edge(edge, mask);
    if (flow != nullptr) {
        _continue_at(edge.target, flow);
    }
}

void ScheduleEmitter::_emit_terminator(
    const schedule::BasicBlock &block) {
    if (auto diamond = _find_predicated_memory_diamond(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_predicated_memory_diamond(
            block, *split, *diamond, nullptr);
        return;
    }
    auto reuse_coherent_mask =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE");
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
                    if (reuse_coherent_mask) {
                        _result.coherent_mask_reuse_count += 2u;
                    }
                    auto *true_mask = _builder.CreateAnd(
                        _active_mask, condition);
                    auto *false_mask = _builder.CreateAnd(
                        _active_mask, _builder.CreateNot(condition));
                    auto *true_nonempty =
                        _builder.CreateOrReduce(true_mask);
                    auto *false_nonempty =
                        _builder.CreateOrReduce(false_mask);
                    auto *divergent = _builder.CreateAnd(
                        true_nonempty, false_nonempty);
                    auto *divergent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.divergent", _entry);
                    auto *coherent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent", _entry);
                    _builder.CreateCondBr(
                        divergent, divergent_path,
                        coherent_path);

                    _builder.SetInsertPoint(divergent_path);
                    if (control.convergence) {
                        _declare_convergence(
                            *control.convergence,
                            _builder.getTrue());
                    }
                    auto *true_flow = _route_edge(
                        control.true_edge, true_mask);
                    auto *false_flow = _route_edge(
                        control.false_edge, false_mask);
                    if (true_flow == nullptr ||
                        false_flow == nullptr) {
                        return;
                    }
                    _resume(control.true_edge.target, true_flow);
                    _resume(control.false_edge.target, false_flow);
                    _builder.CreateBr(_scheduler_loop);

                    _builder.SetInsertPoint(coherent_path);
                    auto *true_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent.true", _entry);
                    auto *false_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.coherent.false", _entry);
                    _builder.CreateCondBr(
                        true_nonempty, true_path, false_path);
                    _builder.SetInsertPoint(true_path);
                    _emit_arrival(
                        control.true_edge,
                        reuse_coherent_mask ?
                            _active_mask : true_mask);
                    _builder.SetInsertPoint(false_path);
                    _emit_arrival(
                        control.false_edge,
                        reuse_coherent_mask ?
                            _active_mask : false_mask);
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
                std::is_same_v<T, schedule::SwitchTerminator>) {
                auto *selector_value = _source.value(control.selector);
                auto *selector = _load_value(control.selector);
                if (selector == nullptr || selector_value == nullptr) {
                    return;
                }
                if (selector_value->value_class ==
                    schedule::ValueClass::varying) {
                    if (reuse_coherent_mask) {
                        _result.coherent_mask_reuse_count +=
                            control.cases.size() + 1u;
                    }
                    auto *selector_type = ::llvm::cast<::llvm::VectorType>(
                        selector->getType());
                    auto *element_type = ::llvm::cast<::llvm::IntegerType>(
                        selector_type->getElementType());
                    std::vector<::llvm::Value *> case_masks;
                    case_masks.reserve(control.cases.size());
                    auto *remaining = _active_mask;
                    ::llvm::Value *has_previous_path = _builder.getFalse();
                    ::llvm::Value *divergent = _builder.getFalse();
                    auto record_path = [&](::llvm::Value *mask) noexcept {
                        auto *nonempty = _builder.CreateOrReduce(mask);
                        divergent = _builder.CreateOr(
                            divergent,
                            _builder.CreateAnd(
                                has_previous_path, nonempty));
                        has_previous_path = _builder.CreateOr(
                            has_previous_path, nonempty);
                    };
                    for (auto &&item : control.cases) {
                        auto *case_value = ::llvm::ConstantInt::get(
                            element_type, item.value);
                        auto *matches = _builder.CreateICmpEQ(
                            selector,
                            _builder.CreateVectorSplat(
                                _width, case_value));
                        auto *case_mask = _builder.CreateAnd(
                            remaining, matches);
                        case_masks.emplace_back(case_mask);
                        record_path(case_mask);
                        remaining = _builder.CreateAnd(
                            remaining, _builder.CreateNot(matches));
                    }
                    auto *default_mask = remaining;
                    record_path(default_mask);
                    auto *divergent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.divergent", _entry);
                    auto *coherent_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.coherent", _entry);
                    _builder.CreateCondBr(
                        divergent, divergent_path,
                        coherent_path);

                    _builder.SetInsertPoint(divergent_path);
                    if (control.convergence) {
                        _declare_convergence(
                            *control.convergence,
                            _builder.getTrue());
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        auto *flow = _route_edge(
                            control.cases[i].edge, case_masks[i]);
                        if (flow == nullptr) { return; }
                        _resume(
                            control.cases[i].edge.target, flow);
                    }
                    auto *default_flow = _route_edge(
                        control.default_edge, default_mask);
                    if (default_flow == nullptr) { return; }
                    _resume(
                        control.default_edge.target, default_flow);
                    _builder.CreateBr(_scheduler_loop);

                    _builder.SetInsertPoint(coherent_path);
                    auto *default_path =
                        ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "varying.switch.coherent.default", _entry);
                    auto *seed_selector =
                        _builder.CreateExtractElement(
                            selector, _seed_lane);
                    auto *llvm_switch = _builder.CreateSwitch(
                        seed_selector, default_path,
                        control.cases.size());
                    std::vector<::llvm::BasicBlock *> case_paths;
                    case_paths.reserve(control.cases.size());
                    for (auto &&item : control.cases) {
                        auto *case_path =
                            ::llvm::BasicBlock::Create(
                                _module.getContext(),
                                "varying.switch.coherent.case", _entry);
                        case_paths.emplace_back(case_path);
                        llvm_switch->addCase(
                            ::llvm::ConstantInt::get(
                                element_type, item.value),
                            case_path);
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        _builder.SetInsertPoint(case_paths[i]);
                        _emit_arrival(
                            control.cases[i].edge,
                            reuse_coherent_mask ?
                                _active_mask : case_masks[i]);
                    }
                    _builder.SetInsertPoint(default_path);
                    _emit_arrival(
                        control.default_edge,
                        reuse_coherent_mask ?
                            _active_mask : default_mask);
                } else {
                    auto *default_path = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "uniform.switch.default", _entry);
                    std::vector<::llvm::BasicBlock *> case_paths;
                    case_paths.reserve(control.cases.size());
                    auto *llvm_switch = _builder.CreateSwitch(
                        selector, default_path, control.cases.size());
                    auto *selector_type = ::llvm::cast<
                        ::llvm::IntegerType>(selector->getType());
                    for (auto &&item : control.cases) {
                        auto *case_path = ::llvm::BasicBlock::Create(
                            _module.getContext(),
                            "uniform.switch.case", _entry);
                        case_paths.emplace_back(case_path);
                        llvm_switch->addCase(
                            ::llvm::ConstantInt::get(
                                selector_type, item.value),
                            case_path);
                    }
                    for (auto i = size_t{0u};
                         i < control.cases.size(); i++) {
                        _builder.SetInsertPoint(case_paths[i]);
                        _emit_arrival(
                            control.cases[i].edge, _active_mask);
                    }
                    _builder.SetInsertPoint(default_path);
                    _emit_arrival(
                        control.default_edge, _active_mask);
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
                for (auto frame = uint32_t{0u}; frame < _width; frame++) {
                    auto *index = _builder.getInt32(frame);
                    auto *active_frames = _builder.CreateLoad(
                        _frame_active->getAllocatedType(),
                        _frame_active);
                    auto *frame_active = _frame_is_active(
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
                    auto *released_frames = _builder.CreateAnd(
                        active_frames,
                        _builder.CreateNot(_frame_bit(index)));
                    _builder.CreateStore(
                        _builder.CreateSelect(
                            complete, released_frames, active_frames),
                        _frame_active);

                    auto *parents = _builder.CreateLoad(
                        _frame_parent_token->getAllocatedType(),
                        _frame_parent_token);
                    auto *parent = _builder.CreateExtractElement(
                        parents, index);
                    auto *static_ids = _builder.CreateLoad(
                        _frame_static_id->getAllocatedType(),
                        _frame_static_id);
                    auto *static_id = _builder.CreateExtractElement(
                        static_ids, index);
                    if (_convergence_targets != nullptr) {
                        auto *target = _load_convergence_target(static_id);
                        // The target entry performs the same dynamic cascade
                        // as an ordinary CFG arrival. Deferring it avoids one
                        // copy of the full frame logic per return-site frame.
                        _resume(target, released, parent);
                    }
                }
                _builder.CreateBr(_scheduler_loop);
            } else if constexpr (
                std::is_same_v<T, schedule::UnreachableTerminator>) {
                _builder.CreateUnreachable();
            } else {
                _fail("unsupported Schedule IR terminator reached LLVM emission");
            }
        },
        block.terminator);
}

void ScheduleEmitter::_emit_direct_terminator(
    const schedule::BasicBlock &block,
    const std::vector<::llvm::BasicBlock *> &blocks) {
    if (auto diamond = _find_predicated_memory_diamond(block)) {
        auto *split = std::get_if<schedule::SplitTerminator>(
            &block.terminator);
        _emit_predicated_memory_diamond(
            block, *split, *diamond, &blocks);
        return;
    }
    auto scalar = [&](::llvm::Value *value) -> ::llvm::Value * {
        if (value == nullptr) { return nullptr; }
        return value->getType()->isVectorTy() ?
                   _builder.CreateExtractElement(value, uint64_t{0u}) :
                   value;
    };
    auto emit_edge = [&](const schedule::ControlEdge &edge) {
        _apply_assignments(edge.assignments, _active_mask);
        if (!_failed()) {
            _builder.CreateBr(blocks[edge.target.value]);
        }
    };
    std::visit(
        [&](const auto &control) {
            using T = std::decay_t<decltype(control)>;
            if constexpr (std::is_same_v<
                              T, schedule::BranchTerminator>) {
                emit_edge(control.edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SplitTerminator>) {
                auto *condition = scalar(
                    _load_value(control.condition));
                if (condition == nullptr) { return; }
                auto *true_path = ::llvm::BasicBlock::Create(
                    _module.getContext(), "direct.true", _entry);
                auto *false_path = ::llvm::BasicBlock::Create(
                    _module.getContext(), "direct.false", _entry);
                _builder.CreateCondBr(
                    condition, true_path, false_path);
                _builder.SetInsertPoint(true_path);
                emit_edge(control.true_edge);
                _builder.SetInsertPoint(false_path);
                emit_edge(control.false_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SwitchTerminator>) {
                auto *selector = scalar(
                    _load_value(control.selector));
                if (selector == nullptr) { return; }
                auto *default_path = ::llvm::BasicBlock::Create(
                    _module.getContext(),
                    "direct.switch.default", _entry);
                std::vector<::llvm::BasicBlock *> case_paths;
                case_paths.reserve(control.cases.size());
                auto *llvm_switch = _builder.CreateSwitch(
                    selector, default_path, control.cases.size());
                auto *selector_type = ::llvm::cast<
                    ::llvm::IntegerType>(selector->getType());
                for (auto &&item : control.cases) {
                    auto *case_path = ::llvm::BasicBlock::Create(
                        _module.getContext(),
                        "direct.switch.case", _entry);
                    case_paths.emplace_back(case_path);
                    llvm_switch->addCase(
                        ::llvm::ConstantInt::get(
                            selector_type, item.value),
                        case_path);
                }
                for (auto i = size_t{0u};
                     i < control.cases.size(); i++) {
                    _builder.SetInsertPoint(case_paths[i]);
                    emit_edge(control.cases[i].edge);
                }
                _builder.SetInsertPoint(default_path);
                emit_edge(control.default_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::JoinTerminator>) {
                auto *point = _source.convergence(
                    control.convergence);
                schedule::ControlEdge edge{point->target};
                edge.assignments = control.assignments;
                emit_edge(edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::ReturnTerminator>) {
                if (control.value) {
                    auto *schedule_value = _source.value(
                        *control.value);
                    auto *value = _as_lane_vector(
                        _load_value(*control.value),
                        *schedule_value);
                    if (value != nullptr) {
                        _builder.CreateMaskedStore(
                            value, _return_buffer,
                            ::llvm::Align{
                                schedule_value->type->alignment()},
                            _active_mask);
                    }
                }
                _builder.CreateRetVoid();
            } else if constexpr (std::is_same_v<
                                     T,
                                     schedule::UnreachableTerminator>) {
                _builder.CreateUnreachable();
            } else {
                _fail("unsupported Schedule IR terminator reached direct LLVM emission");
            }
        },
        block.terminator);
}

void ScheduleEmitter::_find_instruction_spills() {
    _spilled_instruction_values.assign(
        _source.values().size(), uint8_t{0u});
    std::vector<schedule::BlockId> emission_blocks;
    emission_blocks.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        emission_blocks.emplace_back(block.id);
    }
    if (_direct_control_flow) {
        for (auto &&block : _source.blocks()) {
            if (auto diamond =
                    _find_predicated_memory_diamond(block)) {
                emission_blocks[diamond->true_block->id.value] = block.id;
                emission_blocks[diamond->false_block->id.value] = block.id;
            }
        }
    }
    auto mark = [&](schedule::ValueId id,
                    schedule::BlockId use_block) noexcept {
        auto *value = _source.value(id);
        if (value != nullptr &&
            value->origin == schedule::ValueOrigin::instruction &&
            value->defining_block &&
            emission_blocks[value->defining_block->value] !=
                emission_blocks[use_block.value]) {
            _spilled_instruction_values[id.value] = 1u;
        }
    };
    for (auto &&block : _source.blocks()) {
        // Predicated direct-control diamonds emit both arms in the split
        // block. Compare definition and use after that emission-block remap:
        // arm-local/outer values remain LLVM SSA, while a value genuinely
        // produced by an earlier emitted block still receives a spill slot.
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
    if (!_direct_control_flow) {
        auto *mask_type = _layout.mask_type();
        auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *zero_i32_lanes = ::llvm::Constant::getNullValue(i32_lanes);
        auto *frame_active_type = ::llvm::IntegerType::get(
            _module.getContext(), _width);
        _live_mask = _builder.CreateAlloca(
            mask_type, nullptr, "live.mask");
        _runnable_mask = _builder.CreateAlloca(
            mask_type, nullptr, "runnable.mask");
        _ready_count = _builder.CreateAlloca(
            _builder.getInt32Ty(), nullptr, "ready.count");
        auto *ready_masks = ::llvm::ArrayType::get(
            mask_type, _width);
        _ready_masks = _builder.CreateAlloca(
            ready_masks, nullptr, "ready.mask");
        auto *ready_targets = ::llvm::ArrayType::get(
            _builder.getInt32Ty(), _width);
        _ready_targets = _builder.CreateAlloca(
            ready_targets, nullptr, "ready.target");
        auto *ready_tokens = ::llvm::ArrayType::get(
            _builder.getInt32Ty(), _width);
        _ready_tokens = _builder.CreateAlloca(
            ready_tokens, nullptr, "ready.token");
        _current_mask = _builder.CreateAlloca(
            mask_type, nullptr, "current.mask");
        _current_token = _builder.CreateAlloca(
            _builder.getInt32Ty(), nullptr, "current.token");
        _frame_active = _builder.CreateAlloca(
            frame_active_type, nullptr, "frame.active");
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
        _builder.CreateStore(
            _builder.getInt32(0u), _ready_count);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(ready_masks),
            _ready_masks);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(ready_targets),
            _ready_targets);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(ready_tokens),
            _ready_tokens);
        _builder.CreateStore(zero_mask, _current_mask);
        _builder.CreateStore(
            _builder.getInt32(0u), _current_token);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(frame_active_type),
            _frame_active);
        _builder.CreateStore(zero_i32_lanes, _frame_static_id);
        _builder.CreateStore(zero_i32_lanes, _frame_parent_token);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(frame_masks),
            _frame_expected);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(frame_masks),
            _frame_arrived);

        std::vector<::llvm::Constant *> convergence_targets;
        convergence_targets.reserve(
            _source.convergence_points().size());
        for (auto &&point : _source.convergence_points()) {
            convergence_targets.emplace_back(
                _builder.getInt32(point.target.value));
        }
        if (!convergence_targets.empty() && _width >= 4u) {
            auto *array = ::llvm::ArrayType::get(
                _builder.getInt32Ty(), convergence_targets.size());
            auto *initializer = ::llvm::ConstantArray::get(
                array, convergence_targets);
            auto *global = new ::llvm::GlobalVariable(
                _module, array, true,
                ::llvm::GlobalValue::PrivateLinkage, initializer,
                "convergence.targets");
            global->setUnnamedAddr(
                ::llvm::GlobalValue::UnnamedAddr::Global);
            global->setAlignment(::llvm::Align{4u});
            _convergence_targets = global;
        } else if (!convergence_targets.empty()) {
            _convergence_targets =
                ::llvm::ConstantVector::get(convergence_targets);
        }
        _target_convergence_depths.assign(
            _source.blocks().size(), 0u);
        for (auto &&point : _source.convergence_points()) {
            ++_target_convergence_depths[point.target.value];
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
            auto value_size = value == nullptr ? 0u :
                                                 _abi_size(value->type);
            auto value_alignment = value == nullptr ? 1u :
                                                      _abi_alignment(value->type);
            if (value == nullptr || value->type == nullptr ||
                !_is_local_lvalue(*instruction.result) ||
                value_size == 0u) {
                _fail("thread-local allocation has an invalid data type");
                return;
            }
            auto byte_count = static_cast<uint64_t>(_width) *
                              value_size;
            auto *storage_type = ::llvm::ArrayType::get(
                _builder.getInt8Ty(), byte_count);
            auto *storage = _builder.CreateAlloca(
                storage_type, nullptr, value->name + ".local");
            storage->setAlignment(
                ::llvm::Align{value_alignment});
            auto *offsets = _lane_offsets(
                _lane_ids(), value_size);
            _local_allocations[instruction.result->value] =
                _local_handle(
                    _builder.CreateVectorSplat(_width, storage),
                    offsets);
        }
    }
    for (auto slot = size_t{0u};
         slot < _ray_query_status_storage.size(); slot++) {
        auto *storage = _builder.CreateAlloca(
            _builder.getInt64Ty(), nullptr,
            "ray.query.status.slot." + std::to_string(slot));
        storage->setAlignment(::llvm::Align{alignof(uint64_t)});
        _builder.CreateStore(_builder.getInt64(0u), storage);
        _ray_query_status_storage[slot] = storage;
        auto *callback_type = ::llvm::FixedVectorType::get(
            ::llvm::PointerType::getUnqual(_module.getContext()), _width);
        auto *callback = _builder.CreateAlloca(
            callback_type, nullptr,
            "ray.query.status.callback.slot." + std::to_string(slot));
        callback->setAlignment(::llvm::Align{alignof(void *)});
        _builder.CreateAlignedStore(
            ::llvm::Constant::getNullValue(callback_type),
            callback, ::llvm::Align{alignof(void *)});
        _ray_query_status_callback_storage[slot] = callback;
        if (slot < _ray_query_state_handle_storage.size()) {
            auto *state_handles = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.state.handles.slot." + std::to_string(slot));
            state_handles->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                state_handles, ::llvm::Align{alignof(void *)});
            _ray_query_state_handle_storage[slot] = state_handles;
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
                     _direct_control_flow &&
                             value.value_class ==
                                 schedule::ValueClass::cohort_uniform ?
                         _layout.expression_type(value) :
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
        _result.state_slot_count++;
        _result.spilled_instruction_count += spill;
    }
}

void ScheduleEmitter::_partition_state_residency() {
    // LLVM's default O2 pipeline promotes eligible allocas through the global
    // dispatcher. That is profitable for hot state, but a large set of rarely
    // accessed slots creates wide scheduler PHIs and ultimately physical
    // register spills. Keep hot state promotable and pin cold state to its L1
    // stack slot when cold state dominates the function's state set.
    static constexpr auto max_cold_accesses = size_t{6u};
    auto count_accesses = [](const ::llvm::AllocaInst *slot) noexcept {
        auto count = size_t{0u};
        for (auto *user : slot->users()) {
            count += ::llvm::isa<::llvm::LoadInst>(user) ||
                     ::llvm::isa<::llvm::StoreInst>(user);
        }
        return count;
    };
    auto slot_count = size_t{0u};
    auto cold_count = size_t{0u};
    for (auto *slot : _state_slots) {
        if (slot == nullptr) { continue; }
        slot_count++;
        cold_count += count_accesses(slot) <= max_cold_accesses;
    }
    _result.cold_state_slot_count = cold_count;
    if (slot_count == 0u || cold_count * 2u < slot_count ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_COLD_STATE_PARTITION")) {
        return;
    }
    for (auto *slot : _state_slots) {
        if (slot == nullptr ||
            count_accesses(slot) > max_cold_accesses) {
            continue;
        }
        for (auto *user : slot->users()) {
            if (auto *load = ::llvm::dyn_cast<::llvm::LoadInst>(user)) {
                load->setVolatile(true);
            } else if (auto *store =
                           ::llvm::dyn_cast<::llvm::StoreInst>(user)) {
                store->setVolatile(true);
            }
        }
        _result.stack_pinned_state_slot_count++;
    }
}

bool ScheduleEmitter::_can_emit_direct_control_flow() const noexcept {
    std::vector<bool> covered_convergences(
        _source.convergence_points().size(), false);
    for (auto &&block : _source.blocks()) {
        auto supported = std::visit(
            [&](const auto &control) noexcept {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<
                                  T, schedule::SplitTerminator>) {
                    auto *condition = _source.value(control.condition);
                    if (condition != nullptr &&
                        schedule::is_uniform(condition->value_class)) {
                        return true;
                    }
                    auto diamond =
                        _find_predicated_memory_diamond(block);
                    if (diamond && control.convergence) {
                        covered_convergences[control.convergence->value] =
                            true;
                    }
                    return diamond.has_value();
                } else if constexpr (std::is_same_v<
                                         T,
                                         schedule::SwitchTerminator>) {
                    auto *selector = _source.value(control.selector);
                    return selector != nullptr &&
                           schedule::is_uniform(
                               selector->value_class);
                } else {
                    return std::is_same_v<
                               T, schedule::BranchTerminator> ||
                           std::is_same_v<
                               T, schedule::ReturnTerminator> ||
                           std::is_same_v<
                               T, schedule::UnreachableTerminator>;
                }
            },
            block.terminator);
        if (!supported) { return false; }
    }
    return std::all_of(
        covered_convergences.begin(), covered_convergences.end(),
        [](bool covered) noexcept { return covered; });
}

void ScheduleEmitter::_build_direct(::llvm::Value *initial_mask) {
    auto &context = _module.getContext();
    std::vector<bool> inlined_blocks(
        _source.blocks().size(), false);
    for (auto &&block : _source.blocks()) {
        if (auto diamond = _find_predicated_memory_diamond(block)) {
            inlined_blocks[diamond->true_block->id.value] = true;
            inlined_blocks[diamond->false_block->id.value] = true;
        }
    }
    std::vector<::llvm::BasicBlock *> blocks(
        _source.blocks().size(), nullptr);
    for (auto &&block : _source.blocks()) {
        if (!inlined_blocks[block.id.value]) {
            blocks[block.id.value] = ::llvm::BasicBlock::Create(
                context,
                "direct.schedule." + std::to_string(block.id.value),
                _entry);
        }
    }
    auto *activate = ::llvm::BasicBlock::Create(
        context, "direct.activate", _entry);
    auto *inactive = ::llvm::BasicBlock::Create(
        context, "direct.inactive", _entry);
    auto *active = _builder.CreateOrReduce(initial_mask);
    _builder.CreateCondBr(active, activate, inactive);

    _builder.SetInsertPoint(inactive);
    _builder.CreateRetVoid();

    _builder.SetInsertPoint(activate);
    _active_mask = _width == 1u ?
                       static_cast<::llvm::Value *>(
                           ::llvm::ConstantVector::getSplat(
                               ::llvm::ElementCount::getFixed(1u),
                               _builder.getTrue())) :
                       initial_mask;
    // With a statically row-aligned packet, any nonempty dispatch-edge mask
    // is a prefix and therefore contains lane zero. Direct control never
    // changes that mask, so keep the seed in a register constant instead of
    // repeating first-active extraction in hot memory loops.
    auto lane_zero_is_active =
        _width == 1u ||
        (_static_block_size[0u] >= _width &&
         _static_block_size[0u] % _width == 0u);
    _seed_lane = lane_zero_is_active ?
                     static_cast<::llvm::Value *>(
                         _builder.getInt32(0u)) :
                     _safe_first_lane(_active_mask);
    _builder.CreateBr(blocks[_source.entry().value]);
    for (auto &&block : _source.blocks()) {
        if (inlined_blocks[block.id.value]) { continue; }
        _builder.SetInsertPoint(blocks[block.id.value]);
        _locals.clear();
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        _emit_direct_terminator(block, blocks);
        if (_failed()) { return; }
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
    _result.schedule_block_count = _source.blocks().size();
    _result.convergence_point_count =
        _source.convergence_points().size();
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
    _direct_control_flow =
        _width == 1u ||
        (!luisa::compute::detail::env_flag(
             "LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG") &&
         _can_emit_direct_control_flow());
    _result.direct_control_flow = _direct_control_flow;
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
    if (_direct_control_flow) {
        _build_direct(initial_mask);
        return;
    }
    _builder.CreateStore(initial_mask, _live_mask);
    _resume(_source.entry(), initial_mask);

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
    _schedule_blocks.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        _schedule_blocks.emplace_back(::llvm::BasicBlock::Create(
            context,
            "schedule." + std::to_string(block.id.value), _entry));
    }
    _builder.CreateBr(_scheduler_loop);

    _builder.SetInsertPoint(_scheduler_loop);
    auto *ready_count = _builder.CreateLoad(
        _ready_count->getAllocatedType(), _ready_count);
    _builder.CreateCondBr(
        _builder.CreateICmpNE(
            ready_count, _builder.getInt32(0u)),
        dispatch, done);

    _builder.SetInsertPoint(dispatch);
    auto *ready_index = _builder.CreateSub(
        ready_count, _builder.getInt32(1u));
    _builder.CreateStore(ready_index, _ready_count);
    auto *ready_mask_ptr = _ready_element_pointer(
        _ready_masks, ready_index);
    auto *popped_mask = _builder.CreateLoad(
        _layout.mask_type(), ready_mask_ptr);
    _builder.CreateStore(popped_mask, _current_mask);
    auto *ready_target_ptr = _ready_element_pointer(
        _ready_targets, ready_index);
    auto *pc = _builder.CreateLoad(
        _builder.getInt32Ty(), ready_target_ptr);
    auto *ready_token_ptr = _ready_element_pointer(
        _ready_tokens, ready_index);
    auto *token = _builder.CreateLoad(
        _builder.getInt32Ty(), ready_token_ptr);
    _builder.CreateStore(token, _current_token);
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask);
    _builder.CreateStore(
        _builder.CreateAnd(
            runnable, _builder.CreateNot(popped_mask)),
        _runnable_mask);
    auto *dispatch_switch = _builder.CreateSwitch(
        pc,
        invalid,
        static_cast<unsigned>(_schedule_blocks.size()));
    for (auto &&block : _source.blocks()) {
        dispatch_switch->addCase(
            _builder.getInt32(block.id.value),
            _schedule_blocks[block.id.value]);
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
        _builder.SetInsertPoint(
            _schedule_blocks[block.id.value]);
        _locals.clear();
        _active_mask = _builder.CreateLoad(
            _layout.mask_type(), _current_mask);
        if (_target_convergence_depths[block.id.value] != 0u) {
            auto *flow = _cascade_at_convergence_target(
                _builder.getInt32(block.id.value), _active_mask);
            auto *ready = ::llvm::BasicBlock::Create(
                context,
                "convergence.target." +
                    std::to_string(block.id.value) + ".ready",
                _entry);
            _builder.CreateCondBr(
                _builder.CreateOrReduce(flow),
                ready, _scheduler_loop);
            _builder.SetInsertPoint(ready);
            _active_mask = flow;
        }
        _seed_lane = _safe_first_lane(_active_mask);
        for (auto &&instruction : block.instructions) {
            _emit_instruction(instruction);
            if (_failed()) { return; }
        }
        _emit_terminator(block);
        if (_failed()) { return; }
    }

    _partition_state_residency();
}

}// namespace luisa::compute::simd::detail
