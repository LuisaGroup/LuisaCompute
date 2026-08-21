#include "llvm_schedule_emitter.h"

#include <llvm/IR/Attributes.h>

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
            value = _cast(instruction, operand_sanitization_mask);
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
        case schedule::Opcode::ray_query_pipeline:
            _ray_query_pipeline(instruction);
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
        case schedule::Opcode::print:
            _print(instruction);
            break;
        case schedule::Opcode::assert_:
            _assert(instruction);
            break;
        case schedule::Opcode::clock:
            value = _clock(instruction);
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
    if (destination == nullptr || source == nullptr) { return; }
    // The coalescer proves that the logical source and destination have
    // noninterfering per-lane live ranges. Their masked state move is then an
    // identity; cross-copy source interference keeps sequential emission safe.
    if (destination->origin == schedule::ValueOrigin::state_slot &&
        source->origin == schedule::ValueOrigin::state_slot &&
        assignment.destination.value < _state_slots.size() &&
        assignment.source.value < _state_slots.size() &&
        _state_slots[assignment.destination.value] != nullptr &&
        _state_slots[assignment.destination.value] ==
            _state_slots[assignment.source.value]) {
        return;
    }
    auto *value = _load_value(assignment.source);
    if (value == nullptr) { return; }
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

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_frame_metadata(
    ::llvm::AllocaInst *frames, ::llvm::Value *index) {
    auto *type = frames->getAllocatedType();
    if (auto *array = ::llvm::dyn_cast<::llvm::ArrayType>(type)) {
        auto *pointer = _builder.CreateInBoundsGEP(
            array, frames, {_builder.getInt32(0u), index});
        return _builder.CreateLoad(array->getElementType(), pointer);
    }
    auto *values = _builder.CreateLoad(type, frames);
    return _builder.CreateExtractElement(values, index);
}

void ScheduleEmitter::_store_frame_metadata(
    ::llvm::AllocaInst *frames, ::llvm::Value *index,
    ::llvm::Value *value) {
    auto *type = frames->getAllocatedType();
    if (auto *array = ::llvm::dyn_cast<::llvm::ArrayType>(type)) {
        auto *pointer = _builder.CreateInBoundsGEP(
            array, frames, {_builder.getInt32(0u), index});
        _builder.CreateStore(value, pointer);
        return;
    }
    auto *values = _builder.CreateLoad(type, frames);
    _builder.CreateStore(
        _builder.CreateInsertElement(values, value, index), frames);
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
    auto *current_static = _load_frame_metadata(
        _frame_static_id, current_index);
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

    auto *old_static = _load_frame_metadata(
        _frame_static_id, free_index);
    auto *new_static = _builder.CreateSelect(
        allocate, _builder.getInt32(convergence.value), old_static);
    _store_frame_metadata(
        _frame_static_id, free_index, new_static);

    auto *old_parent = _load_frame_metadata(
        _frame_parent_token, free_index);
    auto *new_parent = _builder.CreateSelect(
        allocate, current_token, old_parent);
    _store_frame_metadata(
        _frame_parent_token, free_index, new_parent);

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
    auto *static_id = _load_frame_metadata(
        _frame_static_id, index);
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
    auto *parent_token = _load_frame_metadata(
        _frame_parent_token, index);
    _builder.CreateStore(
        _builder.CreateSelect(complete, parent_token, token),
        _current_token);
    return _builder.CreateSelect(_splat(matches), released, flow);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_cascade_at_convergence_target(
    ::llvm::Value *target, ::llvm::Value *flow) {
    if (_convergence_targets == nullptr) { return flow; }
    auto guard_empty_chain =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_CONVERGENCE_TOKEN_GUARD");
    if (guard_empty_chain) {
        _result.convergence_token_guard_count++;
    }
    auto *preheader = _builder.GetInsertBlock();
    auto *loop = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade.exit", _entry);
    if (guard_empty_chain) {
        auto *token = _builder.CreateLoad(
            _current_token->getAllocatedType(), _current_token,
            "convergence.current.token");
        _builder.CreateCondBr(
            _builder.CreateICmpNE(
                token, _builder.getInt32(0u),
                "convergence.token.present"),
            loop, exit);
    } else {
        _builder.CreateBr(loop);
    }
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
    if (guard_empty_chain) {
        auto *next_token = _builder.CreateLoad(
            _current_token->getAllocatedType(), _current_token,
            "convergence.parent.token");
        more = _builder.CreateAnd(
            more,
            _builder.CreateICmpNE(
                next_token, _builder.getInt32(0u),
                "convergence.parent.present"));
    }
    auto *latch = _builder.GetInsertBlock();
    _builder.CreateCondBr(more, loop, exit);
    current_flow->addIncoming(next_flow, latch);
    depth->addIncoming(next_depth, latch);
    _builder.SetInsertPoint(exit);
    if (guard_empty_chain) {
        auto *result = _builder.CreatePHI(
            _layout.mask_type(), 2u,
            "convergence.cascade.result");
        result->addIncoming(flow, preheader);
        result->addIncoming(next_flow, latch);
        return result;
    }
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
    if (edge.loop_back) {
        _advance_cooperative_loop_epoch(*edge.loop_back, mask);
        if (_failed()) { return nullptr; }
    }
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

void ScheduleEmitter::_find_instruction_spills() {
    _spilled_instruction_values.assign(
        _source.values().size(), uint8_t{0u});
    std::vector<schedule::BlockId> emission_blocks;
    emission_blocks.reserve(_source.blocks().size());
    for (auto &&block : _source.blocks()) {
        emission_blocks.emplace_back(block.id);
    }
    std::vector<uint8_t> chained_blocks(
        _source.blocks().size(), uint8_t{0u});
    if (!_predicated_acyclic_control_flow) {
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto region =
                    _find_chained_predicated_region(block)) {
                chained_blocks[block.id.value] = 1u;
                for (auto *inlined : region->inlined_blocks) {
                    emission_blocks[inlined->id.value] = block.id;
                    chained_blocks[inlined->id.value] = 1u;
                }
            }
        }
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto diamond =
                    _find_guarded_predicated_math_diamond(block)) {
                for (auto *arm : diamond->true_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                for (auto *arm : diamond->false_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
            }
        }
        for (auto &&block : _source.blocks()) {
            if (chained_blocks[block.id.value] != 0u) { continue; }
            if (auto region = _find_nested_predicated_region(block)) {
                emission_blocks[region->nested_split_block->id.value] =
                    block.id;
                for (auto *arm : region->nested_diamond.true_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                for (auto *arm : region->nested_diamond.false_blocks) {
                    emission_blocks[arm->id.value] = block.id;
                }
                emission_blocks[region->nested_merge_block->id.value] =
                    block.id;
                emission_blocks[region->other_block->id.value] = block.id;
            }
        }
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
    if (!_direct_control_flow &&
        !_predicated_acyclic_control_flow) {
        auto *mask_type = _layout.mask_type();
        auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
        // Dynamic whole-vector frame updates are cheaper through promoted
        // vectors at W1--W8, but measurably increase register/stack pressure
        // at W16. Keep this width policy independently A/B-testable.
        _use_scalar_frame_metadata =
            _width == 16u &&
            !luisa::compute::detail::env_flag(
                "LUISA_SIMD_DISABLE_SCALAR_FRAME_METADATA");
        _result.scalar_frame_metadata =
            _use_scalar_frame_metadata;
        auto *i32_lanes = _use_scalar_frame_metadata ?
                              static_cast<::llvm::Type *>(
                                  ::llvm::ArrayType::get(
                                      _builder.getInt32Ty(), _width)) :
                              static_cast<::llvm::Type *>(
                                  ::llvm::FixedVectorType::get(
                                      _builder.getInt32Ty(), _width));
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

        auto *loop_epoch_type = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        _cooperative_loop_epochs.reserve(
            _result.block_barrier_loop_epoch_count);
        for (auto index = size_t{0u};
             index < _result.block_barrier_loop_epoch_count; index++) {
            auto *epoch = _builder.CreateAlloca(
                loop_epoch_type, nullptr,
                "cooperative.loop.epoch." + std::to_string(index));
            _builder.CreateStore(
                ::llvm::Constant::getNullValue(loop_epoch_type), epoch);
            _cooperative_loop_epochs.emplace_back(epoch);
        }

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
    _shared_memory_size = 0u;
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
            if (_is_shared_lvalue(*instruction.result)) {
                auto alignment = std::max<size_t>(
                    value_alignment, 16u);
                auto offset = _align_up(
                    _shared_memory_size, alignment);
                if (offset > simd_max_shared_memory_bytes ||
                    value_size >
                        simd_max_shared_memory_bytes - offset) {
                    _fail("SIMD cooperative shared memory exceeds the runtime limit");
                    return;
                }
                _shared_memory_size = offset + value_size;
                auto *shared_address = _byte_pointer(
                    _launch_config,
                    offsetof(SIMDPacketLaunchConfig, shared_memory));
                auto *shared = _builder.CreateLoad(
                    ::llvm::PointerType::getUnqual(
                        _module.getContext()),
                    shared_address, "shared.memory");
                auto *base = offset == 0u ?
                                 shared :
                                 _builder.CreateInBoundsPtrAdd(
                                     shared,
                                     _builder.getInt64(offset));
                auto *zero_offsets = ::llvm::Constant::getNullValue(
                    ::llvm::FixedVectorType::get(
                        _builder.getInt64Ty(), _width));
                _local_allocations[instruction.result->value] =
                    _local_handle(
                        _builder.CreateVectorSplat(_width, base),
                        zero_offsets);
            } else {
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
    }
    _result.shared_memory_size = _shared_memory_size;
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
        if (slot < _ray_query_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.pipeline.w1.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot <
            _ray_query_surface_filter_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_surface_filter_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot <
            _ray_query_empty_surface_filter_pipeline_callback_storage.size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.empty.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_empty_surface_filter_pipeline_callback_storage[slot] =
                pipeline_callback;
        }
        if (slot < _ray_query_empty_surface_filter_accel_storage.size()) {
            auto *accel = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.empty.surface.filter.accel.slot." +
                    std::to_string(slot));
            accel->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type), accel,
                ::llvm::Align{alignof(void *)});
            _ray_query_empty_surface_filter_accel_storage[slot] = accel;
        }
        if (slot <
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                .size()) {
            auto *pipeline_callback = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.direct.output.surface.filter.pipeline.callback.slot." +
                    std::to_string(slot));
            pipeline_callback->setAlignment(
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type),
                pipeline_callback,
                ::llvm::Align{alignof(void *)});
            _ray_query_direct_output_surface_filter_pipeline_callback_storage
                [slot] = pipeline_callback;
        }
        if (slot <
            _ray_query_direct_output_surface_filter_accel_storage.size()) {
            auto *accel = _builder.CreateAlloca(
                callback_type, nullptr,
                "ray.query.direct.output.surface.filter.accel.slot." +
                    std::to_string(slot));
            accel->setAlignment(::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                ::llvm::Constant::getNullValue(callback_type), accel,
                ::llvm::Align{alignof(void *)});
            _ray_query_direct_output_surface_filter_accel_storage[slot] =
                accel;
        }
        if (slot <
            _ray_query_surface_filter_ray_packet_storage.size()) {
            auto *lane_type = ::llvm::FixedVectorType::get(
                _builder.getInt32Ty(), _width);
            auto *packet_type = ::llvm::ArrayType::get(
                lane_type, simd_host_accel_ray_packet_field_count);
            auto *ray_packet = _builder.CreateAlloca(
                packet_type, nullptr,
                "ray.query.surface.filter.ray.packet.slot." +
                    std::to_string(slot));
            ray_packet->setAlignment(::llvm::Align{
                _width * sizeof(uint32_t)});
            _ray_query_surface_filter_ray_packet_storage[slot] =
                ray_packet;
            auto *call_packet = _builder.CreateAlloca(
                packet_type, nullptr,
                "ray.query.surface.filter.call.ray.packet.slot." +
                    std::to_string(slot));
            call_packet->setAlignment(::llvm::Align{
                _width * sizeof(uint32_t)});
            _ray_query_surface_filter_ray_packet_call_storage[slot] =
                call_packet;
        }
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
    _coalesce_state_slots();
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
    std::vector<::llvm::AllocaInst *> physical_slots;
    physical_slots.reserve(_state_slots.size());
    for (auto *slot : _state_slots) {
        if (slot != nullptr &&
            std::find(physical_slots.cbegin(),
                      physical_slots.cend(), slot) ==
                physical_slots.cend()) {
            physical_slots.emplace_back(slot);
        }
    }
    auto slot_count = physical_slots.size();
    auto cold_count = size_t{0u};
    for (auto *slot : physical_slots) {
        cold_count += count_accesses(slot) <= max_cold_accesses;
    }
    _result.cold_state_slot_count = cold_count;
    // Coalescing has already collapsed mutually exclusive PHI versions into
    // a much smaller physical set. Pinning a majority of that set recreates
    // the very copy/load traffic the liveness proof removed, so leave the
    // compact state promotable. The uncoalesced oracle retains the established
    // cold-majority partition.
    if (_result.coalesced_state_slot_count != 0u ||
        slot_count == 0u || cold_count * 2u < slot_count ||
        luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_COLD_STATE_PARTITION")) {
        return;
    }
    for (auto *slot : physical_slots) {
        if (count_accesses(slot) > max_cold_accesses) {
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

void ScheduleEmitter::_build() {
    auto &context = _module.getContext();
    auto handler_entry = _is_handler_entry();
    auto parameter_types = std::vector<::llvm::Type *>{};
    if (handler_entry) {
        if (_is_surface_filter_handler_entry()) {
            parameter_types = {
                ::llvm::Type::getInt32Ty(context),
                ::llvm::Type::getInt64Ty(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context)};
        } else {
            parameter_types = {
                ::llvm::Type::getInt32Ty(context),
                ::llvm::Type::getInt64Ty(context),
                ::llvm::PointerType::getUnqual(context),
                ::llvm::PointerType::getUnqual(context)};
            parameter_types.reserve(3u + _parameters.size());
            for (auto index = size_t{1u};
                 index < _parameters.size(); index++) {
                auto *type = _handler_parameter_type(
                    *_parameters[index]);
                if (type == nullptr) { return; }
                parameter_types.emplace_back(type);
            }
        }
    } else {
        parameter_types = {
            ::llvm::PointerType::getUnqual(context),
            ::llvm::PointerType::getUnqual(context),
            ::llvm::PointerType::getUnqual(context),
            ::llvm::Type::getInt32Ty(context)};
    }
    auto *function_type = ::llvm::FunctionType::get(
        !handler_entry && _cooperative_block ?
            static_cast<::llvm::Type *>(
                ::llvm::PointerType::getUnqual(context)) :
            static_cast<::llvm::Type *>(
                ::llvm::Type::getVoidTy(context)),
        parameter_types,
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
        function_type,
        handler_entry ? ::llvm::GlobalValue::InternalLinkage :
                        ::llvm::GlobalValue::ExternalLinkage,
        _entry_name, _module);
    if (handler_entry) {
        _entry->setDSOLocal(true);
        _entry->addFnAttr(::llvm::Attribute::InlineHint);
        if (_is_surface_filter_handler_entry()) {
            for (auto index : {2u, 3u}) {
                _entry->addParamAttr(index, ::llvm::Attribute::NonNull);
                _entry->addParamAttr(index, ::llvm::Attribute::ReadOnly);
            }
            _entry->addParamAttr(4u, ::llvm::Attribute::NonNull);
            _entry->addParamAttr(4u, ::llvm::Attribute::NoAlias);
        }
    }
    _result.schedule_block_count = _source.blocks().size();
    _result.convergence_point_count =
        _source.convergence_points().size();
    auto argument = _entry->arg_begin();
    if (handler_entry) {
        _active_lane_count = &*argument++;
        _active_lane_count->setName("lane_count");
        _handler_active_mask_bits = &*argument++;
        _handler_active_mask_bits->setName("active_mask_bits");
        if (_is_surface_filter_handler_entry()) {
            auto *pointer_type =
                ::llvm::PointerType::getUnqual(context);
            _argument_buffer =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _launch_config =
                ::llvm::ConstantPointerNull::get(pointer_type);
            _surface_filter_ray_packet = &*argument++;
            _surface_filter_ray_packet->setName("ray_packet");
            _surface_filter_hit_packet = &*argument++;
            _surface_filter_hit_packet->setName("hit_packet");
            _surface_filter_committed_mask_bits = &*argument;
            _surface_filter_committed_mask_bits->setName(
                "committed_mask_bits");
        } else {
            _argument_buffer = &*argument++;
            _argument_buffer->setName("state_pointer_lanes");
            _launch_config = &*argument++;
            _launch_config->setName("launch_config");
            for (auto index = size_t{1u};
                 index < _parameters.size(); index++) {
                argument->setName(
                    "capture." + std::to_string(index - 1u));
                ++argument;
            }
        }
        _return_buffer = ::llvm::ConstantPointerNull::get(
            ::llvm::PointerType::getUnqual(context));
    } else {
        _argument_buffer = &*argument++;
        _argument_buffer->setName("argument_buffer");
        _return_buffer = &*argument++;
        _return_buffer->setName("return_lanes");
        _launch_config = &*argument++;
        _launch_config->setName("launch_config");
        _active_lane_count = &*argument;
        _active_lane_count->setName("active_lane_count");
    }
    // The runtime owns launch_config separately from every resource and from
    // the packed descriptor buffer. The packet body only observes it. These
    // attributes let an inlined batch hoist immutable launch geometry across
    // packet iterations without changing the portable packet ABI.
    if (!handler_entry &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES")) {
        _entry->addParamAttr(0u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
        _entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
        if (!_cooperative_block) {
            _entry->addParamAttr(2u, ::llvm::Attribute::ReadOnly);
        }
    }

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "prologue", _entry);
    _builder.SetInsertPoint(prologue);
    if (handler_entry) {
        _entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
        _entry->addParamAttr(3u, ::llvm::Attribute::NoAlias);
        _entry->addParamAttr(3u, ::llvm::Attribute::NonNull);
        _entry->addParamAttr(3u, ::llvm::Attribute::ReadOnly);
        _trap_if(
            _builder.CreateICmpNE(
                _active_lane_count, _builder.getInt32(_width)),
            "ray.query.handler.width.mismatch");
        auto lane_bits = (uint64_t{1u} << _width) - 1u;
        _trap_if(
            _builder.CreateICmpNE(
                _builder.CreateAnd(
                    _handler_active_mask_bits,
                    _builder.getInt64(~lane_bits)),
                _builder.getInt64(0u)),
            "ray.query.handler.mask.out.of.range");
    }
    if (_cooperative_block) {
        _begin_cooperative_coroutine();
        if (_failed()) { return; }
    }
    _direct_control_flow =
        !_cooperative_block &&
        (_width == 1u ||
         (!luisa::compute::detail::env_flag(
              "LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG") &&
          _can_emit_direct_control_flow()));
    _result.direct_control_flow = _direct_control_flow;
    if (!_direct_control_flow &&
        _enable_predicated_acyclic_control_flow) {
        if (auto order = _find_predicated_acyclic_order()) {
            _predicated_acyclic_control_flow = true;
            _predicated_acyclic_order = std::move(*order);
        }
    }
    _result.predicated_acyclic_control_flow =
        _predicated_acyclic_control_flow;
    _allocate_state();
    if (_failed()) { return; }
    _create_external_values();
    if (_failed()) { return; }

    auto *lane_ids = _lane_ids();
    auto *initial_mask = static_cast<::llvm::Value *>(nullptr);
    if (handler_entry) {
        auto *packed_mask_type = ::llvm::IntegerType::get(
            context, _width);
        initial_mask = _builder.CreateBitCast(
            _builder.CreateTrunc(
                _handler_active_mask_bits, packed_mask_type),
            _layout.mask_type(), "handler.active.mask");
    } else {
        auto *count = _builder.CreateVectorSplat(
            _width, _active_lane_count);
        initial_mask = _builder.CreateICmpULT(lane_ids, count);
    }
    _ensure_launch_vectors();
    // Runtime block IDs are always drawn from ceil(dispatch_size / block_size)
    // in each dimension. A statically unit-sized dimension therefore has
    // thread_id == 0 and block_id < dispatch_size for every launched block;
    // it cannot contribute a dispatch-edge tail. Avoid materializing that
    // redundant vector compare, especially for the two unit dimensions of a
    // 1D kernel. Standalone packet callers have the same valid-block contract.
    auto elide_unit_dimension_masks =
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_UNIT_DIMENSION_MASK_ELISION");
    if (handler_entry) {
        // The caller supplies the exact sparse physical-lane cohort. Dispatch
        // geometry remains available to handler expressions but must not
        // widen or narrow this semantic candidate mask.
    } else if (_enable_linear_1d_packet_tail_narrowing &&
               !_cooperative_block) {
        // The runtime-only packet wrapper narrows its final 1D packet to the
        // exact dispatch and block remainder, and skips packets with no live
        // lanes. The active-lane prefix above is therefore the complete
        // dispatch mask; rebuilding dispatch_id < dispatch_size in every
        // packet would be redundant. Cooperative wrappers deliberately issue
        // every static packet in a block so all participating packets can
        // rendezvous. They must retain the full dispatch-extent mask for a
        // partial edge block, including a mixed final packet.
        _result.linear_1d_packet_tail_narrowing_count++;
    } else {
        for (auto i = uint32_t{0u}; i < 3u; i++) {
            if (elide_unit_dimension_masks &&
                _static_block_size[i] == 1u) {
                _result.unit_dimension_mask_elision_count++;
                continue;
            }
            initial_mask = _builder.CreateAnd(
                initial_mask,
                _builder.CreateICmpULT(
                    _dispatch_id[i],
                    _builder.CreateVectorSplat(
                        _width, _dispatch_size[i])));
        }
    }
    if (_cooperative_block) {
        _initialize_cooperative_packet(initial_mask);
    }
    if (_direct_control_flow) {
        _build_direct(initial_mask);
        return;
    }
    if (_predicated_acyclic_control_flow) {
        _build_predicated_acyclic(initial_mask);
        return;
    }
    _builder.CreateStore(initial_mask, _live_mask);
    _resume(_source.entry(), initial_mask);

    _scheduler_loop = ::llvm::BasicBlock::Create(
        context, "scheduler.loop", _entry);
    auto *dispatch = ::llvm::BasicBlock::Create(
        context, "scheduler.dispatch", _entry);
    _scheduler_dispatch_route = ::llvm::BasicBlock::Create(
        context, "scheduler.dispatch.route", _entry);
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
    std::optional<StructuredEarlyExitLoop> structured_early_exit_loop;
    for (auto &&block : _source.blocks()) {
        if (auto candidate =
                _find_structured_early_exit_loop(block)) {
            structured_early_exit_loop = std::move(*candidate);
            break;
        }
    }
    std::vector<uint8_t> structured_emitted_blocks(
        _source.blocks().size(), uint8_t{0u});
    std::vector<uint8_t> structured_absorbed_blocks(
        _source.blocks().size(), uint8_t{0u});
    if (structured_early_exit_loop) {
        for (auto *block :
             structured_early_exit_loop->emitted_blocks) {
            structured_emitted_blocks[block->id.value] = 1u;
        }
        for (auto *block :
             structured_early_exit_loop->absorbed_blocks) {
            structured_absorbed_blocks[block->id.value] = 1u;
        }
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
    _builder.CreateBr(_scheduler_dispatch_route);

    _builder.SetInsertPoint(_scheduler_dispatch_route);
    _scheduler_dispatch_pc = _builder.CreatePHI(
        _builder.getInt32Ty(), 1u, "scheduler.dispatch.pc");
    _scheduler_dispatch_pc->addIncoming(pc, dispatch);
    auto *dispatch_switch = _builder.CreateSwitch(
        _scheduler_dispatch_pc,
        invalid,
        static_cast<unsigned>(_schedule_blocks.size()));
    for (auto &&block : _source.blocks()) {
        if (structured_emitted_blocks[block.id.value] != 0u ||
            structured_absorbed_blocks[block.id.value] != 0u) {
            continue;
        }
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
    if (_cooperative_block) {
        _finish_entry();
    } else {
        _builder.CreateRetVoid();
    }

    std::vector<uint8_t> locally_inlined_blocks(
        _source.blocks().size(), uint8_t{0u});
    std::vector<uint8_t> chained_blocks(
        _source.blocks().size(), uint8_t{0u});
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto region =
                _find_chained_predicated_region(block)) {
            chained_blocks[block.id.value] = 1u;
            for (auto *inlined : region->inlined_blocks) {
                locally_inlined_blocks[inlined->id.value] = 1u;
                chained_blocks[inlined->id.value] = 1u;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto diamond =
                _find_guarded_predicated_math_diamond(block)) {
            for (auto *arm : diamond->true_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            for (auto *arm : diamond->false_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
        }
    }
    for (auto &&block : _source.blocks()) {
        if (chained_blocks[block.id.value] != 0u) { continue; }
        if (auto region = _find_nested_predicated_region(block)) {
            locally_inlined_blocks[region->nested_split_block->id.value] = 1u;
            for (auto *arm : region->nested_diamond.true_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            for (auto *arm : region->nested_diamond.false_blocks) {
                locally_inlined_blocks[arm->id.value] = 1u;
            }
            locally_inlined_blocks[region->nested_merge_block->id.value] = 1u;
            locally_inlined_blocks[region->other_block->id.value] = 1u;
        }
    }
    if (structured_early_exit_loop) {
        for (auto *block :
             structured_early_exit_loop->absorbed_blocks) {
            locally_inlined_blocks[block->id.value] = 1u;
        }
    }
    for (auto &&block : _source.blocks()) {
        if (structured_emitted_blocks[block.id.value] != 0u) {
            continue;
        }
        _builder.SetInsertPoint(
            _schedule_blocks[block.id.value]);
        if (locally_inlined_blocks[block.id.value] != 0u) {
            _builder.CreateUnreachable();
            continue;
        }
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
        if (structured_early_exit_loop &&
            structured_early_exit_loop->header->id == block.id) {
            _emit_structured_early_exit_loop(
                *structured_early_exit_loop);
            if (_failed()) { return; }
            continue;
        }
        if (auto loop = _find_predicated_loop(block)) {
            _emit_predicated_loop(*loop);
            if (_failed()) { return; }
            continue;
        }
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
