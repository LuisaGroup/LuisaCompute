#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <unordered_set>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/ModRef.h>

#include "../../common/env_flag.h"
#include <luisa/core/stl/string.h>

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
        case schedule::Opcode::call:
            if (instruction.outlined_callable_id) {
                value = _outlined_call(instruction);
            } else if (instruction.contiguous_copy) {
                _contiguous_copy(instruction);
            } else {
                _strided_mma(instruction);
            }
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
        case schedule::Opcode::assume:
            _emit_assume(instruction);
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
                auto kind = _local_soa_kind_of(*instruction.result);
                if (kind == LocalSoAKind::uniform || _width == 1u) {
                    _builder.CreateStore(value, slot);
                } else if (kind == LocalSoAKind::varying) {
                    auto *old = _builder.CreateLoad(
                        slot->getAllocatedType(), slot);
                    _builder.CreateStore(
                        _merge_varying_soa_handles(
                            value, old, _active_mask),
                        slot);
                } else {
                    auto *old = _builder.CreateLoad(
                        slot->getAllocatedType(), slot);
                    _builder.CreateStore(
                        _merge_local_handles(
                            value, old, _active_mask),
                        slot);
                }
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
                if (_width == 1u) {
                    _builder.CreateStore(lanes, slot);
                } else {
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
    }
    if (!_collectives.succeeded()) { _fail(_collectives.error()); }
}

void ScheduleEmitter::_assign(schedule::EdgeAssignment assignment,
                              ::llvm::Value *mask, ::llvm::Value *value) {
    auto *destination = _source.value(assignment.destination);
    auto *source = _source.value(assignment.source);
    if (destination == nullptr || source == nullptr) { return; }
    // The coalescer proves that the logical source and destination have
    // noninterfering per-lane live ranges. Their masked state move is then an
    // identity. Other sources were snapshotted before any destination write.
    if (destination->origin == schedule::ValueOrigin::state_slot &&
        source->origin == schedule::ValueOrigin::state_slot &&
        assignment.destination.value < _state_slots.size() &&
        assignment.source.value < _state_slots.size() &&
        _state_slots[assignment.destination.value] != nullptr &&
        _state_slots[assignment.destination.value] ==
            _state_slots[assignment.source.value]) {
        return;
    }
    if (value == nullptr) { return; }
    auto *slot = _state_slots[assignment.destination.value];
    if (_is_local_lvalue(assignment.destination)) {
        auto kind = _local_soa_kind_of(assignment.destination);
        if (kind == LocalSoAKind::uniform || _width == 1u) {
            _builder.CreateStore(value, slot);
        } else if (kind == LocalSoAKind::varying) {
            auto *old = _builder.CreateLoad(
                slot->getAllocatedType(), slot);
            _builder.CreateStore(
                _merge_varying_soa_handles(value, old, mask), slot);
        } else {
            auto *old = _builder.CreateLoad(
                slot->getAllocatedType(), slot);
            _builder.CreateStore(
                _merge_local_handles(value, old, mask), slot);
        }
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
    if (_width == 1u) {
        _builder.CreateStore(lanes, slot);
        return;
    }
    auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
    auto *merged = _masked_merge(lanes, old, destination->type, mask);
    if (merged != nullptr) { _builder.CreateStore(merged, slot); }
}

void ScheduleEmitter::_apply_assignments(
    const std::vector<schedule::EdgeAssignment> &assignments,
    ::llvm::Value *mask) {
    // PHI edge copies are simultaneous. Interference between distinct slots
    // does not protect a source that is itself another copy's destination
    // (a <- b, b <- a). Load every old value before updating any state slot.
    std::vector<::llvm::Value *> sources;
    sources.reserve(assignments.size());
    for (auto assignment : assignments) {
        sources.emplace_back(_load_value(assignment.source));
        if (_failed()) { return; }
    }
    for (auto i = size_t{0u}; i < assignments.size(); i++) {
        _assign(assignments[i], mask, sources[i]);
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
    auto *bits_type = ::llvm::cast<::llvm::IntegerType>(_frame_active_type);
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
    ::llvm::Value *frames, ::llvm::Value *index) {
    auto *array = ::llvm::cast<::llvm::ArrayType>(_ready_masks_type);
    return _builder.CreateInBoundsGEP(
        array, frames, {_builder.getInt32(0u), index});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_frame_metadata(
    ::llvm::Value *frames, ::llvm::Value *index) {
    auto *type = _frame_metadata_type;
    if (auto *array = ::llvm::dyn_cast<::llvm::ArrayType>(type)) {
        auto *pointer = _builder.CreateInBoundsGEP(
            array, frames, {_builder.getInt32(0u), index});
        return _builder.CreateLoad(array->getElementType(), pointer);
    }
    auto *values = _builder.CreateLoad(type, frames);
    return _builder.CreateExtractElement(values, index);
}

void ScheduleEmitter::_store_frame_metadata(
    ::llvm::Value *frames, ::llvm::Value *index,
    ::llvm::Value *value) {
    auto *type = _frame_metadata_type;
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
    ::llvm::Value *array, ::llvm::Type *array_type, ::llvm::Value *index) {
    auto *typed = ::llvm::cast<::llvm::ArrayType>(array_type);
    return _builder.CreateInBoundsGEP(
        typed, array,
        {_builder.getInt32(0u), index});
}

[[nodiscard]] ::llvm::StructType *ScheduleEmitter::_get_or_create_sched_state_type() {
    auto &ctx = _module.getContext();
    const auto name = "__luisa_simd_sched_state_w" + std::to_string(_width);
    if (auto *existing = ::llvm::StructType::getTypeByName(ctx, name)) {
        return existing;
    }
    auto *mask_type = _layout.mask_type();
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *frame_active_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *ready_masks_ty = ::llvm::ArrayType::get(mask_type, _width);
    auto *ready_i32_ty = ::llvm::ArrayType::get(i32_ty, _width);
    return ::llvm::StructType::create(
        ctx,
        {mask_type,        // 0: live_mask
         mask_type,        // 1: runnable_mask
         i32_ty,           // 2: ready_count
         ready_masks_ty,   // 3: ready_masks
         ready_i32_ty,     // 4: ready_targets
         ready_i32_ty,     // 5: ready_tokens
         mask_type,        // 6: current_mask
         i32_ty,           // 7: current_token
         frame_active_ty,  // 8: frame_active
         ready_i32_ty,     // 9: frame_static_id
         ready_i32_ty,     // 10: frame_parent_token
         ready_masks_ty,   // 11: frame_expected
         ready_masks_ty},  // 12: frame_arrived
        name);
}

[[nodiscard]] ::llvm::Function *ScheduleEmitter::_get_or_create_sched_declare_helper() {
    auto &ctx = _module.getContext();
    const auto fn_name = "__luisa_simd_sched_declare_w" + std::to_string(_width);
    if (auto *existing = _module.getFunction(fn_name)) {
        return existing;
    }
    auto *sched_ty = _get_or_create_sched_state_type();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *i1_ty = ::llvm::Type::getInt1Ty(ctx);
    auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *mask_arg_ty = _width <= 32u ?
                            static_cast<::llvm::Type *>(i32_ty) :
                            static_cast<::llvm::Type *>(::llvm::Type::getInt64Ty(ctx));
    auto *frame_active_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *ready_masks_ty = ::llvm::ArrayType::get(mask_int_ty, _width);
    auto *ready_i32_ty = ::llvm::ArrayType::get(i32_ty, _width);

    // Pure-GPR leaf helper: zero vector/FP register touches so PreserveAll is
    // safe on x86-64 (SSE/AVX2/AVX-512) and AArch64 without partial-vector saves.
    const auto sched_cc =
        luisa::compute::detail::env_flag("LUISA_SIMD_SCHED_HELPER_FASTCC") ?
            ::llvm::CallingConv::Fast :
            ::llvm::CallingConv::PreserveAll;

    auto *fn_ty = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(ctx), {ptr_ty, i32_ty, i1_ty, mask_arg_ty}, false);
    auto *fn = ::llvm::Function::Create(
        fn_ty, ::llvm::GlobalValue::InternalLinkage, fn_name, _module);
    fn->setDSOLocal(true);
    fn->setCallingConv(sched_cc);
    fn->addFnAttr(::llvm::Attribute::NoInline);
    fn->addFnAttr(::llvm::Attribute::NoUnwind);
    fn->addFnAttr(::llvm::Attribute::WillReturn);
    fn->addParamAttr(0u, ::llvm::Attribute::NoAlias);
    fn->addParamAttr(0u, ::llvm::Attribute::NonNull);
#if LLVM_VERSION_MAJOR >= 21
    fn->addParamAttr(
        0u, ::llvm::Attribute::getWithCaptureInfo(
                ctx, ::llvm::CaptureInfo::none()));
#else
    fn->addParamAttr(0u, ::llvm::Attribute::NoCapture);
#endif

    auto *entry_bb = ::llvm::BasicBlock::Create(ctx, "entry", fn);
    auto *body_bb = ::llvm::BasicBlock::Create(ctx, "body", fn);
    auto *ret_bb = ::llvm::BasicBlock::Create(ctx, "ret", fn);
    auto *trap_bb = ::llvm::BasicBlock::Create(ctx, "overflow.trap", fn);
    auto *alloc_bb = ::llvm::BasicBlock::Create(ctx, "alloc", fn);

    ::llvm::IRBuilder<> b{entry_bb};
    auto *sched = fn->getArg(0u);
    auto *conv_id = fn->getArg(1u);
    auto *divergent = fn->getArg(2u);
    auto *active_mask_bits = fn->getArg(3u);

    b.CreateCondBr(divergent, body_bb, ret_bb);

    b.SetInsertPoint(ret_bb);
    b.CreateRetVoid();

    b.SetInsertPoint(body_bb);
    auto *cur_tok_ptr = b.CreateStructGEP(sched_ty, sched, 7u);
    auto *frame_act_ptr = b.CreateStructGEP(sched_ty, sched, 8u);
    auto *frame_sid_ptr = b.CreateStructGEP(sched_ty, sched, 9u);
    auto *frame_par_ptr = b.CreateStructGEP(sched_ty, sched, 10u);
    auto *frame_exp_ptr = b.CreateStructGEP(sched_ty, sched, 11u);
    auto *frame_arr_ptr = b.CreateStructGEP(sched_ty, sched, 12u);

    auto *current_token = b.CreateLoad(i32_ty, cur_tok_ptr);
    auto *has_current = b.CreateICmpNE(current_token, b.getInt32(0u));
    auto *current_index = b.CreateSelect(
        has_current, b.CreateSub(current_token, b.getInt32(1u)), b.getInt32(0u));
    auto *active_frames = b.CreateLoad(frame_active_ty, frame_act_ptr);
    auto *cur_bit = b.CreateShl(
        ::llvm::ConstantInt::get(frame_active_ty, 1u),
        b.CreateZExtOrTrunc(current_index, frame_active_ty));
    auto *current_active = b.CreateICmpNE(
        b.CreateAnd(active_frames, cur_bit),
        ::llvm::Constant::getNullValue(frame_active_ty));
    auto *cur_sid = b.CreateLoad(
        i32_ty,
        b.CreateInBoundsGEP(ready_i32_ty, frame_sid_ptr, {b.getInt32(0u), current_index}));
    auto *reuse = b.CreateAnd(
        has_current,
        b.CreateAnd(current_active, b.CreateICmpEQ(cur_sid, conv_id)));
    b.CreateCondBr(reuse, ret_bb, alloc_bb);

    b.SetInsertPoint(alloc_bb);
    auto *free_frames = b.CreateNot(active_frames);
    auto *has_free = b.CreateICmpNE(
        free_frames, ::llvm::Constant::getNullValue(frame_active_ty));
    auto *do_alloc_bb = ::llvm::BasicBlock::Create(ctx, "do.alloc", fn);
    b.CreateCondBr(has_free, do_alloc_bb, trap_bb);

    b.SetInsertPoint(trap_bb);
#if LLVM_VERSION_MAJOR >= 22
    auto *trap_fn = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
    auto *trap_fn = ::llvm::Intrinsic::getDeclaration(
#endif
        &_module, ::llvm::Intrinsic::trap);
    b.CreateCall(trap_fn);
    b.CreateUnreachable();

    b.SetInsertPoint(do_alloc_bb);
    auto *active_mask = b.CreateTrunc(active_mask_bits, mask_int_ty);
    auto *raw_free = b.CreateBinaryIntrinsic(
        ::llvm::Intrinsic::cttz, free_frames, b.getFalse());
    auto *free_index = b.CreateZExtOrTrunc(raw_free, i32_ty);
    auto *free_bit = b.CreateShl(
        ::llvm::ConstantInt::get(frame_active_ty, 1u),
        b.CreateZExtOrTrunc(free_index, frame_active_ty));
    b.CreateStore(b.CreateOr(active_frames, free_bit), frame_act_ptr);
    b.CreateStore(
        conv_id,
        b.CreateInBoundsGEP(ready_i32_ty, frame_sid_ptr, {b.getInt32(0u), free_index}));
    b.CreateStore(
        current_token,
        b.CreateInBoundsGEP(ready_i32_ty, frame_par_ptr, {b.getInt32(0u), free_index}));
    b.CreateStore(
        active_mask,
        b.CreateInBoundsGEP(ready_masks_ty, frame_exp_ptr, {b.getInt32(0u), free_index}));
    b.CreateStore(
        ::llvm::Constant::getNullValue(mask_int_ty),
        b.CreateInBoundsGEP(ready_masks_ty, frame_arr_ptr, {b.getInt32(0u), free_index}));
    b.CreateStore(b.CreateAdd(free_index, b.getInt32(1u)), cur_tok_ptr);
    b.CreateRetVoid();
    return fn;
}

[[nodiscard]] ::llvm::Function *ScheduleEmitter::_get_or_create_sched_resume_helper() {
    auto &ctx = _module.getContext();
    const auto fn_name = "__luisa_simd_sched_resume_w" + std::to_string(_width);
    if (auto *existing = _module.getFunction(fn_name)) {
        return existing;
    }
    auto *sched_ty = _get_or_create_sched_state_type();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *mask_arg_ty = _width <= 32u ?
                            static_cast<::llvm::Type *>(i32_ty) :
                            static_cast<::llvm::Type *>(::llvm::Type::getInt64Ty(ctx));
    auto *ready_masks_ty = ::llvm::ArrayType::get(mask_int_ty, _width);
    auto *ready_i32_ty = ::llvm::ArrayType::get(i32_ty, _width);

    const auto sched_cc =
        luisa::compute::detail::env_flag("LUISA_SIMD_SCHED_HELPER_FASTCC") ?
            ::llvm::CallingConv::Fast :
            ::llvm::CallingConv::PreserveAll;

    auto *fn_ty = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(ctx), {ptr_ty, i32_ty, mask_arg_ty, i32_ty}, false);
    auto *fn = ::llvm::Function::Create(
        fn_ty, ::llvm::GlobalValue::InternalLinkage, fn_name, _module);
    fn->setDSOLocal(true);
    fn->setCallingConv(sched_cc);
    fn->addFnAttr(::llvm::Attribute::NoInline);
    fn->addFnAttr(::llvm::Attribute::NoUnwind);
    fn->addFnAttr(::llvm::Attribute::WillReturn);
    fn->addParamAttr(0u, ::llvm::Attribute::NoAlias);
    fn->addParamAttr(0u, ::llvm::Attribute::NonNull);
#if LLVM_VERSION_MAJOR >= 21
    fn->addParamAttr(
        0u, ::llvm::Attribute::getWithCaptureInfo(
                ctx, ::llvm::CaptureInfo::none()));
#else
    fn->addParamAttr(0u, ::llvm::Attribute::NoCapture);
#endif

    auto *entry_bb = ::llvm::BasicBlock::Create(ctx, "entry", fn);
    auto *push_bb = ::llvm::BasicBlock::Create(ctx, "push", fn);
    auto *ok_bb = ::llvm::BasicBlock::Create(ctx, "ok", fn);
    auto *trap_bb = ::llvm::BasicBlock::Create(ctx, "overflow.trap", fn);
    auto *ret_bb = ::llvm::BasicBlock::Create(ctx, "ret", fn);

    ::llvm::IRBuilder<> b{entry_bb};
    auto *sched = fn->getArg(0u);
    auto *target = fn->getArg(1u);
    auto *mask_bits = fn->getArg(2u);
    auto *token = fn->getArg(3u);

    auto *nonempty = b.CreateICmpNE(
        mask_bits, ::llvm::Constant::getNullValue(mask_arg_ty));
    b.CreateCondBr(nonempty, push_bb, ret_bb);

    b.SetInsertPoint(ret_bb);
    b.CreateRetVoid();

    b.SetInsertPoint(push_bb);
    auto *ready_cnt_ptr = b.CreateStructGEP(sched_ty, sched, 2u);
    auto *count = b.CreateLoad(i32_ty, ready_cnt_ptr);
    b.CreateCondBr(b.CreateICmpUGE(count, b.getInt32(_width)), trap_bb, ok_bb);

    b.SetInsertPoint(trap_bb);
#if LLVM_VERSION_MAJOR >= 22
    auto *trap_fn = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
    auto *trap_fn = ::llvm::Intrinsic::getDeclaration(
#endif
        &_module, ::llvm::Intrinsic::trap);
    b.CreateCall(trap_fn);
    b.CreateUnreachable();

    b.SetInsertPoint(ok_bb);
    auto *mask = b.CreateTrunc(mask_bits, mask_int_ty);
    auto *runnable_ptr = b.CreateStructGEP(sched_ty, sched, 1u);
    auto *ready_masks_ptr = b.CreateStructGEP(sched_ty, sched, 3u);
    auto *ready_targets_ptr = b.CreateStructGEP(sched_ty, sched, 4u);
    auto *ready_tokens_ptr = b.CreateStructGEP(sched_ty, sched, 5u);
    b.CreateStore(
        mask,
        b.CreateInBoundsGEP(ready_masks_ty, ready_masks_ptr, {b.getInt32(0u), count}));
    b.CreateStore(
        target,
        b.CreateInBoundsGEP(ready_i32_ty, ready_targets_ptr, {b.getInt32(0u), count}));
    b.CreateStore(
        token,
        b.CreateInBoundsGEP(ready_i32_ty, ready_tokens_ptr, {b.getInt32(0u), count}));
    b.CreateStore(b.CreateAdd(count, b.getInt32(1u)), ready_cnt_ptr);
    auto *runnable = b.CreateLoad(mask_int_ty, runnable_ptr);
    b.CreateStore(b.CreateOr(runnable, mask), runnable_ptr);
    b.CreateRetVoid();
    return fn;
}

[[nodiscard]] ::llvm::Function *ScheduleEmitter::_get_or_create_sched_cascade_helper() {
    auto &ctx = _module.getContext();
    const auto fn_name = "__luisa_simd_sched_cascade_w" + std::to_string(_width);
    if (auto *existing = _module.getFunction(fn_name)) {
        return existing;
    }
    auto *sched_ty = _get_or_create_sched_state_type();
    auto *ptr_ty = ::llvm::PointerType::getUnqual(ctx);
    auto *i32_ty = ::llvm::Type::getInt32Ty(ctx);
    auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *mask_arg_ty = _width <= 32u ?
                            static_cast<::llvm::Type *>(i32_ty) :
                            static_cast<::llvm::Type *>(::llvm::Type::getInt64Ty(ctx));
    auto *zero_mask = ::llvm::Constant::getNullValue(mask_int_ty);
    auto *frame_active_ty = ::llvm::IntegerType::get(ctx, _width);
    auto *ready_masks_ty = ::llvm::ArrayType::get(mask_int_ty, _width);
    auto *ready_i32_ty = ::llvm::ArrayType::get(i32_ty, _width);

    const auto sched_cc =
        luisa::compute::detail::env_flag("LUISA_SIMD_SCHED_HELPER_FASTCC") ?
            ::llvm::CallingConv::Fast :
            ::llvm::CallingConv::PreserveAll;

    auto *fn_ty = ::llvm::FunctionType::get(
        mask_arg_ty, {ptr_ty, ptr_ty, i32_ty, mask_arg_ty}, false);
    auto *fn = ::llvm::Function::Create(
        fn_ty, ::llvm::GlobalValue::InternalLinkage, fn_name, _module);
    fn->setDSOLocal(true);
    fn->setCallingConv(sched_cc);
    fn->addFnAttr(::llvm::Attribute::NoInline);
    fn->addFnAttr(::llvm::Attribute::NoUnwind);
    fn->addFnAttr(::llvm::Attribute::WillReturn);
    fn->addParamAttr(0u, ::llvm::Attribute::NoAlias);
    fn->addParamAttr(0u, ::llvm::Attribute::NonNull);
    fn->addParamAttr(1u, ::llvm::Attribute::NoAlias);
    fn->addParamAttr(1u, ::llvm::Attribute::NonNull);
    fn->addParamAttr(1u, ::llvm::Attribute::ReadOnly);
#if LLVM_VERSION_MAJOR >= 21
    fn->addParamAttr(
        0u, ::llvm::Attribute::getWithCaptureInfo(
                ctx, ::llvm::CaptureInfo::none()));
    fn->addParamAttr(
        1u, ::llvm::Attribute::getWithCaptureInfo(
                ctx, ::llvm::CaptureInfo::none()));
#else
    fn->addParamAttr(0u, ::llvm::Attribute::NoCapture);
    fn->addParamAttr(1u, ::llvm::Attribute::NoCapture);
#endif

    auto *entry_bb = ::llvm::BasicBlock::Create(ctx, "entry", fn);
    auto *loop_bb = ::llvm::BasicBlock::Create(ctx, "loop", fn);
    auto *apply_bb = ::llvm::BasicBlock::Create(ctx, "apply", fn);
    auto *exit_bb = ::llvm::BasicBlock::Create(ctx, "exit", fn);

    ::llvm::IRBuilder<> b{entry_bb};
    auto *sched = fn->getArg(0u);
    auto *conv_targets = fn->getArg(1u);
    auto *target = fn->getArg(2u);
    auto *initial_flow = b.CreateTrunc(fn->getArg(3u), mask_int_ty);

    auto *live_ptr = b.CreateStructGEP(sched_ty, sched, 0u);
    auto *runnable_ptr = b.CreateStructGEP(sched_ty, sched, 1u);
    auto *cur_tok_ptr = b.CreateStructGEP(sched_ty, sched, 7u);
    auto *frame_act_ptr = b.CreateStructGEP(sched_ty, sched, 8u);
    auto *frame_sid_ptr = b.CreateStructGEP(sched_ty, sched, 9u);
    auto *frame_par_ptr = b.CreateStructGEP(sched_ty, sched, 10u);
    auto *frame_exp_ptr = b.CreateStructGEP(sched_ty, sched, 11u);
    auto *frame_arr_ptr = b.CreateStructGEP(sched_ty, sched, 12u);
    b.CreateBr(loop_bb);

    b.SetInsertPoint(loop_bb);
    auto *current_flow = b.CreatePHI(mask_int_ty, 2u, "flow");
    auto *depth = b.CreatePHI(i32_ty, 2u, "depth");
    current_flow->addIncoming(initial_flow, entry_bb);
    depth->addIncoming(b.getInt32(0u), entry_bb);

    auto *any = b.CreateICmpNE(current_flow, zero_mask);
    auto *token = b.CreateLoad(i32_ty, cur_tok_ptr);
    auto *has_token = b.CreateAnd(any, b.CreateICmpNE(token, b.getInt32(0u)));
    auto *index = b.CreateSelect(
        has_token, b.CreateSub(token, b.getInt32(1u)), b.getInt32(0u));
    auto *active_frames = b.CreateLoad(frame_active_ty, frame_act_ptr);
    auto *idx_bit = b.CreateShl(
        ::llvm::ConstantInt::get(frame_active_ty, 1u),
        b.CreateZExtOrTrunc(index, frame_active_ty));
    auto *frame_active = b.CreateICmpNE(
        b.CreateAnd(active_frames, idx_bit),
        ::llvm::Constant::getNullValue(frame_active_ty));
    auto *static_id = b.CreateLoad(
        i32_ty,
        b.CreateInBoundsGEP(ready_i32_ty, frame_sid_ptr, {b.getInt32(0u), index}));
    auto *dyn_target = b.CreateLoad(
        i32_ty, b.CreateInBoundsGEP(i32_ty, conv_targets, static_id));
    auto *matches = b.CreateAnd(
        has_token,
        b.CreateAnd(frame_active, b.CreateICmpEQ(dyn_target, target)));
    b.CreateCondBr(matches, apply_bb, exit_bb);

    b.SetInsertPoint(apply_bb);
    auto *expected_ptr = b.CreateInBoundsGEP(
        ready_masks_ty, frame_exp_ptr, {b.getInt32(0u), index});
    auto *arrived_ptr = b.CreateInBoundsGEP(
        ready_masks_ty, frame_arr_ptr, {b.getInt32(0u), index});
    auto *expected = b.CreateLoad(mask_int_ty, expected_ptr);
    auto *arrived = b.CreateLoad(mask_int_ty, arrived_ptr);
    auto *new_arrived = b.CreateOr(arrived, current_flow);
    auto *live = b.CreateLoad(mask_int_ty, live_ptr);
    auto *expected_live = b.CreateAnd(expected, live);
    auto *complete = b.CreateICmpEQ(new_arrived, expected_live);

    b.CreateStore(b.CreateSelect(complete, zero_mask, new_arrived), arrived_ptr);
    b.CreateStore(b.CreateSelect(complete, zero_mask, expected), expected_ptr);
    auto *released_frames = b.CreateAnd(active_frames, b.CreateNot(idx_bit));
    b.CreateStore(b.CreateSelect(complete, released_frames, active_frames), frame_act_ptr);
    auto *runnable = b.CreateLoad(mask_int_ty, runnable_ptr);
    b.CreateStore(b.CreateAnd(runnable, b.CreateNot(current_flow)), runnable_ptr);
    auto *released = b.CreateSelect(complete, new_arrived, zero_mask);
    auto *parent_token = b.CreateLoad(
        i32_ty,
        b.CreateInBoundsGEP(ready_i32_ty, frame_par_ptr, {b.getInt32(0u), index}));
    auto *next_token = b.CreateSelect(complete, parent_token, token);
    b.CreateStore(next_token, cur_tok_ptr);

    auto *next_depth = b.CreateAdd(depth, b.getInt32(1u));
    auto *more = b.CreateAnd(
        complete,
        b.CreateAnd(
            b.CreateICmpNE(released, zero_mask),
            b.CreateAnd(
                b.CreateICmpULT(next_depth, b.getInt32(_width)),
                b.CreateICmpNE(next_token, b.getInt32(0u)))));
    auto *latch_bb = b.GetInsertBlock();
    b.CreateCondBr(more, loop_bb, exit_bb);
    current_flow->addIncoming(released, latch_bb);
    depth->addIncoming(next_depth, latch_bb);

    b.SetInsertPoint(exit_bb);
    auto *ret_flow = b.CreatePHI(mask_int_ty, 2u, "ret.flow");
    ret_flow->addIncoming(current_flow, loop_bb);
    ret_flow->addIncoming(released, latch_bb);
    b.CreateRet(b.CreateZExt(ret_flow, mask_arg_ty));
    return fn;
}

void ScheduleEmitter::_trap_if(::llvm::Value *condition, luisa::string_view label) {
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
    if (_outline_scheduler_helpers) {
        auto &ctx = _module.getContext();
        auto *helper = _get_or_create_sched_declare_helper();
        auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
        auto *mask_arg_ty = _width <= 32u ?
                                static_cast<::llvm::Type *>(_builder.getInt32Ty()) :
                                static_cast<::llvm::Type *>(_builder.getInt64Ty());
        auto *mask_bits = _builder.CreateZExt(
            _builder.CreateBitCast(_active_mask, mask_int_ty), mask_arg_ty);
        auto *call = _builder.CreateCall(
            helper,
            {_sched_state, _builder.getInt32(convergence.value),
             divergent, mask_bits});
        call->setCallingConv(helper->getCallingConv());
        return;
    }
    auto *current_token = _builder.CreateLoad(
        _builder.getInt32Ty(), _current_token);
    auto *has_current = _builder.CreateICmpNE(
        current_token, _builder.getInt32(0u));
    auto *current_index = _builder.CreateSelect(
        has_current,
        _builder.CreateSub(current_token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active_type, _frame_active);
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
        _builder.getInt32Ty(), _current_token);
    auto *has_token = _builder.CreateAnd(
        any, _builder.CreateICmpNE(token, _builder.getInt32(0u)));
    auto *index = _builder.CreateSelect(
        has_token,
        _builder.CreateSub(token, _builder.getInt32(1u)),
        _builder.getInt32(0u));
    auto *active_frames = _builder.CreateLoad(
        _frame_active_type, _frame_active);
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
        _layout.mask_type(), _live_mask);
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
        _layout.mask_type(), _runnable_mask);
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
    if (_outline_scheduler_helpers &&
        ::llvm::isa<::llvm::GlobalVariable>(_convergence_targets)) {
        auto *helper = _get_or_create_sched_cascade_helper();
        auto *preheader = _builder.GetInsertBlock();
        auto *check_target = ::llvm::BasicBlock::Create(
            _module.getContext(), "convergence.cascade.check", _entry);
        auto *slow = ::llvm::BasicBlock::Create(
            _module.getContext(), "convergence.cascade.call", _entry);
        auto *exit = ::llvm::BasicBlock::Create(
            _module.getContext(), "convergence.cascade.exit", _entry);
        if (guard_empty_chain) {
            auto *token = _builder.CreateLoad(
                _builder.getInt32Ty(), _current_token,
                "convergence.current.token");
            _builder.CreateCondBr(
                _builder.CreateICmpNE(
                    token, _builder.getInt32(0u),
                    "convergence.token.present"),
                check_target, exit);
            _builder.SetInsertPoint(check_target);
            // Inline the top-frame target check before paying a helper call and
            // caller-saved SIMD register spills on non-matching inner blocks.
            auto *index = _builder.CreateSub(
                token, _builder.getInt32(1u), "convergence.top.index");
            auto *static_id = _builder.CreateLoad(
                _builder.getInt32Ty(),
                _builder.CreateInBoundsGEP(
                    _ready_i32_type, _frame_static_id,
                    {_builder.getInt32(0u), index}),
                "convergence.top.sid");
            auto *dyn_target = _builder.CreateLoad(
                _builder.getInt32Ty(),
                _builder.CreateInBoundsGEP(
                    _builder.getInt32Ty(), _convergence_targets, static_id),
                "convergence.top.target");
            auto *target_br = _builder.CreateCondBr(
                _builder.CreateICmpEQ(
                    dyn_target, target, "convergence.top.matches"),
                slow, exit);
            target_br->setMetadata(
                ::llvm::LLVMContext::MD_prof,
                ::llvm::MDBuilder{_module.getContext()}.createBranchWeights(1u, 32u));
        } else {
            _builder.CreateBr(slow);
        }
        _builder.SetInsertPoint(slow);
        auto &ctx = _module.getContext();
        auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
        auto *mask_arg_ty = _width <= 32u ?
                                static_cast<::llvm::Type *>(_builder.getInt32Ty()) :
                                static_cast<::llvm::Type *>(_builder.getInt64Ty());
        auto *flow_bits = _builder.CreateZExt(
            _builder.CreateBitCast(flow, mask_int_ty), mask_arg_ty);
        auto *call = _builder.CreateCall(
            helper,
            {_sched_state, _convergence_targets, target, flow_bits});
        call->setCallingConv(helper->getCallingConv());
        auto *call_mask = _builder.CreateBitCast(
            _builder.CreateTrunc(call, mask_int_ty), _layout.mask_type());
        _builder.CreateBr(exit);
        _builder.SetInsertPoint(exit);
        if (guard_empty_chain) {
            auto *result = _builder.CreatePHI(
                _layout.mask_type(), 3u, "convergence.cascade.result");
            result->addIncoming(flow, preheader);
            result->addIncoming(flow, check_target);
            result->addIncoming(call_mask, slow);
            return result;
        }
        return call_mask;
    }
    auto *preheader = _builder.GetInsertBlock();
    auto *loop = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade", _entry);
    auto *exit = ::llvm::BasicBlock::Create(
        _module.getContext(), "convergence.cascade.exit", _entry);
    if (guard_empty_chain) {
        auto *token = _builder.CreateLoad(
            _builder.getInt32Ty(), _current_token,
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
            _builder.getInt32Ty(), _current_token,
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
    if (_outline_scheduler_helpers) {
        auto &ctx = _module.getContext();
        auto *helper = _get_or_create_sched_resume_helper();
        auto *mask_int_ty = ::llvm::IntegerType::get(ctx, _width);
        auto *mask_arg_ty = _width <= 32u ?
                                static_cast<::llvm::Type *>(_builder.getInt32Ty()) :
                                static_cast<::llvm::Type *>(_builder.getInt64Ty());
        auto *mask_bits = _builder.CreateZExt(
            _builder.CreateBitCast(mask, mask_int_ty), mask_arg_ty);
        auto *call = _builder.CreateCall(
            helper, {_sched_state, target, mask_bits, token});
        call->setCallingConv(helper->getCallingConv());
        return;
    }
    auto *nonempty = _builder.CreateOrReduce(mask);
    auto *count = _builder.CreateLoad(
        _builder.getInt32Ty(), _ready_count);
    _trap_if(
        _builder.CreateAnd(
            nonempty,
            _builder.CreateICmpUGE(
                count, _builder.getInt32(_width))),
        "ready.overflow");
    auto *index = _builder.CreateSelect(
        nonempty, count, _builder.getInt32(0u));
    auto *mask_ptr = _ready_element_pointer(
        _ready_masks, _ready_masks_type, index);
    auto *old_mask = _builder.CreateLoad(
        _layout.mask_type(), mask_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(nonempty, mask, old_mask),
        mask_ptr);
    auto *target_ptr = _ready_element_pointer(
        _ready_targets, _ready_i32_type, index);
    auto *old_target = _builder.CreateLoad(
        _builder.getInt32Ty(), target_ptr);
    _builder.CreateStore(
        _builder.CreateSelect(nonempty, target, old_target),
        target_ptr);
    auto *token_ptr = _ready_element_pointer(
        _ready_tokens, _ready_i32_type, index);
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
        _layout.mask_type(), _runnable_mask);
    _builder.CreateStore(_builder.CreateOr(runnable, mask),
                         _runnable_mask);
}

void ScheduleEmitter::_resume(
    ::llvm::Value *target, ::llvm::Value *mask) {
    auto *token = _builder.CreateLoad(
        _builder.getInt32Ty(), _current_token);
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

}// namespace luisa::compute::simd::detail
