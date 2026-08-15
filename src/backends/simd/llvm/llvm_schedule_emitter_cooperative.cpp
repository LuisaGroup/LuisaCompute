#include "llvm_schedule_emitter.h"

namespace luisa::compute::simd::detail {

void ScheduleEmitter::_begin_cooperative_coroutine() {
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *null_pointer = ::llvm::ConstantPointerNull::get(pointer_type);
    auto *none = ::llvm::ConstantTokenNone::get(context);
    auto *entry_block = _builder.GetInsertBlock();

    _entry->setPresplitCoroutine();
    _entry->setCoroDestroyOnlyWhenComplete();
    _coroutine_token = _builder.CreateIntrinsic(
        ::llvm::Type::getTokenTy(context), ::llvm::Intrinsic::coro_id,
        {_builder.getInt32(0u), null_pointer, null_pointer,
         null_pointer});
    auto *requires_allocation = _builder.CreateIntrinsic(
        _builder.getInt1Ty(), ::llvm::Intrinsic::coro_alloc,
        {_coroutine_token});
    auto *allocate_block = ::llvm::BasicBlock::Create(
        context, "coro.allocate", _entry);
    auto *begin_block = ::llvm::BasicBlock::Create(
        context, "coro.begin", _entry);
    _builder.CreateCondBr(
        requires_allocation, allocate_block, begin_block);

    _builder.SetInsertPoint(allocate_block);
    auto *frame_size = _builder.CreateIntrinsic(
        _builder.getInt64Ty(), ::llvm::Intrinsic::coro_size, {});
    auto *allocator_address = _byte_pointer(
        _launch_config,
        offsetof(SIMDPacketLaunchConfig, cooperative_frame_alloc));
    auto *allocator = _builder.CreateLoad(
        pointer_type, allocator_address, "cooperative.frame.allocator");
    _trap_if(
        _builder.CreateIsNull(allocator),
        "cooperative.frame.allocator.missing");
    auto *allocator_type = ::llvm::FunctionType::get(
        pointer_type, {_builder.getInt64Ty()}, false);
    auto *frame = _builder.CreateCall(
        allocator_type, allocator, {frame_size},
        "cooperative.frame");
    _trap_if(
        _builder.CreateIsNull(frame),
        "cooperative.frame.allocation.failed");
    auto *allocate_end = _builder.GetInsertBlock();
    _builder.CreateBr(begin_block);

    _builder.SetInsertPoint(begin_block);
    auto *storage = _builder.CreatePHI(
        pointer_type, 2u, "cooperative.frame.storage");
    storage->addIncoming(frame, allocate_end);
    storage->addIncoming(null_pointer, entry_block);
    _coroutine_handle = _builder.CreateIntrinsic(
        pointer_type, ::llvm::Intrinsic::coro_begin,
        {_coroutine_token, storage});

    _coroutine_final = ::llvm::BasicBlock::Create(
        context, "coro.final", _entry);
    _coroutine_cleanup = ::llvm::BasicBlock::Create(
        context, "coro.cleanup", _entry);
    _coroutine_suspend = ::llvm::BasicBlock::Create(
        context, "coro.suspend", _entry);

    ::llvm::IRBuilder<> final_builder{_coroutine_final};
    auto *final_signal = final_builder.CreateIntrinsic(
        final_builder.getInt8Ty(), ::llvm::Intrinsic::coro_suspend,
        {none, final_builder.getInt1(true)});
    auto *final_unreachable = ::llvm::BasicBlock::Create(
        context, "coro.final.unreachable", _entry);
    auto *final_switch = final_builder.CreateSwitch(
        final_signal, _coroutine_suspend, 2u);
    final_switch->addCase(
        final_builder.getInt8(0u), final_unreachable);
    final_switch->addCase(
        final_builder.getInt8(1u), _coroutine_cleanup);
    ::llvm::IRBuilder<> unreachable_builder{final_unreachable};
    unreachable_builder.CreateUnreachable();

    ::llvm::IRBuilder<> cleanup_builder{_coroutine_cleanup};
    auto *memory = cleanup_builder.CreateIntrinsic(
        pointer_type, ::llvm::Intrinsic::coro_free,
        {_coroutine_token, _coroutine_handle});
    auto *free_block = ::llvm::BasicBlock::Create(
        context, "coro.free", _entry);
    cleanup_builder.CreateCondBr(
        cleanup_builder.CreateIsNotNull(memory),
        free_block, _coroutine_suspend);

    cleanup_builder.SetInsertPoint(free_block);
    auto *free_address = cleanup_builder.CreateConstInBoundsGEP1_64(
        cleanup_builder.getInt8Ty(), _launch_config,
        offsetof(SIMDPacketLaunchConfig, cooperative_frame_free));
    auto *free_callback = cleanup_builder.CreateLoad(
        pointer_type, free_address, "cooperative.frame.free");
    auto *free_type = ::llvm::FunctionType::get(
        cleanup_builder.getVoidTy(), {pointer_type}, false);
    auto *has_free_callback = cleanup_builder.CreateIsNotNull(
        free_callback);
    auto *call_free = ::llvm::BasicBlock::Create(
        context, "coro.free.call", _entry);
    cleanup_builder.CreateCondBr(
        has_free_callback, call_free, _coroutine_suspend);
    cleanup_builder.SetInsertPoint(call_free);
    cleanup_builder.CreateCall(free_type, free_callback, {memory});
    cleanup_builder.CreateBr(_coroutine_suspend);

    ::llvm::IRBuilder<> suspend_builder{_coroutine_suspend};
    static_cast<void>(suspend_builder.CreateIntrinsic(
        suspend_builder.getInt1Ty(), ::llvm::Intrinsic::coro_end,
        {_coroutine_handle, suspend_builder.getInt1(false), none}));
    suspend_builder.CreateRet(_coroutine_handle);

    _builder.SetInsertPoint(begin_block);
}

[[nodiscard]] ::llvm::Value *
ScheduleEmitter::_cooperative_barrier_slot() {
    auto *pointer_type = ::llvm::PointerType::getUnqual(
        _module.getContext());
    auto *ids_address = _byte_pointer(
        _launch_config,
        offsetof(SIMDPacketLaunchConfig, barrier_ids));
    auto *ids = _builder.CreateLoad(
        pointer_type, ids_address, "cooperative.barrier.ids");
    return _builder.CreateInBoundsGEP(
        _builder.getInt32Ty(), ids, _packet_index,
        "cooperative.barrier.slot");
}

void ScheduleEmitter::_initialize_cooperative_packet(
    ::llvm::Value *initial_mask) {
    auto *thread_index = _load_launch_u32(
        offsetof(SIMDPacketLaunchConfig, thread_index));
    _packet_index = _width == 1u ?
                        thread_index :
                        _builder.CreateUDiv(
                            thread_index,
                            _builder.getInt32(_width),
                            "cooperative.packet.index");
    _packet_participating = _builder.CreateOrReduce(initial_mask);
    _packet_participating->setName(
        "cooperative.packet.participating");
    _builder.CreateStore(
        _builder.getInt32(simd_cooperative_packet_running),
        _cooperative_barrier_slot());
}

void ScheduleEmitter::_emit_block_barrier(
    const schedule::BlockBarrierTerminator &barrier) {
    auto *live = _builder.CreateLoad(
        _live_mask->getAllocatedType(), _live_mask,
        "barrier.live.mask");
    auto *same_mask = _builder.CreateAndReduce(
        _builder.CreateICmpEQ(_active_mask, live));
    auto *ready_count = _builder.CreateLoad(
        _ready_count->getAllocatedType(), _ready_count,
        "barrier.ready.count");
    auto *runnable = _builder.CreateLoad(
        _runnable_mask->getAllocatedType(), _runnable_mask,
        "barrier.runnable.mask");
    auto *invalid = _builder.CreateOr(
        _builder.CreateNot(same_mask),
        _builder.CreateOr(
            _builder.CreateICmpNE(
                ready_count, _builder.getInt32(0u)),
            _builder.CreateOrReduce(runnable)));
    _trap_if(invalid, "nonuniform.block.barrier");

    auto *flow = _route_edge(barrier.resume_edge, _active_mask);
    if (flow == nullptr) { return; }
    _builder.CreateStore(
        _builder.getInt32(barrier.barrier_id),
        _cooperative_barrier_slot());
    _builder.CreateFence(::llvm::AtomicOrdering::Release);
    auto *signal = _builder.CreateIntrinsic(
        _builder.getInt8Ty(), ::llvm::Intrinsic::coro_suspend,
        {::llvm::ConstantTokenNone::get(_module.getContext()),
         _builder.getInt1(false)});
    auto *resume = ::llvm::BasicBlock::Create(
        _module.getContext(), "block.barrier.resume", _entry);
    auto *suspend_switch = _builder.CreateSwitch(
        signal, _coroutine_suspend, 2u);
    suspend_switch->addCase(_builder.getInt8(0u), resume);
    suspend_switch->addCase(
        _builder.getInt8(1u), _coroutine_cleanup);

    _builder.SetInsertPoint(resume);
    _builder.CreateFence(::llvm::AtomicOrdering::Acquire);
    _continue_at(barrier.resume_edge.target, flow);
}

void ScheduleEmitter::_finish_entry() {
    auto *status = _builder.CreateSelect(
        _packet_participating,
        _builder.getInt32(simd_cooperative_packet_complete),
        _builder.getInt32(simd_cooperative_packet_inactive));
    _builder.CreateStore(status, _cooperative_barrier_slot());
    _builder.CreateBr(_coroutine_final);
}

}// namespace luisa::compute::simd::detail
