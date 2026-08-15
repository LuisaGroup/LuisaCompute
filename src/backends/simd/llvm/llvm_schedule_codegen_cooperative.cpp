#include "llvm_schedule_codegen.h"

#include <limits>
#include <vector>

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Function *build_cooperative_packet_batch_entry(
    ::llvm::Module &module, ::llvm::Function *packet_entry,
    uint32_t specialization_width, uint32_t static_packet_count,
    size_t shared_memory_size, size_t block_barrier_count,
    std::string &error) {
    auto &context = module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *i1_type = ::llvm::Type::getInt1Ty(context);
    auto *i32_type = ::llvm::Type::getInt32Ty(context);
    auto *i64_type = ::llvm::Type::getInt64Ty(context);
    auto *void_type = ::llvm::Type::getVoidTy(context);
    if (block_barrier_count >
        std::numeric_limits<uint32_t>::max()) {
        error = "cooperative SIMD wrapper has too many static barriers";
        return nullptr;
    }
    if (packet_entry == nullptr || static_packet_count == 0u ||
        !packet_entry->getReturnType()->isPointerTy() ||
        packet_entry->arg_size() != 4u) {
        error = "cooperative SIMD wrapper requires a static packet coroutine";
        return nullptr;
    }
    auto name = packet_entry->getName().str() + ".cooperative_block";
    if (module.getFunction(name) != nullptr) {
        error = "duplicate cooperative SIMD entry '" + name + "'";
        return nullptr;
    }
    auto *wrapper_type = ::llvm::FunctionType::get(
        void_type,
        {pointer_type, pointer_type, pointer_type, i32_type}, false);
    auto *wrapper = ::llvm::Function::Create(
        wrapper_type, ::llvm::GlobalValue::ExternalLinkage,
        name, module);
    wrapper->setDSOLocal(true);
    wrapper->addParamAttr(0u, ::llvm::Attribute::NoAlias);
    wrapper->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
    wrapper->addParamAttr(2u, ::llvm::Attribute::NoAlias);
    wrapper->addParamAttr(2u, ::llvm::Attribute::NonNull);
    packet_entry->setLinkage(::llvm::GlobalValue::InternalLinkage);
    packet_entry->setDSOLocal(true);

    auto argument = wrapper->arg_begin();
    auto *argument_buffer = &*argument++;
    auto *return_lanes = &*argument++;
    auto *launch_config = &*argument++;
    auto *packet_count = &*argument;
    argument_buffer->setName("argument_buffer");
    return_lanes->setName("return_lanes");
    launch_config->setName("launch_config");
    packet_count->setName("packet_count");

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "cooperative.prologue", wrapper);
    auto *count_valid = ::llvm::BasicBlock::Create(
        context, "cooperative.count.valid", wrapper);
    auto *callback_valid = ::llvm::BasicBlock::Create(
        context, "cooperative.callback.valid", wrapper);
    auto *memory_valid = ::llvm::BasicBlock::Create(
        context, "cooperative.memory.valid", wrapper);
    auto *init_head = ::llvm::BasicBlock::Create(
        context, "cooperative.init.head", wrapper);
    auto *init_body = ::llvm::BasicBlock::Create(
        context, "cooperative.init.body", wrapper);
    auto *init_done = ::llvm::BasicBlock::Create(
        context, "cooperative.init.done", wrapper);
    auto *phase_begin = ::llvm::BasicBlock::Create(
        context, "cooperative.phase.begin", wrapper);
    auto *scan_head = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.head", wrapper);
    auto *scan_done_test = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.done.test", wrapper);
    auto *scan_finished = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.finished", wrapper);
    auto *scan_complete = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.complete", wrapper);
    auto *scan_complete_valid = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.complete.valid", wrapper);
    auto *scan_alive = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.alive", wrapper);
    auto *scan_alive_valid = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.alive.valid", wrapper);
    auto *scan_record = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.record", wrapper);
    auto *scan_update = ::llvm::BasicBlock::Create(
        context, "cooperative.scan.update", wrapper);
    auto *phase_post = ::llvm::BasicBlock::Create(
        context, "cooperative.phase.post", wrapper);
    auto *phase_live = ::llvm::BasicBlock::Create(
        context, "cooperative.phase.live", wrapper);
    auto *resume_head = ::llvm::BasicBlock::Create(
        context, "cooperative.resume.head", wrapper);
    auto *resume_invoke = ::llvm::BasicBlock::Create(
        context, "cooperative.resume.invoke", wrapper);
    auto *resume_update = ::llvm::BasicBlock::Create(
        context, "cooperative.resume.update", wrapper);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "cooperative.exit", wrapper);
    auto *trap = ::llvm::BasicBlock::Create(
        context, "cooperative.invalid", wrapper);

    ::llvm::IRBuilder<> builder{prologue};
    auto *handle_array_type = ::llvm::ArrayType::get(
        pointer_type, static_packet_count);
    auto *barrier_array_type = ::llvm::ArrayType::get(
        i32_type, static_packet_count);
    auto *handles = builder.CreateAlloca(
        handle_array_type, nullptr, "cooperative.handles");
    auto *barrier_ids = builder.CreateAlloca(
        barrier_array_type, nullptr, "cooperative.barrier.ids");
    auto *any_alive = builder.CreateAlloca(
        i1_type, nullptr, "cooperative.any.alive");
    auto *any_complete = builder.CreateAlloca(
        i1_type, nullptr, "cooperative.any.complete");
    auto *barrier_seen = builder.CreateAlloca(
        i1_type, nullptr, "cooperative.barrier.seen");
    auto *barrier_value = builder.CreateAlloca(
        i32_type, nullptr, "cooperative.barrier.value");
    builder.CreateStore(
        ::llvm::Constant::getNullValue(handle_array_type), handles);
    std::vector<::llvm::Constant *> initial_ids(
        static_packet_count,
        builder.getInt32(simd_cooperative_packet_inactive));
    builder.CreateStore(
        ::llvm::ConstantArray::get(barrier_array_type, initial_ids),
        barrier_ids);
    auto *expected_packet_count = builder.getInt32(
        static_packet_count);
    builder.CreateCondBr(
        builder.CreateICmpEQ(packet_count, expected_packet_count),
        count_valid, trap);

    builder.SetInsertPoint(count_valid);
    auto byte_pointer = [&](size_t offset) {
        return builder.CreateConstInBoundsGEP1_64(
            builder.getInt8Ty(), launch_config, offset);
    };
    auto *begin_callback = builder.CreateLoad(
        pointer_type,
        byte_pointer(offsetof(
            SIMDPacketLaunchConfig, cooperative_block_begin)),
        "cooperative.block.begin");
    builder.CreateCondBr(
        builder.CreateIsNotNull(begin_callback),
        callback_valid, trap);

    builder.SetInsertPoint(callback_valid);
    auto *begin_type = ::llvm::FunctionType::get(
        pointer_type, {i64_type}, false);
    auto *shared_memory = builder.CreateCall(
        begin_type, begin_callback,
        {builder.getInt64(shared_memory_size)},
        "cooperative.shared.memory");
    builder.CreateCondBr(
        builder.CreateIsNotNull(shared_memory),
        memory_valid, trap);

    builder.SetInsertPoint(memory_valid);
    builder.CreateStore(
        shared_memory,
        byte_pointer(offsetof(
            SIMDPacketLaunchConfig, shared_memory)));
    auto *barrier_base = builder.CreateInBoundsGEP(
        barrier_array_type, barrier_ids,
        {builder.getInt32(0u), builder.getInt32(0u)});
    builder.CreateStore(
        barrier_base,
        byte_pointer(offsetof(
            SIMDPacketLaunchConfig, barrier_ids)));
    auto *thread_index_address = byte_pointer(
        offsetof(SIMDPacketLaunchConfig, thread_index));
    builder.CreateBr(init_head);

    builder.SetInsertPoint(init_head);
    auto *init_index = builder.CreatePHI(
        i32_type, 2u, "cooperative.init.index");
    init_index->addIncoming(builder.getInt32(0u), memory_valid);
    builder.CreateCondBr(
        builder.CreateICmpULT(init_index, expected_packet_count),
        init_body, init_done);

    builder.SetInsertPoint(init_body);
    auto *thread_index = builder.CreateMul(
        init_index, builder.getInt32(specialization_width));
    builder.CreateStore(thread_index, thread_index_address);
    auto *handle = builder.CreateCall(
        packet_entry,
        {argument_buffer, return_lanes, launch_config,
         builder.getInt32(specialization_width)},
        "cooperative.handle");
    auto *handle_slot = builder.CreateInBoundsGEP(
        handle_array_type, handles,
        {builder.getInt32(0u), init_index});
    builder.CreateStore(handle, handle_slot);
    auto *init_next = builder.CreateAdd(
        init_index, builder.getInt32(1u));
    builder.CreateBr(init_head);
    init_index->addIncoming(init_next, init_body);

    builder.SetInsertPoint(init_done);
    builder.CreateBr(phase_begin);

    builder.SetInsertPoint(phase_begin);
    builder.CreateStore(builder.getFalse(), any_alive);
    builder.CreateStore(builder.getFalse(), any_complete);
    builder.CreateStore(builder.getFalse(), barrier_seen);
    builder.CreateStore(builder.getInt32(0u), barrier_value);
    builder.CreateBr(scan_head);

    builder.SetInsertPoint(scan_head);
    auto *scan_index = builder.CreatePHI(
        i32_type, 2u, "cooperative.scan.index");
    scan_index->addIncoming(builder.getInt32(0u), phase_begin);
    auto *scan_handle_slot = builder.CreateInBoundsGEP(
        handle_array_type, handles,
        {builder.getInt32(0u), scan_index});
    auto *scan_handle = builder.CreateLoad(
        pointer_type, scan_handle_slot);
    builder.CreateCondBr(
        builder.CreateIsNull(scan_handle),
        scan_update, scan_done_test);

    builder.SetInsertPoint(scan_done_test);
    auto *done = builder.CreateIntrinsic(
        i1_type, ::llvm::Intrinsic::coro_done, {scan_handle});
    builder.CreateCondBr(done, scan_complete, scan_alive);

    auto barrier_slot = [&](::llvm::Value *index) {
        return builder.CreateInBoundsGEP(
            barrier_array_type, barrier_ids,
            {builder.getInt32(0u), index});
    };
    builder.SetInsertPoint(scan_complete);
    auto *complete_status = builder.CreateLoad(
        i32_type, barrier_slot(scan_index));
    auto *is_inactive = builder.CreateICmpEQ(
        complete_status,
        builder.getInt32(simd_cooperative_packet_inactive));
    auto *is_complete = builder.CreateICmpEQ(
        complete_status,
        builder.getInt32(simd_cooperative_packet_complete));
    builder.CreateCondBr(
        builder.CreateOr(is_inactive, is_complete),
        scan_complete_valid, trap);

    builder.SetInsertPoint(scan_complete_valid);
    builder.CreateIntrinsic(
        void_type, ::llvm::Intrinsic::coro_destroy, {scan_handle});
    builder.CreateStore(
        ::llvm::ConstantPointerNull::get(pointer_type),
        scan_handle_slot);
    auto *old_complete = builder.CreateLoad(
        i1_type, any_complete);
    builder.CreateStore(
        builder.CreateOr(old_complete, is_complete), any_complete);
    builder.CreateBr(scan_update);

    builder.SetInsertPoint(scan_alive);
    auto *alive_status = builder.CreateLoad(
        i32_type, barrier_slot(scan_index));
    builder.CreateCondBr(
        builder.CreateICmpULT(
            alive_status, builder.getInt32(
                              static_cast<uint32_t>(
                                  block_barrier_count))),
        scan_alive_valid, trap);

    builder.SetInsertPoint(scan_alive_valid);
    auto *seen = builder.CreateLoad(i1_type, barrier_seen);
    auto *expected_barrier = builder.CreateLoad(
        i32_type, barrier_value);
    auto *mismatch = builder.CreateAnd(
        seen,
        builder.CreateICmpNE(alive_status, expected_barrier));
    builder.CreateCondBr(mismatch, trap, scan_record);

    builder.SetInsertPoint(scan_record);
    builder.CreateStore(builder.getTrue(), any_alive);
    builder.CreateStore(builder.getTrue(), barrier_seen);
    builder.CreateStore(alive_status, barrier_value);
    builder.CreateBr(scan_update);

    builder.SetInsertPoint(scan_update);
    auto *scan_next = builder.CreateAdd(
        scan_index, builder.getInt32(1u));
    builder.CreateCondBr(
        builder.CreateICmpULT(scan_next, expected_packet_count),
        scan_head, scan_finished);
    scan_index->addIncoming(scan_next, scan_update);

    builder.SetInsertPoint(scan_finished);
    auto *has_alive = builder.CreateLoad(i1_type, any_alive);
    auto *has_complete = builder.CreateLoad(i1_type, any_complete);
    builder.CreateCondBr(
        builder.CreateAnd(has_alive, has_complete), trap, phase_post);

    builder.SetInsertPoint(phase_post);
    builder.CreateCondBr(has_alive, phase_live, exit);

    builder.SetInsertPoint(phase_live);
    builder.CreateFence(::llvm::AtomicOrdering::AcquireRelease);
    builder.CreateBr(resume_head);

    builder.SetInsertPoint(resume_head);
    auto *resume_index = builder.CreatePHI(
        i32_type, 2u, "cooperative.resume.index");
    resume_index->addIncoming(builder.getInt32(0u), phase_live);
    auto *resume_handle_slot = builder.CreateInBoundsGEP(
        handle_array_type, handles,
        {builder.getInt32(0u), resume_index});
    auto *resume_handle = builder.CreateLoad(
        pointer_type, resume_handle_slot);
    builder.CreateCondBr(
        builder.CreateIsNotNull(resume_handle),
        resume_invoke, resume_update);

    builder.SetInsertPoint(resume_invoke);
    builder.CreateStore(
        builder.getInt32(simd_cooperative_packet_running),
        barrier_slot(resume_index));
    builder.CreateIntrinsic(
        void_type, ::llvm::Intrinsic::coro_resume, {resume_handle});
    builder.CreateBr(resume_update);

    builder.SetInsertPoint(resume_update);
    auto *resume_next = builder.CreateAdd(
        resume_index, builder.getInt32(1u));
    builder.CreateCondBr(
        builder.CreateICmpULT(resume_next, expected_packet_count),
        resume_head, phase_begin);
    resume_index->addIncoming(resume_next, resume_update);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();

    builder.SetInsertPoint(trap);
#if LLVM_VERSION_MAJOR >= 22
    auto *trap_intrinsic = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
    auto *trap_intrinsic = ::llvm::Intrinsic::getDeclaration(
#endif
        &module, ::llvm::Intrinsic::trap);
    builder.CreateCall(trap_intrinsic);
    builder.CreateUnreachable();
    return wrapper;
}

}// namespace luisa::compute::simd::detail
