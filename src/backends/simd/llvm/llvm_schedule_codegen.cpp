#include "llvm_schedule_codegen.h"
#include "llvm_schedule_emitter.h"

#include <cstddef>
#include <limits>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <luisa/xir/op.h>
#include <luisa/xir/special_register.h>

#include "../../common/env_flag.h"

namespace luisa::compute::simd {

namespace {

enum class PacketBatchLowering {
    dynamic_loop,
    unrolled_calls,
    inlined_loop,
};

void apply_packet_wrapper_abi_attributes(
    ::llvm::Function *entry) noexcept {
    if (luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES")) {
        return;
    }
    // The wrappers mutate launch_config while advancing packets/blocks, so
    // they cannot inherit the packet body's readonly fact for parameter 2.
    // The packed descriptor record is still immutable, and both ABI objects
    // are separately owned for the complete wrapper call.
    entry->addParamAttr(0u, ::llvm::Attribute::NoAlias);
    entry->addParamAttr(0u, ::llvm::Attribute::ReadOnly);
    entry->addParamAttr(2u, ::llvm::Attribute::NoAlias);
    entry->addParamAttr(2u, ::llvm::Attribute::NonNull);
}

[[nodiscard]] bool is_linear_block_agnostic(
    const schedule::Function &function) noexcept {
    std::vector<schedule::ValueId> dispatch_ids;
    for (auto &&value : function.values()) {
        if (value.origin != schedule::ValueOrigin::special_register) {
            continue;
        }
        auto *metadata = std::get_if<
            schedule::SpecialRegisterValueMetadata>(&value.metadata);
        if (metadata == nullptr) { return false; }
        auto tag = static_cast<xir::DerivedSpecialRegisterTag>(
            metadata->tag);
        if (tag == xir::DerivedSpecialRegisterTag::THREAD_ID ||
            tag == xir::DerivedSpecialRegisterTag::BLOCK_ID) {
            return false;
        }
        if (tag == xir::DerivedSpecialRegisterTag::DISPATCH_ID) {
            dispatch_ids.emplace_back(value.id);
        }
    }
    auto is_zero_constant = [&](schedule::ValueId id) noexcept {
        auto *value = function.value(id);
        auto *metadata = value == nullptr ? nullptr :
                                            std::get_if<
                                                schedule::ConstantValueMetadata>(
                                                &value->metadata);
        if (value == nullptr ||
            value->origin != schedule::ValueOrigin::constant ||
            metadata == nullptr || metadata->bytes.empty()) {
            return false;
        }
        for (auto byte : metadata->bytes) {
            if (byte != std::byte{0u}) { return false; }
        }
        return true;
    };
    for (auto &&block : function.blocks()) {
        if (std::holds_alternative<schedule::BlockBarrierTerminator>(
                block.terminator)) {
            return false;
        }
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::alloca) {
                return false;
            }
            for (auto operand_index = size_t{0u};
                 operand_index < instruction.operands.size();
                 operand_index++) {
                auto operand = instruction.operands[operand_index];
                auto is_dispatch_id = false;
                for (auto id : dispatch_ids) {
                    is_dispatch_id |= operand == id;
                }
                if (!is_dispatch_id) { continue; }
                if (operand_index != 0u ||
                    instruction.opcode != schedule::Opcode::arithmetic ||
                    instruction.source_op != static_cast<uint32_t>(
                                                 xir::ArithmeticOp::EXTRACT) ||
                    instruction.operands.size() != 2u ||
                    !is_zero_constant(instruction.operands[1u])) {
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] ::llvm::Function *build_packet_batch_entry(
    ::llvm::Module &module, ::llvm::Function *packet_entry,
    uint32_t specialization_width, uint32_t static_packet_count,
    uint32_t static_block_size_x, PacketBatchLowering lowering,
    bool enable_linear_1d_packet_tail_narrowing,
    bool enable_linear_1d_block_coalescing,
    std::string &error) {
    auto name = packet_entry->getName().str() + ".packet_batch";
    if (module.getFunction(name) != nullptr) {
        error = "duplicate SIMD packet-batch entry '" + name + "'";
        return nullptr;
    }
    auto &context = module.getContext();
    auto *i32_type = ::llvm::Type::getInt32Ty(context);
    auto *zero = ::llvm::ConstantInt::get(i32_type, 0u);
    auto *one = ::llvm::ConstantInt::get(i32_type, 1u);
    auto *width = ::llvm::ConstantInt::get(
        i32_type, specialization_width);
    auto *batch_entry = ::llvm::Function::Create(
        packet_entry->getFunctionType(),
        ::llvm::GlobalValue::ExternalLinkage, name, module);
    apply_packet_wrapper_abi_attributes(batch_entry);
    // The runtime discovers only the block entry. Keeping the packet body
    // internal both makes PIC calls non-preemptable and lets GlobalDCE remove
    // the original after the W8 loop has inlined its sole body copy.
    packet_entry->setLinkage(::llvm::GlobalValue::InternalLinkage);
    packet_entry->setDSOLocal(true);
    batch_entry->setDSOLocal(true);
    auto argument = batch_entry->arg_begin();
    auto *argument_buffer = &*argument++;
    auto *return_lanes = &*argument++;
    auto *launch_config = &*argument++;
    auto *packet_count = &*argument;
    argument_buffer->setName("argument_buffer");
    return_lanes->setName("return_lanes");
    launch_config->setName("launch_config");
    packet_count->setName("packet_count");

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "packet.batch.prologue", batch_entry);
    ::llvm::IRBuilder<> builder{prologue};
    auto *thread_index_address = builder.CreateConstInBoundsGEP1_64(
        ::llvm::Type::getInt8Ty(context), launch_config,
        offsetof(SIMDPacketLaunchConfig, thread_index),
        "thread.index.address");
    auto *base_thread_index = builder.CreateLoad(
        i32_type, thread_index_address, "base.thread.index");
    if (!enable_linear_1d_packet_tail_narrowing) {
        if (lowering == PacketBatchLowering::unrolled_calls) {
            for (auto packet = uint32_t{0u};
                 packet < static_packet_count; packet++) {
                auto *thread_index = packet == 0u ?
                                         base_thread_index :
                                         builder.CreateAdd(
                                             base_thread_index,
                                             ::llvm::ConstantInt::get(
                                                 i32_type,
                                                 packet * specialization_width),
                                             "packet.thread.index");
                builder.CreateStore(thread_index, thread_index_address);
                builder.CreateCall(
                    packet_entry,
                    {argument_buffer, return_lanes, launch_config, width});
            }
            builder.CreateRetVoid();
            return batch_entry;
        }

        auto *loop = ::llvm::BasicBlock::Create(
            context, "packet.batch.loop", batch_entry);
        auto *exit = ::llvm::BasicBlock::Create(
            context, "packet.batch.exit", batch_entry);
        auto *empty = builder.CreateICmpEQ(
            packet_count, zero, "packet.batch.empty");
        builder.CreateCondBr(empty, exit, loop);

        builder.SetInsertPoint(loop);
        auto *packet_index = builder.CreatePHI(
            i32_type, 2u, "packet.index");
        packet_index->addIncoming(zero, prologue);
        auto *thread_offset = builder.CreateMul(
            packet_index, width, "packet.thread.offset");
        auto *thread_index = builder.CreateAdd(
            base_thread_index, thread_offset, "packet.thread.index");
        builder.CreateStore(thread_index, thread_index_address);
        auto *packet_call = builder.CreateCall(
            packet_entry,
            {argument_buffer, return_lanes, launch_config, width});
        if (lowering == PacketBatchLowering::inlined_loop) {
            packet_call->addFnAttr(::llvm::Attribute::AlwaysInline);
        }
        auto *next_packet = builder.CreateAdd(
            packet_index, one, "packet.index.next");
        auto *has_more = builder.CreateICmpULT(
            next_packet, packet_count, "packet.batch.has.more");
        builder.CreateCondBr(has_more, loop, exit);
        packet_index->addIncoming(next_packet, loop);

        builder.SetInsertPoint(exit);
        builder.CreateRetVoid();
        return batch_entry;
    }

    // A runtime 1D packet range is normally a prefix of one block. A proven
    // block-agnostic body may instead receive a range spanning consecutive
    // blocks from the block-batch wrapper. Split either form into a
    // constant-width main path plus at most one narrowed tail packet. This
    // makes the packet body's active-lane mask constant all-on in the hot
    // path without cloning the kernel or extending its domain at an edge.
    auto *i8_type = ::llvm::Type::getInt8Ty(context);
    auto *i64_type = ::llvm::Type::getInt64Ty(context);
    auto *zero64 = ::llvm::ConstantInt::get(i64_type, 0u);
    auto *width64 = ::llvm::ConstantInt::get(
        i64_type, specialization_width);
    auto *block_size64 = ::llvm::ConstantInt::get(
        i64_type, static_block_size_x);
    auto load_launch_u32 = [&](size_t offset,
                               std::string_view value_name) {
        auto *address = builder.CreateConstInBoundsGEP1_64(
            i8_type, launch_config, offset,
            std::string{value_name} + ".address");
        return builder.CreateLoad(
            i32_type, address, std::string{value_name});
    };
    auto minimum = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                       std::string_view value_name) {
        return builder.CreateSelect(
            builder.CreateICmpULT(lhs, rhs), lhs, rhs,
            std::string{value_name});
    };
    auto *block_id = load_launch_u32(
        offsetof(SIMDPacketLaunchConfig, block_id), "block.x");
    auto *dispatch_size = load_launch_u32(
        offsetof(SIMDPacketLaunchConfig, dispatch_size),
        "dispatch.size.x");
    auto *base64 = builder.CreateZExt(
        base_thread_index, i64_type, "base.thread.index.i64");
    auto *block_origin = builder.CreateMul(
        builder.CreateZExt(block_id, i64_type), block_size64,
        "block.origin.x");
    auto *dispatch64 = builder.CreateZExt(
        dispatch_size, i64_type, "dispatch.size.x.i64");
    auto *origin_in_range = builder.CreateICmpULE(
        block_origin, dispatch64, "block.origin.in.range");
    auto *safe_origin = builder.CreateSelect(
        origin_in_range, block_origin, zero64,
        "block.origin.safe");
    auto *global_start = builder.CreateAdd(
        safe_origin, base64, "packet.range.start");
    auto *inside_dispatch = builder.CreateAnd(
        origin_in_range,
        builder.CreateICmpULT(
            global_start, dispatch64),
        "packet.range.inside.dispatch");
    auto *dispatch_remaining = builder.CreateSelect(
        inside_dispatch,
        builder.CreateSub(
            dispatch64, global_start),
        zero64, "dispatch.remaining");
    auto *inside_block = builder.CreateICmpULT(
        base64, block_size64, "packet.range.inside.block");
    auto *block_remaining = builder.CreateSelect(
        inside_block,
        builder.CreateSub(block_size64, base64),
        zero64, "block.remaining");
    auto *range_remaining = enable_linear_1d_block_coalescing ?
                                dispatch_remaining :
                                minimum(
                                    dispatch_remaining,
                                    block_remaining,
                                    "packet.range.remaining");
    auto *requested_threads = builder.CreateMul(
        builder.CreateZExt(packet_count, i64_type), width64,
        "packet.range.requested.threads");
    auto *active_threads = minimum(
        range_remaining, requested_threads,
        "packet.range.active.threads");
    auto *static_packet_count_value = ::llvm::ConstantInt::get(
        i32_type, static_packet_count);
    auto *static_thread_count = ::llvm::ConstantInt::get(
        i64_type,
        static_cast<uint64_t>(static_packet_count) *
            specialization_width);
    auto *complete_static_range =
        enable_linear_1d_block_coalescing ?
            builder.CreateAnd(
                builder.CreateICmpNE(packet_count, zero),
                builder.CreateICmpUGE(
                    range_remaining, requested_threads),
                "packet.range.complete.coalesced") :
            builder.CreateAnd(
                builder.CreateICmpEQ(
                    packet_count, static_packet_count_value),
                builder.CreateICmpUGE(
                    range_remaining, static_thread_count),
                "packet.range.complete.static");
    auto *full = ::llvm::BasicBlock::Create(
        context, "packet.batch.full", batch_entry);
    auto *partial = ::llvm::BasicBlock::Create(
        context, "packet.batch.partial", batch_entry);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "packet.batch.exit", batch_entry);
    builder.CreateCondBr(complete_static_range, full, partial);

    builder.SetInsertPoint(full);
    if (lowering == PacketBatchLowering::unrolled_calls) {
        for (auto packet = uint32_t{0u};
             packet < static_packet_count; packet++) {
            auto *thread_index = packet == 0u ?
                                     base_thread_index :
                                     builder.CreateAdd(
                                         base_thread_index,
                                         ::llvm::ConstantInt::get(
                                             i32_type,
                                             packet * specialization_width),
                                         "packet.thread.index");
            builder.CreateStore(thread_index, thread_index_address);
            builder.CreateCall(
                packet_entry,
                {argument_buffer, return_lanes, launch_config, width});
        }
        builder.CreateBr(exit);
    } else {
        auto *full_loop = ::llvm::BasicBlock::Create(
            context, "packet.batch.full.loop", batch_entry);
        builder.CreateBr(full_loop);
        builder.SetInsertPoint(full_loop);
        auto *packet_index = builder.CreatePHI(
            i32_type, 2u, "packet.index");
        packet_index->addIncoming(zero, full);
        auto *thread_offset = builder.CreateMul(
            packet_index, width, "packet.thread.offset");
        auto *thread_index = builder.CreateAdd(
            base_thread_index, thread_offset, "packet.thread.index");
        builder.CreateStore(thread_index, thread_index_address);
        auto *packet_call = builder.CreateCall(
            packet_entry,
            {argument_buffer, return_lanes, launch_config, width});
        if (lowering == PacketBatchLowering::inlined_loop) {
            packet_call->addFnAttr(::llvm::Attribute::AlwaysInline);
        }
        auto *next_packet = builder.CreateAdd(
            packet_index, one, "packet.index.next");
        auto *has_more = builder.CreateICmpULT(
            next_packet, packet_count, "packet.batch.has.more");
        builder.CreateCondBr(has_more, full_loop, exit);
        packet_index->addIncoming(next_packet, full_loop);
    }

    builder.SetInsertPoint(partial);
    auto *full_packet_count64 = builder.CreateUDiv(
        active_threads, width64, "packet.full.count.i64");
    auto *full_packet_count = builder.CreateTrunc(
        full_packet_count64, i32_type, "packet.full.count");
    auto *tail_lane_count = builder.CreateTrunc(
        builder.CreateURem(
            active_threads, width64,
            "packet.tail.lane.count.i64"),
        i32_type, "packet.tail.lane.count");
    auto *partial_loop = ::llvm::BasicBlock::Create(
        context, "packet.batch.partial.full.loop", batch_entry);
    auto *tail_check = ::llvm::BasicBlock::Create(
        context, "packet.batch.tail.check", batch_entry);
    auto *tail_call = ::llvm::BasicBlock::Create(
        context, "packet.batch.tail.call", batch_entry);
    auto *partial_finish = ::llvm::BasicBlock::Create(
        context, "packet.batch.partial.finish", batch_entry);
    builder.CreateCondBr(
        builder.CreateICmpNE(full_packet_count, zero),
        partial_loop, tail_check);

    builder.SetInsertPoint(partial_loop);
    auto *partial_packet_index = builder.CreatePHI(
        i32_type, 2u, "partial.packet.index");
    partial_packet_index->addIncoming(zero, partial);
    auto *partial_thread_offset = builder.CreateMul(
        partial_packet_index, width,
        "partial.packet.thread.offset");
    auto *partial_thread_index = builder.CreateAdd(
        base_thread_index, partial_thread_offset,
        "partial.packet.thread.index");
    builder.CreateStore(
        partial_thread_index, thread_index_address);
    builder.CreateCall(
        packet_entry,
        {argument_buffer, return_lanes, launch_config, width});
    auto *next_partial_packet = builder.CreateAdd(
        partial_packet_index, one,
        "partial.packet.index.next");
    auto *has_more_full_packets = builder.CreateICmpULT(
        next_partial_packet, full_packet_count,
        "partial.packet.has.more.full");
    builder.CreateCondBr(
        has_more_full_packets, partial_loop, tail_check);
    partial_packet_index->addIncoming(
        next_partial_packet, partial_loop);

    builder.SetInsertPoint(tail_check);
    builder.CreateCondBr(
        builder.CreateICmpNE(tail_lane_count, zero),
        tail_call, partial_finish);

    builder.SetInsertPoint(tail_call);
    auto *tail_thread_offset = builder.CreateMul(
        full_packet_count, width, "packet.tail.thread.offset");
    auto *tail_thread_index = builder.CreateAdd(
        base_thread_index, tail_thread_offset,
        "packet.tail.thread.index");
    builder.CreateStore(tail_thread_index, thread_index_address);
    builder.CreateCall(
        packet_entry,
        {argument_buffer, return_lanes, launch_config,
         tail_lane_count});
    builder.CreateBr(partial_finish);

    builder.SetInsertPoint(partial_finish);
    auto *has_requested_packets = builder.CreateICmpNE(
        packet_count, zero, "packet.batch.has.requested");
    auto *last_requested_packet = builder.CreateSub(
        packet_count, one, "packet.batch.last.requested");
    auto *last_requested_offset = builder.CreateMul(
        last_requested_packet, width,
        "packet.batch.last.requested.offset");
    auto *last_requested_index = builder.CreateAdd(
        base_thread_index, last_requested_offset,
        "packet.batch.last.requested.index");
    builder.CreateStore(
        builder.CreateSelect(
            has_requested_packets, last_requested_index,
            base_thread_index),
        thread_index_address);
    builder.CreateBr(exit);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    return batch_entry;
}

[[nodiscard]] ::llvm::Function *build_block_batch_entry(
    ::llvm::Module &module, ::llvm::Function *packet_batch_entry,
    uint32_t specialization_width, uint32_t static_packet_count,
    bool enable_linear_1d_block_coalescing, std::string &error) {
    if (static_packet_count == 0u) {
        error = "SIMD block-batch entry requires a static packet count";
        return nullptr;
    }
    auto name = packet_batch_entry->getName().str() + ".blocks";
    if (module.getFunction(name) != nullptr) {
        error = "duplicate SIMD block-batch entry '" + name + "'";
        return nullptr;
    }
    auto &context = module.getContext();
    auto *i8_type = ::llvm::Type::getInt8Ty(context);
    auto *i32_type = ::llvm::Type::getInt32Ty(context);
    auto *zero = ::llvm::ConstantInt::get(i32_type, 0u);
    auto *one = ::llvm::ConstantInt::get(i32_type, 1u);
    auto *packet_count = ::llvm::ConstantInt::get(
        i32_type, static_packet_count);
    auto *block_entry = ::llvm::Function::Create(
        packet_batch_entry->getFunctionType(),
        ::llvm::GlobalValue::ExternalLinkage, name, module);
    apply_packet_wrapper_abi_attributes(block_entry);
    packet_batch_entry->setLinkage(
        ::llvm::GlobalValue::InternalLinkage);
    packet_batch_entry->setDSOLocal(true);
    block_entry->setDSOLocal(true);
    auto argument = block_entry->arg_begin();
    auto *argument_buffer = &*argument++;
    auto *return_lanes = &*argument++;
    auto *launch_config = &*argument++;
    auto *block_count = &*argument;
    argument_buffer->setName("argument_buffer");
    return_lanes->setName("return_lanes");
    launch_config->setName("launch_config");
    block_count->setName("block_count");

    ::llvm::BasicBlock *generic_prologue = nullptr;
    if (enable_linear_1d_block_coalescing) {
        auto *guard = ::llvm::BasicBlock::Create(
            context, "block.batch.coalescing.guard", block_entry);
        auto *coalesced = ::llvm::BasicBlock::Create(
            context, "block.batch.coalesced", block_entry);
        generic_prologue = ::llvm::BasicBlock::Create(
            context, "block.batch.generic.prologue", block_entry);
        ::llvm::IRBuilder<> builder{guard};
        auto address = [&](size_t offset,
                           std::string_view value_name) {
            return builder.CreateConstInBoundsGEP1_64(
                i8_type, launch_config, offset, value_name);
        };
        auto *block_x_address = address(
            offsetof(SIMDPacketLaunchConfig, block_id),
            "block.x.address");
        auto *thread_index_address = address(
            offsetof(SIMDPacketLaunchConfig, thread_index),
            "thread.index.address");
        auto *dispatch_y_address = address(
            offsetof(SIMDPacketLaunchConfig, dispatch_size) +
                sizeof(uint32_t),
            "dispatch.y.address");
        auto *dispatch_z_address = address(
            offsetof(SIMDPacketLaunchConfig, dispatch_size) +
                2u * sizeof(uint32_t),
            "dispatch.z.address");
        auto *initial_block_x = builder.CreateLoad(
            i32_type, block_x_address, "initial.block.x");
        auto *dispatch_y = builder.CreateLoad(
            i32_type, dispatch_y_address, "dispatch.y");
        auto *dispatch_z = builder.CreateLoad(
            i32_type, dispatch_z_address, "dispatch.z");
        auto *linear_dispatch = builder.CreateAnd(
            builder.CreateICmpEQ(dispatch_y, one),
            builder.CreateICmpEQ(dispatch_z, one),
            "block.batch.linear.dispatch");
        auto *can_coalesce = builder.CreateAnd(
            builder.CreateICmpNE(block_count, zero),
            linear_dispatch,
            "block.batch.can.coalesce");
        builder.CreateCondBr(
            can_coalesce, coalesced, generic_prologue);

        builder.SetInsertPoint(coalesced);
        builder.CreateStore(zero, thread_index_address);
        auto *total_packet_count = builder.CreateMul(
            block_count, packet_count,
            "block.batch.coalesced.packet.count");
        builder.CreateCall(
            packet_batch_entry,
            {argument_buffer, return_lanes, launch_config,
             total_packet_count});
        auto *last_block = builder.CreateAdd(
            initial_block_x,
            builder.CreateSub(block_count, one),
            "block.batch.last.block.x");
        builder.CreateStore(last_block, block_x_address);
        builder.CreateStore(
            ::llvm::ConstantInt::get(
                i32_type,
                (static_packet_count - 1u) *
                    specialization_width),
            thread_index_address);
        builder.CreateRetVoid();
    }

    auto *prologue = generic_prologue == nullptr ?
                         ::llvm::BasicBlock::Create(
                             context, "block.batch.prologue", block_entry) :
                         generic_prologue;
    ::llvm::IRBuilder<> builder{prologue};
    auto address = [&](size_t offset, std::string_view value_name) {
        return builder.CreateConstInBoundsGEP1_64(
            i8_type, launch_config, offset, value_name);
    };
    auto *block_x_address = address(
        offsetof(SIMDPacketLaunchConfig, block_id),
        "block.x.address");
    auto *block_y_address = address(
        offsetof(SIMDPacketLaunchConfig, block_id) + sizeof(uint32_t),
        "block.y.address");
    auto *block_z_address = address(
        offsetof(SIMDPacketLaunchConfig, block_id) + 2u * sizeof(uint32_t),
        "block.z.address");
    auto *thread_index_address = address(
        offsetof(SIMDPacketLaunchConfig, thread_index),
        "thread.index.address");
    auto *grid_x_address = address(
        offsetof(SIMDPacketLaunchConfig, grid_size),
        "grid.x.address");
    auto *grid_y_address = address(
        offsetof(SIMDPacketLaunchConfig, grid_size) + sizeof(uint32_t),
        "grid.y.address");
    auto *initial_block_x = builder.CreateLoad(
        i32_type, block_x_address, "initial.block.x");
    auto *initial_block_y = builder.CreateLoad(
        i32_type, block_y_address, "initial.block.y");
    auto *initial_block_z = builder.CreateLoad(
        i32_type, block_z_address, "initial.block.z");
    auto *grid_x = builder.CreateLoad(
        i32_type, grid_x_address, "grid.x");
    auto *grid_y = builder.CreateLoad(
        i32_type, grid_y_address, "grid.y");
    auto *loop = ::llvm::BasicBlock::Create(
        context, "block.batch.loop", block_entry);
    auto *exit = ::llvm::BasicBlock::Create(
        context, "block.batch.exit", block_entry);
    auto *empty = builder.CreateICmpEQ(
        block_count, zero, "block.batch.empty");
    builder.CreateCondBr(empty, exit, loop);

    builder.SetInsertPoint(loop);
    auto *block_index = builder.CreatePHI(
        i32_type, 2u, "block.index");
    auto *block_x = builder.CreatePHI(
        i32_type, 2u, "block.x");
    auto *block_y = builder.CreatePHI(
        i32_type, 2u, "block.y");
    auto *block_z = builder.CreatePHI(
        i32_type, 2u, "block.z");
    block_index->addIncoming(zero, prologue);
    block_x->addIncoming(initial_block_x, prologue);
    block_y->addIncoming(initial_block_y, prologue);
    block_z->addIncoming(initial_block_z, prologue);
    builder.CreateStore(block_x, block_x_address);
    builder.CreateStore(block_y, block_y_address);
    builder.CreateStore(block_z, block_z_address);
    builder.CreateStore(zero, thread_index_address);
    builder.CreateCall(
        packet_batch_entry,
        {argument_buffer, return_lanes, launch_config, packet_count});

    auto *incremented_x = builder.CreateAdd(
        block_x, one, "block.x.incremented");
    auto *wrap_x = builder.CreateICmpEQ(
        incremented_x, grid_x, "block.x.wrap");
    auto *next_x = builder.CreateSelect(
        wrap_x, zero, incremented_x, "block.x.next");
    auto *increment_y = builder.CreateZExt(
        wrap_x, i32_type, "block.y.increment");
    auto *incremented_y = builder.CreateAdd(
        block_y, increment_y, "block.y.incremented");
    auto *at_grid_y = builder.CreateICmpEQ(
        incremented_y, grid_y, "block.y.at.end");
    auto *wrap_y = builder.CreateAnd(
        wrap_x, at_grid_y, "block.y.wrap");
    auto *next_y = builder.CreateSelect(
        wrap_y, zero, incremented_y, "block.y.next");
    auto *increment_z = builder.CreateZExt(
        wrap_y, i32_type, "block.z.increment");
    auto *next_z = builder.CreateAdd(
        block_z, increment_z, "block.z.next");
    auto *next_block = builder.CreateAdd(
        block_index, one, "block.index.next");
    auto *has_more = builder.CreateICmpULT(
        next_block, block_count, "block.batch.has.more");
    builder.CreateCondBr(has_more, loop, exit);
    block_index->addIncoming(next_block, loop);
    block_x->addIncoming(next_x, loop);
    block_y->addIncoming(next_y, loop);
    block_z->addIncoming(next_z, loop);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    return block_entry;
}

}// namespace

LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name,
    bool enable_fast_math,
    std::array<uint32_t, 3u> static_block_size,
    bool enable_uniform_buffer_broadcast,
    bool enable_lane_affine_buffer,
    bool enable_paired_leaf_gather,
    uint32_t dispatch_worker_count,
    bool enable_native_predicated_loop,
    bool enable_packet_batch_entry,
    bool enable_inlined_packet_batch,
    bool enable_block_batch_entry) {
    auto enable_linear_1d_packet_tail_narrowing =
        enable_packet_batch_entry &&
        specialization_width != 0u &&
        static_block_size[0u] != 0u &&
        static_block_size[0u] % specialization_width == 0u &&
        static_block_size[1u] == 1u &&
        static_block_size[2u] == 1u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LINEAR_1D_PACKET_TAIL_NARROWING");
    auto schedule_instruction_count = size_t{0u};
    for (auto &&block : function.blocks()) {
        schedule_instruction_count += block.instructions.size();
    }
    // W16 loop inlining has opposite effects on the two representative 1D
    // shapes: it removes repeated descriptor loads and large call shells from
    // a small straight-line AoS kernel, but increases loop/register pressure
    // for a mixed-mask CFG. Keep this deliberately bounded instead of cloning
    // arbitrary Schedule state machines into the packet loop.
    auto enable_bounded_w16_packet_inline =
        enable_linear_1d_packet_tail_narrowing &&
        specialization_width == 16u &&
        function.blocks().size() == 1u &&
        schedule_instruction_count >= 8u &&
        schedule_instruction_count <= 32u &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_W16_LINEAR_1D_PACKET_INLINE");
    auto enable_linear_1d_block_coalescing_candidate =
        enable_block_batch_entry &&
        enable_linear_1d_packet_tail_narrowing &&
        function.blocks().size() == 1u &&
        schedule_instruction_count <= 32u &&
        is_linear_block_agnostic(function) &&
        !luisa::compute::detail::env_flag(
            "LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING");
    auto result = detail::ScheduleEmitter{
        module, function, specialization_width, entry_name,
        enable_fast_math, static_block_size,
        enable_uniform_buffer_broadcast,
        enable_lane_affine_buffer,
        enable_paired_leaf_gather,
        dispatch_worker_count,
        enable_native_predicated_loop,
        enable_packet_batch_entry,
        enable_linear_1d_packet_tail_narrowing}
                      .run();
    if (result.succeeded() && enable_packet_batch_entry) {
        constexpr auto max_specialized_packet_count = uint64_t{32u};
        auto block_thread_count = uint64_t{1u};
        for (auto dimension : static_block_size) {
            if (dimension == 0u) {
                block_thread_count = 0u;
                break;
            }
            if (block_thread_count >
                std::numeric_limits<uint64_t>::max() / dimension) {
                block_thread_count = 0u;
                break;
            }
            block_thread_count *= dimension;
        }
        auto packet_count = block_thread_count == 0u ||
                                    block_thread_count %
                                            specialization_width !=
                                        0u ?
                                uint64_t{0u} :
                                block_thread_count /
                                    specialization_width;
        auto lowering = PacketBatchLowering::dynamic_loop;
        if (packet_count != 0u &&
            packet_count <= max_specialized_packet_count) {
            if (enable_inlined_packet_batch &&
                (specialization_width == 8u ||
                 enable_bounded_w16_packet_inline)) {
                lowering = PacketBatchLowering::inlined_loop;
            } else if (specialization_width == 8u ||
                       specialization_width == 16u) {
                lowering = PacketBatchLowering::unrolled_calls;
            }
        }
        auto static_packet_count =
            packet_count <=
                    std::numeric_limits<uint32_t>::max() ?
                static_cast<uint32_t>(packet_count) :
                0u;
        auto enable_linear_1d_block_coalescing =
            enable_linear_1d_block_coalescing_candidate &&
            result.direct_control_flow &&
            lowering == PacketBatchLowering::inlined_loop &&
            static_packet_count != 0u;
        result.packet_batch_entry = build_packet_batch_entry(
            module, result.entry, specialization_width,
            static_packet_count, static_block_size[0u], lowering,
            enable_linear_1d_packet_tail_narrowing,
            enable_linear_1d_block_coalescing,
            result.error);
        if (result.packet_batch_entry != nullptr &&
            enable_block_batch_entry &&
            result.direct_control_flow &&
            static_packet_count != 0u) {
            result.block_batch_entry = build_block_batch_entry(
                module, result.packet_batch_entry,
                specialization_width, static_packet_count,
                enable_linear_1d_block_coalescing, result.error);
            if (result.block_batch_entry != nullptr &&
                enable_linear_1d_block_coalescing) {
                result.linear_1d_block_coalescing_count++;
            }
        }
    }
    return result;
}

}// namespace luisa::compute::simd
