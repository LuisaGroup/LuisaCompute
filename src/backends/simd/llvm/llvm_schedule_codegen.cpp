#include "llvm_schedule_codegen.h"
#include "llvm_schedule_emitter.h"

#include <cstddef>
#include <limits>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::simd {

namespace {

enum class PacketBatchLowering {
    dynamic_loop,
    unrolled_calls,
    inlined_loop,
};

[[nodiscard]] ::llvm::Function *build_packet_batch_entry(
    ::llvm::Module &module, ::llvm::Function *packet_entry,
    uint32_t specialization_width, uint32_t static_packet_count,
    PacketBatchLowering lowering,
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

[[nodiscard]] ::llvm::Function *build_block_batch_entry(
    ::llvm::Module &module, ::llvm::Function *packet_batch_entry,
    uint32_t static_packet_count, std::string &error) {
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

    auto *prologue = ::llvm::BasicBlock::Create(
        context, "block.batch.prologue", block_entry);
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
    auto result = detail::ScheduleEmitter{
        module, function, specialization_width, entry_name,
        enable_fast_math, static_block_size,
        enable_uniform_buffer_broadcast,
        enable_lane_affine_buffer,
        enable_paired_leaf_gather,
        dispatch_worker_count,
        enable_native_predicated_loop}
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
                specialization_width == 8u) {
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
        result.packet_batch_entry = build_packet_batch_entry(
            module, result.entry, specialization_width,
            static_packet_count, lowering,
            result.error);
        if (result.packet_batch_entry != nullptr &&
            enable_block_batch_entry &&
            result.direct_control_flow &&
            static_packet_count != 0u) {
            result.block_batch_entry = build_block_batch_entry(
                module, result.packet_batch_entry,
                static_packet_count, result.error);
        }
    }
    return result;
}

}// namespace luisa::compute::simd
