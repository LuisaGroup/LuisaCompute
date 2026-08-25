#include "block_barrier.h"

#include <utility>
#include <vector>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/thread_group.h>

namespace luisa::compute::simd::schedule {

namespace {

[[nodiscard]] bool is_block_barrier(
    const xir::Instruction *instruction) noexcept {
    return instruction != nullptr &&
           instruction->isa<xir::ThreadGroupInst>() &&
           static_cast<const xir::ThreadGroupInst *>(instruction)->op() ==
               xir::ThreadGroupOp::SYNCHRONIZE_BLOCK;
}

}// namespace

BlockBarrierCanonicalizationResult canonicalize_block_barriers(
    xir::Function *function) noexcept {
    BlockBarrierCanonicalizationResult result;
    if (function == nullptr || function->definition() == nullptr ||
        function->definition()->body_block() == nullptr) {
        result.error =
            "block-barrier canonicalization requires a function body";
        return result;
    }

    std::vector<xir::BasicBlock *> blocks;
    function->definition()->traverse_basic_blocks(
        [&](xir::BasicBlock *block) noexcept {
            blocks.emplace_back(block);
        });
    xir::XIRBuilder builder;
    for (auto block_index = size_t{0u};
         block_index < blocks.size(); block_index++) {
        auto *block = blocks[block_index];
        std::vector<xir::Instruction *> instructions;
        for (auto *instruction : block->instructions()) {
            instructions.emplace_back(instruction);
        }
        auto barrier_index = instructions.size();
        for (auto i = size_t{0u}; i < instructions.size(); i++) {
            if (is_block_barrier(instructions[i])) {
                barrier_index = i;
                break;
            }
        }
        if (barrier_index == instructions.size()) { continue; }
        result.barrier_count++;
        if (!block->is_terminated() || instructions.empty() ||
            instructions.back() != block->terminator()) {
            result.error =
                "block barrier appears in an unterminated XIR block";
            return result;
        }
        if (barrier_index + 1u == instructions.size() - 1u) {
            continue;
        }

        std::vector<xir::BasicBlock *> original_successors;
        block->traverse_successors(
            false, [&](xir::BasicBlock *successor) noexcept {
                original_successors.emplace_back(successor);
            });
        auto *resume = function->create_basic_block();
        for (auto i = barrier_index + 1u;
             i < instructions.size(); i++) {
            // Instruction::remove_self detaches every operand Use. Reinsert
            // through the destination list's virtual insert-before hook so
            // parent_block and all operand use-lists are restored. A raw
            // builder append uses insert-after on the previous instruction
            // and is therefore not a valid cross-block move operation.
            resume->instructions().push_back(
                instructions[i]->remove_self());
        }
        for (auto *successor : original_successors) {
            for (auto *instruction : successor->instructions()) {
                if (!instruction->isa<xir::PhiInst>()) { break; }
                auto *phi = static_cast<xir::PhiInst *>(instruction);
                for (auto i = size_t{0u};
                     i < phi->incoming_count(); i++) {
                    auto incoming = phi->incoming(i);
                    if (incoming.block == block) {
                        phi->set_incoming(i, incoming.value, resume);
                    }
                }
            }
        }
        builder.set_insertion_point(block);
        builder.br(resume);
        blocks.emplace_back(resume);
        result.split_block_count++;
    }
    return result;
}

}// namespace luisa::compute::simd::schedule
