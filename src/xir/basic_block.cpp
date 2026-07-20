#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

void BasicBlock::_do_traverse_predecessors(bool exclude_self, void *ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> visited;
    // we can find all predecessors by traversing all users of the block and find their containing blocks
    for (auto &&use : use_list()) {
        auto user = use->user();
        LUISA_ASSERT(user != nullptr && user->isa<Instruction>(), "Invalid user of basic block.");
        auto user_block = static_cast<Instruction *>(user)->parent_block();
        LUISA_DEBUG_ASSERT(user_block != nullptr, "Invalid parent block.");
        if (!exclude_self || user_block != this) {
            // A terminator may legally reference the same target more than once
            // (e.g. a degenerate conditional branch or switch). CFG traversal is
            // block-based rather than edge-based, so report each predecessor once.
            if (std::find(visited.begin(), visited.end(), user_block) == visited.end()) {
                visited.emplace_back(user_block);
                visit(ctx, user_block);
            }
        }
    }
}

void BasicBlock::_do_traverse_successors(bool exclude_self, void *ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> visited;
    // we can find all successors by finding the block operands of the terminator instruction
    auto terminator = this->terminator();
    for (auto op_use : terminator->operand_uses()) {
        LUISA_DEBUG_ASSERT(op_use != nullptr, "Invalid operand use.");
        if (auto op = op_use->value(); op != nullptr && (!exclude_self || op != this) && op->isa<BasicBlock>()) {
            auto *successor = static_cast<BasicBlock *>(op);
            // See the predecessor traversal above: multiple block operands can
            // represent the same CFG successor, but clients expect a block set.
            if (std::find(visited.begin(), visited.end(), successor) == visited.end()) {
                visited.emplace_back(successor);
                visit(ctx, successor);
            }
        }
    }
}

BasicBlock::BasicBlock(Function *function) noexcept
    : Super{function, nullptr}, _instructions{this} {}

bool BasicBlock::is_terminated() const noexcept {
    return !_instructions.empty() && _instructions.back()->is_terminator();
}

TerminatorInstruction *BasicBlock::terminator() noexcept {
    LUISA_DEBUG_ASSERT(is_terminated(), "Basic block is not terminated.");
    return static_cast<TerminatorInstruction *>(_instructions.back());
}

const TerminatorInstruction *BasicBlock::terminator() const noexcept {
    return const_cast<BasicBlock *>(this)->terminator();
}

}// namespace luisa::compute::xir
