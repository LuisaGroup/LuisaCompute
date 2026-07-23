#pragma once

#include <llvm/IR/BasicBlock.h>

namespace lc::llvm_codegen {

namespace detail {

template<typename BasicBlock>
[[nodiscard]] inline llvm::Instruction *terminator_or_null(
    BasicBlock *block) noexcept {
    // Detect the API itself instead of guessing from LLVM_VERSION_MAJOR. This
    // remains correct for downstream LLVM branches that backport the safer
    // construction-time probe without changing their advertised major.
    if constexpr (requires(BasicBlock *candidate) {
                      candidate->getTerminatorOrNull();
                  }) {
        return block->getTerminatorOrNull();
    } else {
        return block->getTerminator();
    }
}

}// namespace detail

// Newer LLVM releases provide getTerminatorOrNull() because getTerminator()
// may assert for incomplete blocks. Older releases expose the nullable
// behavior through getTerminator(). Keep this API boundary centralized.
[[nodiscard]] inline llvm::Instruction *terminator_or_null(
    llvm::BasicBlock *block) noexcept {
    return detail::terminator_or_null(block);
}

}// namespace lc::llvm_codegen
