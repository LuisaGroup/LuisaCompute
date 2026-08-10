#pragma once

#include <cstdint>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/coro.h>

namespace luisa::compute::xir::detail {

// Suspend/resume is a semantic CFG edge split across two token-bearing
// instructions rather than an operand edge. CFG transforms that compute
// executable reachability must add these edges or they will erase every
// continuation block as an ordinary orphan.
class CoroTransferGraph {
private:
    luisa::unordered_map<uint32_t, luisa::vector<BasicBlock *>>
        _resume_blocks;

public:
    explicit CoroTransferGraph(FunctionDefinition *definition) noexcept {
        if (definition == nullptr) { return; }
        // A resume block is deliberately disconnected from the ordinary CFG:
        // the matching suspend token is its incoming semantic edge. Walking
        // from body_block therefore cannot discover the very blocks needed to
        // construct this relation. Index the function's owned block set first,
        // then use the resulting transfer graph for executable traversal.
        for (auto *block : definition->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<CoroResumeInst>()) {
                    auto *resume =
                        static_cast<CoroResumeInst *>(instruction);
                    _resume_blocks[resume->token()].emplace_back(block);
                }
            }
        }
    }

    template<typename Visit>
    void traverse_successors(BasicBlock *block, Visit &&visit) const noexcept {
        if (block == nullptr || !block->is_terminated() ||
            !block->terminator()->isa<CoroSuspendInst>()) {
            return;
        }
        auto *suspend =
            static_cast<CoroSuspendInst *>(block->terminator());
        if (auto iter = _resume_blocks.find(suspend->token());
            iter != _resume_blocks.end()) {
            // Retain every match for malformed duplicate-token IR. Validation
            // rejects ambiguity later; reachability must not choose one resume
            // arbitrarily and mutate the diagnostic input.
            for (auto *resume_block : iter->second) {
                visit(resume_block);
            }
        }
    }
};

}// namespace luisa::compute::xir::detail
