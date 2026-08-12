#pragma once

#include <cstdint>
#include <limits>

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
    bool _must_analysis_shape_valid{false};
    luisa::unordered_map<uint32_t, luisa::vector<BasicBlock *>>
        _suspend_blocks;
    luisa::unordered_map<uint32_t, luisa::vector<BasicBlock *>>
        _resume_blocks;

public:
    explicit CoroTransferGraph(FunctionDefinition *definition) noexcept {
        if (definition == nullptr || definition->body_block() == nullptr) {
            return;
        }
        _must_analysis_shape_valid = true;
        // A resume block is deliberately disconnected from the ordinary CFG:
        // the matching suspend token is its incoming semantic edge. Walking
        // from body_block therefore cannot discover the very blocks needed to
        // construct this relation. Index the function's owned block set first,
        // then use the resulting transfer graph for executable traversal.
        for (auto *block : definition->basic_blocks()) {
            if (block == nullptr || block->parent_function() != definition ||
                !block->is_terminated()) {
                _must_analysis_shape_valid = false;
                continue;
            }
            auto resume_count = size_t{0u};
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<CoroSuspendInst>()) {
                    auto *suspend =
                        static_cast<CoroSuspendInst *>(instruction);
                    _suspend_blocks[suspend->token()].emplace_back(block);
                } else if (instruction->isa<CoroResumeInst>()) {
                    ++resume_count;
                    auto *resume =
                        static_cast<CoroResumeInst *>(instruction);
                    _resume_blocks[resume->token()].emplace_back(block);
                }
            }
            if (resume_count > 1u ||
                (block == definition->body_block() && resume_count != 0u)) {
                _must_analysis_shape_valid = false;
            }
        }
    }

    // Dominance and other must analyses cannot choose among ambiguous token
    // transfers. Require one nonzero terminator suspend and one resume for
    // every token. Reachability clients may still use traverse_successors on
    // malformed input to retain every possible continuation for diagnostics.
    [[nodiscard]] bool has_unique_complete_token_pairs() const noexcept {
        if (!_must_analysis_shape_valid ||
            _suspend_blocks.size() != _resume_blocks.size()) {
            return false;
        }
        for (auto &[token, suspends] : _suspend_blocks) {
            if (token == 0u ||
                token == std::numeric_limits<uint32_t>::max() ||
                suspends.size() != 1u) {
                return false;
            }
            auto resume_iter = _resume_blocks.find(token);
            if (resume_iter == _resume_blocks.end() ||
                resume_iter->second.size() != 1u) {
                return false;
            }
            auto *suspend_block = suspends.front();
            if (suspend_block == nullptr || !suspend_block->is_terminated() ||
                !suspend_block->terminator()->isa<CoroSuspendInst>() ||
                static_cast<CoroSuspendInst *>(suspend_block->terminator())
                        ->token() != token) {
                return false;
            }
        }
        return true;
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
