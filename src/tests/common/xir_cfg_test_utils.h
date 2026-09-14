#pragma once

#include <array>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::test {

// Check the physical structured-CFG contract independently of pass counters.
// A binary branch may be a loop prepare, or have a direct target owned by a
// construct enclosing its source. Ordinary divergence still needs an If.
[[nodiscard]] inline bool raw_conditional_has_structured_owner(
    compute::xir::Function *function, compute::xir::ConditionalBranchInst *branch,
    const compute::xir::DomTree &dominance) noexcept {
    using namespace compute::xir;
    auto *block = branch->parent_block();
    for (auto *owner : function->basic_blocks()) {
        auto *term = owner->terminator();
        BasicBlock *merge = nullptr;
        std::array<BasicBlock *, 3u> boundaries{};
        if (term->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(term);
            merge = loop->merge_block();
            if (loop->prepare_block() == block &&
                branch->true_block() == loop->body_block() && branch->false_block() == merge) { return true; }
            boundaries = {merge, loop->prepare_block(), loop->update_block()};
        } else if (term->isa<SimpleLoopInst>()) {
            auto *loop = static_cast<SimpleLoopInst *>(term);
            merge = loop->merge_block();
            boundaries = {merge, loop->body_block(), nullptr};
        } else if (term->isa<SwitchInst>()) {
            merge = static_cast<SwitchInst *>(term)->merge_block();
            boundaries[0] = merge;
        } else if (term->isa<IfInst>()) {
            merge = static_cast<IfInst *>(term)->merge_block();
            boundaries[0] = merge;
        }
        if (merge == nullptr || !dominance.strictly_dominates(owner, block) ||
            (merge != block && dominance.dominates(merge, block))) { continue; }
        for (auto *boundary : boundaries) {
            if (boundary != nullptr &&
                (boundary == branch->true_block() || boundary == branch->false_block())) { return true; }
        }
    }
    return false;
}

}// namespace luisa::test
