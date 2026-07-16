#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_break_continue.h>

namespace luisa::compute::xir {

namespace detail {

static void lower_break_continue_in_function(Function *function, LowerBreakContinueInfo &info) noexcept {
    auto def = function->definition();
    if (def == nullptr) { return; }
    luisa::vector<BasicBlock *> break_blocks;
    luisa::vector<BasicBlock *> continue_blocks;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (!block->is_terminated()) { return; }
        auto terminator = block->terminator();
        if (terminator->isa<BreakInst>()) {
            break_blocks.emplace_back(block);
        } else if (terminator->isa<ContinueInst>()) {
            continue_blocks.emplace_back(block);
        }
    });
    if (break_blocks.empty() && continue_blocks.empty()) { return; }
    XIRBuilder b;
    for (auto block : break_blocks) {
        auto break_inst = static_cast<BreakInst *>(block->terminator());
        auto target = break_inst->target_block();
        LUISA_DEBUG_ASSERT(target != nullptr, "BreakInst with null target block.");
        break_inst->remove_self();
        b.set_insertion_point(block);
        b.br(target);
    }
    for (auto block : continue_blocks) {
        auto continue_inst = static_cast<ContinueInst *>(block->terminator());
        auto target = continue_inst->target_block();
        LUISA_DEBUG_ASSERT(target != nullptr, "ContinueInst with null target block.");
        continue_inst->remove_self();
        b.set_insertion_point(block);
        b.br(target);
    }
    info.lowered_break_count += break_blocks.size();
    info.lowered_continue_count += continue_blocks.size();
}

}// namespace detail

LowerBreakContinueInfo lower_break_continue_pass_run_on_function(Function *function) noexcept {
    LowerBreakContinueInfo info;
    detail::lower_break_continue_in_function(function, info);
    return info;
}

LowerBreakContinueInfo lower_break_continue_pass_run_on_module(Module *module) noexcept {
    LowerBreakContinueInfo info;
    for (auto f : module->function_list()) {
        detail::lower_break_continue_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
