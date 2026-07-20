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

struct LowerBreakContinueWorklist {
    luisa::vector<BreakInst *> breaks;
    luisa::vector<ContinueInst *> continues;
};

[[nodiscard]] static LowerBreakContinueWorklist collect_break_continue_in_function(
    Function *function, LowerBreakContinueInfo &info) noexcept {
    LowerBreakContinueWorklist worklist;
    if (function == nullptr) { return worklist; }
    auto def = function->definition();
    if (def == nullptr) { return worklist; }
    for (auto *block : def->basic_blocks()) {
        if (block == nullptr || !block->is_terminated()) { continue; }
        auto *terminator = block->terminator();
        if (terminator->isa<BreakInst>()) {
            auto *break_inst = static_cast<BreakInst *>(terminator);
            auto *target = break_inst->target_block();
            if (target == nullptr || target->parent_function() != function) {
                ++info.rejected_break_count;
            } else {
                worklist.breaks.emplace_back(break_inst);
            }
        } else if (terminator->isa<ContinueInst>()) {
            auto *continue_inst = static_cast<ContinueInst *>(terminator);
            auto *target = continue_inst->target_block();
            if (target == nullptr || target->parent_function() != function) {
                ++info.rejected_continue_count;
            } else {
                worklist.continues.emplace_back(continue_inst);
            }
        }
    }
    return worklist;
}

static void lower_break_continue_worklist(
    LowerBreakContinueWorklist &&worklist, LowerBreakContinueInfo &info) noexcept {
    XIRBuilder b;
    for (auto *break_inst : worklist.breaks) {
        auto *block = break_inst->parent_block();
        auto *target = break_inst->target_block();
        break_inst->remove_self();
        b.set_insertion_point(block);
        b.br(target);
    }
    for (auto *continue_inst : worklist.continues) {
        auto *block = continue_inst->parent_block();
        auto *target = continue_inst->target_block();
        continue_inst->remove_self();
        b.set_insertion_point(block);
        b.br(target);
    }
    info.lowered_break_count += worklist.breaks.size();
    info.lowered_continue_count += worklist.continues.size();
}

}// namespace detail

LowerBreakContinueInfo lower_break_continue_pass_run_on_function(Function *function) noexcept {
    LowerBreakContinueInfo info;
    auto worklist = detail::collect_break_continue_in_function(function, info);
    if (info.succeeded()) {
        detail::lower_break_continue_worklist(std::move(worklist), info);
    }
    return info;
}

LowerBreakContinueInfo lower_break_continue_pass_run_on_module(Module *module) noexcept {
    LowerBreakContinueInfo info;
    if (module == nullptr) { return info; }
    luisa::vector<detail::LowerBreakContinueWorklist> worklists;
    for (auto f : module->function_list()) {
        worklists.emplace_back(detail::collect_break_continue_in_function(f, info));
    }
    if (!info.succeeded()) { return info; }
    for (auto &&worklist : worklists) {
        detail::lower_break_continue_worklist(std::move(worklist), info);
    }
    return info;
}

}// namespace luisa::compute::xir
