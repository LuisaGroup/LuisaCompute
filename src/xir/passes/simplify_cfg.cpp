#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/simplify_cfg.h>

namespace luisa::compute::xir {

namespace detail {

static void retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr) return;
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto br = static_cast<BranchInst *>(term);
            if (br->target_block() == from) br->set_target_block(to);
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto cb = static_cast<ConditionalBranchInst *>(term);
            if (cb->true_block() == from) cb->set_true_target(to);
            if (cb->false_block() == from) cb->set_false_target(to);
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto sw = static_cast<SwitchInst *>(term);
            if (sw->default_block() == from) sw->set_default_block(to);
            for (size_t i = 0u; i < sw->case_count(); ++i) {
                if (sw->case_block(i) == from) sw->set_case_block(i, to);
            }
            break;
        }
        default: break;
    }
}

static bool fold_constant_cond_br(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    luisa::vector<ConditionalBranchInst *> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb == nullptr) { return; }
        auto t = bb->terminator();
        if (t != nullptr && t->isa<ConditionalBranchInst>()) {
            auto cb = static_cast<ConditionalBranchInst *>(t);
            auto cond = cb->condition();
            if (cond != nullptr && cond->isa<Constant>()) {
                auto cc = static_cast<Constant *>(cond);
                if (cc->type() != nullptr && cc->type()->is_bool()) {
                    targets.push_back(cb);
                }
            }
        }
    });
    if (targets.empty()) return false;
    for (auto cb : targets) {
        auto bb = cb->parent_block();
        if (bb == nullptr) { continue; }
        auto c = static_cast<Constant *>(cb->condition());
        auto taken = c->as<bool>() ? cb->true_block() : cb->false_block();
        if (taken == nullptr) { continue; }
        cb->remove_self();
        XIRBuilder b;
        b.set_insertion_point(bb);
        b.br(taken);
        ++info.folded_constant_cond_br_count;
    }
    return true;
}

static bool thread_empty_blocks(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    auto entry = def->body_block();
    luisa::vector<BasicBlock *> candidates;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb == entry) return;
        auto &insts = bb->instructions();
        if (insts.empty()) return;
        if (insts.front() != bb->terminator()) return;
        auto t = bb->terminator();
        if (t == nullptr || !t->isa<BranchInst>()) return;
        auto br = static_cast<BranchInst *>(t);
        if (br->target_block() == bb) return;
        candidates.push_back(bb);
    });
    if (candidates.empty()) return false;
    bool any = false;
    for (auto bb : candidates) {
        if (bb->terminator() == nullptr) continue;
        auto br = static_cast<BranchInst *>(bb->terminator());
        auto target = br->target_block();
        luisa::vector<BasicBlock *> preds;
        bb->traverse_predecessors(true, [&](BasicBlock *p) noexcept {
            preds.push_back(p);
        });
        bool redirected = false;
        for (auto p : preds) {
            if (p == bb) continue;
            retarget_terminator(p->terminator(), bb, target);
            redirected = true;
        }
        if (redirected) {
            ++info.threaded_empty_block_count;
            any = true;
        }
    }
    return any;
}

static bool remove_unreachable_blocks(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    auto entry = def->body_block();
    if (entry == nullptr) return false;
    luisa::unordered_set<BasicBlock *> reachable;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        reachable.insert(bb);
    });
    luisa::vector<BasicBlock *> dead;
    for (auto bb : def->basic_blocks()) {
        if (bb == entry) continue;
        if (!reachable.contains(bb)) dead.push_back(bb);
    }
    if (dead.empty()) return false;
    for (auto bb : dead) {
        bb->remove_self();
        ++info.removed_unreachable_block_count;
    }
    return true;
}

}// namespace detail

SimplifyCFGInfo simplify_cfg_pass_run_on_function(Function *function) noexcept {
    SimplifyCFGInfo info;
    if (function == nullptr || !function->is_definition()) return info;
    auto def = static_cast<FunctionDefinition *>(function);
    if (def->body_block() == nullptr) return info;
    bool changed = true;
    while (changed) {
        changed = false;
        if (detail::fold_constant_cond_br(def, info)) changed = true;
        if (detail::thread_empty_blocks(def, info)) changed = true;
        if (detail::remove_unreachable_blocks(def, info)) changed = true;
    }
    return info;
}

SimplifyCFGInfo simplify_cfg_pass_run_on_module(Module *module) noexcept {
    SimplifyCFGInfo info;
    if (module == nullptr) return info;
    for (auto f : module->function_list()) {
        auto sub = simplify_cfg_pass_run_on_function(f);
        info.folded_constant_cond_br_count += sub.folded_constant_cond_br_count;
        info.threaded_empty_block_count += sub.threaded_empty_block_count;
        info.merged_straight_line_count += sub.merged_straight_line_count;
        info.removed_unreachable_block_count += sub.removed_unreachable_block_count;
    }
    return info;
}

}// namespace luisa::compute::xir
