#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool retarget_terminator(Instruction *term, BasicBlock *from, BasicBlock *to) noexcept {
    if (term == nullptr || from == nullptr || to == nullptr) { return false; }
    auto changed = false;
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto br = static_cast<BranchInst *>(term);
            if (br->target_block() == from) {
                br->set_target_block(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto cb = static_cast<ConditionalBranchInst *>(term);
            if (cb->true_block() == from) {
                cb->set_true_target(to);
                changed = true;
            }
            if (cb->false_block() == from) {
                cb->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::IF: {
            auto if_inst = static_cast<IfInst *>(term);
            if (if_inst->true_block() == from) {
                if_inst->set_true_target(to);
                changed = true;
            }
            if (if_inst->false_block() == from) {
                if_inst->set_false_target(to);
                changed = true;
            }
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto sw = static_cast<SwitchInst *>(term);
            if (sw->default_block() == from) {
                sw->set_default_block(to);
                changed = true;
            }
            for (size_t i = 0; i < sw->case_count(); ++i) {
                if (sw->case_block(i) == from) {
                    sw->set_case_block(i, to);
                    changed = true;
                }
            }
            break;
        }
        default: break;
    }
    return changed;
}

static bool fold_constant_cond_br(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    luisa::vector<ConditionalBranchInst *> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb == nullptr) { return; }
        auto t = bb->terminator();
        if (t != nullptr && t->isa<ConditionalBranchInst>()) {
            auto cb = static_cast<ConditionalBranchInst *>(t);
            auto cond = cb->condition();
            if (cb->true_block() != nullptr && cb->true_block() == cb->false_block()) {
                targets.push_back(cb);
            } else if (cond != nullptr && cond->isa<Constant>()) {
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
        BasicBlock *taken = nullptr;
        if (cb->true_block() != nullptr && cb->true_block() == cb->false_block()) {
            taken = cb->true_block();
        } else {
            auto c = static_cast<Constant *>(cb->condition());
            taken = c->as<bool>() ? cb->true_block() : cb->false_block();
        }
        if (taken == nullptr) { continue; }
        cb->remove_self();
        XIRBuilder b;
        b.set_insertion_point(bb);
        b.br(taken);
        ++info.folded_constant_cond_br_count;
    }
    return true;
}

template<typename Visit>
static void traverse_structural_successors(BasicBlock *block, Visit &&visit) noexcept {
    if (block == nullptr || !block->is_terminated()) { return; }
    auto *term = block->terminator();
    for (auto use : term->operand_uses()) {
        if (auto *value = use->value(); value != nullptr && value->isa<BasicBlock>()) {
            visit(static_cast<BasicBlock *>(value));
        }
    }
    if (auto *merge = term->control_flow_merge(); merge != nullptr) {
        if (auto *merge_block = merge->merge_block(); merge_block != nullptr) { visit(merge_block); }
    }
    if (term->isa<LoopInst>()) {
        auto *loop = static_cast<LoopInst *>(term);
        if (auto *body = loop->body_block(); body != nullptr) { visit(body); }
        if (auto *update = loop->update_block(); update != nullptr) { visit(update); }
    }
}

static luisa::unordered_set<BasicBlock *> collect_structurally_reachable_blocks(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> reachable;
    luisa::vector<BasicBlock *> work;
    auto add = [&](BasicBlock *block) noexcept {
        if (block != nullptr && reachable.emplace(block).second) { work.emplace_back(block); }
    };
    add(def->body_block());
    while (!work.empty()) {
        auto *block = work.back();
        work.pop_back();
        traverse_structural_successors(block, add);
    }
    return reachable;
}

static luisa::unordered_set<BasicBlock *> collect_loop_headers(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> headers;
    luisa::unordered_set<BasicBlock *> visiting;
    luisa::unordered_set<BasicBlock *> visited;
    auto visit = [&](auto &&self, BasicBlock *block) noexcept -> void {
        if (block == nullptr || visited.contains(block)) { return; }
        visiting.emplace(block);
        traverse_structural_successors(block, [&](BasicBlock *succ) noexcept {
            if (visiting.contains(succ)) {
                headers.emplace(succ);
            } else if (!visited.contains(succ)) {
                self(self, succ);
            }
        });
        visiting.erase(block);
        visited.emplace(block);
    };
    visit(visit, def->body_block());
    return headers;
}

static luisa::unordered_set<BasicBlock *> collect_structural_targets(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        auto term = bb->terminator();
        if (term == nullptr) return;
        if (auto merge = term->control_flow_merge()) {
            if (auto merge_block = merge->merge_block()) targets.emplace(merge_block);
        }
        if (term->isa<LoopInst>()) {
            auto loop = static_cast<LoopInst *>(term);
            if (auto prepare = loop->prepare_block()) { targets.emplace(prepare); }
            if (auto body = loop->body_block()) { targets.emplace(body); }
            if (auto update = loop->update_block()) { targets.emplace(update); }
        } else if (term->isa<SimpleLoopInst>()) {
            auto loop = static_cast<SimpleLoopInst *>(term);
            if (auto body = loop->body_block()) { targets.emplace(body); }
        }
    });
    return targets;
}

static bool thread_empty_blocks(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    auto entry = def->body_block();
    auto loop_headers = collect_loop_headers(def);
    auto structural_targets = collect_structural_targets(def);
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
        if (br->target_block() != nullptr && loop_headers.contains(br->target_block())) return;
        if (structural_targets.contains(bb)) return;
        candidates.push_back(bb);
    });
    if (candidates.empty()) return false;
    bool any = false;
    for (auto bb : candidates) {
        if (bb->terminator() == nullptr) continue;
        auto br = static_cast<BranchInst *>(bb->terminator());
        auto target = br->target_block();
        if (target == nullptr) { continue; }
        bool target_has_phi = false;
        for (auto inst : target->instructions()) {
            if (inst->isa<PhiInst>()) {
                target_has_phi = true;
                break;
            }
        }
        if (target_has_phi) continue;
        luisa::vector<BasicBlock *> preds;
        bb->traverse_predecessors(true, [&](BasicBlock *p) noexcept {
            preds.push_back(p);
        });
        bool redirected = false;
        for (auto p : preds) {
            if (p == bb) continue;
            redirected |= retarget_terminator(p->terminator(), bb, target);
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
    auto reachable = collect_structurally_reachable_blocks(def);
    luisa::vector<BasicBlock *> dead;
    for (auto bb : def->basic_blocks()) {
        if (bb == entry) continue;
        if (!reachable.contains(bb)) dead.push_back(bb);
    }
    if (dead.empty()) return false;
    luisa::unordered_set<BasicBlock *> dead_set{dead.begin(), dead.end()};
    for (auto bb : def->basic_blocks()) {
        if (dead_set.contains(bb)) continue;
        for (auto inst : bb->instructions()) {
            if (!inst->isa<PhiInst>()) continue;
            auto phi = static_cast<PhiInst *>(inst);
            for (size_t i = phi->incoming_count(); i-- > 0;) {
                if (dead_set.contains(phi->incoming(i).block)) {
                    phi->remove_incoming(i);
                }
            }
        }
    }
    for (auto bb : dead) {
        while (!bb->instructions().empty()) {
            bb->instructions().front()->remove_self();
        }
        bb->remove_self();
        ++info.removed_unreachable_block_count;
    }
    return true;
}

[[nodiscard]] static bool block_has_phi(BasicBlock *bb) noexcept {
    auto has_phi = false;
    bb->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) has_phi = true;
    });
    return has_phi;
}

[[nodiscard]] static bool has_phi_sensitive_successor(BasicBlock *bb, BasicBlock *succ) noexcept {
    auto sensitive = false;
    succ->traverse_successors(false, [&](BasicBlock *target) noexcept {
        if (target == bb || target == succ || block_has_phi(target)) sensitive = true;
    });
    return sensitive;
}

static bool merge_straight_line_blocks(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    auto entry = def->body_block();
    auto structural_targets = collect_structural_targets(def);
    BasicBlock *candidate_block = nullptr;
    BasicBlock *candidate_successor = nullptr;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (candidate_block != nullptr) return;
        auto term = bb->terminator();
        if (term == nullptr || !term->isa<BranchInst>()) return;
        auto br = static_cast<BranchInst *>(term);
        auto succ = br->target_block();
        if (succ == nullptr || succ == bb || succ == entry) return;
        if (structural_targets.contains(bb) || structural_targets.contains(succ)) return;
        if (block_has_phi(succ)) return;
        if (has_phi_sensitive_successor(bb, succ)) return;
        size_t pred_count = 0;
        BasicBlock *pred = nullptr;
        succ->traverse_predecessors(false, [&](BasicBlock *p) noexcept {
            ++pred_count;
            pred = p;
        });
        if (pred_count == 1 && pred == bb) {
            candidate_block = bb;
            candidate_successor = succ;
        }
    });
    if (candidate_block == nullptr || candidate_successor == nullptr) return false;
    auto bb = candidate_block;
    auto succ = candidate_successor;
    if (bb->terminator() == nullptr || !bb->terminator()->isa<BranchInst>()) return false;
    auto br = static_cast<BranchInst *>(bb->terminator());
    if (br->target_block() != succ || block_has_phi(succ)) return false;
    if (has_phi_sensitive_successor(bb, succ)) return false;
    size_t pred_count = 0;
    BasicBlock *pred = nullptr;
    succ->traverse_predecessors(false, [&](BasicBlock *p) noexcept {
        ++pred_count;
        pred = p;
    });
    if (pred_count != 1 || pred != bb) return false;
    br->remove_self();
    XIRBuilder b;
    b.set_insertion_point(bb);
    while (!succ->instructions().empty()) {
        auto inst = succ->instructions().front()->remove_self();
        b.append(std::move(inst));
    }
    succ->remove_self();
    ++info.merged_straight_line_count;
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
        if (detail::fold_constant_cond_br(def, info)) {
            changed = true;
            continue;
        }
        if (detail::thread_empty_blocks(def, info)) {
            changed = true;
            continue;
        }
        if (detail::merge_straight_line_blocks(def, info)) {
            changed = true;
            continue;
        }
        if (detail::remove_unreachable_blocks(def, info)) {
            changed = true;
            continue;
        }
    }
    return info;
}

SimplifyCFGInfo simplify_cfg_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SimplifyCFGInfo info;
    if (module == nullptr) return info;
    for (auto f : module->function_list()) {
        auto sub = simplify_cfg_pass_run_on_function(f);
        info.folded_constant_cond_br_count += sub.folded_constant_cond_br_count;
        info.folded_switch_count += sub.folded_switch_count;
        info.threaded_empty_block_count += sub.threaded_empty_block_count;
        info.merged_straight_line_count += sub.merged_straight_line_count;
        info.removed_unreachable_block_count += sub.removed_unreachable_block_count;
    }
    if (report != nullptr) {
        report->set("folded_constant_cond_br", info.folded_constant_cond_br_count);
        report->set("folded_switch", info.folded_switch_count);
        report->set("threaded_empty_block", info.threaded_empty_block_count);
        report->set("merged_straight_line", info.merged_straight_line_count);
        report->set("removed_unreachable_block", info.removed_unreachable_block_count);
    }
    return info;
}

}// namespace luisa::compute::xir
