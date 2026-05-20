#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
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

[[nodiscard]] static luisa::optional<SwitchInst::case_value_type> try_evaluate_static_switch_condition(Value *value) noexcept {
    if (value == nullptr || !value->isa<Constant>()) return luisa::nullopt;
    auto c = static_cast<Constant *>(value);
    switch (c->type()->tag()) {
        case Type::Tag::BOOL: return c->as<bool>() ? 1 : 0;
        case Type::Tag::INT8: return c->as<int8_t>();
        case Type::Tag::UINT8: return c->as<uint8_t>();
        case Type::Tag::INT16: return c->as<int16_t>();
        case Type::Tag::UINT16: return c->as<uint16_t>();
        case Type::Tag::INT32: return c->as<int32_t>();
        case Type::Tag::UINT32: return static_cast<SwitchInst::case_value_type>(c->as<uint32_t>());
        case Type::Tag::INT64: return static_cast<SwitchInst::case_value_type>(c->as<int64_t>());
        case Type::Tag::UINT64: return static_cast<SwitchInst::case_value_type>(c->as<uint64_t>());
        default: break;
    }
    return luisa::nullopt;
}

static bool fold_switches(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    luisa::vector<SwitchInst *> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (auto t = bb->terminator(); t != nullptr && t->isa<SwitchInst>()) {
            auto sw = static_cast<SwitchInst *>(t);
            auto default_block = sw->default_block();
            if (default_block == nullptr) return;
            auto common = default_block;
            auto all_same = true;
            for (size_t i = 0u; i < sw->case_count(); ++i) {
                if (sw->case_block(i) != common) {
                    all_same = false;
                    break;
                }
            }
            if (all_same || try_evaluate_static_switch_condition(sw->value())) {
                targets.emplace_back(sw);
            }
        }
    });
    if (targets.empty()) return false;
    for (auto sw : targets) {
        auto bb = sw->parent_block();
        if (bb == nullptr) continue;
        auto target = sw->default_block();
        if (auto static_value = try_evaluate_static_switch_condition(sw->value())) {
            for (size_t i = 0u; i < sw->case_count(); ++i) {
                if (sw->case_value(i) == *static_value) {
                    target = sw->case_block(i);
                    break;
                }
            }
        }
        if (target == nullptr) continue;
        sw->remove_self();
        XIRBuilder b;
        b.set_insertion_point(bb);
        b.br(target);
        ++info.folded_switch_count;
    }
    return true;
}

// Detect blocks that already have a back-edge predecessor (i.e., loop headers under any reasonable
// dominance-aware analysis). A back-edge is a CFG edge p -> bb where bb dominates p; without a full
// dominator computation here, we approximate using reverse-postorder index: any predecessor that
// appears AFTER bb in RPO is a back-edge source. This is sufficient for guarding jump-threading of
// blocks whose only successor is a loop header, where threading would collapse a structured if-merge
// onto the loop's continue role (illegal in SPIR-V structured control flow).
static luisa::unordered_set<BasicBlock *> collect_loop_headers(FunctionDefinition *def) noexcept {
    luisa::unordered_map<BasicBlock *, size_t> rpo_index;
    size_t idx = 0;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        rpo_index.emplace(bb, idx++);
    });
    luisa::unordered_set<BasicBlock *> headers;
    for (auto &[bb, i] : rpo_index) {
        bb->traverse_predecessors(true, [&](BasicBlock *p) noexcept {
            auto it = rpo_index.find(p);
            if (it != rpo_index.end() && it->second >= i && p != bb) {
                headers.insert(bb);
            }
        });
    }
    return headers;
}

static bool thread_empty_blocks(FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    auto entry = def->body_block();
    auto loop_headers = collect_loop_headers(def);
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
        // Preserve trampoline blocks that sit immediately before a loop header.
        // Threading them out would force a structured merge block (the predecessor of bb)
        // to serve as the loop's continue target, which violates SPIR-V structured CF.
        if (br->target_block() != nullptr && loop_headers.contains(br->target_block())) return;
        candidates.push_back(bb);
    });
    if (candidates.empty()) return false;
    bool any = false;
    for (auto bb : candidates) {
        if (bb->terminator() == nullptr) continue;
        auto br = static_cast<BranchInst *>(bb->terminator());
        auto target = br->target_block();
        bool target_has_phi = false;
        for (auto inst : target->instructions()) {
            if (inst->isa<PhiInst>()) { target_has_phi = true; break; }
        }
        if (target_has_phi) continue;
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

static luisa::unordered_set<BasicBlock *> collect_structural_targets(FunctionDefinition *def) noexcept {
    luisa::unordered_set<BasicBlock *> targets;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        auto term = bb->terminator();
        if (term == nullptr) return;
        if (auto merge = term->control_flow_merge()) {
            if (auto merge_block = merge->merge_block()) targets.emplace(merge_block);
        }
    });
    return targets;
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
        size_t pred_count = 0u;
        BasicBlock *pred = nullptr;
        succ->traverse_predecessors(false, [&](BasicBlock *p) noexcept {
            ++pred_count;
            pred = p;
        });
        if (pred_count == 1u && pred == bb) {
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
    size_t pred_count = 0u;
    BasicBlock *pred = nullptr;
    succ->traverse_predecessors(false, [&](BasicBlock *p) noexcept {
        ++pred_count;
        pred = p;
    });
    if (pred_count != 1u || pred != bb) return false;
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
        if (detail::fold_constant_cond_br(def, info)) changed = true;
        if (detail::fold_switches(def, info)) changed = true;
        if (detail::thread_empty_blocks(def, info)) changed = true;
        if (detail::merge_straight_line_blocks(def, info)) changed = true;
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
        info.folded_switch_count += sub.folded_switch_count;
        info.threaded_empty_block_count += sub.threaded_empty_block_count;
        info.merged_straight_line_count += sub.merged_straight_line_count;
        info.removed_unreachable_block_count += sub.removed_unreachable_block_count;
    }
    return info;
}

}// namespace luisa::compute::xir
