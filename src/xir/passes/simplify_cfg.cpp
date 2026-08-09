#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
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
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::INDEXED_BRANCH: {
            auto sw = static_cast<
                IndexedBranchTerminatorInstruction *>(term);
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
    luisa::unordered_map<BasicBlock *, LoopInst *> loop_prepares;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block != nullptr && block->is_terminated() &&
            block->terminator()->isa<LoopInst>()) {
            auto *loop = static_cast<LoopInst *>(block->terminator());
            if (loop->prepare_block() != nullptr) {
                loop_prepares.emplace(loop->prepare_block(), loop);
            }
        }
    });
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
                    // A canonical structured Loop prepare may be simplified
                    // to Branch(body) when true. Branch(merge) when false,
                    // however, destroys the prepare/body/update contract
                    // while leaving the owning LoopInst in place.
                    auto prepare = loop_prepares.find(bb);
                    if (!cc->as<bool>() &&
                        prepare != loop_prepares.end() &&
                        cb->true_block() ==
                            prepare->second->body_block() &&
                        cb->false_block() ==
                            prepare->second->merge_block()) {
                        return;
                    }
                    targets.push_back(cb);
                }
            }
        }
    });
    if (targets.empty()) return false;
    auto any = false;
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
        auto *dropped = taken == cb->true_block() ? cb->false_block() : cb->true_block();
        // Replacing the conditional branch removes `bb` as a predecessor of the
        // untaken successor. Keep its phi nodes in sync with that CFG update.
        // Degenerate branches (`true == false`) still retain the edge, so there
        // is no incoming to remove in that case.
        if (dropped != nullptr && dropped != taken) {
            for (auto *inst : dropped->instructions()) {
                if (!inst->isa<PhiInst>()) { continue; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (size_t i = phi->incoming_count(); i-- > 0u;) {
                    if (phi->incoming(i).block == bb) { phi->remove_incoming(i); }
                }
            }
        }
        auto removed = cb->remove_self();
        XIRBuilder b;
        b.set_insertion_point(bb);
        auto *branch = b.br(taken);
        for (auto *metadata : removed->metadata_list()) {
            branch->metadata_list().push_front(metadata->clone());
        }
        ++info.folded_constant_cond_br_count;
        any = true;
    }
    return any;
}

[[nodiscard]] static luisa::optional<
    IndexedBranchTerminatorInstruction::case_value_type>
evaluate_constant_indexed_branch(Value *value) noexcept {
    if (value == nullptr || !value->isa<Constant>()) {
        return luisa::nullopt;
    }
    auto *constant = static_cast<Constant *>(value);
    switch (constant->type()->tag()) {
        case Type::Tag::BOOL: return constant->as<bool>();
        case Type::Tag::INT8:
            return luisa::bit_cast<uint8_t>(constant->as<int8_t>());
        case Type::Tag::UINT8: return constant->as<uint8_t>();
        case Type::Tag::INT16:
            return luisa::bit_cast<uint16_t>(constant->as<int16_t>());
        case Type::Tag::UINT16: return constant->as<uint16_t>();
        case Type::Tag::INT32:
            return luisa::bit_cast<uint32_t>(constant->as<int32_t>());
        case Type::Tag::UINT32: return constant->as<uint32_t>();
        case Type::Tag::INT64:
            return luisa::bit_cast<uint64_t>(constant->as<int64_t>());
        case Type::Tag::UINT64: return constant->as<uint64_t>();
        default: return luisa::nullopt;
    }
}

static bool fold_constant_indexed_branch(
    FunctionDefinition *def, SimplifyCFGInfo &info) noexcept {
    luisa::vector<IndexedBranchInst *> candidates;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        if (block != nullptr && block->is_terminated() &&
            block->terminator()->isa<IndexedBranchInst>()) {
            auto *indexed_branch =
                static_cast<IndexedBranchInst *>(block->terminator());
            if (evaluate_constant_indexed_branch(
                    indexed_branch->value())) {
                candidates.emplace_back(indexed_branch);
            }
        }
    });
    auto any = false;
    for (auto *indexed_branch : candidates) {
        auto *block = indexed_branch->parent_block();
        auto selector =
            evaluate_constant_indexed_branch(indexed_branch->value());
        if (!selector) { continue; }
        auto *taken = indexed_branch->default_block();
        luisa::unordered_set<BasicBlock *> dropped;
        dropped.emplace(indexed_branch->default_block());
        for (auto i = 0u; i < indexed_branch->case_count(); i++) {
            auto *target = indexed_branch->case_block(i);
            dropped.emplace(target);
            if (indexed_branch->case_value(i) == *selector) {
                taken = target;
            }
        }
        // A malformed raw branch may have no default or a null selected case.
        // Do not replace it with Branch(nullptr), and do not claim progress:
        // the caller's fixed-point loop must still terminate on rejected IR.
        if (taken == nullptr) { continue; }
        dropped.erase(taken);
        for (auto *target : dropped) {
            if (target == nullptr) { continue; }
            for (auto *inst : target->instructions()) {
                if (!inst->isa<PhiInst>()) { continue; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (auto i = phi->incoming_count(); i-- > 0u;) {
                    if (phi->incoming(i).block == block) {
                        phi->remove_incoming(i);
                    }
                }
            }
        }
        auto removed = indexed_branch->remove_self();
        XIRBuilder b;
        b.set_insertion_point(block);
        auto *branch = b.br(taken);
        for (auto *metadata : removed->metadata_list()) {
            branch->metadata_list().push_front(metadata->clone());
        }
        ++info.folded_switch_count;
        any = true;
    }
    return any;
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
    luisa::vector<ManagedPtr<Instruction>> removed_instructions;
    for (auto *bb : dead) {
        while (!bb->instructions().empty()) {
            removed_instructions.emplace_back(
                bb->instructions().front()->remove_self());
        }
    }
    // Terminator operands are now detached, so dead blocks no longer keep
    // one another alive through CFG Use nodes. Retain the blocks until all
    // of them have been unlinked for deterministic lifetime safety.
    luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
    removed_blocks.reserve(dead.size());
    for (auto *bb : dead) {
        removed_blocks.emplace_back(bb->remove_self());
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
    ++info.straight_line_scan_count;
    luisa::vector<BasicBlock *> blocks;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        blocks.emplace_back(bb);
    });
    auto changed = false;
    luisa::unordered_set<BasicBlock *> removed;
    // Snapshot entries are raw pointers. Retain detached blocks until every
    // entry has been skipped or visited, so no worklist pointer can dangle.
    luisa::vector<ManagedPtr<BasicBlock>> removed_blocks;
    for (auto *block : blocks) {
        if (removed.contains(block)) { continue; }
        // Contract the maximal chain rooted at this physical-order block.
        // A contraction changes no predecessor relation except replacing
        // (block -> successor -> target) with (block -> target), so only the
        // same source can become newly eligible. All other live sources were
        // already present in `blocks` and are revalidated when visited.
        for (;;) {
            ++info.straight_line_block_visit_count;
            auto *terminator = block->terminator();
            if (terminator == nullptr ||
                !terminator->isa<BranchInst>()) {
                break;
            }
            auto *branch = static_cast<BranchInst *>(terminator);
            auto *successor = branch->target_block();
            if (successor == nullptr || successor == block ||
                successor == entry || removed.contains(successor) ||
                structural_targets.contains(block) ||
                structural_targets.contains(successor) ||
                block_has_phi(successor) ||
                has_phi_sensitive_successor(block, successor)) {
                break;
            }
            auto predecessor_count = size_t{0u};
            BasicBlock *predecessor = nullptr;
            successor->traverse_predecessors(
                false, [&](BasicBlock *candidate) noexcept {
                    ++predecessor_count;
                    predecessor = candidate;
                });
            if (predecessor_count != 1u || predecessor != block) {
                break;
            }

            branch->remove_self();
            XIRBuilder builder;
            builder.set_insertion_point(block);
            while (!successor->instructions().empty()) {
                auto instruction =
                    successor->instructions().front()->remove_self();
                builder.append(std::move(instruction));
            }
            removed.emplace(successor);
            removed_blocks.emplace_back(successor->remove_self());
            ++info.merged_straight_line_count;
            changed = true;
        }
    }
    return changed;
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
        if (detail::fold_constant_indexed_branch(def, info)) {
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
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            auto sub = simplify_cfg_pass_run_on_function(f);
            info.folded_constant_cond_br_count += sub.folded_constant_cond_br_count;
            info.folded_switch_count += sub.folded_switch_count;
            info.threaded_empty_block_count += sub.threaded_empty_block_count;
            info.merged_straight_line_count += sub.merged_straight_line_count;
            info.removed_unreachable_block_count += sub.removed_unreachable_block_count;
            info.straight_line_scan_count += sub.straight_line_scan_count;
            info.straight_line_block_visit_count +=
                sub.straight_line_block_visit_count;
        }
    }
    if (report != nullptr) {
        report->set("folded_constant_cond_br", info.folded_constant_cond_br_count);
        report->set("folded_switch", info.folded_switch_count);
        report->set("threaded_empty_block", info.threaded_empty_block_count);
        report->set("merged_straight_line", info.merged_straight_line_count);
        report->set("removed_unreachable_block", info.removed_unreachable_block_count);
        report->set("straight_line_scan", info.straight_line_scan_count);
        report->set(
            "straight_line_block_visit",
            info.straight_line_block_visit_count);
    }
    return info;
}

}// namespace luisa::compute::xir
