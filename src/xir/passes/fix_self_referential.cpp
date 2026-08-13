#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace detail {

struct PendingFix {
    Instruction *instruction;
    Value *target;
    luisa::vector<size_t> operand_indices;
};

static void collect_self_referential_fixes_on_function(
    Function *function, luisa::vector<PendingFix> &pending,
    FixSelfReferentialInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr || def->body_block() == nullptr) { return; }
    auto dom_tree = compute_dom_tree(function);
    def->traverse_instructions([&](Instruction *inst) noexcept {
        luisa::vector<size_t> self_operands;
        for (size_t i = 0; i < inst->operand_count(); ++i) {
            if (inst->operand(i) == inst) { self_operands.emplace_back(i); }
        }
        if (self_operands.empty()) { return; }
        // A loop-carried Phi may legitimately keep its previous value on a
        // backedge by naming itself as that edge's incoming value. This is
        // valid SSA and maps directly to a self-referencing OpPhi. The repair
        // below is only for malformed ordinary value instructions produced by
        // aggregate-store rewrites.
        if (inst->isa<PhiInst>()) { return; }
        if (!inst->isa<ArithmeticInst>() ||
            static_cast<ArithmeticInst *>(inst)->op() != ArithmeticOp::INSERT) {
            info.unresolved_count += self_operands.size();
            return;
        }
        // INSERT is (aggregate, element, index) -> aggregate. Only its
        // aggregate operand may be repaired from the backing storage. Loading
        // the aggregate into either of the other operand positions would
        // replace a scalar/index value with an aggregate and create
        // type-invalid IR. If any unsupported self-reference is present,
        // reject the whole instruction before scheduling a mutation.
        if (self_operands.size() != 1u || self_operands.front() != 0u) {
            info.unresolved_count += self_operands.size();
            return;
        }
        Value *target = nullptr;
        auto ambiguous = false;
        for (auto &&use : inst->use_list()) {
            auto *user = use->user();
            if (user == nullptr || !user->isa<StoreInst>()) { continue; }
            auto *store = static_cast<StoreInst *>(user);
            if (store->value() != inst) { continue; }
            auto *candidate = store->variable();
            if (candidate == nullptr || !candidate->is_lvalue() ||
                candidate->type() != inst->type()) {
                ambiguous = true;
                break;
            }
            if (target == nullptr) {
                target = candidate;
            } else if (target != candidate) {
                ambiguous = true;
                break;
            }
        }
        if (target != nullptr && target->isa<Instruction>()) {
            auto *target_inst = static_cast<Instruction *>(target);
            auto *target_block = target_inst->parent_block();
            auto *inst_block = inst->parent_block();
            if (target_block == inst_block) {
                auto seen_target = false;
                for (auto *ordered : inst_block->instructions()) {
                    if (ordered == target_inst) { seen_target = true; }
                    if (ordered == inst) { break; }
                }
                ambiguous |= !seen_target;
            } else {
                ambiguous |= !dom_tree.contains(target_block) ||
                             !dom_tree.contains(inst_block) ||
                             !dom_tree.dominates(target_block, inst_block);
            }
        }
        if (target == nullptr || ambiguous) {
            info.unresolved_count += self_operands.size();
            return;
        }
        pending.emplace_back(PendingFix{inst, target, std::move(self_operands)});
    });
}

static void apply_self_referential_fixes(
    const luisa::vector<PendingFix> &pending,
    FixSelfReferentialInfo &info) noexcept {
    for (auto &&fix : pending) {
        auto replacement = luisa::make_managed<LoadInst>(
            fix.instruction->parent_block(), fix.instruction->type(), fix.target);
        auto *load = static_cast<LoadInst *>(
            fix.instruction->insert_before_self(std::move(replacement)));
        for (auto operand_index : fix.operand_indices) {
            fix.instruction->set_operand(operand_index, load);
            info.fixed_count++;
        }
    }
}

}// namespace detail

FixSelfReferentialInfo fix_self_referential_pass_run_on_function(Function *function) noexcept {
    FixSelfReferentialInfo info;
    luisa::vector<detail::PendingFix> pending;
    detail::collect_self_referential_fixes_on_function(
        function, pending, info);
    if (info.succeeded()) {
        detail::apply_self_referential_fixes(pending, info);
    }
    return info;
}

FixSelfReferentialInfo fix_self_referential_pass_run_on_module(Module *module, PassReport *report) noexcept {
    FixSelfReferentialInfo info;
    luisa::vector<detail::PendingFix> pending;
    if (module != nullptr) {
        for (auto f : module->function_list()) {
            detail::collect_self_referential_fixes_on_function(
                f, pending, info);
        }
        if (info.succeeded()) {
            detail::apply_self_referential_fixes(pending, info);
        }
    }
    if (report != nullptr) {
        report->set("fixed_inst", info.fixed_count);
        report->set("unresolved_inst", info.unresolved_count);
    }
    return info;
}

}// namespace luisa::compute::xir
