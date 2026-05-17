#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/early_return_elimination.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static ReturnInst *find_final_return_instruction(FunctionDefinition *def) noexcept {
    for (auto block = def->body_block();;) {
        auto terminator = block->terminator();
        if (terminator->isa<ReturnInst>()) { return static_cast<ReturnInst *>(terminator); }
        auto control_merge = terminator->control_flow_merge();
        if (control_merge == nullptr || control_merge->merge_block() == nullptr) { return nullptr; }
        block = control_merge->merge_block();
    }
}

[[nodiscard]] static luisa::vector<BasicBlock *> build_full_merge_chain(FunctionDefinition *def) noexcept {
    luisa::vector<BasicBlock *> chain;
    for (auto block = def->body_block();;) {
        chain.emplace_back(block);
        auto term = block->terminator();
        if (term->isa<ReturnInst>()) { break; }
        auto cfm = term->control_flow_merge();
        if (cfm == nullptr || cfm->merge_block() == nullptr) { break; }
        block = cfm->merge_block();
    }
    return chain;
}

[[nodiscard]] static bool is_reachable_avoiding(BasicBlock *start, BasicBlock *target, BasicBlock *stop) noexcept {
    if (start == target) { return true; }
    luisa::vector<BasicBlock *> stack;
    luisa::unordered_set<BasicBlock *> visited;
    stack.emplace_back(start);
    visited.emplace(start);
    while (!stack.empty()) {
        auto curr = stack.back();
        stack.pop_back();
        bool found = false;
        curr->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (found || succ == stop || visited.contains(succ)) { return; }
            if (succ == target) { found = true; return; }
            visited.emplace(succ);
            stack.emplace_back(succ);
        });
        if (found) { return true; }
    }
    return false;
}

[[nodiscard]] static BasicBlock *find_merge_target(BasicBlock *early_return_block,
                                                    const luisa::vector<BasicBlock *> &chain) noexcept {
    auto n = chain.size();
    for (size_t i = n - 1u; i > 0u; --i) {
        auto candidate_container = chain[i - 1u];
        auto candidate_merge = chain[i];
        if (is_reachable_avoiding(candidate_container, early_return_block, candidate_merge)) {
            return candidate_merge;
        }
    }
    return chain.empty() ? nullptr : chain.back();
}

[[nodiscard]] static bool is_already_conditionalized(BasicBlock *block, AllocaInst *not_returned_flag) noexcept {
    auto term = block->terminator();
    if (!term->isa<IfInst>()) { return false; }
    auto if_inst = static_cast<IfInst *>(term);
    auto cond = if_inst->condition();
    if (cond == nullptr || !cond->isa<LoadInst>()) { return false; }
    return static_cast<LoadInst *>(cond)->variable() == not_returned_flag;
}

static BasicBlock *conditionalize_block(BasicBlock *block, AllocaInst *not_returned_flag,
                                        FunctionDefinition *def) noexcept {
    if (is_already_conditionalized(block, not_returned_flag)) {
        auto if_inst = static_cast<IfInst *>(block->terminator());
        return if_inst->merge_block();
    }

    auto &insts = block->instructions();
    auto head = insts.head_sentinel();
    auto term = block->terminator();

    luisa::vector<ManagedPtr<Instruction>> non_terms;
    for (auto inst = head->next(); inst != term;) {
        auto next = inst->next();
        non_terms.emplace_back(inst->remove_self());
        inst = next;
    }
    auto managed_term = term->remove_self();

    auto t_new = def->create_basic_block();
    auto f_new = def->create_basic_block();
    auto merge_new = def->create_basic_block();

    auto t_new_tail = t_new->instructions().tail_sentinel();
    for (auto &inst : non_terms) {
        t_new_tail->insert_before_self(std::move(inst));
    }

    XIRBuilder b;
    b.set_insertion_point(t_new);
    b.br(merge_new);

    b.set_insertion_point(f_new);
    b.br(merge_new);

    merge_new->instructions().tail_sentinel()->insert_before_self(std::move(managed_term));

    b.set_insertion_point(head);
    auto load_flag = b.load(Type::of<bool>(), not_returned_flag);
    auto if_inst = b.if_(load_flag);
    if_inst->set_true_target(t_new);
    if_inst->set_false_target(f_new);
    if_inst->set_merge_block(merge_new);

    return merge_new;
}

static void handle_final_return_block(BasicBlock *final_return_block, AllocaInst *not_returned_flag,
                                      AllocaInst *return_value_slot, FunctionDefinition *def) noexcept {
    auto merge_new = conditionalize_block(final_return_block, not_returned_flag, def);

    if (return_value_slot == nullptr) { return; }

    auto if_inst = static_cast<IfInst *>(final_return_block->terminator());
    auto t_new = if_inst->true_block();

    auto t_new_term = t_new->terminator();
    auto ret_type = return_value_slot->type();

    auto merge_new_term = merge_new->terminator();
    LUISA_ASSERT(merge_new_term->isa<ReturnInst>(), "Expected ReturnInst in merge_new block.");
    auto orig_return = static_cast<ReturnInst *>(merge_new_term);
    auto orig_val = orig_return->return_value();

    XIRBuilder b;
    b.set_insertion_point(t_new_term->prev());
    b.store(return_value_slot, orig_val);

    b.set_insertion_point(merge_new->instructions().head_sentinel());
    auto loaded = b.load(ret_type, return_value_slot);

    orig_return->set_return_value(loaded);
}

static void eliminate_early_return(ReturnInst *return_inst, AllocaInst *not_returned_flag,
                                   AllocaInst *return_value_slot, BasicBlock *merge_target,
                                   Module *module) noexcept {
    auto parent = return_inst->parent_block();
    auto bool_type = Type::of<bool>();
    auto const_false = module->create_constant_zero(bool_type);

    XIRBuilder b;
    b.set_insertion_point(parent);
    if (return_value_slot != nullptr) {
        auto val = return_inst->return_value();
        if (val != nullptr) { b.store(return_value_slot, val); }
    }
    b.store(not_returned_flag, const_false);
    b.br(merge_target);
    return_inst->remove_self();
}

static void eliminate_early_return_in_function(Function *function, EarlyReturnEliminationInfo &info) noexcept {
    if (auto def = function->definition()) {
        luisa::vector<ReturnInst *> early_returns;
        auto final_return = find_final_return_instruction(def);
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (auto terminator = block->terminator(); terminator != final_return && terminator->isa<ReturnInst>()) {
                early_returns.emplace_back(static_cast<ReturnInst *>(terminator));
            }
        });
        if (!early_returns.empty()) {
            XIRBuilder b;
            b.set_insertion_point(def->body_block()->instructions().head_sentinel());
            auto bool_type = Type::of<bool>();
            auto not_returned_flag = b.alloca_local(bool_type);
            not_returned_flag->add_comment("early return flag");
            auto const_true = function->parent_module()->create_constant_one(bool_type);
            b.set_insertion_point(def->body_block()->terminator()->prev());
            auto store_inst = b.store(not_returned_flag, const_true);
            store_inst->add_comment("initialize early return flag");

            auto chain = build_full_merge_chain(def);

            luisa::vector<std::pair<ReturnInst *, BasicBlock *>> return_targets;
            return_targets.reserve(early_returns.size());
            for (auto r : early_returns) {
                auto merge_target = find_merge_target(r->parent_block(), chain);
                return_targets.emplace_back(r, merge_target);
            }

            luisa::unordered_set<BasicBlock *> conditionalized;
            for (auto &[r, merge_target] : return_targets) {
                size_t idx = 0u;
                for (size_t i = 0u; i < chain.size(); ++i) {
                    if (chain[i] == merge_target) { idx = i; break; }
                }
                for (auto i = idx; i < chain.size(); ++i) {
                    conditionalized.emplace(chain[i]);
                }
            }

            auto ret_type = function->type();
            AllocaInst *return_value_slot = nullptr;
            if (ret_type != nullptr) {
                b.set_insertion_point(def->body_block()->instructions().head_sentinel());
                return_value_slot = b.alloca_local(ret_type);
                return_value_slot->add_comment("early return value slot");
            }

            auto final_return_block = final_return->parent_block();
            for (auto block : chain) {
                if (!conditionalized.contains(block)) { continue; }
                if (block == final_return_block) {
                    handle_final_return_block(block, not_returned_flag, return_value_slot, def);
                } else {
                    conditionalize_block(block, not_returned_flag, def);
                }
            }

            for (auto &[r, merge_target] : return_targets) {
                eliminate_early_return(r, not_returned_flag, return_value_slot, merge_target,
                                       function->parent_module());
            }
        }
        info.removed_return_count += early_returns.size();
    }
}

}// namespace detail

EarlyReturnEliminationInfo early_return_elimination_pass_run_on_function(Function *function) noexcept {
    EarlyReturnEliminationInfo info;
    detail::eliminate_early_return_in_function(function, info);
    return info;
}

EarlyReturnEliminationInfo early_return_elimination_pass_run_on_module(Module *module) noexcept {
    EarlyReturnEliminationInfo info;
    for (auto f : module->function_list()) {
        detail::eliminate_early_return_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
