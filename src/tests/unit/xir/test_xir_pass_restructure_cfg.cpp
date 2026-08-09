// Test for reconstructing structured XIR control flow from explicit CFGs.

#include "ut/ut.hpp"
#include <luisa/luisa-compute.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(
    const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;

    explicit ScopedEnvironmentVariable(
        const char *env_name, const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(),
            previous ? previous->c_str() : nullptr);
    }

    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def,
                                           DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_non_canonical_loop_prepare(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        if (prepare == nullptr || !prepare->is_terminated()) {
            ++n;
            return;
        }
        auto *prepare_term = prepare->terminator();
        if (prepare_term->isa<BranchInst>()) {
            auto *branch = static_cast<BranchInst *>(prepare_term);
            n += branch->target_block() != loop->body_block();
            return;
        }
        if (!prepare_term->isa<ConditionalBranchInst>()) {
            ++n;
            return;
        }
        auto *cond_br = static_cast<ConditionalBranchInst *>(prepare_term);
        if (cond_br->condition() == nullptr ||
            cond_br->condition()->type() != Type::of<bool>() ||
            cond_br->true_block() != loop->body_block() ||
            cond_br->false_block() != loop->merge_block()) {
            ++n;
        }
    });
    return n;
}

[[nodiscard]] size_t count_canonical_conditional_loop_prepare(
    FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated() ||
            !bb->terminator()->isa<LoopInst>()) {
            return;
        }
        auto *loop =
            static_cast<LoopInst *>(bb->terminator());
        auto *prepare = loop->prepare_block();
        if (prepare == nullptr || !prepare->is_terminated() ||
            !prepare->terminator()
                 ->isa<ConditionalBranchInst>()) {
            return;
        }
        auto *branch =
            static_cast<ConditionalBranchInst *>(
                prepare->terminator());
        n += branch->condition() != nullptr &&
             branch->condition()->type() == Type::of<bool>() &&
             branch->true_block() == loop->body_block() &&
             branch->false_block() == loop->merge_block();
    });
    return n;
}

[[nodiscard]] size_t count_non_canonical_loop_update(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        auto *term = bb->terminator();
        if (!term->isa<LoopInst>()) { return; }
        auto *loop = static_cast<LoopInst *>(term);
        auto *prepare = loop->prepare_block();
        auto *update = loop->update_block();
        if (prepare == nullptr || update == nullptr || !update->is_terminated()) {
            ++n;
            return;
        }
        bool branches_to_prepare = false;
        update->traverse_successors(false, [&](BasicBlock *succ) noexcept {
            if (succ == prepare) { branches_to_prepare = true; }
        });
        if (!branches_to_prepare) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_phi(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_owned_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *block : def->basic_blocks()) { ++n; }
    return n;
}

[[nodiscard]] size_t count_post_merge_selection_reentries(
    luisa::compute::xir::Function *function) noexcept {
    auto *def =
        function == nullptr ? nullptr : function->definition();
    if (def == nullptr) { return 0u; }
    auto dom = compute_dom_tree(function);
    auto count = size_t{0u};
    for (auto *header : def->basic_blocks()) {
        if (header == nullptr || !header->is_terminated() ||
            !dom.contains(header)) {
            continue;
        }
        auto *term = header->terminator();
        if (!term->isa<IfInst>() &&
            !term->isa<SwitchInst>()) {
            continue;
        }
        auto *merge =
            term->control_flow_merge()->merge_block();
        if (merge == nullptr || !dom.contains(merge)) {
            continue;
        }
        auto invalid = false;
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || block == header ||
                block == merge || !dom.contains(block) ||
                !dom.dominates(header, block) ||
                dom.dominates(merge, block)) {
                continue;
            }
            block->traverse_predecessors(
                false,
                [&](BasicBlock *predecessor) noexcept {
                    invalid |=
                        dom.contains(predecessor) &&
                        dom.dominates(merge, predecessor);
                });
        }
        count += invalid ? 1u : 0u;
    }
    return count;
}

[[nodiscard]] bool branch_chain_reaches(BasicBlock *from, BasicBlock *to) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    auto *cur = from;
    while (cur != nullptr && visited.emplace(cur).second) {
        if (cur == to) { return true; }
        if (!cur->is_terminated() || !cur->terminator()->isa<BranchInst>()) { return false; }
        cur = static_cast<BranchInst *>(cur->terminator())->target_block();
    }
    return false;
}

void run_spirv_normalize_before_restructure(Module *m) noexcept {
    auto algebraic_options = AlgebraicSimplifyOptions{};
    (void)lower_ray_query_loop_to_loop_pass_run_on_module(m);
    (void)destructure_cfg_pass_run_on_module(m);
    (void)mem2reg_pass_run_on_module(m);
    (void)algebraic_simplify_pass_run_on_module(m, algebraic_options);
    (void)const_fold_pass_run_on_module(m);
    (void)sccp_pass_run_on_module(m);
    (void)dce_pass_run_on_module(m);
    (void)local_store_forward_pass_run_on_module(m);
    (void)local_load_elimination_pass_run_on_module(m);
    (void)dead_store_elimination_pass_run_on_module(m);
    (void)dce_pass_run_on_module(m);
    (void)gvn_pass_run_on_module(m);
    (void)if_conversion_pass_run_on_module(m);
    (void)phi_cleanup_pass_run_on_module(m);
    (void)unused_callable_removal_pass_run_on_module(m);
    (void)simplify_cfg_pass_run_on_module(m);
    (void)reg2mem_pass_run_on_module(m);
}

void expect_no_structured_cfg(FunctionDefinition *def) noexcept {
    expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 0u);
    expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
    expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
}

}// namespace

void reg_restructure_cfg() {

    "restructure_empty_function_noop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
        expect(info.irreducible_region_count == 0u);
    };

    "restructure_external_function_skipped"_test = [] {
        Module m;
        auto *ext = m.create_external_function(Type::of<void>());
        auto info = restructure_cfg_pass_run_on_function(ext);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
    };

    "restructure_switch_with_duplicate_targets"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector = kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(selector);
        auto *shared_target = sw->create_default_block();
        sw->add_case(1, shared_target);
        sw->add_case(2, shared_target);
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(shared_target);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        size_t successor_count = 0u;
        body->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            expect(successor == shared_target);
            ++successor_count;
        });
        expect(successor_count == 1u);
        size_t predecessor_count = 0u;
        shared_target->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
            expect(predecessor == body);
            ++predecessor_count;
        });
        expect(predecessor_count == 1u);

        // Dominance and post-dominance traversal must accept multiple switch
        // operands that represent the same CFG successor. The restructurer then
        // gives each switch label a unique proxy as required by code generation.
        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
        expect(info.restructured_switch_count == 0u);
        expect(info.canonicalized_cfg_count >= 1u);
        expect(info.changed());
        expect(info.succeeded());
        expect(body->terminator()->isa<SwitchInst>());
        auto *normalized = static_cast<SwitchInst *>(body->terminator());
        expect(normalized->case_count() == 2u);
        auto *default_target = normalized->default_block();
        auto *case_0_target = normalized->case_block(0u);
        auto *case_1_target = normalized->case_block(1u);
        expect(default_target != case_0_target);
        expect(default_target != case_1_target);
        expect(case_0_target != case_1_target);
        expect(branch_chain_reaches(default_target, merge));
        expect(branch_chain_reaches(case_0_target, merge));
        expect(branch_chain_reaches(case_1_target, merge));
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());

        auto idempotent =
            restructure_cfg_pass_run_on_function(kernel);
        expect(!idempotent.changed());
        expect(idempotent.succeeded());
    };

    "restructure_switch_merge_is_not_retargeted_as_executable_edge"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector = kernel->create_value_argument(Type::of<int>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.simple_loop();
        auto *loop_body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();

        b.set_insertion_point(loop_body);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1u);
        auto *switch_merge = switch_inst->create_merge_block();
        // The direct default edge and the case edge share a forwarding block
        // that is also the declarative Switch merge. Splitting the shared
        // continue must rewrite only executable case/default edges; the merge
        // declaration is not a CFG successor to retarget.
        switch_inst->set_default_block(switch_merge);
        b.set_insertion_point(case_block);
        b.br(switch_merge);
        b.set_insertion_point(switch_merge);
        b.continue_(loop_body);

        b.set_insertion_point(loop_merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(switch_inst->merge_block() == switch_merge);
        expect(switch_inst->default_block() != switch_merge);
        expect(switch_inst->default_block()->terminator()->isa<ContinueInst>());
        expect(static_cast<ContinueInst *>(
                   switch_inst->default_block()->terminator())
                   ->target_block() == loop_body);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_irreducible_scc_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *condition0 = kernel->create_value_argument(Type::of<bool>());
        auto *condition1 = kernel->create_value_argument(Type::of<bool>());
        auto *definition = kernel->definition();
        auto *left = definition->create_basic_block();
        auto *right = definition->create_basic_block();
        auto *exit = definition->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry_branch = b.cond_br(condition0, left, right);
        b.set_insertion_point(left);
        auto *left_branch = b.br(right);
        b.set_insertion_point(right);
        auto *right_branch = b.cond_br(condition1, left, exit);
        b.set_insertion_point(exit);
        auto *exit_return = b.return_void();
        auto block_count = definition->basic_blocks().count_size();

        auto info = restructure_cfg_pass_run_on_function(kernel);

        expect(!info.succeeded());
        expect(info.irreducible_region_count == 1u);
        expect(info.restructured_loop_count == 0u);
        expect(info.restructured_if_count == 0u);
        expect(definition->basic_blocks().count_size() == block_count);
        expect(body->terminator() == entry_branch);
        expect(left->terminator() == left_branch);
        expect(right->terminator() == right_branch);
        expect(exit->terminator() == exit_return);
        expect(entry_branch->true_block() == left);
        expect(entry_branch->false_block() == right);
        expect(right_branch->true_block() == left);
        expect(right_branch->false_block() == exit);
    };

    "restructure_phi_input_is_rejected_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *definition = kernel->definition();
        auto *left = definition->create_basic_block();
        auto *right = definition->create_basic_block();
        auto *merge = definition->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry = b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        auto *left_exit = b.br(merge);
        b.set_insertion_point(right);
        auto *right_exit = b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(m.create_constant_zero(Type::of<int>()), left);
        phi->add_incoming(m.create_constant_one(Type::of<int>()), right);
        auto *ret = b.return_void();
        auto block_count = definition->basic_blocks().count_size();

        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(!info.succeeded());
        expect(info.invalid_construct_count == 1u);
        expect(!info.changed());
        expect(definition->basic_blocks().count_size() == block_count);
        expect(body->terminator() == entry);
        expect(left->terminator() == left_exit);
        expect(right->terminator() == right_exit);
        expect(merge->terminator() == ret);
        expect(phi->is_linked());
    };

    "restructure_module_rejection_is_atomic_across_functions"_test = [] {
        Module m;
        XIRBuilder b;

        BasicBlock *valid_body;
        auto *valid = make_kernel_with_body(m, valid_body);
        auto *valid_condition =
            valid->create_value_argument(Type::of<bool>());
        auto *valid_left = valid->create_basic_block();
        auto *valid_right = valid->create_basic_block();
        auto *valid_merge = valid->create_basic_block();
        b.set_insertion_point(valid_body);
        auto *valid_entry =
            b.cond_br(valid_condition, valid_left, valid_right);
        b.set_insertion_point(valid_left);
        auto *valid_left_exit = b.br(valid_merge);
        b.set_insertion_point(valid_right);
        auto *valid_right_exit = b.br(valid_merge);
        b.set_insertion_point(valid_merge);
        b.return_void();

        BasicBlock *invalid_body;
        auto *invalid = make_kernel_with_body(m, invalid_body);
        auto *invalid_condition0 =
            invalid->create_value_argument(Type::of<bool>());
        auto *invalid_condition1 =
            invalid->create_value_argument(Type::of<bool>());
        auto *invalid_left = invalid->create_basic_block();
        auto *invalid_right = invalid->create_basic_block();
        auto *invalid_exit = invalid->create_basic_block();
        b.set_insertion_point(invalid_body);
        auto *invalid_entry = b.cond_br(
            invalid_condition0, invalid_left, invalid_right);
        b.set_insertion_point(invalid_left);
        b.br(invalid_right);
        b.set_insertion_point(invalid_right);
        b.cond_br(invalid_condition1, invalid_left, invalid_exit);
        b.set_insertion_point(invalid_exit);
        b.return_void();

        auto valid_block_count =
            valid->definition()->basic_blocks().count_size();
        auto invalid_block_count =
            invalid->definition()->basic_blocks().count_size();
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(!info.succeeded());
        expect(info.irreducible_region_count == 1u);
        expect(info.boundary_verifier_count == 1u);
        expect(info.intermediate_verifier_count == 0u);
        expect(!info.changed());
        expect(valid->definition()->basic_blocks().count_size() ==
               valid_block_count);
        expect(invalid->definition()->basic_blocks().count_size() ==
               invalid_block_count);
        expect(valid_body->terminator() == valid_entry);
        expect(valid_left->terminator() == valid_left_exit);
        expect(valid_right->terminator() == valid_right_exit);
        expect(invalid_body->terminator() == invalid_entry);
    };

    "restructure_iteration_exhaustion_rolls_back_shadow_cfg"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *continue_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *header = kernel->create_basic_block();
        auto *loop_body = kernel->create_basic_block();
        auto *header_exit = kernel->create_basic_block();
        auto *latch_exit = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *header_branch =
            b.cond_br(condition, loop_body, header_exit);
        b.set_insertion_point(loop_body);
        auto *back_edge =
            b.cond_br(continue_condition, header, latch_exit);
        b.set_insertion_point(header_exit);
        b.return_void();
        b.set_insertion_point(latch_exit);
        b.return_void();

        auto block_count_before = count_owned_blocks(kernel);
        auto function_count_before =
            m.function_list().count_size();
        auto constant_count_before =
            m.constant_list().count_size();
        auto info = restructure_cfg_pass_run_on_function(
            kernel,
            {.main_iteration_limit = 1u,
             .post_iteration_limit = 64u});

        expect(!info.succeeded());
        expect(!info.changed());
        expect(info.iteration_limit_count == 1u);
        expect(kernel->body_block() == entry);
        expect(entry->terminator()->isa<BranchInst>());
        expect(header->terminator() == header_branch);
        expect(loop_body->terminator() == back_edge);
        expect(count_owned_blocks(kernel) == block_count_before);
        expect(m.function_list().count_size() ==
               function_count_before);
        expect(m.constant_list().count_size() ==
               constant_count_before);
        expect(xir_verify_module(&m).succeeded());
        // The failed dry run created exit-selector constants. Rollback must
        // remove both list nodes and hash-bucket entries so reinterning works.
        auto *probe =
            m.create_constant_zero(Type::of<uint32_t>());
        expect(m.constant_list().count_size() ==
               constant_count_before + 1u);
        expect(m.remove_constant_if_unused(probe));
        expect(m.constant_list().count_size() ==
               constant_count_before);
    };

    "restructure_ignores_disconnected_construct_predecessors"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *selection_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *loop_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *latch_condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(entry);
        auto *selection = b.if_(selection_condition);
        auto *selection_true =
            selection->create_true_block();
        auto *selection_false =
            selection->create_false_block();
        auto *selection_merge =
            selection->create_merge_block();
        auto *loop_header = kernel->create_basic_block();
        auto *loop_body = kernel->create_basic_block();
        auto *header_exit = kernel->create_basic_block();
        auto *latch_exit = kernel->create_basic_block();
        auto *disconnected = kernel->create_basic_block();

        b.set_insertion_point(selection_true);
        b.br(selection_merge);
        b.set_insertion_point(selection_false);
        b.br(selection_merge);
        b.set_insertion_point(selection_merge);
        b.br(loop_header);
        b.set_insertion_point(loop_header);
        b.cond_br(loop_condition, loop_body, header_exit);
        b.set_insertion_point(loop_body);
        b.cond_br(
            latch_condition, loop_header, latch_exit);
        b.set_insertion_point(header_exit);
        b.return_void();
        b.set_insertion_point(latch_exit);
        b.return_void();

        // BasicBlock predecessor queries include this owned but unreachable
        // edge. It is not a second dynamic entry into the reachable IfInst.
        b.set_insertion_point(disconnected);
        b.br(selection_true);

        expect(xir_verify_module(&m).succeeded());
        auto block_count = count_owned_blocks(kernel);
        auto info = restructure_cfg_pass_run_on_function(
            kernel,
            {.main_iteration_limit = 1u,
             .post_iteration_limit = 64u});

        // The one-iteration budget deliberately reports only exhaustion. The
        // disconnected use-list edge must neither be cloned nor misreported as
        // a malformed structured construct.
        expect(!info.succeeded());
        expect(!info.changed());
        expect(info.iteration_limit_count == 1u);
        expect(info.invalid_construct_count == 0u);
        expect(count_owned_blocks(kernel) == block_count);
        expect(entry->terminator() == selection);
        expect(disconnected->terminator()->isa<BranchInst>());
        expect(xir_verify_module(&m).succeeded());
    };

    "restructure_normalizes_loop_boundary_guard_before_entry_enforcement"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(module, entry);
        auto *loop_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *boundary_condition =
            kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();

        builder.set_insertion_point(prepare);
        builder.cond_br(
            loop_condition, loop_body, loop_merge);
        builder.set_insertion_point(loop_body);
        auto *boundary = builder.if_(boundary_condition);
        boundary->set_true_target(prepare);
        boundary->set_false_target(loop_merge);
        auto *synthetic_merge = boundary->create_merge_block();
        builder.set_insertion_point(synthetic_merge);
        builder.unreachable_();
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto block_count = count_owned_blocks(kernel);

        auto info = restructure_cfg_pass_run_on_function(
            kernel,
            {.main_iteration_limit = 0u,
             .post_iteration_limit = 64u});

        expect(info.succeeded());
        expect(info.changed());
        expect(info.iteration_limit_count == 0u);
        expect(info.invalid_construct_count == 0u);
        expect(count_owned_blocks(kernel) <= block_count + 4u)
            << "normalizing a physical loop guard may add only constant-size "
               "boundary proxies, not clone the loop prepare region";
        expect(entry->terminator() == loop);
        expect(boundary->merge_block() ==
               boundary->false_block());
        auto *boundary_merge_terminator =
            boundary->false_block()->terminator();
        expect(
            branch_chain_reaches(
                boundary->false_block(), loop_merge) ||
            (boundary_merge_terminator->isa<BreakInst>() &&
             static_cast<BreakInst *>(
                 boundary_merge_terminator)
                     ->target_block() == loop_merge));
        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
    };

    "restructure_module_late_failure_is_atomic_across_functions"_test = [] {
        Module m;
        BasicBlock *first_entry;
        auto *first = make_kernel_with_body(m, first_entry);
        auto *first_condition =
            first->create_value_argument(Type::of<bool>());
        auto *first_true = first->create_basic_block();
        auto *first_false = first->create_basic_block();
        auto *first_merge = first->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(first_entry);
        auto *first_branch =
            b.cond_br(first_condition, first_true, first_false);
        b.set_insertion_point(first_true);
        b.br(first_merge);
        b.set_insertion_point(first_false);
        b.br(first_merge);
        b.set_insertion_point(first_merge);
        b.return_void();

        BasicBlock *second_entry;
        auto *second = make_kernel_with_body(m, second_entry);
        auto *second_condition =
            second->create_value_argument(Type::of<bool>());
        auto *second_header = second->create_basic_block();
        auto *second_body = second->create_basic_block();
        auto *second_exit = second->create_basic_block();
        b.set_insertion_point(second_entry);
        b.br(second_header);
        b.set_insertion_point(second_header);
        auto *second_branch =
            b.cond_br(second_condition, second_body, second_exit);
        b.set_insertion_point(second_body);
        b.br(second_header);
        b.set_insertion_point(second_exit);
        b.return_void();

        auto first_block_count = count_owned_blocks(first);
        auto second_block_count = count_owned_blocks(second);
        auto function_count = m.function_list().count_size();
        auto constant_count = m.constant_list().count_size();
        auto info = restructure_cfg_pass_run_on_module(
            &m, nullptr,
            {.main_iteration_limit = 1u,
             .post_iteration_limit = 64u});

        expect(!info.succeeded());
        expect(!info.changed());
        expect(info.iteration_limit_count == 1u);
        expect(first->body_block() == first_entry);
        expect(first_entry->terminator() == first_branch);
        expect(second->body_block() == second_entry);
        expect(second_header->terminator() == second_branch);
        expect(count_owned_blocks(first) == first_block_count);
        expect(count_owned_blocks(second) == second_block_count);
        expect(m.function_list().count_size() == function_count);
        expect(m.constant_list().count_size() == constant_count);
        expect(xir_verify_module(&m).succeeded());
    };

    "restructure_null_module_is_a_noop"_test = [] {
        auto info = restructure_cfg_pass_run_on_module(nullptr);
        expect(info.succeeded());
        expect(!info.changed());
    };

    "restructure_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_simple_loop_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        auto *lbody = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        auto *cond = m.create_constant_zero(Type::of<bool>());
        auto *cont = k->definition()->create_basic_block();
        b.set_insertion_point(lbody);
        b.cond_br(cond, merge, cont);
        b.set_insertion_point(cont);
        b.continue_(lbody);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 1u);
    };

    "restructure_simple_loop_latch_conditional_to_break_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *loop_body = def->create_basic_block();
        auto *work = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *break_proxy = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        sl->set_body_block(loop_body);
        sl->set_merge_block(merge);
        b.set_insertion_point(loop_body);
        b.br(work);
        b.set_insertion_point(work);
        b.br(latch);
        b.set_insertion_point(latch);
        b.cond_br(cond, break_proxy, loop_body);
        b.set_insertion_point(break_proxy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) <= 1u);
    };

    "restructure_simple_loop_nested_latch_conditional_to_break_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *loop_body = def->create_basic_block();
        auto *then_block = def->create_basic_block();
        auto *else_block = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *break_proxy = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        sl->set_body_block(loop_body);
        sl->set_merge_block(merge);
        b.set_insertion_point(loop_body);
        auto *inner_if = b.if_(cond);
        inner_if->set_true_target(then_block);
        inner_if->set_false_target(else_block);
        inner_if->set_merge_block(inner_merge);
        b.set_insertion_point(then_block);
        b.br(inner_merge);
        b.set_insertion_point(else_block);
        b.br(inner_merge);
        b.set_insertion_point(inner_merge);
        b.br(latch);
        b.set_insertion_point(latch);
        b.cond_br(cond, break_proxy, loop_body);
        b.set_insertion_point(break_proxy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) == 1u);
    };

    "restructure_loop_body_break_or_continue_through_proxy_chain"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *loop_body = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *continue_proxy_0 = def->create_basic_block();
        auto *continue_proxy_1 = def->create_basic_block();
        auto *update = def->create_basic_block();
        auto *merge = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        loop->set_prepare_block(header);
        loop->set_body_block(loop_body);
        loop->set_update_block(update);
        loop->set_merge_block(merge);
        b.set_insertion_point(header);
        b.cond_br(cond, loop_body, merge);
        b.set_insertion_point(loop_body);
        b.cond_br(cond, break_block, continue_proxy_0);
        b.set_insertion_point(break_block);
        b.break_(merge);
        b.set_insertion_point(continue_proxy_0);
        b.br(continue_proxy_1);
        b.set_insertion_point(continue_proxy_1);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(header);
        b.set_insertion_point(merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 2u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.irreducible_region_count == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) == 1u);
    };

    "restructure_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        for (size_t i = 0; i < kFns; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *if_inst = b.if_(cond);
            auto *t = if_inst->create_true_block();
            auto *f = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            b.set_insertion_point(t);
            b.br(merge);
            b.set_insertion_point(f);
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();
        }
        (void)destructure_cfg_pass_run_on_module(&m);
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == kFns);
        for (auto *f : m.function_list()) {
            auto *def = f->definition();
            if (def == nullptr) { continue; }
            expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 1u);
            expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        }
    };

    "restructure_idempotent_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto first = restructure_cfg_pass_run_on_function(k);
        auto second = restructure_cfg_pass_run_on_function(k);
        expect(first.restructured_if_count == 1u);
        expect(second.restructured_if_count == 0u);
    };

    "restructure_construct_entry_dominance_is_linear_per_cfg_version"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *cond =
            m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        constexpr auto construct_count = size_t{256u};
        for (auto i = 0u; i < construct_count; ++i) {
            auto *if_inst = b.if_(cond);
            auto *t = if_inst->create_true_block();
            auto *f = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            b.set_insertion_point(t);
            b.br(merge);
            b.set_insertion_point(f);
            b.br(merge);
            b.set_insertion_point(merge);
        }
        b.return_void();

        auto info =
            restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(!info.changed());
        expect(
            count_terminator_kind(
                k->definition(),
                DerivedInstructionTag::IF) ==
            construct_count);
        // Entry legality for every construct is queried against the same
        // immutable CFG version, so one dominance tree is both necessary and
        // sufficient. The count may grow only after an actual CFG mutation.
        expect(
            info.construct_entry_dom_tree_count == 1u);
        // Loop-boundary membership is another value of this immutable CFG,
        // independent of the number of construct queries.
        expect(
            info.construct_entry_boundary_analysis_count ==
            1u);
        expect(
            info.construct_exit_boundary_analysis_count ==
            1u);
        // Each diamond ends at its merge before the next begins. The sparse
        // dominator event walk suspends the completed construct for the merge
        // subtree, so there are no pairwise parent candidates to inspect.
        expect(
            info.construct_exit_parent_query_count == 0u);
        // Selection-exit legality for all 256 sites observes the same CFG.
        // The loop-boundary relation is materialized once, not rediscovered
        // by a full-function scan for every site.
        expect(
            info.selection_exit_boundary_analysis_count ==
            1u);
        expect(
            info.selection_exit_site_query_count ==
            construct_count);
        expect(
            info.selection_exit_enclosing_loop_query_count ==
            construct_count);
        // The final post-merge audit asks each merge's sparse dominance
        // frontier. These sequential diamonds have empty frontiers, so graph
        // width cannot turn the audit into construct_count * block_count.
        expect(
            info.selection_reentry_audit_selection_query_count ==
            construct_count);
        expect(
            info.selection_reentry_audit_frontier_query_count ==
            0u);
        expect(
            info.selection_reentry_audit_predecessor_query_count ==
            0u);
    };

    "restructure_empty_module_noop"_test = [] {
        Module m;
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(info.restructured_if_count == 0u);
        expect(info.restructured_loop_count == 0u);
        expect(info.boundary_verifier_count == 2u);
        expect(info.intermediate_verifier_count == 0u);
    };

    "restructure_verifies_only_complete_pass_boundaries_by_default"_test = [] {
        ScopedEnvironmentVariable disable_intermediate_verification{
            "LUISA_XIR_VERIFY_INTERMEDIATE", nullptr};
        auto append_diamond = [](
                                  Module &module) noexcept {
            BasicBlock *body;
            auto *kernel =
                make_kernel_with_body(module, body);
            auto *condition =
                kernel->create_value_argument(
                    Type::of<bool>());
            auto *true_block =
                kernel->create_basic_block();
            auto *false_block =
                kernel->create_basic_block();
            auto *merge =
                kernel->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.cond_br(
                condition, true_block, false_block);
            builder.set_insertion_point(true_block);
            builder.br(merge);
            builder.set_insertion_point(false_block);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.return_void();
            return kernel;
        };

        Module module;
        append_diamond(module);
        append_diamond(module);
        auto module_info =
            restructure_cfg_pass_run_on_module(&module);
        expect(module_info.succeeded());
        expect(module_info.restructured_if_count == 2u);
        // The verifier work is a property of the pass boundary, not the
        // number of definitions or transactional dry-run/replay phases.
        expect(module_info.boundary_verifier_count == 2u);
        expect(module_info.intermediate_verifier_count == 0u);

        Module function_module;
        auto *kernel = append_diamond(
            function_module);
        auto function_info =
            restructure_cfg_pass_run_on_function(kernel);
        expect(function_info.succeeded());
        expect(function_info.restructured_if_count == 1u);
        expect(function_info.boundary_verifier_count == 2u);
        expect(function_info.intermediate_verifier_count == 0u);
    };

    "restructure_intermediate_verification_is_explicitly_opt_in"_test = [] {
        ScopedEnvironmentVariable enable_intermediate_verification{
            "LUISA_XIR_VERIFY_INTERMEDIATE", "1"};
        Module module;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(module, body);
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(
            condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.br(merge);
        builder.set_insertion_point(false_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto info =
            restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.boundary_verifier_count == 2u);
        expect(info.intermediate_verifier_count > 0u);
    };

    "restructure_if_preserves_true_false_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        (void)restructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        auto *new_term = def->body_block()->terminator();
        expect(new_term != nullptr);
        expect(new_term->isa<IfInst>());
        auto *rebuilt = static_cast<IfInst *>(new_term);
        // Empty true/false arms branched directly to merge, so during
        // destructure they collapse and restructure retargets them to a
        // fresh structural merge. The arms thus may equal either the
        // original blocks or the structural merge itself.
        auto *rt = rebuilt->true_block();
        auto *rf = rebuilt->false_block();
        auto *rm = rebuilt->merge_block();
        expect(rt != nullptr);
        expect(rf != nullptr);
        expect(rm != nullptr);
        // The structural merge must reach the original merge block. It is
        // either the original merge itself or a freshly-synthesized block
        // whose sole terminator is `br merge`.
        auto *rm_term = rm->terminator();
        expect(rm == merge ||
               (rm_term != nullptr &&
                rm_term->isa<BranchInst>() &&
                static_cast<BranchInst *>(rm_term)->target_block() == merge));
    };

    "restructure_nested_if_from_destructured"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        auto *omerge = outer->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        auto *imerge = inner->create_merge_block();
        b.set_insertion_point(it);
        b.br(imerge);
        b.set_insertion_point(if_);
        b.br(imerge);
        b.set_insertion_point(imerge);
        b.br(omerge);
        b.set_insertion_point(of);
        b.br(omerge);
        b.set_insertion_point(omerge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_if_batch_consumes_a_linear_diamond_chain"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        constexpr auto diamond_count = size_t{64u};
        XIRBuilder builder;
        auto *header = body;
        for (auto i = size_t{0u};
             i < diamond_count;
             ++i) {
            auto *true_block =
                kernel->create_basic_block();
            auto *false_block =
                kernel->create_basic_block();
            auto *merge = kernel->create_basic_block();
            builder.set_insertion_point(header);
            builder.cond_br(
                condition, true_block, false_block);
            builder.set_insertion_point(true_block);
            builder.br(merge);
            builder.set_insertion_point(false_block);
            builder.br(merge);
            header = merge;
        }
        builder.set_insertion_point(header);
        builder.return_void();

        auto info = restructure_cfg_pass_run_on_function(
            kernel,
            {.main_iteration_limit = 1u,
             .post_iteration_limit = 64u});

        // Candidate discovery uses one dom/post-dom snapshot. Every diamond's
        // merge remains inferable from the refreshed dominance tree, so no
        // post-mutation candidate consumes a rebuilt post-dom fallback.
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_if_count ==
               diamond_count);
        expect(
            info.if_batch_post_dom_rebuild_count ==
            0u);
        expect(count_terminator_kind(
                   kernel,
                   DerivedInstructionTag::
                       CONDITIONAL_BRANCH) == 0u);
        expect(count_terminator_kind(
                   kernel,
                   DerivedInstructionTag::IF) ==
               diamond_count);
        expect(xir_verify_module(
                   &m,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets =
                        true})
                   .succeeded());
    };

    "restructure_nested_loop_does_not_capture_outer_tail"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *outer_header = def->create_basic_block();
        auto *outer_body = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *after_inner = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *outer_exit = def->create_basic_block();
        auto *cond = m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        b.cond_br(cond, outer_body, outer_exit);
        b.set_insertion_point(outer_body);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        b.cond_br(cond, inner_body, after_inner);
        b.set_insertion_point(inner_body);
        b.br(inner_latch);
        b.set_insertion_point(inner_latch);
        b.br(inner_header);
        b.set_insertion_point(after_inner);
        b.cond_br(cond, outer_exit, outer_latch);
        b.set_insertion_point(outer_latch);
        b.br(outer_header);
        b.set_insertion_point(outer_exit);
        b.return_void();
        auto info = restructure_cfg_pass_run_on_function(k);
        auto loop_count = count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                          count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP);
        expect(info.restructured_loop_count == 2u);
        expect(loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 0u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_nested_loop_shared_outer_continue_collapses_exit_dispatch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *outer_condition =
            k->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            k->create_value_argument(Type::of<bool>());
        auto *continue_condition =
            k->create_value_argument(Type::of<bool>());
        auto *break_condition =
            k->create_value_argument(Type::of<bool>());

        auto *outer_header = def->create_basic_block();
        auto *outer_body = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *inner_work = def->create_basic_block();
        auto *inner_update = def->create_basic_block();
        auto *outer_update = def->create_basic_block();
        auto *exit = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        b.cond_br(outer_condition, outer_body, exit);
        b.set_insertion_point(outer_body);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        // Normal inner-loop exit is the outer loop's continue path.
        b.cond_br(inner_condition, inner_body, outer_update);
        b.set_insertion_point(inner_body);
        // One arm continues the inner loop; the other may break it.
        b.cond_br(
            continue_condition, inner_update, inner_work);
        b.set_insertion_point(inner_work);
        b.cond_br(
            break_condition, outer_update, inner_update);
        b.set_insertion_point(inner_update);
        b.br(inner_header);
        b.set_insertion_point(outer_update);
        b.br(outer_header);
        b.set_insertion_point(exit);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.restructured_loop_count == 2u);
        expect(count_terminator_kind(
                   def, DerivedInstructionTag::LOOP) ==
               2u);
        expect(count_terminator_kind(
                   def,
                   DerivedInstructionTag::CONDITIONAL_BRANCH) ==
               count_canonical_conditional_loop_prepare(def));
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
        // The only raw conditionals allowed to remain are native loop guards.
        // A state-writing exit proxy can require the corresponding inner
        // guard to be represented as a loop-boundary IfInst instead.
        auto verification = xir_verify_module(
            &m,
            {.require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets =
                 true});
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown verification failure" :
                    verification.errors.front().message);
        auto block_count = size_t{0u};
        def->traverse_basic_blocks(
            [&](BasicBlock *) noexcept { ++block_count; });
        auto rerun = restructure_cfg_pass_run_on_function(k);
        expect(rerun.succeeded());
        auto rerun_block_count = size_t{0u};
        def->traverse_basic_blocks(
            [&](BasicBlock *) noexcept {
                ++rerun_block_count;
            });
        expect(rerun_block_count == block_count);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_nested_loop_boundary_selections_yield_before_dispatch_collapse"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *loop_condition =
            k->create_value_argument(Type::of<bool>());
        auto *outer_condition =
            k->create_value_argument(Type::of<bool>());
        auto *first_boundary_condition =
            k->create_value_argument(Type::of<bool>());
        auto *second_boundary_condition =
            k->create_value_argument(Type::of<bool>());

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.cond_br(loop_condition, loop_body, merge);
        b.set_insertion_point(loop_body);
        auto *outer = b.if_(outer_condition);
        auto *first = outer->create_true_block();
        auto *second = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();

        auto build_boundary_selection =
            [&](BasicBlock *header, Value *condition) noexcept {
                b.set_insertion_point(header);
                auto *selection = b.if_(condition);
                auto *merge_arm = selection->create_true_block();
                auto *break_arm = selection->create_false_block();
                selection->set_merge_block(merge_arm);
                b.set_insertion_point(merge_arm);
                b.break_(merge);
                b.set_insertion_point(break_arm);
                b.break_(merge);
            };
        build_boundary_selection(
            first, first_boundary_condition);
        build_boundary_selection(
            second, second_boundary_condition);
        b.set_insertion_point(outer_merge);
        b.unreachable_();
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(info.selection_exit_round_yield_count > 0u);
        auto verification = xir_verify_module(
            &m,
            {.require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets = true});
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown verification failure" :
                    verification.errors.front().message);
    };

    "restructure_outer_update_path_with_inner_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *outer_continue_entry = def->create_basic_block();
        auto *after_inner = def->create_basic_block();

        b.set_insertion_point(body);
        auto *outer = b.loop();
        auto *outer_prepare = outer->create_prepare_block();
        auto *outer_body = outer->create_body_block();
        auto *outer_update = outer->create_update_block();
        auto *outer_merge = outer->create_merge_block();

        b.set_insertion_point(outer_prepare);
        b.cond_br(cond, outer_body, outer_merge);

        b.set_insertion_point(outer_body);
        auto *body_if = b.if_(cond);
        auto *body_then = body_if->create_true_block();
        auto *body_else = body_if->create_false_block();
        auto *body_if_merge = body_if->create_merge_block();
        b.set_insertion_point(body_then);
        b.br(body_if_merge);
        b.set_insertion_point(body_else);
        b.br(body_if_merge);
        b.set_insertion_point(body_if_merge);
        auto *continue_if = b.if_(cond);
        auto *break_path = continue_if->create_true_block();
        auto *continue_path = continue_if->create_false_block();
        auto *continue_if_merge = continue_if->create_merge_block();
        b.set_insertion_point(break_path);
        b.break_(outer_merge);
        b.set_insertion_point(continue_path);
        b.br(continue_if_merge);
        b.set_insertion_point(continue_if_merge);
        b.br(outer_continue_entry);

        b.set_insertion_point(outer_continue_entry);
        auto *inner = b.loop();
        auto *inner_prepare = inner->create_prepare_block();
        auto *inner_body = inner->create_body_block();
        auto *inner_update = inner->create_update_block();
        auto *inner_merge = inner->create_merge_block();
        b.set_insertion_point(inner_prepare);
        b.cond_br(cond, inner_body, inner_merge);
        b.set_insertion_point(inner_body);
        b.br(inner_update);
        b.set_insertion_point(inner_update);
        b.br(inner_prepare);
        b.set_insertion_point(inner_merge);
        b.br(after_inner);
        b.set_insertion_point(after_inner);
        b.br(outer_update);

        b.set_insertion_point(outer_update);
        b.br(outer_prepare);
        b.set_insertion_point(outer_merge);
        b.return_void();

        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 2u);
        run_spirv_normalize_before_restructure(&m);
        expect_no_structured_cfg(def);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                   count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) ==
               2u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_outer_loop_break_or_update_if_shape"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *body_entry = def->create_basic_block();
        auto *first_then = def->create_basic_block();
        auto *first_else = def->create_basic_block();
        auto *first_merge = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *update_path = def->create_basic_block();
        auto *if_merge_on_update_path = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(header);
        b.set_insertion_point(header);
        b.cond_br(cond, body_entry, exit);
        b.set_insertion_point(body_entry);
        b.cond_br(cond, first_then, first_else);
        b.set_insertion_point(first_then);
        b.br(first_merge);
        b.set_insertion_point(first_else);
        b.br(first_merge);
        b.set_insertion_point(first_merge);
        b.cond_br(cond, break_block, update_path);
        b.set_insertion_point(break_block);
        b.br(exit);
        b.set_insertion_point(update_path);
        b.br(if_merge_on_update_path);
        b.set_insertion_point(if_merge_on_update_path);
        b.br(latch);
        b.set_insertion_point(latch);
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();

        run_spirv_normalize_before_restructure(&m);
        expect_no_structured_cfg(def);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 1u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 1u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
    };

    "restructure_full_pipeline_loop_with_inner_phi_diamond"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *buffer = k->create_resource_argument(
            Type::buffer(Type::of<int>()));
        auto *header = def->create_basic_block();
        auto *loop_body = def->create_basic_block();
        auto *then_block = def->create_basic_block();
        auto *else_block = def->create_basic_block();
        auto *diamond_merge = def->create_basic_block();
        auto *break_block = def->create_basic_block();
        auto *continue_block = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *index = m.create_constant_zero(Type::of<uint>());
        int one_v = 1;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        int two_v = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_v);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(header);
        b.set_insertion_point(header);
        b.cond_br(cond, loop_body, exit);
        b.set_insertion_point(loop_body);
        b.cond_br(cond, then_block, else_block);
        b.set_insertion_point(then_block);
        auto *then_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, one});
        b.call(ResourceWriteOp::BUFFER_WRITE,
               {buffer, index, then_value});
        b.br(diamond_merge);
        b.set_insertion_point(else_block);
        auto *else_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, two});
        b.call(ResourceWriteOp::BUFFER_WRITE,
               {buffer, index, else_value});
        b.br(diamond_merge);
        b.set_insertion_point(diamond_merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(then_value, then_block);
        phi->add_incoming(else_value, else_block);
        auto *break_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {phi, zero});
        b.cond_br(break_cond, break_block, continue_block);
        b.set_insertion_point(break_block);
        b.br(exit);
        b.set_insertion_point(continue_block);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        b.cond_br(cond, inner_body, inner_merge);
        b.set_insertion_point(inner_body);
        b.br(inner_latch);
        b.set_insertion_point(inner_latch);
        b.br(inner_header);
        b.set_insertion_point(inner_merge);
        b.br(outer_latch);
        b.set_insertion_point(outer_latch);
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();

        expect(count_phi(def) == 1u);
        run_spirv_normalize_before_restructure(&m);
        auto lowered_spills = audit_reg2mem_spills_on_module(&m);
        expect(lowered_spills.remaining_phi_spill_count == 1u);
        expect(lowered_spills.remaining_cross_block_spill_count == 0u);
        expect_no_structured_cfg(def);
        expect(count_phi(def) == 0u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_loop_count == 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) +
                   count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) ==
               2u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(count_non_canonical_loop_update(def) == 0u);
        auto restructured_spills = audit_reg2mem_spills_on_module(&m);
        expect(restructured_spills.remaining_phi_spill_count == 1u);
        expect(restructured_spills.remaining_cross_block_spill_count == 0u);
        auto mem2reg = mem2reg_pass_run_on_module(&m);
        expect(mem2reg.promoted_alloca_count > 0u);
        auto recovered_spills = audit_reg2mem_spills_on_module(&m);
        expect(recovered_spills.succeeded());
        expect(count_phi(def) > 0u);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_full_pipeline_ast_kernel_nested_loop_break"_test = [] {
        Kernel1D kernel = [](BufferFloat buf, Float t) noexcept {
            auto idx = dispatch_id().x;
            Float x = buf.read(idx);
            Float acc = def(0.0f);
            Float state = def(x);
            Bool flag = def(false);
            $for (i, 8u) {
                auto hit = state > t;
                $if (hit & (state != acc)) {
                    flag = flag | hit;
                    hit = false;
                    Float tmp = def(0.0f);
                    $for (j, 4u) {
                        tmp += state * cast<float>(j + 1u);
                    };
                    state = tmp * 0.25f;
                };
                acc += state;
                state += 1.0f;
                $if (hit) { $break; };
            };
            buf.write(idx, ite(flag, acc, state));
        };

        auto m = ast_to_xir_translate(kernel.function()->function(), {});
        expect(m != nullptr);
        run_spirv_normalize_before_restructure(m.get());
        for (auto *f : m->function_list()) {
            if (auto *def = f->definition(); def != nullptr) {
                expect_no_structured_cfg(def);
                expect(count_phi(def) == 0u);
            }
        }
        auto info = restructure_cfg_pass_run_on_module(m.get());
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_loop_count > 0u);
        for (auto *f : m->function_list()) {
            if (auto *def = f->definition(); def != nullptr) {
                expect(count_non_canonical_loop_prepare(def) == 0u);
                expect(count_non_canonical_loop_update(def) == 0u);
            }
        }
        expect(xir_verify_module(
                   m.get(), {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_many_sequential_loop_breaks_stays_linear"_test = [] {
        constexpr auto loop_count = 32u;
        Kernel1D kernel = [](BufferUInt limits,
                             BufferUInt break_at,
                             BufferUInt output) noexcept {
            UInt checksum = 0u;
            for (auto site = 0u; site < loop_count; ++site) {
                UInt index = 0u;
                $while (index < limits.read(site)) {
                    $if (index == break_at.read(site)) {
                        $break;
                    };
                    checksum += index + site;
                    index += 1u;
                };
            }
            output.write(dispatch_x(), checksum);
        };

        auto module =
            ast_to_xir_translate(
                kernel.function()->function(), {});
        expect(module != nullptr);
        run_spirv_normalize_before_restructure(
            module.get());
        auto *definition =
            module->function_list().front()
                ->definition();
        expect(definition != nullptr);
        const auto input_block_count =
            count_owned_blocks(definition);

        auto info =
            restructure_cfg_pass_run_on_module(
                module.get());
        const auto output_block_count =
            count_owned_blocks(definition);

        expect(info.succeeded())
            << "independent loop exits must reach a fixed point "
               "(input blocks: "
            << input_block_count << ", output blocks after "
               "transaction: "
            << output_block_count << ", iteration limits: "
            << info.iteration_limit_count << ")";
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_loop_count == loop_count);
        expect(output_block_count <=
               input_block_count * 4u)
            << "restructuring independent loop exits must stay linear "
               "in input CFG size (input blocks: "
            << input_block_count << ", output blocks: "
            << output_block_count << ")";
        expect(count_terminator_kind(
                   definition,
                   DerivedInstructionTag::CONDITIONAL_BRANCH) ==
               count_canonical_conditional_loop_prepare(
                   definition))
            << "only canonical native loop guards may remain raw";
        auto verification = xir_verify_module(
            module.get(),
            {.require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets =
                 true});
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown verification failure" :
                    verification.errors.front().message);
    };

    "restructure_routes_forwarded_loop_exits_through_declared_merge"_test = [] {
        Kernel1D kernel = [](BufferUInt output,
                             Bool a, Bool b,
                             Bool c, Bool d) noexcept {
            $loop {
                $if (a) {
                    $break;
                };
                $if (b) {
                    $if (c) {
                        $break;
                    };
                };
                $if (d) {
                    $continue;
                };
            };
            output.write(0u, 1u);
        };

        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        run_spirv_normalize_before_restructure(
            module.get());
        auto *definition =
            module->function_list().front()->definition();
        expect(definition != nullptr);

        auto info = restructure_cfg_pass_run_on_module(
            module.get());
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);

        LoopInst *recovered_loop = nullptr;
        auto loop_count = size_t{0u};
        definition->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                if (block->is_terminated() &&
                    block->terminator()->isa<LoopInst>()) {
                    recovered_loop = static_cast<LoopInst *>(
                        block->terminator());
                    loop_count++;
                }
            });
        expect(loop_count == 1u);
        expect(recovered_loop != nullptr);
        if (recovered_loop == nullptr) { return; }

        auto *merge = recovered_loop->merge_block();
        expect(merge != nullptr);
        expect(merge != nullptr && merge->is_terminated() &&
               merge->terminator()->isa<BranchInst>());
        if (merge == nullptr || !merge->is_terminated() ||
            !merge->terminator()->isa<BranchInst>()) {
            return;
        }
        auto *continuation =
            static_cast<BranchInst *>(merge->terminator())
                ->target_block();
        expect(continuation != nullptr);

        auto canonical_break_count = size_t{0u};
        definition->traverse_basic_blocks(
            [&](BasicBlock *block) noexcept {
                if (!block->is_terminated() ||
                    !block->terminator()->isa<BreakInst>()) {
                    return;
                }
                canonical_break_count +=
                    static_cast<BreakInst *>(
                        block->terminator())
                                ->target_block() == merge ?
                        1u :
                        0u;
            });
        expect(canonical_break_count > 0u)
            << "an internal edge to the merge's forwarding destination "
               "must become an explicit Break to the declared merge";

        auto merge_predecessor_count = size_t{0u};
        auto bypass_predecessor_count = size_t{0u};
        continuation->traverse_predecessors(
            false,
            [&](BasicBlock *predecessor) noexcept {
                if (predecessor == merge) {
                    merge_predecessor_count++;
                } else {
                    bypass_predecessor_count++;
                }
            });
        expect(merge_predecessor_count == 1u);
        expect(bypass_predecessor_count == 0u)
            << "the loop continuation must have no predecessor that "
               "bypasses its declared single-exit merge";

        auto block_count = count_owned_blocks(definition);
        auto rerun = restructure_cfg_pass_run_on_module(
            module.get());
        expect(rerun.succeeded());
        expect(rerun.iteration_limit_count == 0u);
        expect(count_owned_blocks(definition) == block_count)
            << "loop-boundary canonicalization must be idempotent";
        auto verification = xir_verify_module(
            module.get(),
            {.require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets =
                 true});
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown verification failure" :
                    verification.errors.front().message);
    };

    "restructure_converts_remaining_divergent_conditional"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *ret_a = def->create_basic_block();
        auto *ret_b = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(cond, ret_a, ret_b);
        b.set_insertion_point(ret_a);
        b.return_void();
        b.set_insertion_point(ret_b);
        b.return_void();
        // Skip full pipeline; reg2mem is a no-op here (no phis).
        (void)reg2mem_pass_run_on_function(k);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.restructured_if_count >= 1u) << "conditional branch should be structurized";
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_fixup_nested_if_cross_hierarchy"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *outer_merge = def->create_basic_block();
        auto *inner_merge = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        b.set_insertion_point(it);
        b.br(inner_merge);
        b.set_insertion_point(if_);
        b.br(outer_merge);
        b.set_insertion_point(inner_merge);
        b.br(outer_merge);
        b.set_insertion_point(of);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();
        (void)destructure_cfg_pass_run_on_function(k);
        (void)reg2mem_pass_run_on_function(k);
        expect_no_structured_cfg(def);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) >= 2u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.restructured_if_count >= 2u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
    };

    "restructure_shared_successor_uses_single_exit_protocol"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *outer_condition =
            k->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *arm = def->create_basic_block();
        auto *local_exit = def->create_basic_block();
        auto *shared = def->create_basic_block();
        auto *exit = def->create_basic_block();
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *slot = b.alloca_local(Type::of<int>());
        b.cond_br(outer_condition, header, shared);
        b.set_insertion_point(header);
        b.cond_br(inner_condition, arm, local_exit);
        b.set_insertion_point(arm);
        b.br(shared);
        b.set_insertion_point(local_exit);
        b.br(exit);
        b.set_insertion_point(shared);
        auto *original_store =
            b.store(slot, m.create_constant_one(Type::of<int>()));
        b.br(exit);
        b.set_insertion_point(exit);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.unstructured_branch_count == 0u);
        expect(original_store->parent_block() == shared);
        size_t writes_to_slot = 0u;
        def->traverse_instructions([&](Instruction *instruction) noexcept {
            if (instruction->isa<StoreInst>() &&
                static_cast<StoreInst *>(instruction)->variable() ==
                    slot) {
                ++writes_to_slot;
            }
        });
        // The shared block is not dominated by the inner header. Cloning its
        // reachable subgraph would duplicate this side effect; the explicit
        // single-exit protocol must preserve one copy instead.
        expect(writes_to_slot == 1u);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_nested_selection_exit_to_shared_continuation_converges"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *outer_condition =
            k->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            k->create_value_argument(Type::of<bool>());
        auto *query_type =
            Type::custom("LC_RayQueryAll");
        auto *query_source =
            k->create_reference_argument(query_type);
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *slot = b.alloca_local(Type::of<uint32_t>());
        auto *query_slot = b.alloca_local(query_type);
        auto *outer = b.if_(outer_condition);
        auto *outer_true = outer->create_true_block();
        auto *shared_continuation =
            outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();

        b.set_insertion_point(outer_true);
        auto *inner = b.if_(inner_condition);
        auto *inner_true = inner->create_true_block();
        auto *inner_false = inner->create_false_block();
        auto *inner_merge = inner->create_merge_block();
        b.set_insertion_point(inner_true);
        b.br(shared_continuation);
        b.set_insertion_point(inner_false);
        b.store(
            slot,
            m.create_constant_one(Type::of<uint32_t>()));
        b.br(outer_merge);
        b.set_insertion_point(inner_merge);
        b.unreachable_();
        b.set_insertion_point(shared_continuation);
        b.store(
            slot,
            m.create_constant_zero(Type::of<uint32_t>()));
        auto *query_value =
            b.load(query_type, query_source);
        b.store(query_slot, query_value);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto initial_block_count = count_owned_blocks(def);
        auto info = restructure_cfg_pass_run_on_function(
            k, {.main_iteration_limit = 64u,
                .post_iteration_limit = 8u});
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(info.unstructured_branch_count == 0u);
        auto query_alloca_count = size_t{0u};
        def->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                query_alloca_count +=
                    instruction->isa<AllocaInst>() &&
                    instruction->type() == query_type;
            });
        // Node splitting creates a second mutually exclusive initializer, so
        // affine ray-query state must receive distinct storage on each path.
        expect(query_alloca_count == 2u);
        // A finite structurization may split a shared continuation, but its
        // size must be bounded by the input graph rather than one copy per
        // fixed-point iteration.
        expect(count_owned_blocks(def) <=
               initial_block_count * 4u);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_splits_dispatch_reentry_through_fallback_proxy"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *selector =
            k->create_value_argument(Type::of<uint32_t>());
        auto *root_condition =
            k->create_value_argument(Type::of<bool>());
        auto *nested_condition =
            k->create_value_argument(Type::of<bool>());
        auto *query_type =
            Type::custom("LC_RayQueryAll");
        auto *query_source =
            k->create_reference_argument(query_type);
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *query_slot = b.alloca_local(query_type);
        auto *root = b.if_(root_condition);
        auto *scenario_body =
            root->create_true_block();
        auto *noise_body =
            root->create_false_block();
        auto *root_merge =
            root->create_merge_block();
        b.set_insertion_point(scenario_body);
        auto *selection = b.switch_(selector);
        auto *first_return_arm =
            selection->create_case_block(1u);
        auto *second_return_arm =
            selection->create_case_block(2u);
        auto *cross_arm =
            selection->create_case_block(3u);
        auto *nested_header =
            selection->create_default_block();
        auto *original_merge =
            selection->create_merge_block();
        auto *nested_merge =
            def->create_basic_block();
        auto *first_return =
            def->create_basic_block();
        auto *second_return =
            def->create_basic_block();

        b.set_insertion_point(first_return_arm);
        b.br(first_return);
        b.set_insertion_point(second_return_arm);
        b.br(second_return);
        b.set_insertion_point(cross_arm);
        b.br(nested_merge);
        b.set_insertion_point(nested_header);
        auto *nested = b.if_(nested_condition);
        auto *nested_true = nested->create_true_block();
        auto *nested_false = nested->create_false_block();
        nested->set_merge_block(nested_merge);
        b.set_insertion_point(nested_true);
        b.br(nested_merge);
        b.set_insertion_point(nested_false);
        b.br(nested_merge);
        b.set_insertion_point(nested_merge);
        auto *query_value =
            b.load(query_type, query_source);
        b.store(query_slot, query_value);
        b.return_void();
        b.set_insertion_point(original_merge);
        b.unreachable_();
        b.set_insertion_point(first_return);
        b.return_void();
        b.set_insertion_point(second_return);
        b.return_void();
        constexpr auto noise_selection_count =
            size_t{128u};
        b.set_insertion_point(noise_body);
        for (auto i = size_t{0u};
             i < noise_selection_count;
             ++i) {
            auto *noise =
                b.if_(nested_condition);
            auto *noise_true =
                noise->create_true_block();
            auto *noise_false =
                noise->create_false_block();
            auto *noise_merge =
                noise->create_merge_block();
            b.set_insertion_point(noise_true);
            b.br(noise_merge);
            b.set_insertion_point(noise_false);
            b.br(noise_merge);
            b.set_insertion_point(noise_merge);
        }
        b.return_void();
        b.set_insertion_point(root_merge);
        b.unreachable_();

        expect(xir_verify_module(&m).succeeded());
        auto initial_block_count = count_owned_blocks(def);
        auto first = restructure_cfg_pass_run_on_function(
            k, {.main_iteration_limit = 64u,
                .post_iteration_limit = 8u});
        expect(first.succeeded());
        expect(first.iteration_limit_count == 0u);
        expect(first.unstructured_branch_count == 0u);
        expect(count_post_merge_selection_reentries(k) ==
               0u);
        expect(
            first.selection_reentry_boundary_analysis_count >
            0u);
        expect(
            first.selection_reentry_edge_query_count >
            0u);
        // The 128 structured selections in the sibling root arm are
        // reachable but cannot dominate the re-entered block. Owner queries
        // therefore follow only the destination's dominator ancestors and
        // remain independent of that unrelated graph width.
        expect(
            first.selection_reentry_owner_query_count <
            noise_selection_count);
        expect(
            first.selection_reentry_owner_query_count <=
            first.selection_reentry_edge_query_count *
                8u);
        auto query_alloca_count = size_t{0u};
        def->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                query_alloca_count +=
                    instruction->isa<AllocaInst>() &&
                    instruction->type() == query_type;
            });
        // The outer switch sees three non-local targets: two returns and the
        // nested selection's merge. Its generated state-dispatch ladder
        // reaches the last target through an unconditional fallback proxy.
        // The nested arms retain their original paths to that merge, so the
        // outer header still dominates it while the switch's new merge does
        // not. Node splitting must follow the proxy and split the exact edge
        // that crosses this post-merge selection boundary.
        expect(query_alloca_count == 2u);
        expect(count_owned_blocks(def) <=
               initial_block_count * 5u);
        expect(xir_verify_module(
                   &m,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());

        auto block_count = count_owned_blocks(def);
        auto second = restructure_cfg_pass_run_on_function(
            k, {.main_iteration_limit = 64u,
                .post_iteration_limit = 8u});
        expect(second.succeeded());
        expect(second.iteration_limit_count == 0u);
        expect(count_owned_blocks(def) == block_count);
        expect(count_post_merge_selection_reentries(k) ==
               0u);
    };

    "restructure_state_dispatch_transports_path_local_values"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        auto *condition =
            f->create_value_argument(Type::of<bool>());
        auto *true_return = f->create_basic_block();
        auto *false_return = f->create_basic_block();
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *selection = b.if_(condition);
        auto *true_path = selection->create_true_block();
        auto *false_path = selection->create_false_block();
        auto *unreachable_merge = selection->create_merge_block();
        auto *one = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(true_path);
        auto *path_value = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {one, one});
        b.br(true_return);
        b.set_insertion_point(false_path);
        b.br(false_return);
        b.set_insertion_point(unreachable_merge);
        b.unreachable_();
        b.set_insertion_point(true_return);
        auto *path_return = b.return_(path_value);
        b.set_insertion_point(false_return);
        b.return_(m.create_constant_zero(Type::of<int>()));

        expect(xir_verify_module(&m).succeeded());
        auto info = restructure_cfg_pass_run_on_function(f);
        expect(info.succeeded());
        expect(info.unstructured_branch_count == 0u);
        auto spills = audit_reg2mem_spills_on_function(f);
        expect(spills.remaining_phi_spill_count == 0u);
        expect(spills.remaining_cross_block_spill_count == 1u);
        expect(path_return->return_value() != path_value);
        expect(path_return->return_value()->isa<LoadInst>());
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());

        auto promoted = mem2reg_pass_run_on_function(f);
        expect(promoted.promoted_alloca_count >= 1u);
        expect(audit_reg2mem_spills_on_function(f).succeeded());
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_separates_structured_loop_prepare_role"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition =
            k->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *old_prepare = loop->create_prepare_block();
        auto *old_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(old_prepare);
        auto *guard = b.if_(condition);
        guard->set_true_target(old_body);
        auto *exit_arm = guard->create_false_block();
        guard->set_merge_block(exit_arm);
        b.set_insertion_point(exit_arm);
        b.break_(merge);
        b.set_insertion_point(old_body);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(old_prepare);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        expect(count_non_canonical_loop_prepare(
                   k->definition()) == 1u);
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(count_non_canonical_loop_prepare(
                   k->definition()) == 0u);
        auto *new_prepare = loop->prepare_block();
        expect(new_prepare != old_prepare);
        expect(loop->body_block() == old_prepare);
        expect(old_prepare->terminator() == guard);
        expect(new_prepare->terminator()->isa<BranchInst>());
        if (new_prepare->terminator()->isa<BranchInst>()) {
            expect(static_cast<BranchInst *>(
                       new_prepare->terminator())
                       ->target_block() == old_prepare);
        }
        auto *canonical_update = loop->update_block();
        expect(canonical_update->terminator()->isa<BranchInst>());
        if (canonical_update->terminator()->isa<BranchInst>()) {
            expect(static_cast<BranchInst *>(
                       canonical_update->terminator())
                       ->target_block() == new_prepare);
        }
        expect(xir_verify_module(
                   &m,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets =
                        true})
                   .succeeded());
    };

    "restructure_preserves_nontrivial_loop_update_exit"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition =
            k->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *old_update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        b.continue_(old_update);

        // This is the boundary shape produced when an update-region state
        // dispatch decides between another iteration and an early exit.
        b.set_insertion_point(old_update);
        auto *guard = b.if_(condition);
        auto *continue_arm = guard->create_true_block();
        auto *break_arm = guard->create_false_block();
        guard->set_merge_block(break_arm);
        b.set_insertion_point(continue_arm);
        b.continue_(old_update);
        b.set_insertion_point(break_arm);
        b.break_(loop_merge);
        b.set_insertion_point(loop_merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);

        auto *new_update = loop->update_block();
        expect(new_update != old_update);
        expect(new_update->terminator()->isa<BranchInst>());
        if (new_update->terminator()->isa<BranchInst>()) {
            expect(static_cast<BranchInst *>(
                       new_update->terminator())
                       ->target_block() == prepare);
        }

        // Entry into the old update remains executable, while completion
        // advances through the new canonical trampoline.
        expect(loop_body->terminator()->isa<BranchInst>());
        if (loop_body->terminator()->isa<BranchInst>()) {
            expect(static_cast<BranchInst *>(
                       loop_body->terminator())
                       ->target_block() == old_update);
        }
        expect(continue_arm->terminator()->isa<ContinueInst>());
        if (continue_arm->terminator()->isa<ContinueInst>()) {
            expect(static_cast<ContinueInst *>(
                       continue_arm->terminator())
                       ->target_block() == new_update);
        }
        expect(break_arm->terminator()->isa<BreakInst>());
        if (break_arm->terminator()->isa<BreakInst>()) {
            expect(static_cast<BreakInst *>(
                       break_arm->terminator())
                       ->target_block() == loop_merge);
        }
        expect(old_update->terminator() == guard);
        expect(xir_verify_module(
                   &m,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets =
                        true})
                   .succeeded());
    };

    "restructure_routes_nested_switch_exits_through_merges"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *selector = m.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *outer = b.switch_(selector);
        auto *outer_default = outer->create_default_block();
        auto *outer_case = outer->create_case_block(1);
        auto *outer_merge = outer->create_merge_block();
        auto *ret = def->create_basic_block();

        b.set_insertion_point(outer_default);
        auto *inner = b.switch_(selector);
        auto *inner_default = inner->create_default_block();
        auto *inner_case = inner->create_case_block(1);
        auto *inner_merge = inner->create_merge_block();

        b.set_insertion_point(inner_default);
        b.br(ret);
        b.set_insertion_point(inner_case);
        b.br(ret);
        b.set_insertion_point(inner_merge);
        b.unreachable_();

        b.set_insertion_point(outer_case);
        b.br(ret);
        b.set_insertion_point(outer_merge);
        b.unreachable_();
        b.set_insertion_point(ret);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.irreducible_region_count == 0u);
        expect(inner->merge_block() != inner_merge);
        expect(outer->merge_block() != outer_merge);
        expect(static_cast<BranchInst *>(inner_default->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner_case->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner->merge_block()->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer_case->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer->merge_block()->terminator())->target_block() == ret);
    };

    "restructure_routes_nested_if_exits_through_merges"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *outer_then = outer->create_true_block();
        auto *outer_else = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *ret = def->create_basic_block();

        b.set_insertion_point(outer_then);
        auto *inner = b.if_(cond);
        auto *inner_then = inner->create_true_block();
        auto *inner_else = inner->create_false_block();
        auto *inner_merge = inner->create_merge_block();

        b.set_insertion_point(inner_then);
        b.br(inner_merge);
        b.set_insertion_point(inner_else);
        b.br(outer_merge);
        b.set_insertion_point(inner_merge);
        b.br(outer_merge);

        b.set_insertion_point(outer_else);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.br(ret);
        b.set_insertion_point(ret);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.iteration_limit_count == 0u);
        expect(info.irreducible_region_count == 0u);
        expect(inner->merge_block() != inner_merge);
        expect(outer->merge_block() != outer_merge);
        expect(static_cast<BranchInst *>(inner_then->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner_else->terminator())->target_block() == inner->merge_block());
        expect(static_cast<BranchInst *>(inner->merge_block()->terminator())->target_block() == outer->merge_block());
        expect(static_cast<BranchInst *>(outer_else->terminator())->target_block() == outer->merge_block());
        expect(branch_chain_reaches(outer->merge_block(), ret));
    };

    "restructure_drains_more_than_64_independent_selection_exits"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *ret = def->create_basic_block();
        struct Site {
            IfInst *selection;
            BasicBlock *exit_arm;
            BasicBlock *original_merge;
        };
        luisa::vector<Site> sites;
        sites.reserve(65u);
        XIRBuilder b;
        auto *cursor = body;
        for (size_t i = 0u; i < 65u; i++) {
            b.set_insertion_point(cursor);
            auto *selection = b.if_(condition);
            auto *next = selection->create_true_block();
            auto *exit_arm = selection->create_false_block();
            auto *original_merge = selection->create_merge_block();
            b.set_insertion_point(exit_arm);
            b.br(ret);
            b.set_insertion_point(original_merge);
            b.unreachable_();
            sites.emplace_back(Site{selection, exit_arm, original_merge});
            cursor = next;
        }
        b.set_insertion_point(cursor);
        b.br(ret);
        b.set_insertion_point(ret);
        b.return_void();

        auto first = restructure_cfg_pass_run_on_function(k);
        expect(first.succeeded());
        expect(first.iteration_limit_count == 0u);
        expect(first.unstructured_branch_count == 0u);
        luisa::vector<BasicBlock *> rewritten_merges;
        rewritten_merges.reserve(sites.size());
        for (auto site : sites) {
            auto *merge = site.selection->merge_block();
            rewritten_merges.emplace_back(merge);
            expect(merge != site.original_merge);
            expect(site.exit_arm->terminator()->isa<BranchInst>());
            if (site.exit_arm->terminator()->isa<BranchInst>()) {
                expect(static_cast<BranchInst *>(site.exit_arm->terminator())->target_block() == merge);
            }
            expect(branch_chain_reaches(merge, ret));
        }
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());

        auto count_blocks = [&]() noexcept {
            size_t count = 0u;
            for ([[maybe_unused]] auto *block : def->basic_blocks()) { count++; }
            return count;
        };
        auto block_count = count_blocks();
        auto second = restructure_cfg_pass_run_on_function(k);
        expect(second.succeeded());
        expect(second.iteration_limit_count == 0u);
        expect(count_blocks() == block_count);
        for (size_t i = 0u; i < sites.size(); i++) {
            expect(sites[i].selection->merge_block() == rewritten_merges[i]);
        }
    };

    "restructure_structurizes_raw_branch_inside_structured_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *inner_tail = k->create_basic_block();
        b.set_insertion_point(outer_true);
        b.cond_br(condition, outer_merge, inner_tail);
        b.set_insertion_point(inner_tail);
        b.br(outer_merge);
        b.set_insertion_point(outer_false);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.unstructured_branch_count == 0u);
        expect(info.invalid_construct_count == 0u);
        expect(info.iteration_limit_count == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(xir_verify_module(
                   &m, {.require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_structurizes_one_sided_branch_before_nested_selection"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(condition);
        auto *raw_header = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *nested_header = k->create_basic_block();

        b.set_insertion_point(raw_header);
        b.cond_br(condition, nested_header, outer_merge);
        b.set_insertion_point(nested_header);
        auto *nested = b.if_(condition);
        auto *nested_true = nested->create_true_block();
        auto *nested_false = nested->create_false_block();
        auto *nested_merge = nested->create_merge_block();
        b.set_insertion_point(nested_true);
        b.br(nested_merge);
        b.set_insertion_point(nested_false);
        b.br(nested_merge);
        b.set_insertion_point(nested_merge);
        b.br(outer_merge);
        b.set_insertion_point(outer_false);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        auto verification = xir_verify_module(
            &m, {.require_unique_merge_blocks = true});
        expect(info.succeeded());
        expect(info.restructured_if_count >= 1u);
        expect(raw_header->terminator()->isa<IfInst>());
        auto *structured = static_cast<IfInst *>(raw_header->terminator());
        expect(structured->merge_block() != outer_merge);
        expect(nested_header->terminator() == nested);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        expect(verification.succeeded());
    };

    "restructure_does_not_reenter_selection_after_its_merge"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *loop_condition =
            k->create_value_argument(Type::of<bool>());
        auto *outer_condition =
            k->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            k->create_value_argument(Type::of<bool>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *outer_header = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        auto *outer_true = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *shared_continue = def->create_basic_block();

        b.set_insertion_point(prepare);
        b.cond_br(
            loop_condition, outer_header, loop_merge);
        b.set_insertion_point(outer_header);
        b.cond_br(
            outer_condition, outer_true, inner_header);
        b.set_insertion_point(outer_true);
        b.br(shared_continue);
        b.set_insertion_point(inner_header);
        auto *inner = b.if_(inner_condition);
        auto *inner_true = inner->create_true_block();
        auto *inner_false = inner->create_false_block();
        auto *inner_merge = inner->create_merge_block();
        b.set_insertion_point(inner_true);
        b.br(inner_merge);
        b.set_insertion_point(inner_false);
        b.br(inner_merge);
        b.set_insertion_point(inner_merge);
        b.br(shared_continue);
        b.set_insertion_point(shared_continue);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        b.return_void();

        auto info =
            restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(outer_header->terminator()->isa<IfInst>());
        // The selection's real convergence is immediately before the
        // enclosing loop continue. A private merge chosen in front of the
        // nested false-arm selection must not let that post-merge path jump
        // back into a block also reached by the true arm.
        expect(count_post_merge_selection_reentries(k) ==
               0u);
        expect(xir_verify_module(
                   &m,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets =
                        true})
                   .succeeded());
    };

    "restructure_structurizes_loop_early_exit_ladder"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *first_guard = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *first_break = def->create_basic_block();
        auto *second_guard = def->create_basic_block();
        auto *second_break = def->create_basic_block();
        auto *third_guard = def->create_basic_block();
        auto *third_break = def->create_basic_block();
        auto *continue_block = def->create_basic_block();

        b.set_insertion_point(prepare);
        b.cond_br(condition, first_guard, merge);
        b.set_insertion_point(first_guard);
        b.cond_br(condition, first_break, second_guard);
        b.set_insertion_point(first_break);
        b.break_(merge);
        b.set_insertion_point(second_guard);
        b.cond_br(condition, second_break, third_guard);
        b.set_insertion_point(second_break);
        b.break_(merge);
        b.set_insertion_point(third_guard);
        b.cond_br(condition, third_break, continue_block);
        b.set_insertion_point(third_break);
        b.break_(merge);
        b.set_insertion_point(continue_block);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        auto verification = xir_verify_module(
            &m, {.require_unique_merge_blocks = true});
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(first_guard->terminator()->isa<IfInst>());
        expect(second_guard->terminator()->isa<IfInst>());
        expect(third_guard->terminator()->isa<IfInst>());
        expect(count_terminator_kind(def, DerivedInstructionTag::CONDITIONAL_BRANCH) == 1u);
        expect(count_non_canonical_loop_prepare(def) == 0u);
        expect(verification.succeeded());
    };

    "restructure_rebuilds_switch_from_indexed_branch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *selector = k->create_value_argument(Type::of<uint32_t>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sw = b.switch_(selector);
        auto *case_block = sw->create_case_block(1);
        auto *default_block = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto destructure_info = destructure_cfg_pass_run_on_module(&m);
        auto if_conversion_info = if_conversion_pass_run_on_module(&m);
        expect(destructure_info.succeeded());
        expect(destructure_info.destructured_switch_count == 1u);
        expect(body->terminator()->isa<IndexedBranchInst>());
        expect(if_conversion_info.succeeded());
        auto restructure_info = restructure_cfg_pass_run_on_module(&m);
        expect(restructure_info.succeeded());
        expect(restructure_info.restructured_switch_count == 1u);
        expect(count_terminator_kind(
                   k->definition(), DerivedInstructionTag::INDEXED_BRANCH) ==
               0u);
        expect(count_terminator_kind(
                   k->definition(), DerivedInstructionTag::SWITCH) == 1u);
        expect(body->terminator()->isa<SwitchInst>());
        auto *rebuilt = static_cast<SwitchInst *>(body->terminator());
        expect(rebuilt->value() == selector);
        expect(rebuilt->case_count() == 1u);
        expect(rebuilt->case_value(0u) == 1u);
        expect(rebuilt->case_block(0u) == case_block);
        expect(rebuilt->default_block() == default_block);
        expect(rebuilt->merge_block() != nullptr);
        expect(xir_verify_module(
                   &m, {.require_no_unstructured_control_flow = true,
                        .require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_switch_roundtrip_preserves_selector_widths_aliases_and_adjacent_passes"_test = [] {
        struct SelectorCase {
            const Type *type;
            std::array<uint64_t, 2u> literals;
        };
        const std::array selector_cases{
            SelectorCase{Type::of<bool>(), {0u, 1u}},
            SelectorCase{Type::of<int8_t>(), {0xffu, 0x80u}},
            SelectorCase{Type::of<uint8_t>(), {0xffu, 0x80u}},
            SelectorCase{Type::of<int16_t>(), {0xffffu, 0x8000u}},
            SelectorCase{Type::of<uint16_t>(), {0xffffu, 0x8000u}},
            SelectorCase{
                Type::of<int32_t>(), {0xffffffffu, 0x80000000u}},
            SelectorCase{
                Type::of<uint32_t>(), {0xffffffffu, 0x80000000u}},
            SelectorCase{
                Type::of<int64_t>(),
                {std::numeric_limits<uint64_t>::max(),
                 uint64_t{1u} << 63u}},
            SelectorCase{
                Type::of<uint64_t>(),
                {std::numeric_limits<uint64_t>::max(),
                 uint64_t{1u} << 63u}},
        };

        for (auto &&selector_case : selector_cases) {
            Module module;
            BasicBlock *body;
            auto *kernel = make_kernel_with_body(module, body);
            auto *selector =
                kernel->create_value_argument(selector_case.type);
            auto *shared_default_and_case =
                kernel->create_basic_block();
            auto *second_case = kernel->create_basic_block();
            auto *exit = kernel->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *indexed = builder.indexed_branch(selector);
            indexed->set_default_block(shared_default_and_case);
            indexed->add_case(
                selector_case.literals[0u],
                shared_default_and_case);
            indexed->add_case(
                selector_case.literals[1u], second_case);
            builder.set_insertion_point(shared_default_and_case);
            builder.br(exit);
            builder.set_insertion_point(second_case);
            builder.br(exit);
            builder.set_insertion_point(exit);
            builder.return_void();

            expect(xir_verify_module(&module).succeeded());
            auto check_structured_switch = [&]() noexcept {
                expect(body->terminator()->isa<SwitchInst>());
                if (!body->terminator()->isa<SwitchInst>()) { return; }
                auto *switch_inst =
                    static_cast<SwitchInst *>(body->terminator());
                expect(switch_inst->value() == selector);
                expect(switch_inst->case_count() == 2u);
                expect(switch_inst->case_value(0u) ==
                       selector_case.literals[0u]);
                expect(switch_inst->case_value(1u) ==
                       selector_case.literals[1u]);
                expect(branch_chain_reaches(
                    switch_inst->default_block(), exit));
                expect(branch_chain_reaches(
                    switch_inst->case_block(0u), exit));
                expect(branch_chain_reaches(
                    switch_inst->case_block(1u), exit));
                expect(xir_verify_module(
                           &module,
                           {.require_no_unstructured_control_flow = true,
                            .require_unique_merge_blocks = true})
                           .succeeded());
            };

            auto first =
                restructure_cfg_pass_run_on_function(kernel);
            expect(first.succeeded());
            expect(first.restructured_switch_count == 1u);
            expect(first.iteration_limit_count == 0u);
            check_structured_switch();

            auto destructured =
                destructure_cfg_pass_run_on_function(kernel);
            expect(destructured.succeeded());
            expect(destructured.destructured_switch_count == 1u);
            expect(body->terminator()->isa<IndexedBranchInst>());
            // Adjacent raw-CFG passes are required to preserve a dynamic
            // selector, canonical case literals, and default/case aliases.
            (void)simplify_cfg_pass_run_on_function(kernel);
            (void)sccp_pass_run_on_function(kernel);
            expect(xir_verify_module(&module).succeeded());
            expect(body->terminator()->isa<IndexedBranchInst>());
            if (body->terminator()->isa<IndexedBranchInst>()) {
                auto *raw =
                    static_cast<IndexedBranchInst *>(
                        body->terminator());
                expect(raw->case_count() == 2u);
                expect(raw->case_value(0u) ==
                       selector_case.literals[0u]);
                expect(raw->case_value(1u) ==
                       selector_case.literals[1u]);
            }

            auto second =
                restructure_cfg_pass_run_on_function(kernel);
            expect(second.succeeded());
            expect(second.restructured_switch_count == 1u);
            expect(second.iteration_limit_count == 0u);
            check_structured_switch();
        }
    };

    "restructure_recovers_cyclic_indexed_branch_without_cloning_loop_body"_test = [] {
        struct SelectorCase {
            const Type *type;
            uint64_t back_edge_literal;
            uint64_t exit_literal;
        };
        const std::array selector_cases{
            SelectorCase{Type::of<bool>(), 0u, 1u},
            SelectorCase{Type::of<int8_t>(), 0xffu, 0x80u},
            SelectorCase{Type::of<uint8_t>(), 0xffu, 0x80u},
            SelectorCase{Type::of<int16_t>(), 0xffffu, 0x8000u},
            SelectorCase{Type::of<uint16_t>(), 0xffffu, 0x8000u},
            SelectorCase{
                Type::of<int32_t>(), 0xffffffffu, 0x80000000u},
            SelectorCase{
                Type::of<uint32_t>(), 0xffffffffu, 0x80000000u},
            SelectorCase{
                Type::of<int64_t>(),
                std::numeric_limits<uint64_t>::max(),
                uint64_t{1u} << 63u},
            SelectorCase{
                Type::of<uint64_t>(),
                std::numeric_limits<uint64_t>::max(),
                uint64_t{1u} << 63u},
        };

        auto exercise = [](SelectorCase selector_case,
                           size_t payload_block_count) noexcept {
            Module module;
            BasicBlock *body;
            auto *kernel = make_kernel_with_body(module, body);
            auto *selector =
                kernel->create_value_argument(selector_case.type);
            auto *definition = kernel->definition();
            auto *header = definition->create_basic_block();
            luisa::vector<BasicBlock *> payload;
            payload.reserve(payload_block_count);
            for (auto i = size_t{0u}; i < payload_block_count; ++i) {
                payload.emplace_back(definition->create_basic_block());
            }
            auto *latch = definition->create_basic_block();
            auto *case_exit = definition->create_basic_block();
            auto *default_exit = definition->create_basic_block();

            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.br(header);
            builder.set_insertion_point(header);
            builder.br(payload.empty() ? latch : payload.front());
            for (auto i = size_t{0u}; i < payload.size(); ++i) {
                builder.set_insertion_point(payload[i]);
                builder.br(i + 1u == payload.size() ?
                               latch :
                               payload[i + 1u]);
            }
            builder.set_insertion_point(latch);
            auto *indexed = builder.indexed_branch(selector);
            // The first case is a natural-loop back edge; the second proves
            // that cyclic multi-way branches, rather than a one-case special
            // form, are handled by the same lowering. Boundary bit patterns
            // exercise every verifier-supported selector width and signedness.
            indexed->add_case(
                selector_case.back_edge_literal, header);
            indexed->add_case(
                selector_case.exit_literal, case_exit);
            indexed->set_default_block(default_exit);
            builder.set_insertion_point(case_exit);
            builder.return_void();
            builder.set_insertion_point(default_exit);
            builder.return_void();

            auto block_count_before =
                definition->basic_blocks().count_size();
            expect(xir_verify_module(&module).succeeded());
            auto info =
                restructure_cfg_pass_run_on_function(kernel);
            auto block_count_after =
                definition->basic_blocks().count_size();
            expect(info.succeeded());
            expect(info.restructured_loop_count == 1u);
            expect(info.canonicalized_cfg_count != 0u);
            expect(count_terminator_kind(
                       definition,
                       DerivedInstructionTag::INDEXED_BRANCH) == 0u);
            expect(count_terminator_kind(
                       definition,
                       DerivedInstructionTag::SWITCH) == 0u);
            expect(block_count_after >= block_count_before);
            expect(xir_verify_module(
                       &module,
                       {.require_no_unstructured_control_flow = true,
                        .require_unique_merge_blocks = true})
                       .succeeded());
            return block_count_after - block_count_before;
        };

        for (auto selector_case : selector_cases) {
            auto small_body_growth = exercise(selector_case, 4u);
            auto large_body_growth = exercise(selector_case, 128u);
            expect(small_body_growth == large_body_growth)
                << "structural recovery overhead must be independent of the "
                   "number of blocks in the natural-loop body for selector "
                << selector_case.type->description();
        }
    };

    "restructure_canonicalizes_zero_case_indexed_branch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector =
            kernel->create_value_argument(Type::of<int8_t>());
        auto *definition = kernel->definition();
        auto *default_block = definition->create_basic_block();
        auto *exit = definition->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *indexed_branch = builder.indexed_branch(selector);
        indexed_branch->set_default_block(default_block);
        builder.set_insertion_point(default_block);
        builder.br(exit);
        builder.set_insertion_point(exit);
        builder.return_void();

        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.restructured_switch_count == 0u);
        expect(info.canonicalized_cfg_count != 0u);
        expect(body->terminator()->isa<BranchInst>());
        if (body->terminator()->isa<BranchInst>()) {
            expect(static_cast<BranchInst *>(body->terminator())
                       ->target_block() == default_block);
        }
        expect(xir_verify_module(
                   &m,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_rebuilds_terminal_indexed_branch_without_postdominator"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector =
            kernel->create_value_argument(Type::of<int8_t>());
        auto *definition = kernel->definition();
        auto *case_block = definition->create_basic_block();
        auto *default_block = definition->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *indexed_branch = builder.indexed_branch(selector);
        indexed_branch->add_case(
            std::numeric_limits<uint64_t>::max(), case_block);
        indexed_branch->set_default_block(default_block);
        builder.set_insertion_point(case_block);
        builder.return_void();
        builder.set_insertion_point(default_block);
        builder.return_void();

        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.restructured_switch_count == 1u);
        expect(body->terminator()->isa<SwitchInst>());
        auto *switch_inst =
            static_cast<SwitchInst *>(body->terminator());
        expect(switch_inst->case_count() == 1u);
        expect(switch_inst->case_value(0u) == uint64_t{0xffu});
        expect(switch_inst->case_block(0u) == case_block);
        expect(switch_inst->default_block() == default_block);
        expect(switch_inst->merge_block()->terminator()->isa<UnreachableInst>());
        expect(xir_verify_module(
                   &m,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_keeps_dominated_terminal_arm_inside_enclosing_selection"_test = [] {
        Module module;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(module, body);
        auto *outer_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *definition = kernel->definition();
        XIRBuilder builder;

        builder.set_insertion_point(body);
        auto *outer = builder.if_(outer_condition);
        auto *raw_switch_header = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();

        auto *case_block = definition->create_basic_block();
        auto *default_unreachable =
            definition->create_basic_block();
        builder.set_insertion_point(raw_switch_header);
        auto *indexed = builder.indexed_branch(selector);
        indexed->add_case(1u, case_block);
        indexed->set_default_block(default_unreachable);
        builder.set_insertion_point(case_block);
        builder.br(outer_merge);
        builder.set_insertion_point(default_unreachable);
        builder.unreachable_(
            "the inlined callable has no implementation for this tag");
        builder.set_insertion_point(outer_false);
        builder.br(outer_merge);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.iteration_limit_count == 0u);
        expect(outer->merge_block() == outer_merge)
            << "a header-dominated terminal block has no outgoing edge from "
               "the enclosing selection and is not a non-local exit";
        expect(default_unreachable->terminator()->isa<UnreachableInst>());
        expect(xir_verify_module(
                   &module,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_rebuilds_terminal_indexed_branch_with_aliased_cases"_test = [] {
        Module module;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(module, body);
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *definition = kernel->definition();
        auto *shared_return = definition->create_basic_block();
        auto *default_unreachable = definition->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(body);
        auto *indexed_branch = builder.indexed_branch(selector);
        indexed_branch->set_default_block(default_unreachable);
        for (auto case_value = uint64_t{0u};
             case_value < 5u; ++case_value) {
            indexed_branch->add_case(
                case_value, shared_return);
        }
        builder.set_insertion_point(shared_return);
        builder.return_void();
        builder.set_insertion_point(default_unreachable);
        builder.unreachable_();

        expect(xir_verify_module(&module).succeeded());
        auto info =
            restructure_cfg_pass_run_on_function(kernel);
        expect(info.succeeded());
        expect(info.restructured_switch_count == 1u);
        expect(info.iteration_limit_count == 0u);
        expect(count_post_merge_selection_reentries(kernel) ==
               0u);
        expect(body->terminator()->isa<SwitchInst>());
        auto *switch_inst =
            static_cast<SwitchInst *>(body->terminator());
        luisa::unordered_set<BasicBlock *> arm_entries;
        arm_entries.emplace(switch_inst->default_block());
        for (auto i = size_t{0u};
             i < switch_inst->case_count(); ++i) {
            auto *case_entry = switch_inst->case_block(i);
            expect(arm_entries.emplace(case_entry).second);
            expect(branch_chain_reaches(
                case_entry, shared_return));
        }
        expect(xir_verify_module(
                   &module,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());

        auto block_count = count_owned_blocks(definition);
        auto second =
            restructure_cfg_pass_run_on_function(kernel);
        expect(second.succeeded());
        expect(!second.changed());
        expect(count_owned_blocks(definition) == block_count);
    };

    "restructure_roundtrips_loop_switch_nested_break_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *selector =
            k->create_value_argument(Type::of<uint32_t>());
        auto *a = k->create_value_argument(Type::of<bool>());
        auto *b_condition =
            k->create_value_argument(Type::of<bool>());
        auto *c = k->create_value_argument(Type::of<bool>());
        auto *d = k->create_value_argument(Type::of<bool>());
        XIRBuilder builder;

        builder.set_insertion_point(body);
        auto *loop = builder.simple_loop();
        auto *loop_body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();

        builder.set_insertion_point(loop_body);
        auto *outer = builder.if_(a);
        auto *outer_true = outer->create_true_block();
        auto *outer_merge = outer->create_merge_block();
        outer->set_false_target(outer_merge);

        builder.set_insertion_point(outer_true);
        auto *break_guard = builder.if_(b_condition);
        auto *break_arm = break_guard->create_true_block();
        auto *continue_guard =
            break_guard->create_false_block();
        auto *break_guard_merge =
            break_guard->create_merge_block();

        builder.set_insertion_point(break_arm);
        builder.break_(loop_merge);
        builder.set_insertion_point(continue_guard);
        auto *nested_continue = builder.if_(c);
        auto *continue_arm =
            nested_continue->create_true_block();
        auto *continue_fallthrough =
            nested_continue->create_false_block();
        auto *continue_merge =
            nested_continue->create_merge_block();
        builder.set_insertion_point(continue_arm);
        builder.continue_(loop_body);
        builder.set_insertion_point(continue_fallthrough);
        builder.br(continue_merge);
        builder.set_insertion_point(continue_merge);
        builder.br(break_guard_merge);
        builder.set_insertion_point(break_guard_merge);
        builder.br(outer_merge);

        builder.set_insertion_point(outer_merge);
        auto *switch_inst = builder.switch_(selector);
        auto *case_zero = switch_inst->create_case_block(0u);
        auto *case_one = switch_inst->create_case_block(1u);
        auto *default_block =
            switch_inst->create_default_block();
        auto *switch_merge = switch_inst->create_merge_block();
        builder.set_insertion_point(case_zero);
        builder.br(switch_merge);
        builder.set_insertion_point(case_one);
        auto *case_guard = builder.if_(d);
        auto *case_continue = case_guard->create_true_block();
        auto *case_fallthrough =
            case_guard->create_false_block();
        auto *case_merge = case_guard->create_merge_block();
        builder.set_insertion_point(case_continue);
        builder.continue_(loop_body);
        builder.set_insertion_point(case_fallthrough);
        builder.br(case_merge);
        builder.set_insertion_point(case_merge);
        builder.br(switch_merge);
        builder.set_insertion_point(default_block);
        builder.br(switch_merge);
        builder.set_insertion_point(switch_merge);
        builder.break_(loop_merge);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        auto destructured =
            destructure_cfg_pass_run_on_function(k);
        expect(destructured.succeeded());
        expect(destructured.destructured_switch_count == 1u);
        expect(count_terminator_kind(
                   k->definition(),
                   DerivedInstructionTag::INDEXED_BRANCH) == 1u);

        auto restructured =
            restructure_cfg_pass_run_on_function(k);
        expect(restructured.succeeded());
        expect(restructured.iteration_limit_count == 0u);
        expect(restructured.restructured_switch_count == 1u);
        expect(count_terminator_kind(
                   k->definition(),
                   DerivedInstructionTag::INDEXED_BRANCH) == 0u);
        expect(count_terminator_kind(
                   k->definition(),
                   DerivedInstructionTag::SWITCH) == 1u);
        expect(count_terminator_kind(
                   k->definition(),
                   DerivedInstructionTag::CONDITIONAL_BRANCH) == 0u);
        auto block_count = size_t{0u};
        k->definition()->traverse_basic_blocks(
            [&](BasicBlock *) noexcept { ++block_count; });
        expect(block_count < 64u)
            << "restructuring must keep recovered selection subgraphs "
               "linear in size (actual block count: "
            << block_count << ")";
        auto rerun =
            restructure_cfg_pass_run_on_function(k);
        expect(rerun.succeeded());
        auto rerun_block_count = size_t{0u};
        k->definition()->traverse_basic_blocks(
            [&](BasicBlock *) noexcept {
                ++rerun_block_count;
            });
        expect(rerun_block_count == block_count)
            << "restructure_cfg must be idempotent after rebuilding a "
               "nested Switch (before: "
            << block_count << ", after: "
            << rerun_block_count << ")";
        expect(xir_verify_module(
                   &m,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets =
                        true})
                   .succeeded());
    };

    "restructure_reports_unhandled_raw_conditional"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *return_block = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *branch = b.cond_br(condition, return_block, nullptr);
        b.set_insertion_point(return_block);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.unstructured_branch_count == 1u);
        expect(info.invalid_construct_count >= 1u);
        expect(body->terminator() == branch);
    };

    "restructure_rebuilds_indexed_branch_in_unreachable_owned_shell"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *selector =
            k->create_value_argument(Type::of<uint32_t>());
        auto *dead_header = k->create_basic_block();
        auto *dead_default = k->create_basic_block();
        auto *dead_case = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(dead_header);
        auto *indexed = b.indexed_branch(selector);
        indexed->set_default_block(dead_default);
        indexed->add_case(7u, dead_case);
        b.set_insertion_point(dead_default);
        b.return_void();
        b.set_insertion_point(dead_case);
        b.return_void();

        auto info = restructure_cfg_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.restructured_switch_count == 1u);
        expect(dead_header->terminator()->isa<SwitchInst>());
        expect(xir_verify_module(
                   &m,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());
    };

    "restructure_rejects_non_integer_indexed_selector_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *selector =
            kernel->create_value_argument(Type::of<float>());
        auto *default_block = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *indexed = builder.indexed_branch(selector);
        indexed->set_default_block(default_block);
        builder.set_insertion_point(default_block);
        builder.return_void();
        auto block_count = count_owned_blocks(kernel->definition());

        auto info = restructure_cfg_pass_run_on_function(kernel);
        expect(!info.succeeded());
        expect(info.invalid_construct_count == 1u);
        expect(!info.changed());
        expect(count_owned_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == indexed);
        if (body->terminator() == indexed) {
            expect(indexed->value() == selector);
            expect(indexed->default_block() == default_block);
        }
    };

    "restructure_duplicate_narrow_cases_reject_module_atomically"_test = [] {
        Module m;
        BasicBlock *valid_body;
        auto *valid_kernel = make_kernel_with_body(m, valid_body);
        auto *condition =
            valid_kernel->create_value_argument(Type::of<bool>());
        auto *valid_true = valid_kernel->create_basic_block();
        auto *valid_false = valid_kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(valid_body);
        auto *valid_branch =
            builder.cond_br(condition, valid_true, valid_false);
        builder.set_insertion_point(valid_true);
        builder.return_void();
        builder.set_insertion_point(valid_false);
        builder.return_void();

        BasicBlock *invalid_body;
        auto *invalid_kernel = make_kernel_with_body(m, invalid_body);
        auto *selector =
            invalid_kernel->create_value_argument(Type::of<int8_t>());
        auto *case_zero = invalid_kernel->create_basic_block();
        auto *case_wrapped = invalid_kernel->create_basic_block();
        auto *default_block = invalid_kernel->create_basic_block();
        builder.set_insertion_point(invalid_body);
        auto *invalid_indexed = builder.indexed_branch(selector);
        invalid_indexed->add_case(0u, case_zero);
        invalid_indexed->add_case(uint64_t{0x100u}, case_wrapped);
        invalid_indexed->set_default_block(default_block);
        builder.set_insertion_point(case_zero);
        builder.return_void();
        builder.set_insertion_point(case_wrapped);
        builder.return_void();
        builder.set_insertion_point(default_block);
        builder.return_void();

        auto valid_block_count =
            count_owned_blocks(valid_kernel->definition());
        auto invalid_block_count =
            count_owned_blocks(invalid_kernel->definition());
        auto info = restructure_cfg_pass_run_on_module(&m);
        expect(!info.succeeded());
        expect(info.invalid_construct_count == 1u);
        expect(!info.changed());
        expect(count_owned_blocks(valid_kernel->definition()) ==
               valid_block_count);
        expect(count_owned_blocks(invalid_kernel->definition()) ==
               invalid_block_count);
        expect(valid_body->terminator() == valid_branch);
        expect(invalid_body->terminator() == invalid_indexed);
        if (invalid_body->terminator() == invalid_indexed) {
            expect(invalid_indexed->case_value(0u) == 0u);
            expect(invalid_indexed->case_value(1u) == 0u);
        }
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_restructure_cfg();
    return 0;
}
