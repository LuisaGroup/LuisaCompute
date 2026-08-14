// Test for XIR scalar, memory, CFG, and interprocedural transformation passes.
// This test covers successful rewrites, conservative no-op cases, malformed-input
// rejection, and verifier-preserving behavior across the shared pass pipeline.

#include "ut/ut.hpp"
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/alias_analysis.h>
#include <luisa/xir/passes/call_graph.h>
#include <luisa/xir/passes/convergence_region.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/div_rem_pairs.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/early_cse.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/lex_scope_analysis.h>
#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/passes/loop_vectorization.h>
#include <luisa/xir/passes/lower_break_continue.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/outline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/pointer_usage.h>
#include <luisa/xir/passes/post_dom_tree.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/reassociate.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/slp_vectorization.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/uniformity_analysis.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>
#include <luisa/core/stl/unordered_map.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

static KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

static size_t count_reachable_insts(FunctionDefinition *f, DerivedInstructionTag tag) noexcept {
    size_t count = 0u;
    f->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == tag) { count++; }
    });
    return count;
}

static size_t count_reachable_blocks(FunctionDefinition *f) noexcept {
    size_t count = 0u;
    f->traverse_basic_blocks([&](BasicBlock *) noexcept { count++; });
    return count;
}

static StoreInst *find_store_before(Instruction *before, Value *variable, Value *value) noexcept {
    auto *block = before == nullptr ? nullptr : before->parent_block();
    if (block == nullptr) { return nullptr; }
    for (auto *inst : block->instructions()) {
        if (inst == before) { break; }
        if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            if ((variable == nullptr || store->variable() == variable) &&
                (value == nullptr || store->value() == value)) {
                return store;
            }
        }
    }
    return nullptr;
}

void reg_pass_entry_totality() {
    "xir_pass_null_entry_points_are_total"_test = [] {
        expect(cvp_pass_run_on_function(nullptr).replaced_inst_count == 0u);
        expect(cvp_pass_run_on_module(nullptr).replaced_inst_count == 0u);
        expect(fuse_consecutive_buffer_reads_pass_run_on_function(nullptr)
                   .fused_group_count == 0u);
        expect(fuse_consecutive_buffer_reads_pass_run_on_module(nullptr)
                   .fused_group_count == 0u);
        expect(fix_self_referential_pass_run_on_function(nullptr)
                   .fixed_count == 0u);
        expect(fix_self_referential_pass_run_on_module(nullptr)
                   .fixed_count == 0u);
        expect(early_return_elimination_pass_run_on_function(nullptr)
                   .removed_return_count == 0u);
        expect(early_return_elimination_pass_run_on_module(nullptr)
                   .removed_return_count == 0u);
        expect(sroa_pass_run_on_function(nullptr)
                   .decomposed_alloca_count == 0u);
        expect(sroa_pass_run_on_module(nullptr)
                   .decomposed_alloca_count == 0u);
        expect(slp_vectorization_pass_run_on_function(nullptr)
                   .vectorized_tree_count == 0u);
        expect(slp_vectorization_pass_run_on_module(nullptr)
                   .vectorized_tree_count == 0u);
        expect(licm_pass_run_on_function(nullptr).hoisted_count == 0u);
        expect(licm_pass_run_on_module(nullptr).hoisted_count == 0u);
        expect(alias_analysis_pass_run_on_function(nullptr)
                   .queried_count == 0u);
        expect(alias_analysis_pass_run_on_module(nullptr)
                   .queried_count == 0u);
        expect(autodiff_pass_run_on_function(nullptr)
                   .transformed_scope_count == 0u);
        expect(autodiff_pass_run_on_module(nullptr)
                   .transformed_scope_count == 0u);
        expect(lex_scope_analysis_pass_run_on_function(nullptr, {})
                   .lexical_scope_breakers.empty());

        auto dom = compute_dom_tree(nullptr);
        expect(dom.root() == nullptr);
        expect(dom.nodes().empty());
        auto post_dom = compute_post_dom_tree(nullptr);
        expect(post_dom.root() == nullptr);
        expect(post_dom.nodes().empty());

        Module m;
        auto *declaration = m.create_callable(Type::of<int>());
        expect(compute_dom_tree(declaration).root() == nullptr);
        expect(compute_post_dom_tree(declaration).root() == nullptr);
        expect(!autodiff_pass_run_on_function(declaration).changed());
        expect(lex_scope_analysis_pass_run_on_function(declaration, {})
                   .lexical_scope_breakers.empty());
    };

    "xir_pass_null_module_reports_have_stable_schema"_test = [] {
        auto expect_zero_report =
            [](PassReport &report, size_t expected_entry_count) noexcept {
                expect(report.entries().size() == expected_entry_count);
                for (auto &&entry : report.entries()) {
                    expect(entry.value == 0u);
                }
            };
        auto check_zero_report =
            [&](size_t expected_entry_count, auto &&run) noexcept {
                PassReport report;
                run(&report);
                expect_zero_report(report, expected_entry_count);
            };

        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)algebraic_simplify_pass_run_on_module(
                nullptr, {}, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)alias_analysis_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)const_fold_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)cvp_pass_run_on_module(nullptr, report);
        });
        check_zero_report(5u, [](PassReport *report) noexcept {
            (void)dce_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)dead_arg_elim_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)dead_store_elimination_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(10u, [](PassReport *report) noexcept {
            (void)destructure_cfg_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)div_rem_pairs_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)early_cse_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)early_return_elimination_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)fix_self_referential_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(3u, [](PassReport *report) noexcept {
            (void)fuse_consecutive_buffer_reads_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(3u, [](PassReport *report) noexcept {
            (void)gvn_pass_run_on_module(nullptr, report);
        });
        check_zero_report(4u, [](PassReport *report) noexcept {
            (void)if_conversion_pass_run_on_module(nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)indvar_simplify_pass_run_on_module(nullptr, report);
        });
        check_zero_report(32u, [](PassReport *report) noexcept {
            (void)inline_pass_run_on_module(nullptr, report);
        });
        check_zero_report(32u, [](PassReport *report) noexcept {
            (void)inline_all_pass_run_on_module(nullptr, report);
        });
        check_zero_report(32u, [](PassReport *report) noexcept {
            (void)inline_all_pass_run_on_module(
                nullptr, InlineOptions{}, report);
        });
        check_zero_report(32u, [](PassReport *report) noexcept {
            (void)inline_call_sites_pass_run_on_module(
                nullptr, luisa::span<CallInst *const>{},
                InlineOptions{}, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)licm_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)local_load_elimination_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)local_store_forward_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)loop_fusion_pass_run_on_module(nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)loop_rotation_pass_run_on_module(nullptr, report);
        });
        check_zero_report(3u, [](PassReport *report) noexcept {
            (void)loop_vectorization_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)lower_ray_query_loop_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)lower_ray_query_loop_to_loop_pass_run_on_module(
                nullptr, report);
        });
        check_zero_report(4u, [](PassReport *report) noexcept {
            (void)mem2reg_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)phi_cleanup_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)promote_ref_arg_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)reassociate_pass_run_on_module(nullptr, report);
        });
        check_zero_report(3u, [](PassReport *report) noexcept {
            (void)reg2mem_pass_run_on_module(nullptr, report);
        });
        check_zero_report(4u, [](PassReport *report) noexcept {
            (void)audit_reg2mem_spills_on_module(nullptr, report);
        });
        check_zero_report(76u, [](PassReport *report) noexcept {
            (void)restructure_cfg_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)scalarizer_pass_run_on_module(nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)sccp_pass_run_on_module(nullptr, report);
        });
        check_zero_report(7u, [](PassReport *report) noexcept {
            (void)simplify_cfg_pass_run_on_module(nullptr, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)simplify_libcalls_pass_run_on_module(nullptr, report);
        });
        check_zero_report(3u, [](PassReport *report) noexcept {
            (void)slp_vectorization_pass_run_on_module(nullptr, report);
        });
        check_zero_report(2u, [](PassReport *report) noexcept {
            (void)sroa_pass_run_on_module(nullptr, {}, report);
        });
        check_zero_report(1u, [](PassReport *report) noexcept {
            (void)unused_callable_removal_pass_run_on_module(
                nullptr, report);
        });

        // Analyses distinguish a null module from an empty module: the stable
        // schema reports one invalid function while all work counters remain
        // zero.
        auto expect_null_analysis_report =
            [](PassReport &report, size_t expected_entry_count) noexcept {
                expect(report.entries().size() == expected_entry_count);
                size_t invalid_entries = 0u;
                for (auto &&entry : report.entries()) {
                    if (entry.key == "invalid_function") {
                        expect(entry.value == 1u);
                        ++invalid_entries;
                    } else {
                        expect(entry.value == 0u);
                    }
                }
                expect(invalid_entries == 1u);
            };
        {
            PassReport report;
            auto info =
                pointer_usage_pass_run_on_module(nullptr, &report);
            expect(info.invalid_function_count == 1u);
            expect(!info.succeeded());
            expect_null_analysis_report(report, 6u);
        }
        {
            PassReport report;
            auto info = scev_pass_run_on_module(nullptr, &report);
            expect(info.invalid_function_count == 1u);
            expect(!info.succeeded());
            expect_null_analysis_report(report, 3u);
        }
    };

    "xir_pass_bodyless_callable_entry_points_are_total_noops"_test = [] {
        Module module;
        auto *declaration =
            module.create_callable(Type::of<int>());

        expect(algebraic_simplify_pass_run_on_function(declaration)
                   .simplified_inst_count == 0u);
        expect(const_fold_pass_run_on_function(declaration)
                   .folded_inst_count == 0u);
        expect(!dce_pass_run_on_function(declaration).changed());
        expect(dead_store_elimination_pass_run_on_function(declaration)
                   .eliminated_store_count == 0u);
        expect(early_cse_pass_run_on_function(declaration)
                   .eliminated_inst_count == 0u);
        expect(local_load_elimination_pass_run_on_function(declaration)
                   .removed_load_count == 0u);
        expect(local_store_forward_pass_run_on_function(declaration)
                   .removed_load_count == 0u);
        expect(!trace_gep_pass_run_on_function(declaration).changed());
        expect(!transpose_gep_pass_run_on_function(declaration).changed());
        expect(outline_pass_run_on_function(&module, declaration)
                   .outlined_func_count == 0u);

        // Sibling scalar/SSA/CFG entries share the same declaration-only
        // contract. Running them together catches future helpers that regress
        // to blindly traversing a null body block.
        (void)cvp_pass_run_on_function(declaration);
        (void)gvn_pass_run_on_function(declaration);
        (void)reassociate_pass_run_on_function(declaration);
        (void)div_rem_pairs_pass_run_on_function(declaration);
        (void)fix_self_referential_pass_run_on_function(declaration);
        (void)fuse_consecutive_buffer_reads_pass_run_on_function(
            declaration);
        (void)if_conversion_pass_run_on_function(declaration);
        (void)licm_pass_run_on_function(declaration);
        (void)loop_rotation_pass_run_on_function(declaration);
        (void)loop_fusion_pass_run_on_function(declaration);
        (void)loop_vectorization_pass_run_on_function(declaration);
        (void)lower_break_continue_pass_run_on_function(declaration);
        (void)lower_ray_query_loop_pass_run_on_function(declaration);
        (void)lower_ray_query_loop_to_loop_pass_run_on_function(
            declaration);
        (void)mem2reg_pass_run_on_function(declaration);
        (void)phi_cleanup_pass_run_on_function(declaration);
        (void)reg2mem_pass_run_on_function(declaration);
        (void)scalarizer_pass_run_on_function(declaration);
        (void)sccp_pass_run_on_function(declaration);
        (void)simplify_cfg_pass_run_on_function(declaration);
        (void)simplify_libcalls_pass_run_on_function(declaration);
        (void)slp_vectorization_pass_run_on_function(declaration);
        (void)sroa_pass_run_on_function(declaration);
        (void)autodiff_pass_run_on_function(declaration);
        expect(destructure_cfg_pass_run_on_function(declaration)
                   .succeeded());
        expect(restructure_cfg_pass_run_on_function(declaration)
                   .succeeded());
        (void)dead_arg_elim_pass_run_on_function(declaration);
        (void)early_return_elimination_pass_run_on_function(
            declaration);
        (void)alias_analysis_pass_run_on_function(declaration);
        auto pointer_usage =
            pointer_usage_pass_run_on_function(declaration);
        expect(pointer_usage.succeeded());
        PointerUsageAnalysis pointer_usage_analysis;
        expect(pointer_usage_analysis.analyze(declaration).succeeded());
        expect(pointer_usage_analysis.is_current());
        expect(pointer_usage_analysis.function() == declaration);
        auto scev = scev_pass_run_on_function(declaration);
        expect(scev.succeeded());
        SCEVAnalysis scev_analysis;
        expect(scev_analysis.analyze(declaration).succeeded());
        expect(scev_analysis.is_current());
        expect(scev_analysis.function() == declaration);
        (void)lex_scope_analysis_pass_run_on_function(declaration, {});
        (void)compute_convergence_regions(
            declaration, compute_dom_tree(declaration));
        UniformityAnalysis uniformity;
        uniformity.analyze(declaration);

        // Whole-module overloads must preserve declaration-like callables,
        // including passes that rewrite signatures or remove unused internal
        // definitions.
        (void)promote_ref_arg_pass_run_on_module(&module);
        (void)inline_all_pass_run_on_module(&module);
        (void)unused_callable_removal_pass_run_on_module(&module);
        (void)outline_pass_run_on_module(&module);
        expect(pointer_usage_pass_run_on_module(&module).succeeded());
        expect(scev_pass_run_on_module(&module).succeeded());
        (void)compute_call_graph(&module);
        auto destructure =
            destructure_cfg_pass_run_on_module(&module);
        auto restructure =
            restructure_cfg_pass_run_on_module(&module);
        expect(!destructure.changed());
        expect(destructure.succeeded());
        expect(!restructure.changed());
        expect(restructure.succeeded());
        expect(declaration->body_block() == nullptr);
        expect(declaration->parent_module() == &module);

        // Bodyless kernels are malformed rather than external declarations;
        // retain that distinction for CFG and CFG-dependent analyses.
        Module malformed_module;
        auto *bodyless_kernel = malformed_module.create_kernel();
        expect(!destructure_cfg_pass_run_on_function(bodyless_kernel)
                    .succeeded());
        expect(!restructure_cfg_pass_run_on_function(bodyless_kernel)
                    .succeeded());
        expect(!pointer_usage_pass_run_on_function(bodyless_kernel)
                    .succeeded());
        expect(!scev_pass_run_on_function(bodyless_kernel).succeeded());
    };
}

static bool block_local_defs_precede_uses(BasicBlock *block) noexcept {
    luisa::unordered_set<const Instruction *> seen;
    for (auto *inst : block->instructions()) {
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            auto *operand = inst->operand(i);
            if (operand->isa<Instruction>()) {
                auto *operand_inst = static_cast<const Instruction *>(operand);
                if (operand_inst->parent_block() == block &&
                    seen.find(operand_inst) == seen.end()) {
                    return false;
                }
            }
        }
        seen.emplace(inst);
    }
    return true;
}

// ---- algebraic_simplify: integer identities ----

void reg_algebraic_simplify() {

    "algsimpl_int_add_zero_rhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 7;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_add_zero_lhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 5;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_sub_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 3;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_sub_self"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = 9;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {x, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_one_rhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 4;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_one_lhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 4;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {one, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 99;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_div_one"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 8;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_div_zero_numerator"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 5;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {zero, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_zero_div_dynamic_denominator_is_not_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *denominator = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *div = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_DIV,
            {zero, denominator});
        auto *ret = b.return_(div);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == div);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_zero_div_zero_is_not_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<uint64_t>());
        auto *body = f->create_body_block();
        auto *zero = m.create_constant_zero(Type::of<uint64_t>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *div = b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_DIV,
            {zero, zero});
        auto *ret = b.return_(div);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == div);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_uint64_div_power_of_two_preserves_wide_shift_type"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<uint64_t>());
        auto *value = f->create_value_argument(Type::of<uint64_t>());
        auto *body = f->create_body_block();
        uint64_t divisor_value = uint64_t{1u} << 63u;
        auto *divisor = m.create_constant(
            Type::of<uint64_t>(), &divisor_value);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *div = b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_DIV,
            {value, divisor});
        auto *ret = b.return_(div);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *shift = static_cast<ArithmeticInst *>(
            ret->return_value());
        expect(shift->op() == ArithmeticOp::BINARY_SHIFT_RIGHT);
        expect(shift->operand(1u)->type() == Type::of<uint64_t>());
        expect(static_cast<Constant *>(shift->operand(1u))
                   ->as<uint64_t>() == 63u);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_uint_vector_div_mod_power_of_two_broadcasts_typed_constants"_test = [] {
        auto run = [](ArithmeticOp source_op,
                      ArithmeticOp expected_op,
                      uint32_t expected_value) noexcept {
            Module m;
            auto *type = Type::of<uint4>();
            auto *f = m.create_callable(type);
            auto *value = f->create_value_argument(type);
            auto *body = f->create_body_block();
            uint32_t divisor_data[4] = {8u, 8u, 8u, 8u};
            auto *divisor = m.create_constant(type, divisor_data);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *source = b.call(
                type, source_op, {value, divisor});
            auto *ret = b.return_(source);

            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 1u);
            expect(ret->return_value()->isa<ArithmeticInst>());
            auto *replacement = static_cast<ArithmeticInst *>(
                ret->return_value());
            expect(replacement->op() == expected_op);
            expect(replacement->operand(1u)->type() == type);
            auto *constant = static_cast<Constant *>(
                replacement->operand(1u));
            auto *lanes = static_cast<const uint32_t *>(
                constant->data());
            for (auto lane = 0u; lane < 4u; ++lane) {
                expect(lanes[lane] == expected_value);
            }
            expect(xir_verify_module(&m).succeeded());
        };
        run(ArithmeticOp::BINARY_DIV,
            ArithmeticOp::BINARY_SHIFT_RIGHT, 3u);
        run(ArithmeticOp::BINARY_MOD,
            ArithmeticOp::BINARY_BIT_AND, 7u);
    };

    "algsimpl_nonuniform_vector_divisor_is_not_rewritten"_test = [] {
        Module m;
        auto *type = Type::of<uint4>();
        auto *f = m.create_callable(type);
        auto *value = f->create_value_argument(type);
        auto *body = f->create_body_block();
        uint32_t divisor_data[4] = {2u, 4u, 8u, 16u};
        auto *divisor = m.create_constant(type, divisor_data);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *div = b.call(
            type, ArithmeticOp::BINARY_DIV, {value, divisor});
        auto *ret = b.return_(div);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == div);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_int_bitor_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 42;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_OR, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_bitxor_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 13;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_XOR, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_shift_left_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 7;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_float_add_zero_not_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.5f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_float_sub_positive_zero_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == x);
    };

    "algsimpl_float_sub_negative_zero_not_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float negative_zero_v = -0.0f;
        auto *negative_zero = m.create_constant(Type::of<float>(), &negative_zero_v);
        auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, negative_zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
    };

    "algsimpl_float_vector_sub_signed_zero_distinguished"_test = [] {
        Module m;
        auto type = Type::of<float2>();
        auto *f = m.create_callable(type);
        auto *x = f->create_value_argument(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float negative_zero_data[2] = {-0.0f, 0.0f};
        auto *negative_zero = m.create_constant(type, negative_zero_data);
        auto *sub = b.call(type, ArithmeticOp::BINARY_SUB, {x, negative_zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
    };

    "algsimpl_float_vector_unary_minus_zero_not_simplified"_test = [] {
        Module m;
        auto type = Type::of<float2>();
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_data[2] = {0.0f, -0.0f};
        auto *zero = m.create_constant(type, zero_data);
        auto *neg = b.call(type, ArithmeticOp::UNARY_MINUS, {zero});
        auto *ret = b.return_(neg);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == neg);
    };

    "algsimpl_float_matrix_unary_minus_zero_not_simplified"_test = [] {
        Module m;
        auto type = Type::of<float2x2>();
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_data[4] = {0.0f, -0.0f, 0.0f, 0.0f};
        auto *zero = m.create_constant(type, zero_data);
        auto *neg = b.call(type, ArithmeticOp::UNARY_MINUS, {zero});
        auto *ret = b.return_(neg);

        auto info = algebraic_simplify_pass_run_on_function(f);

        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == neg);
    };

    "algsimpl_float_mul_one_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float one_v = 1.0f, x_v = 3.14f;
        auto *one = m.create_constant(Type::of<float>(), &one_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_float_mul_zero_not_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 2.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_no_simplification"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 3; ++i) {
            BasicBlock *body;
            make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t zero_v = 0, x_v = i + 1;
            auto *zero = m.create_constant(Type::of<int>(), &zero_v);
            auto *x = m.create_constant(Type::of<int>(), &x_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, zero});
            b.return_void();
        }
        auto info = algebraic_simplify_pass_run_on_module(&m);
        expect(info.simplified_inst_count == 3u);
    };

    "algsimpl_select_const_condition"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = m.create_callable(Type::of<int>());
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t f_v = 1, t_v = 2;
        auto *f = m.create_constant(Type::of<int>(), &f_v);
        auto *t = m.create_constant(Type::of<int>(), &t_v);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *select = b.call(Type::of<int>(), ArithmeticOp::SELECT, {f, t, cond});
        auto *ret = b.return_(select);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == t);
    };

    "algsimpl_float_mul_zero_keeps_nan_inf_semantics"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.5f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k, {.enable_fast_math = true});
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_insert_into_aggregate"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto type = Type::vector(Type::of<int>(), 2u);
        int32_t x_v = 1, y_v = 2, z_v = 3;
        uint8_t index_v = 1u;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *y = m.create_constant(Type::of<int>(), &y_v);
        auto *z = m.create_constant(Type::of<int>(), &z_v);
        auto *index = m.create_constant(Type::of<uint8_t>(), &index_v);
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {x, y});
        b.call(type, ArithmeticOp::INSERT, {aggregate, z, index});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_extract_accepts_all_integer_constant_widths"_test = [] {
        auto run = [](const Type *index_type, const void *index_data) {
            Module m;
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *vector_type = Type::vector(Type::of<int>(), 2u);
            int32_t x_value = 11;
            int32_t y_value = 17;
            auto *x = m.create_constant(Type::of<int>(), &x_value);
            auto *y = m.create_constant(Type::of<int>(), &y_value);
            auto *index = m.create_constant(index_type, index_data);
            auto *aggregate = b.call(vector_type, ArithmeticOp::AGGREGATE, {x, y});
            auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {aggregate, index});
            auto *ret = b.return_(extract);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 1u);
            expect(ret->return_value() == y);
        };
        int8_t i8 = 1;
        uint8_t u8 = 1u;
        int16_t i16 = 1;
        uint16_t u16 = 1u;
        int32_t i32 = 1;
        uint32_t u32 = 1u;
        int64_t i64 = 1;
        uint64_t u64 = 1u;
        run(Type::of<int8_t>(), &i8);
        run(Type::of<uint8_t>(), &u8);
        run(Type::of<int16_t>(), &i16);
        run(Type::of<uint16_t>(), &u16);
        run(Type::of<int32_t>(), &i32);
        run(Type::of<uint32_t>(), &u32);
        run(Type::of<int64_t>(), &i64);
        run(Type::of<uint64_t>(), &u64);
    };

    "algsimpl_aggregate_swizzle_accepts_mixed_integer_widths"_test = [] {
        Module m;
        auto *type = Type::vector(Type::of<float>(), 3u);
        auto *f = m.create_callable(type);
        auto *value = f->create_value_argument(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int8_t index0_value = 0;
        uint16_t index1_value = 1u;
        int64_t index2_value = 2;
        auto *index0 = m.create_constant(Type::of<int8_t>(), &index0_value);
        auto *index1 = m.create_constant(Type::of<uint16_t>(), &index1_value);
        auto *index2 = m.create_constant(Type::of<int64_t>(), &index2_value);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index0});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index1});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index2});
        auto *ret = b.return_(b.call(type, ArithmeticOp::AGGREGATE, {x, y, z}));
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == value);
    };

    "algsimpl_insert_chain_accepts_mixed_integer_widths"_test = [] {
        Module m;
        auto *type = Type::vector(Type::of<int>(), 3u);
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_value = 3;
        int32_t y_value = 5;
        int32_t z_value = 7;
        uint8_t index0_value = 0u;
        int16_t index1_value = 1;
        uint64_t index2_value = 2u;
        auto *x = m.create_constant(Type::of<int>(), &x_value);
        auto *y = m.create_constant(Type::of<int>(), &y_value);
        auto *z = m.create_constant(Type::of<int>(), &z_value);
        auto *index0 = m.create_constant(Type::of<uint8_t>(), &index0_value);
        auto *index1 = m.create_constant(Type::of<int16_t>(), &index1_value);
        auto *index2 = m.create_constant(Type::of<uint64_t>(), &index2_value);
        auto *insert0 = b.call(type, ArithmeticOp::INSERT, {m.create_undefined(type), x, index0});
        auto *insert1 = b.call(type, ArithmeticOp::INSERT, {insert0, y, index1});
        auto *insert2 = b.call(type, ArithmeticOp::INSERT, {insert1, z, index2});
        auto *ret = b.return_(insert2);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *aggregate = static_cast<ArithmeticInst *>(ret->return_value());
        expect(aggregate->op() == ArithmeticOp::AGGREGATE);
        expect(aggregate->operand(0u) == x);
        expect(aggregate->operand(1u) == y);
        expect(aggregate->operand(2u) == z);
    };

    "algsimpl_invalid_constant_indices_are_conservative"_test = [] {
        auto run = [](const Type *index_type, const void *index_data) {
            Module m;
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *vector_type = Type::vector(Type::of<int>(), 2u);
            auto *zero = m.create_constant_zero(Type::of<int>());
            auto *one = m.create_constant_one(Type::of<int>());
            auto *index = m.create_constant(index_type, index_data);
            auto *aggregate = b.call(vector_type, ArithmeticOp::AGGREGATE, {zero, one});
            auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {aggregate, index});
            auto *ret = b.return_(extract);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 0u);
            expect(ret->return_value() == extract);
        };
        int8_t negative = -1;
        float noninteger = 0.0f;
        uint64_t out_of_bounds = std::numeric_limits<uint64_t>::max();
        run(Type::of<int8_t>(), &negative);
        run(Type::of<float>(), &noninteger);
        run(Type::of<uint64_t>(), &out_of_bounds);
    };

    "algsimpl_zero_sized_array_insert_is_conservative"_test = [] {
        Module m;
        auto *array_type = Type::array(Type::of<int>(), 0u);
        auto *f = m.create_callable(array_type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto invalid_index =
            std::numeric_limits<uint64_t>::max();
        auto *index = m.create_constant(
            Type::of<uint64_t>(), &invalid_index);
        auto *insert = b.call(
            array_type, ArithmeticOp::INSERT,
            {m.create_undefined(array_type),
             m.create_constant_one(Type::of<int>()), index});
        auto *ret = b.return_(insert);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == insert);
        expect(insert->parent_block() == body);
        expect(insert->operand(2u) == index);
    };

    "algsimpl_identity_extract_aggregate_to_original_vector"_test = [] {
        Module m;
        BasicBlock *body;
        auto type = Type::vector(Type::of<float>(), 3u);
        auto *k = m.create_callable(type);
        auto *v = k->create_value_argument(type);
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_i = 0u, y_i = 1u, z_i = 2u;
        auto *x_index = m.create_constant(Type::of<uint>(), &x_i);
        auto *y_index = m.create_constant(Type::of<uint>(), &y_i);
        auto *z_index = m.create_constant(Type::of<uint>(), &z_i);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, x_index});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, y_index});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, z_index});
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {x, y, z});
        auto *ret = b.return_(aggregate);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == v);
    };

    "algsimpl_extract_aggregate_to_shuffle"_test = [] {
        Module m;
        BasicBlock *body;
        auto type = Type::vector(Type::of<float>(), 3u);
        auto *k = m.create_callable(type);
        auto *v = k->create_value_argument(type);
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_i = 0u, y_i = 1u;
        auto *x_index = m.create_constant(Type::of<uint>(), &x_i);
        auto *y_index = m.create_constant(Type::of<uint>(), &y_i);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, x_index});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, y_index});
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {y, x, x});
        auto *ret = b.return_(aggregate);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() != v);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *shuffle = static_cast<ArithmeticInst *>(ret->return_value());
        expect(shuffle->op() == ArithmeticOp::SHUFFLE);
        expect(shuffle->operand(0) == v);
        expect(shuffle->operand(1) == y_index);
        expect(shuffle->operand(2) == x_index);
        expect(shuffle->operand(3) == x_index);
    };

    "algsimpl_float_add_zero_keeps_signed_zero_semantics"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.5f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k, {.enable_fast_math = true});
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_float_sub_self_requires_fast_math"_test = [] {
        {
            Module m;
            auto *f = m.create_callable(Type::of<float>());
            auto *x = f->create_value_argument(Type::of<float>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, x});
            auto *ret = b.return_(sub);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 0u);
            expect(ret->return_value() == sub);
        }
        {
            Module m;
            auto *f = m.create_callable(Type::of<float>());
            auto *x = f->create_value_argument(Type::of<float>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, x});
            auto *ret = b.return_(sub);
            auto info = algebraic_simplify_pass_run_on_function(f, {.enable_fast_math = true});
            expect(info.simplified_inst_count == 1u);
            expect(ret->return_value()->isa<Constant>());
            expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.0f);
        }
    };

    "algsimpl_float_vector_sub_self_requires_fast_math"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float2>());
        auto *x = f->create_value_argument(Type::of<float2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sub = b.call(Type::of<float2>(), ArithmeticOp::BINARY_SUB, {x, x});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
    };

    "algsimpl_nested_extract_from_aggregate_preserves_path"_test = [] {
        Module m;
        auto *inner_type = Type::array(Type::of<int>(), 2u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        auto *inner = b.call(inner_type, ArithmeticOp::AGGREGATE, {one, two});
        auto *outer = b.call(outer_type, ArithmeticOp::AGGREGATE, {inner, inner});
        auto *index0 = m.create_constant_zero(Type::of<uint>());
        auto *index1 = m.create_constant_one(Type::of<uint>());
        auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {outer, index0, index1});
        auto *ret = b.return_(extract);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == extract);
    };

    "algsimpl_nested_insert_into_aggregate_preserves_path"_test = [] {
        Module m;
        auto *inner_type = Type::array(Type::of<int>(), 2u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *f = m.create_callable(outer_type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *inner = b.call(inner_type, ArithmeticOp::AGGREGATE, {zero, zero});
        auto *outer = b.call(outer_type, ArithmeticOp::AGGREGATE, {inner, inner});
        auto *index0 = m.create_constant_zero(Type::of<uint>());
        auto *index1 = m.create_constant_one(Type::of<uint>());
        auto *insert = b.call(outer_type, ArithmeticOp::INSERT, {outer, zero, index0, index1});
        auto *ret = b.return_(insert);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == insert);
        expect(insert->operand(0u) == outer);
    };

    "algsimpl_extract_skips_disjoint_insert_and_reports_change"_test = [] {
        Module m;
        auto *type = Type::of<int2>();
        auto *f = m.create_callable(Type::of<int>());
        auto *base = f->create_value_argument(type);
        auto *replacement = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *insert = b.call(
            type, ArithmeticOp::INSERT, {base, replacement, one});
        auto *extract = b.call(
            Type::of<int>(), ArithmeticOp::EXTRACT, {insert, zero});
        auto *ret = b.return_(extract);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == extract);
        expect(extract->operand(0u) == base);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_overwritten_insert_reports_change"_test = [] {
        Module m;
        auto *type = Type::of<int2>();
        auto *f = m.create_callable(type);
        auto *base = f->create_value_argument(type);
        auto *first = f->create_value_argument(Type::of<int>());
        auto *second = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *index = m.create_constant_one(Type::of<uint>());
        auto *inner = b.call(
            type, ArithmeticOp::INSERT, {base, first, index});
        auto *outer = b.call(
            type, ArithmeticOp::INSERT, {inner, second, index});
        auto *ret = b.return_(outer);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == outer);
        expect(outer->operand(0u) == base);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_annotated_identity_is_retained"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *add = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, zero});
        add->set_location("algebraic_metadata.cpp", 14);
        auto *ret = b.return_(add);

        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == add);
        expect(add->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "algsimpl_null_inputs_are_noops"_test = [] {
        expect(algebraic_simplify_pass_run_on_function(nullptr)
                   .simplified_inst_count == 0u);
        PassReport report;
        expect(algebraic_simplify_pass_run_on_module(
                   nullptr, {}, &report)
                   .simplified_inst_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- const_fold ----

void reg_const_fold() {

    "constfold_int_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_sub"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_mul"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 6, b_v = 7;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_div"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 20, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_div_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 5, b_v = 0;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_uint_div_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t a_v = 5u, b_v = 0u;
        auto *a = m.create_constant(Type::of<uint>(), &a_v);
        auto *bv = m.create_constant(Type::of<uint>(), &b_v);
        b.call(Type::of<uint>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_mod_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 7, b_v = 0;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_shift_left"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_shift_left_overflow_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 32;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *shift = b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv});
        auto *ret = b.return_(shift);
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == shift);
    };

    "constfold_int_shift_right_overflow_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 8, b_v = 33;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *shift = b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_RIGHT, {a, bv});
        auto *ret = b.return_(shift);
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == shift);
    };

    "constfold_int_unary_minus"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = 5;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x}));
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == -x_v);
    };

    "constfold_int_unary_minus_int_min"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = std::numeric_limits<int32_t>::min();
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x}));
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == x_v);
    };

    "constfold_signed_overflow_wraps_without_ub"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        auto add_case = [&](ArithmeticOp op, int32_t lhs, int32_t rhs) noexcept {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        };
        add_case(ArithmeticOp::BINARY_ADD, std::numeric_limits<int32_t>::max(), 1);
        add_case(ArithmeticOp::BINARY_SUB, std::numeric_limits<int32_t>::min(), 1);
        add_case(ArithmeticOp::BINARY_MUL, std::numeric_limits<int32_t>::max(), 2);
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 3u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::max());
        expect(static_cast<Constant *>(returns[2]->return_value())->as<int32_t>() == -2);
    };

    "constfold_int_min_div_mod_negative_one_not_folded"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_DIV, ArithmeticOp::BINARY_MOD}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = std::numeric_limits<int32_t>::min();
            int32_t rhs = -1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        expect(returns[0]->return_value()->isa<ArithmeticInst>());
        expect(returns[1]->return_value()->isa<ArithmeticInst>());
    };

    "constfold_signed_shifts_are_defined"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_SHIFT_LEFT, ArithmeticOp::BINARY_SHIFT_RIGHT}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = op == ArithmeticOp::BINARY_SHIFT_LEFT ? -1 : -4;
            int32_t rhs = 1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 2u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == -2);
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == -2);
    };

    "constfold_signed_shift_boundaries"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        auto add_case = [&](ArithmeticOp op, int32_t lhs, int32_t rhs) noexcept {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        };
        add_case(ArithmeticOp::BINARY_SHIFT_LEFT, 1, 31);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, std::numeric_limits<int32_t>::min(), 31);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, std::numeric_limits<int32_t>::min(), 0);
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 3u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == -1);
        expect(static_cast<Constant *>(returns[2]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
    };

    "constfold_negative_shift_counts_are_not_folded"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_SHIFT_LEFT, ArithmeticOp::BINARY_SHIFT_RIGHT}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = 1, rhs = -1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        expect(returns[0]->return_value()->isa<ArithmeticInst>());
        expect(returns[1]->return_value()->isa<ArithmeticInst>());
    };

    "constfold_shift_count_uses_its_declared_integer_width"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        auto add_case = [&]<typename T>(ArithmeticOp op, T rhs) noexcept {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = 8;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<T>(), &rhs);
            returns.emplace_back(
                b.return_(b.call(Type::of<int>(), op, {a, bv})));
        };
        // Both values have zero in their low 32 bits. Reading the shift
        // operand as int32_t would therefore miscompile both operations as a
        // shift by zero.
        add_case(ArithmeticOp::BINARY_SHIFT_LEFT, uint64_t{1} << 32u);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, -(int64_t{1} << 32u));

        expect(xir_verify_module(&m).succeeded());
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        for (auto *ret : returns) {
            expect(ret->return_value()->isa<ArithmeticInst>());
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_valid_narrow_unsigned_shift_count_is_folded"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t lhs = 1;
        uint8_t rhs = 3u;
        auto *a = m.create_constant(Type::of<int>(), &lhs);
        auto *bv = m.create_constant(Type::of<uint8_t>(), &rhs);
        auto *ret = b.return_(
            b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv}));

        expect(xir_verify_module(&m).succeeded());
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == 8);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_abs_int_min_wraps_without_ub"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t value = std::numeric_limits<int32_t>::min();
        auto *v = m.create_constant(Type::of<int>(), &value);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::ABS, {v}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == value);
    };

    "constfold_float_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float a_v = 1.5f, b_v = 2.5f;
        auto *a = m.create_constant(Type::of<float>(), &a_v);
        auto *bv = m.create_constant(Type::of<float>(), &b_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_target_dependent_sqrt_remains_in_ir"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<float>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 4.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *sqrt = b.call(Type::of<float>(), ArithmeticOp::SQRT, {x});
        auto *ret = b.return_(sqrt);
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == sqrt);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_target_dependent_transcendental_and_vector_pow_remain_in_ir"_test = [] {
        Module m;
        auto *scalar = m.create_callable(Type::of<float>());
        auto *scalar_body = scalar->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(scalar_body);
        float scalar_value = 0.25f;
        auto *scalar_constant =
            m.create_constant(Type::of<float>(), &scalar_value);
        auto *sin =
            b.call(Type::of<float>(), ArithmeticOp::SIN, {scalar_constant});
        auto *scalar_ret = b.return_(sin);

        auto *vector = m.create_callable(Type::of<float2>());
        auto *vector_body = vector->create_body_block();
        b.set_insertion_point(vector_body);
        float2 base_value{2.0f, 3.0f};
        float2 exponent_value{0.5f, 1.25f};
        auto *base = m.create_constant(Type::of<float2>(), &base_value);
        auto *exponent =
            m.create_constant(Type::of<float2>(), &exponent_value);
        auto *pow =
            b.call(Type::of<float2>(), ArithmeticOp::POW, {base, exponent});
        auto *vector_ret = b.return_(pow);

        expect(xir_verify_module(&m).succeeded());
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        expect(scalar_ret->return_value() == sin);
        expect(vector_ret->return_value() == pow);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_comparison_less"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_non_const_operand_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *undef = m.create_undefined(Type::of<int>());
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, undef});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 4; ++i) {
            BasicBlock *body;
            make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = i + 1, b_v = i + 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_void();
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 4u);
    };

    "constfold_uint_unary_minus"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_v = 3u;
        auto *x = m.create_constant(Type::of<uint>(), &x_v);
        b.call(Type::of<uint>(), ArithmeticOp::UNARY_MINUS, {x});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_bitand"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 0xFF, b_v = 0x0F;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_AND, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_float_clamp"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.5f, lo_v = 0.0f, hi_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_lerp_keeps_backend_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 0.5f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "constfold_float_special_values_remain_target_independent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float4>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float nan_v = std::numeric_limits<float>::quiet_NaN();
        float positive_zero_v = 0.0f;
        float negative_zero_v = -0.0f;
        float one_v = 1.0f;
        auto *nan = m.create_constant(Type::of<float>(), &nan_v);
        auto *positive_zero = m.create_constant(Type::of<float>(), &positive_zero_v);
        auto *negative_zero = m.create_constant(Type::of<float>(), &negative_zero_v);
        auto *one = m.create_constant(Type::of<float>(), &one_v);
        auto *min_zero = b.call(Type::of<float>(), ArithmeticOp::MIN,
                                {positive_zero, negative_zero});
        auto *step_nan = b.call(Type::of<float>(), ArithmeticOp::STEP, {one, nan});
        auto *saturate_zero = b.call(Type::of<float>(), ArithmeticOp::SATURATE,
                                     {negative_zero});
        auto *clamp_zero = b.call(Type::of<float>(), ArithmeticOp::CLAMP,
                                  {negative_zero, positive_zero, one});
        [[maybe_unused]] auto clamp_zero_lock = clamp_zero->lock();
        auto *smooth = b.call(Type::of<float>(), ArithmeticOp::SMOOTHSTEP,
                              {positive_zero, one, one});
        auto *result = b.call(Type::of<float4>(), ArithmeticOp::AGGREGATE,
                              {min_zero, step_nan, saturate_zero, smooth});
        auto *ret = b.return_(result);

        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == result);
        expect(result->operand(0u) == min_zero);
        expect(result->operand(1u) == step_nan);
        expect(result->operand(2u) == saturate_zero);
        expect(result->operand(3u) == smooth);
        expect(clamp_zero->is_linked());
    };

    "constfold_pow_int_preserves_large_exponent_parity"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = -1.0f;
        int32_t exponent_value = 16777217;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<int>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == -1.0f);
    };

    "constfold_pow_int_decodes_signed_narrow_exponent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = 2.0f;
        int8_t exponent_value = -1;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<int8_t>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.5f);
    };

    "constfold_pow_int_decodes_unsigned_64_bit_exponent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = -1.0f;
        uint64_t exponent_value = uint64_t{1} << 32u;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<uint64_t>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 1.0f);
    };

    "constfold_pow_int_decodes_vector_exponents_per_lane"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float2 base_value{2.0f, 2.0f};
        byte2 exponent_value{-1, 3};
        auto *base = m.create_constant(Type::of<float2>(), &base_value);
        auto *exponent = m.create_constant(Type::of<byte2>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float2>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        auto result = static_cast<Constant *>(ret->return_value())->as<float2>();
        expect(result.x == 0.5f);
        expect(result.y == 8.0f);
    };

    "constfold_round_does_not_cross_half_boundary"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto value = std::nextafter(0.5f, 0.0f);
        auto *constant = m.create_constant(Type::of<float>(), &value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::ROUND, {constant}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.0f);
    };

    "constfold_rint_is_host_rounding_mode_independent"_test = [] {
        auto previous_rounding = std::fegetround();
        auto changed_rounding = std::fesetround(FE_UPWARD) == 0;
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float value = 1.25f;
        auto *constant = m.create_constant(Type::of<float>(), &value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::RINT, {constant}));
        auto info = const_fold_pass_run_on_function(f);
        if (previous_rounding != -1) { static_cast<void>(std::fesetround(previous_rounding)); }
        expect(changed_rounding);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 1.0f);
    };

    "constfold_basic_float_arithmetic_is_host_rounding_mode_independent"_test = [] {
        auto previous_rounding = std::fegetround();
        auto changed_rounding = std::fesetround(FE_UPWARD) == 0;
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float one_value = 1.0f;
        float half_ulp_value = std::ldexp(1.0f, -24);
        auto *one = m.create_constant(Type::of<float>(), &one_value);
        auto *half_ulp = m.create_constant(Type::of<float>(), &half_ulp_value);
        auto *ret = b.return_(
            b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                   {one, half_ulp}));
        auto info = const_fold_pass_run_on_function(f);
        auto rounding_after_fold = std::fegetround();
        if (previous_rounding != -1) {
            static_cast<void>(std::fesetround(previous_rounding));
        }
        expect(changed_rounding);
        expect(rounding_after_fold == FE_UPWARD);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() ==
               1.0f);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_subnormal_float_arithmetic_remains_target_independent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto denormal_value = std::numeric_limits<float>::denorm_min();
        auto *denormal =
            m.create_constant(Type::of<float>(), &denormal_value);
        auto *sum = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {denormal, denormal});
        auto *ret = b.return_(sum);

        expect(xir_verify_module(&m).succeeded());
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == sum);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_vector_nan_result_remains_target_independent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto infinity = std::numeric_limits<float>::infinity();
        float2 lhs_value{infinity, 1.0f};
        float2 rhs_value{-infinity, 2.0f};
        auto *lhs = m.create_constant(Type::of<float2>(), &lhs_value);
        auto *rhs = m.create_constant(Type::of<float2>(), &rhs_value);
        auto *sum = b.call(
            Type::of<float2>(), ArithmeticOp::BINARY_ADD, {lhs, rhs});
        auto *ret = b.return_(sum);

        expect(xir_verify_module(&m).succeeded());
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == sum);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_nan_not_equal_is_left_for_target_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<bool>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float nan_value = std::numeric_limits<float>::quiet_NaN();
        float one_value = 1.0f;
        auto *nan = m.create_constant(Type::of<float>(), &nan_value);
        auto *one = m.create_constant(Type::of<float>(), &one_value);
        auto *cmp = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {nan, one});
        auto *ret = b.return_(cmp);
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == cmp);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_vector_nan_comparison_is_left_for_target_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<bool2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto nan = std::numeric_limits<float>::quiet_NaN();
        float2 lhs_value{nan, 1.0f};
        float2 rhs_value{0.0f, nan};
        auto *lhs = m.create_constant(Type::of<float2>(), &lhs_value);
        auto *rhs = m.create_constant(Type::of<float2>(), &rhs_value);
        auto *cmp = b.call(
            Type::of<bool2>(), ArithmeticOp::BINARY_EQUAL, {lhs, rhs});
        auto *ret = b.return_(cmp);
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == cmp);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_annotated_instruction_is_not_replaced_by_pooled_constant"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        sum->add_comment("constant fold metadata owner");
        auto *ret = b.return_(sum);

        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == sum);
        expect(sum->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "constfold_null_inputs_are_noops"_test = [] {
        expect(const_fold_pass_run_on_function(nullptr)
                   .folded_inst_count == 0u);
        PassReport report;
        expect(const_fold_pass_run_on_module(nullptr, &report)
                   .folded_inst_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- dce ----

void reg_dce() {

    "dce_unused_arithmetic_removed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_inst_count == 1u);
    };

    "dce_detaches_dead_chain_before_releasing_definitions"_test = [] {
        Module m;
        auto *f = m.create_callable(nullptr);
        Value *value = f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<uint>());
        constexpr auto chain_length = 64u;
        for (auto i = 0u; i < chain_length; ++i) {
            value = b.call(
                Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                {value, one});
        }
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_inst_count == chain_length);
        // The sparse solver seeds from one scan of the chain plus Return and
        // then follows only newly empty use-lists. It must process every dead
        // instruction exactly once, independent of the chain depth.
        expect(info.dead_code_instruction_scan_count == chain_length + 1u);
        expect(info.dead_code_worklist_pop_count == chain_length);
        expect(xir_verify_module(&m).succeeded());
        expect(body->instructions().front()->isa<ReturnInst>());
    };

    "dce_schedules_repeated_operand_once"_test = [] {
        Module m;
        auto *f = m.create_callable(nullptr);
        auto *argument =
            f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *common = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {argument, one});
        b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
               {common, common});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_inst_count == 2u);
        expect(info.dead_code_instruction_scan_count == 3u);
        expect(info.dead_code_worklist_pop_count == 2u);
        expect(xir_verify_module(&m).succeeded());
        expect(body->instructions().front()->isa<ReturnInst>());
    };

    "dce_detaches_write_only_alloca_graph_before_release"_test = [] {
        Module m;
        auto *f = m.create_callable(nullptr);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *array =
            b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *element = b.gep(Type::of<uint>(), array, {zero});
        b.store(element, one);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_inst_count == 3u);
        expect(xir_verify_module(&m).succeeded());
        expect(body->instructions().front()->isa<ReturnInst>());
    };

    "dce_propagates_from_write_only_alloca_without_rescan"_test = [] {
        Module m;
        auto *f = m.create_callable(nullptr);
        auto *argument =
            f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *array =
            b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *element = b.gep(Type::of<uint>(), array, {zero});
        auto *stored = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {argument, one});
        b.store(element, stored);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_inst_count == 4u);
        // alloca, GEP, add, store, Return are classified once. Removing the
        // store exposes add through its real use-list transition.
        expect(info.dead_code_instruction_scan_count == 5u);
        expect(info.dead_code_worklist_pop_count == 1u);
        expect(xir_verify_module(&m).succeeded());
        expect(body->instructions().front()->isa<ReturnInst>());
    };

    "dce_solves_cascading_write_only_allocas_to_fixed_point"_test = [] {
        Module m;
        auto *f = m.create_callable(nullptr);
        auto *argument =
            f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *source = b.alloca_local(Type::of<uint>());
        b.store(source, argument);
        auto *loaded = b.load(Type::of<uint>(), source);
        auto *sink = b.alloca_local(Type::of<uint>());
        b.store(sink, loaded);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto first = dce_pass_run_on_function(f);
        // Removing sink exposes the load from source; removing that load is
        // the exact event that makes source write-only. Both monotone rules
        // must converge in one invocation without a whole-function rescan.
        expect(first.removed_inst_count == 5u);
        expect(first.dead_code_instruction_scan_count == 6u);
        expect(first.dead_code_worklist_pop_count == 1u);
        expect(xir_verify_module(&m).succeeded());
        expect(body->instructions().front()->isa<ReturnInst>());

        auto second = dce_pass_run_on_function(f);
        expect(!second.changed());
        expect(second.dead_code_instruction_scan_count == 1u);
        expect(second.dead_code_worklist_pop_count == 0u);
    };

    "dce_preserves_unused_volatile_resource_reads"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *buffer = k->create_resource_argument(Type::buffer(Type::of<int>()));
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *index = m.create_constant_zero(Type::of<uint>());
        b.call(Type::of<int>(), ResourceReadOp::BUFFER_READ, {buffer, index});
        b.call(Type::of<int>(), ResourceReadOp::BUFFER_VOLATILE_READ, {buffer, index});
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        size_t ordinary_count = 0u;
        size_t volatile_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<ResourceReadInst>()) { return; }
            auto op = static_cast<ResourceReadInst *>(inst)->op();
            ordinary_count += op == ResourceReadOp::BUFFER_READ ? 1u : 0u;
            volatile_count += op == ResourceReadOp::BUFFER_VOLATILE_READ ? 1u : 0u;
        });
        expect(info.removed_inst_count == 1u);
        expect(ordinary_count == 0u);
        expect(volatile_count == 1u);
    };

    "dce_no_dead_code"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_inst_count == 0u);
    };

    "dce_preserves_live_coroutine_resume_and_removes_dead_token_pair"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *entry = callable->create_body_block();
        auto *dead_suspend = callable->create_basic_block();
        auto *dead_resume = callable->create_basic_block();
        auto *live_suspend = callable->create_basic_block();
        auto *live_resume = callable->create_basic_block();
        constexpr uint32_t dead_token = 17u;
        constexpr uint32_t live_token = 91u;

        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.cond_br(
            module.create_constant_zero(Type::of<bool>()),
            dead_suspend, live_suspend);
        builder.set_insertion_point(dead_suspend);
        builder.coro_suspend(dead_token, "dead", nullptr);
        builder.set_insertion_point(dead_resume);
        builder.coro_resume(dead_token, nullptr);
        builder.br(live_suspend);
        builder.set_insertion_point(live_suspend);
        builder.coro_suspend(live_token, "live", nullptr);
        builder.set_insertion_point(live_resume);
        builder.coro_resume(live_token, nullptr);
        builder.coro_terminate();

        expect(xir_verify_module(&module).succeeded());
        auto info = dce_pass_run_on_function(callable);
        expect(info.removed_block_count == 2u)
            << "DCE must remove the dead suspend/resume component only";
        expect(xir_verify_module(&module).succeeded());
        auto suspend_count = size_t{0u};
        auto resume_count = size_t{0u};
        // Resume blocks are semantic successors of suspend tokens, not
        // ordinary CFG successors, so inspect the owned block set here.
        for (auto *block : callable->basic_blocks()) {
            for (auto *instruction : block->instructions()) {
                if (instruction->isa<CoroSuspendInst>()) {
                    ++suspend_count;
                    expect(static_cast<CoroSuspendInst *>(instruction)->token() ==
                           live_token);
                } else if (instruction->isa<CoroResumeInst>()) {
                    ++resume_count;
                    expect(static_cast<CoroResumeInst *>(instruction)->token() ==
                           live_token);
                }
            }
        }
        expect(suspend_count == 1u);
        expect(resume_count == 1u);
    };

    "dce_reports_inserted_terminator_and_is_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        expect(!body->is_terminated());

        auto first = dce_pass_run_on_function(k);
        expect(first.changed());
        expect(first.inserted_terminator_count == 1u);
        expect(first.removed_inst_count == 0u);
        expect(first.removed_block_count == 0u);
        expect(body->is_terminated());
        expect(body->terminator()->isa<UnreachableInst>());

        auto second = dce_pass_run_on_function(k);
        expect(!second.changed());
        expect(second.inserted_terminator_count == 0u);
        expect(body->terminator()->isa<UnreachableInst>());
    };

    "dce_null_entry_points_are_noops"_test = [] {
        auto function_info = dce_pass_run_on_function(nullptr);
        expect(!function_info.changed());
        expect(function_info.inserted_terminator_count == 0u);
        PassReport report;
        auto module_info =
            dce_pass_run_on_module(nullptr, &report);
        expect(!module_info.changed());
        expect(report.entries().size() == 5u);
    };

    "dce_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_void();
        }
        auto info = dce_pass_run_on_module(&m);
        expect(info.removed_inst_count == 2u);
    };

    "dce_repairs_phi_after_unreachable_predecessor_removed"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        auto *live = f->create_basic_block();
        auto *dead = f->create_basic_block();
        auto *merge = f->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(live);

        b.set_insertion_point(live);
        int32_t live_v = 1;
        auto *live_c = m.create_constant(Type::of<int>(), &live_v);
        auto *live_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {live_c, live_c});
        b.br(merge);

        b.set_insertion_point(dead);
        int32_t dead_v = 2;
        auto *dead_c = m.create_constant(Type::of<int>(), &dead_v);
        auto *dead_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {dead_c, dead_c});
        b.br(merge);

        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(live_value, live);
        phi->add_incoming(dead_value, dead);
        b.return_(phi);

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_block_count >= 1u);
        expect(phi->incoming_count() == 1u);
        expect(phi->incoming(0u).block == live);
        expect(phi->incoming(0u).value == live_value);
    };

    "dce_zero_incoming_entry_phi_becomes_undef"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *phi = b.phi(Type::of<int>());
        auto *ret = b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info = dce_pass_run_on_function(f);
        expect(info.removed_inst_count == 1u);
        expect(ret->return_value() != nullptr);
        expect(ret->return_value()->isa<Undefined>());
        expect(xir_verify_module(&m).succeeded());
    };

    "dce_unreachable_block_cleanup_is_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *dead = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(dead);
        b.unreachable_();
        auto first = dce_pass_run_on_function(k);
        auto second = dce_pass_run_on_function(k);
        expect(first.removed_inst_count == 0u);
        expect(second.removed_inst_count == 0u);
        expect(second.removed_block_count == 0u);
    };

    "dce_exec_reachability_retains_structural_shell_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *true_c = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(body);
        auto *if_inst = b.if_(true_c);
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        auto *if_merge = if_inst->create_merge_block();
        b.set_insertion_point(if_true);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        int32_t selector_v = 1;
        auto *selector = m.create_constant(Type::of<int>(), &selector_v);
        auto *sw = b.switch_(selector);
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        auto *sw_merge = sw->create_merge_block();
        b.set_insertion_point(sw_default);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(true_c, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 0u);
        expect(count_reachable_blocks(k) == 10u);
        expect(body->terminator()->isa<IfInst>());
        expect(if_true->terminator()->isa<BranchInst>());
        expect(if_false->terminator()->isa<UnreachableInst>());
        expect(if_merge->terminator()->isa<SwitchInst>());
        expect(sw_case->terminator()->isa<BranchInst>());
        expect(sw_default->terminator()->isa<UnreachableInst>());
        expect(sw_merge->terminator()->isa<LoopInst>());
        auto *result_loop = static_cast<LoopInst *>(sw_merge->terminator());
        expect(result_loop->prepare_block() == prepare);
        expect(result_loop->body_block() == loop_body);
        expect(result_loop->update_block() == update);
        expect(result_loop->merge_block() == loop_merge);
        // Generic DCE intentionally folds the constant-true prepare into the
        // canonical unconditional form. It also clears the inactive merge
        // payload below, which is why the SPIR-V post-restructure boundary uses
        // targeted inactive-role cleanup instead of generic DCE.
        expect(prepare->terminator()->isa<BranchInst>());
        expect(loop_merge->parent_function() == k);
        expect(loop_merge->terminator()->isa<UnreachableInst>());
    };

    "dce_constant_cond_br_becomes_taken_branch"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *taken = k->create_basic_block();
        auto *dead = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_constant_one(Type::of<bool>()), taken, dead);
        b.set_insertion_point(taken);
        b.return_void();
        b.set_insertion_point(dead);
        b.return_void();

        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 1u);
        expect(entry->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(entry->terminator())->target_block() == taken);
        expect(count_reachable_blocks(k) == 2u);
    };

    "dce_constant_if_preserves_taken_break_in_loop_scope"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.break_(merge);
        b.set_insertion_point(if_false);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop_body->terminator()->isa<IfInst>());
        expect(if_true->terminator()->isa<BreakInst>());
        expect(static_cast<BreakInst *>(if_true->terminator())->target_block() == merge);
        expect(if_false->terminator()->isa<UnreachableInst>());
        expect(loop->body_block() == loop_body);
        expect(loop->update_block() == update);
        expect(update->terminator()->isa<UnreachableInst>());
        expect(loop->merge_block() == merge);
    };

    "dce_constant_switch_preserves_taken_continue_in_loop_scope"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        int32_t selector_value = 1;
        auto *selector = m.create_constant(Type::of<int>(), &selector_value);
        auto *switch_inst = b.switch_(selector);
        auto *switch_case = switch_inst->create_case_block(1);
        auto *switch_default = switch_inst->create_default_block();
        b.set_insertion_point(switch_case);
        b.continue_(update);
        b.set_insertion_point(switch_default);
        b.break_(merge);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop_body->terminator()->isa<SwitchInst>());
        expect(switch_case->terminator()->isa<ContinueInst>());
        expect(static_cast<ContinueInst *>(switch_case->terminator())->target_block() == update);
        expect(switch_default->terminator()->isa<UnreachableInst>());
        expect(loop->body_block() == loop_body);
        expect(loop->update_block() == update);
        expect(loop->merge_block() == merge);
        expect(merge->parent_function() == k);
        expect(merge->terminator()->isa<UnreachableInst>());
    };

    "dce_evaluates_int64_switch_condition_without_truncation"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *entry = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        int64_t selector_value = (int64_t{1} << 32u) + 1;
        auto *selector = m.create_constant(Type::of<int64_t>(), &selector_value);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1);
        auto *default_block = switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_inst_count > 0u);
        expect(entry->terminator() == switch_inst);
        expect(switch_inst->value() == selector);
        expect(switch_inst->value()->type() == Type::of<int64_t>());
        expect(static_cast<Constant *>(switch_inst->value())->as<int64_t>() == selector_value);
        expect(switch_inst->case_block(0u) == case_block);
        expect(switch_inst->default_block() == default_block);
        expect(case_block->terminator()->isa<UnreachableInst>());
        expect(default_block->terminator()->isa<BranchInst>());
    };

    "dce_loop_preserves_dead_body_and_update_shells"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(m.create_constant_zero(Type::of<bool>()), loop_body, merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop->prepare_block() == prepare);
        expect(prepare->terminator()->isa<ConditionalBranchInst>());
        if (prepare->terminator()->isa<ConditionalBranchInst>()) {
            auto *branch = static_cast<ConditionalBranchInst *>(prepare->terminator());
            expect(branch->condition()->isa<Constant>());
            expect(!static_cast<Constant *>(branch->condition())->as<bool>());
            expect(branch->true_block() == loop_body);
            expect(branch->false_block() == merge);
        }
        expect(loop->body_block() == loop_body);
        expect(loop_body->terminator()->isa<UnreachableInst>());
        expect(loop->update_block() == update);
        expect(update->terminator()->isa<UnreachableInst>());
        expect(loop->merge_block() == merge);
        expect(count_reachable_blocks(k) == 4u);
        expect(xir_verify_module(&m).succeeded());
    };

    "dce_preserves_unreachable_if_merge_shell"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *entry = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *if_inst = b.if_(condition);
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(if_true);
        b.unreachable_("executable unreachable");
        b.set_insertion_point(if_false);
        b.return_void();
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(if_inst->merge_block() == merge);
        expect(merge->parent_function() == k);
        expect(merge->terminator()->isa<UnreachableInst>());
        expect(if_true->terminator()->isa<UnreachableInst>());
        expect(static_cast<UnreachableInst *>(if_true->terminator())->message() == "executable unreachable");
        expect(count_reachable_blocks(k) == 3u);
    };

    "dce_keeps_reachable_self_loop_and_removes_disconnected_cycle"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *dead_a = k->create_basic_block();
        auto *dead_b = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(entry);
        b.set_insertion_point(dead_a);
        b.br(dead_b);
        b.set_insertion_point(dead_b);
        b.br(dead_a);

        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 2u);
        expect(entry->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(entry->terminator())->target_block() == entry);
        expect(count_reachable_blocks(k) == 1u);
    };
}

// ---- gvn ----

void reg_gvn() {

    "gvn_duplicate_add_replaced"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *final = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, add2});
        b.return_(final);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count >= 1u);
    };

    "gvn_no_duplicate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4, c_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *c = m.create_constant(Type::of<int>(), &c_v);
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, c});
        b.return_(add2);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
    };

    "gvn_strict_float_reversed_add_not_merged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = k->create_value_argument(Type::of<float>());
        auto *bv = k->create_value_argument(Type::of<float>());
        auto *add0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {bv, a});
        auto *pair = b.call(Type::of<float2>(), ArithmeticOp::AGGREGATE, {add0, add1});
        b.return_(pair);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
        expect(pair->operand(0) == add0);
        expect(pair->operand(1) == add1);
    };

    "gvn_integer_reversed_add_merged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = k->create_value_argument(Type::of<int>());
        auto *bv = k->create_value_argument(Type::of<int>());
        auto *add0 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {bv, a});
        auto add1_locked = add1->lock();
        auto *pair = b.call(Type::of<int2>(), ArithmeticOp::AGGREGATE, {add0, add1});
        b.return_(pair);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(pair->operand(0) == add0);
        expect(pair->operand(1) == add0);
        expect(add1_locked->use_list().empty());
    };

    "gvn_does_not_merge_mutable_accel_query_across_write"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *accel =
            k->create_resource_argument(Type::of<Accel>());
        auto *instance =
            m.create_constant_zero(Type::of<uint32_t>());
        auto *new_user_id =
            m.create_constant_one(Type::of<uint32_t>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sink = b.alloca_local(Type::of<uint2>());
        auto *before = b.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        b.call(
            ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
            {accel, instance, new_user_id});
        auto *after = b.call(
            Type::of<uint32_t>(),
            ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        auto *pair = b.call(
            Type::of<uint2>(), ArithmeticOp::AGGREGATE,
            {before, after});
        b.store(sink, pair);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = gvn_pass_run_on_function(k);

        expect(info.replaced_inst_count == 0u);
        expect(pair->operand(0u) == before);
        expect(pair->operand(1u) == after);
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_still_merges_immutable_buffer_size_queries"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *buffer = k->create_resource_argument(
            Type::buffer(Type::of<float>()));
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sink = b.alloca_local(Type::of<uint2>());
        auto *size0 = b.call(
            Type::of<uint32_t>(), ResourceQueryOp::BUFFER_SIZE,
            {buffer});
        auto *size1 = b.call(
            Type::of<uint32_t>(), ResourceQueryOp::BUFFER_SIZE,
            {buffer});
        auto *pair = b.call(
            Type::of<uint2>(), ArithmeticOp::AGGREGATE,
            {size0, size1});
        b.store(sink, pair);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = gvn_pass_run_on_function(k);

        expect(info.replaced_inst_count == 1u);
        expect(pair->operand(0u) == size0);
        expect(pair->operand(1u) == size0);
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_does_not_merge_implicit_derivative_sampling_across_cfg"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *texture = k->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *body = k->create_body_block();
        auto *branch = k->create_basic_block();
        auto *merge = k->create_basic_block();
        auto *uv = m.create_constant_zero(Type::of<float2>());
        auto *selector = m.create_constant_zero(Type::of<uint>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *sample0 = b.call(
            Type::of<float4>(), ResourceQueryOp::TEXTURE2D_SAMPLE,
            {texture, uv, selector, selector});
        b.cond_br(condition, branch, merge);

        b.set_insertion_point(branch);
        auto *sample1 = b.call(
            Type::of<float4>(), ResourceQueryOp::TEXTURE2D_SAMPLE,
            {texture, uv, selector, selector});
        auto *sink = b.alloca_local(Type::of<float4>());
        auto *sum = b.call(
            Type::of<float4>(), ArithmeticOp::BINARY_ADD,
            {sample0, sample1});
        b.store(sink, sum);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
        expect(sum->operand(0u) == sample0);
        expect(sum->operand(1u) == sample1);
        expect(sample0->is_linked());
        expect(sample1->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_still_merges_explicit_lod_sampling_across_cfg"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *texture = k->create_resource_argument(
            Type::texture(Type::of<float>(), 2u));
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *body = k->create_body_block();
        auto *branch = k->create_basic_block();
        auto *merge = k->create_basic_block();
        auto *uv = m.create_constant_zero(Type::of<float2>());
        auto *lod = m.create_constant_zero(Type::of<float>());
        auto *selector = m.create_constant_zero(Type::of<uint>());
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *sample0 = b.call(
            Type::of<float4>(), ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL,
            {texture, uv, lod, selector, selector});
        b.cond_br(condition, branch, merge);

        b.set_insertion_point(branch);
        auto *sample1 = b.call(
            Type::of<float4>(), ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL,
            {texture, uv, lod, selector, selector});
        auto *sink = b.alloca_local(Type::of<float4>());
        auto *sum = b.call(
            Type::of<float4>(), ArithmeticOp::BINARY_ADD,
            {sample0, sample1});
        b.store(sink, sum);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(sum->operand(0u) == sample0);
        expect(sum->operand(1u) == sample0);
        expect(!sample1->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            auto *final = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, add2});
            b.return_(final);
        }
        auto info = gvn_pass_run_on_module(&m);
        expect(info.replaced_inst_count >= 2u);
    };

    "gvn_annotated_duplicate_keeps_distinct_metadata_owner"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int2>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *first = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *annotated = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        annotated->set_location("gvn_metadata.cpp", 12);
        auto *pair = b.call(
            Type::of<int2>(), ArithmeticOp::AGGREGATE,
            {first, annotated});
        b.return_(pair);

        expect(xir_verify_module(&m).succeeded());
        auto info = gvn_pass_run_on_function(f);
        expect(info.replaced_inst_count == 0u);
        expect(pair->operand(0u) == first);
        expect(pair->operand(1u) == annotated);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_annotated_leader_does_not_absorb_plain_duplicate"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int2>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *annotated = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        annotated->set_location("gvn_leader.cpp", 19);
        auto *plain = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *pair = b.call(
            Type::of<int2>(), ArithmeticOp::AGGREGATE,
            {annotated, plain});
        b.return_(pair);

        expect(xir_verify_module(&m).succeeded());
        auto info = gvn_pass_run_on_function(f);
        expect(info.replaced_inst_count == 0u);
        expect(pair->operand(0u) == annotated);
        expect(pair->operand(1u) == plain);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(plain->metadata_list().empty());
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_optimizes_continuations_without_crossing_suspend"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *x = k->create_value_argument(Type::of<int>());
        auto *entry = k->create_body_block();
        auto *resume = k->create_basic_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;

        b.set_insertion_point(entry);
        auto *entry_sink = b.alloca_local(Type::of<int>());
        auto *before = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        b.store(entry_sink, before);
        b.coro_suspend(7u, "gvn-boundary", nullptr);

        b.set_insertion_point(resume);
        b.coro_resume(7u, nullptr);
        auto *resume_sink = b.alloca_local(Type::of<int2>());
        auto *after0 = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *after1 = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *pair = b.call(
            Type::of<int2>(), ArithmeticOp::AGGREGATE,
            {after0, after1});
        auto *store = b.store(resume_sink, pair);
        b.coro_terminate();

        expect(xir_verify_module(&m).succeeded());
        auto info = gvn_pass_run_on_function(k);

        // `after0` must remain continuation-local: replacing it with `before`
        // would create a new frame use. The second expression is in the same
        // continuation and is therefore safely coalesced with `after0`.
        expect(info.rejected_cross_suspend_count >= 1u);
        expect(info.replaced_inst_count == 1u);
        expect(pair->operand(0u) == after0);
        expect(pair->operand(1u) == after0);
        expect(store->value() == pair);
        expect(after0->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "gvn_null_inputs_are_noops"_test = [] {
        expect(!gvn_pass_run_on_function(nullptr).changed());
        expect(!gvn_pass_run_on_module(nullptr).changed());
    };
}

// ---- sccp ----

void reg_sccp() {

    "sccp_bodyless_definition_is_ignored"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto info = sccp_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(info.removed_branch_count == 0u);
    };

    "sccp_const_propagation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 2, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count >= 1u);
    };

    "sccp_no_constants_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *u1 = m.create_undefined(Type::of<int>());
        auto *u2 = m.create_undefined(Type::of<int>());
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {u1, u2});
        b.return_(add);
        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "sccp_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_(add);
        }
        auto info = sccp_pass_run_on_module(&m);
        expect(info.folded_inst_count >= 2u);
    };

    "sccp_loop_carried_phi_not_folded"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);

        auto *header = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *update = k->create_basic_block();
        auto *merge = k->create_basic_block();

        // entry -> header
        b.br(header);

        // header: phi and loop condition
        b.set_insertion_point(header);
        auto *phi = b.phi(Type::of<int>());
        int32_t zero_v = 0, four_v = 4;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *four = m.create_constant(Type::of<int>(), &four_v);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, four});
        b.cond_br(cond, loop_body, merge);

        // loop_body: load produces BOTTOM, add is loop-carried
        b.set_insertion_point(loop_body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *load = b.load(Type::of<int>(), alloca);
        auto *i_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, load});
        b.br(update);

        // update: back-edge to header
        b.set_insertion_point(update);
        b.br(header);

        // merge
        b.set_insertion_point(merge);
        b.return_void();

        phi->add_incoming(zero, entry);
        phi->add_incoming(i_next, update);

        auto info = sccp_pass_run_on_function(k);
        expect(info.removed_branch_count == 0u);
        expect(header->terminator()->derived_instruction_tag() == DerivedInstructionTag::CONDITIONAL_BRANCH);
    };

    "sccp_annotated_constant_value_keeps_instruction_owner"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        sum->set_location("sccp_metadata.cpp", 8);
        auto *ret = b.return_(sum);

        auto info = sccp_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == sum);
        expect(sum->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "sccp_null_inputs_are_noops"_test = [] {
        expect(!sccp_pass_run_on_function(nullptr).changed());
        PassReport report;
        expect(!sccp_pass_run_on_module(nullptr, &report).changed());
        expect(report.entries().size() == 2u);
    };
}

// ---- simplify_libcalls ----

void reg_simplify_libcalls() {

    "simplify_libcalls_lerp_t_zero_keeps_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 0.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "simplify_libcalls_lerp_t_one_keeps_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "simplify_libcalls_clamp_01_to_saturate"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 0.5f, lo_v = 0.0f, hi_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        auto *clamp = b.call(
            Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        clamp->set_location("simplify_libcalls.cpp", 41);
        clamp->add_comment("preserve clamp metadata");
        auto *ret = b.return_(clamp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *replacement =
            static_cast<ArithmeticInst *>(ret->return_value());
        expect(replacement->op() == ArithmeticOp::SATURATE);
        expect(replacement->operand(0) == x);
        auto *location = replacement->find_metadata<LocationMD>();
        expect(location != nullptr);
        expect(location != nullptr && location->line() == 41);
        expect(replacement->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "simplify_libcalls_clamp_negative_zero_not_saturated"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float lo_v = -0.0f, hi_v = 1.0f;
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        auto *clamp = b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        auto *ret = b.return_(clamp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == clamp);
    };

    "simplify_libcalls_uniform_double_vector_clamp_is_saturated"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<double2>());
        auto *x = f->create_value_argument(Type::of<double2>());
        auto *body = f->create_body_block();
        auto lo_value = make_double2(0.0);
        auto hi_value = make_double2(1.0);
        auto *lo =
            m.create_constant(Type::of<double2>(), &lo_value);
        auto *hi =
            m.create_constant(Type::of<double2>(), &hi_value);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *clamp = b.call(
            Type::of<double2>(), ArithmeticOp::CLAMP, {x, lo, hi});
        auto *ret = b.return_(clamp);

        auto info = simplify_libcalls_pass_run_on_function(f);

        expect(info.simplified_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *replacement =
            static_cast<ArithmeticInst *>(ret->return_value());
        expect(replacement->op() == ArithmeticOp::SATURATE);
        expect(replacement->operand(0u) == x);
        expect(xir_verify_module(&m).succeeded());
    };

    "simplify_libcalls_mixed_signed_zero_vector_is_not_saturated"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<double2>());
        auto *x = f->create_value_argument(Type::of<double2>());
        auto *body = f->create_body_block();
        auto lo_value = make_double2(0.0, -0.0);
        auto hi_value = make_double2(1.0);
        auto *lo =
            m.create_constant(Type::of<double2>(), &lo_value);
        auto *hi =
            m.create_constant(Type::of<double2>(), &hi_value);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *clamp = b.call(
            Type::of<double2>(), ArithmeticOp::CLAMP, {x, lo, hi});
        auto *ret = b.return_(clamp);

        auto info = simplify_libcalls_pass_run_on_function(f);

        expect(info.simplified_count == 0u);
        expect(ret->return_value() == clamp);
        expect(xir_verify_module(&m).succeeded());
    };

    "simplify_libcalls_no_simplification"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.0f, y_v = 2.0f, t_v = 0.5f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        b.return_void();
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 0u);
    };

    "simplify_libcalls_step_zero_edge_keeps_sign_dependent_result"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<float>());
        auto *step = b.call(Type::of<float>(), ArithmeticOp::STEP, {zero, x});
        auto *ret = b.return_(step);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == step);
    };

    "simplify_libcalls_module_runs_all_functions"_test = [] {
        Module m;
        luisa::vector<Value *> arguments;
        luisa::vector<ReturnInst *> returns;
        for (int i = 0; i < 2; ++i) {
            auto *f = m.create_callable(Type::of<uint>());
            auto *x = f->create_value_argument(Type::of<uint>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *abs = b.call(Type::of<uint>(), ArithmeticOp::ABS, {x});
            arguments.emplace_back(x);
            returns.emplace_back(b.return_(abs));
        }
        auto info = simplify_libcalls_pass_run_on_module(&m);
        expect(info.simplified_count == 2u);
        expect(returns[0]->return_value() == arguments[0]);
        expect(returns[1]->return_value() == arguments[1]);
    };

    "simplify_libcalls_annotated_identity_is_left_unchanged"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<uint>());
        auto *x = f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *abs = b.call(Type::of<uint>(), ArithmeticOp::ABS, {x});
        abs->set_location("annotated_identity.cpp", 17);
        auto *ret = b.return_(abs);

        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == abs);
        expect(abs->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "simplify_libcalls_annotated_select_identity_is_left_unchanged"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *condition = f->create_value_argument(Type::of<bool>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *select = b.call(
            Type::of<int>(), ArithmeticOp::SELECT,
            {x, x, condition});
        select->add_comment("select metadata has no unique identity owner");
        auto *ret = b.return_(select);

        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == select);
        expect(select->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "simplify_libcalls_null_inputs_are_noops"_test = [] {
        expect(simplify_libcalls_pass_run_on_function(nullptr)
                   .simplified_count == 0u);
        expect(simplify_libcalls_pass_run_on_module(nullptr)
                   .simplified_count == 0u);
    };
}

// ---- reassociate ----

void reg_reassociate() {

    "reassociate_chained_add"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<int>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2, c_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *c = m.create_constant(Type::of<int>(), &c_v);
        auto *bc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {bv, c});
        auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bc});
        b.return_(abc);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count == 1u);
        auto second = reassociate_pass_run_on_function(k);
        expect(second.reassociated_inst_count == 0u)
            << "a canonical left-associated chain must be a fixed point";
        expect(xir_verify_module(&m).succeeded());
    };

    "reassociate_equal_rank_operands_preserve_ir_order"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *a = f->create_value_argument(Type::of<int>());
        auto *b_arg = f->create_value_argument(Type::of<int>());
        auto *c = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ab = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, b_arg});
        auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ab, c});
        auto *ret = b.return_(abc);
        auto info = reassociate_pass_run_on_function(f);
        expect(info.reassociated_inst_count == 0u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *result = static_cast<ArithmeticInst *>(ret->return_value());
        expect(result->operand(1u) == c);
        expect(result->operand(0u)->isa<ArithmeticInst>());
        auto *lhs = static_cast<ArithmeticInst *>(result->operand(0u));
        expect(lhs->operand(0u) == a);
        expect(lhs->operand(1u) == b_arg);
    };

    "reassociate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count == 0u);
    };

    "reassociate_strict_float_chain_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *a = k->create_value_argument(Type::of<float>());
        auto *bv = k->create_value_argument(Type::of<float>());
        auto *c = k->create_value_argument(Type::of<float>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ab = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *abc = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {ab, c});
        auto *ret = b.return_(abc);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count == 0u);
        expect(ret->return_value() == abc);
        expect(abc->operand(0) == ab);
        expect(abc->operand(1) == c);
    };

    "reassociate_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            auto *k = m.create_callable(Type::of<int>());
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2, c_v = 3;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *c = m.create_constant(Type::of<int>(), &c_v);
            auto *bc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {bv, c});
            auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bc});
            b.return_(abc);
        }
        auto info = reassociate_pass_run_on_module(&m);
        expect(info.reassociated_inst_count >= 2u);
        auto second = reassociate_pass_run_on_module(&m);
        expect(second.reassociated_inst_count == 0u)
            << "module reassociation must converge after canonicalization";
        expect(xir_verify_module(&m).succeeded());
    };

    "reassociate_chained_mul_is_idempotent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<uint>());
        auto *a = f->create_value_argument(Type::of<uint>());
        auto *b_arg = f->create_value_argument(Type::of<uint>());
        auto *c = f->create_value_argument(Type::of<uint>());
        auto *d = f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *cd = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL, {c, d});
        auto *bcd = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL, {b_arg, cd});
        auto *abcd = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL, {a, bcd});
        builder.return_(abcd);

        auto first = reassociate_pass_run_on_function(f);
        expect(first.reassociated_inst_count >= 1u);
        auto second = reassociate_pass_run_on_function(f);
        expect(second.reassociated_inst_count == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "reassociate_preserves_annotated_internal_add_and_mul_nodes"_test = [] {
        auto run_case = [](ArithmeticOp op) noexcept {
            Module m;
            auto *f = m.create_callable(Type::of<uint>());
            auto *a = f->create_value_argument(Type::of<uint>());
            auto *b_arg = f->create_value_argument(Type::of<uint>());
            auto *c = f->create_value_argument(Type::of<uint>());
            auto *body = f->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *annotated = builder.call(
                Type::of<uint>(), op, {a, b_arg});
            annotated->add_comment(
                "internal reassociation metadata owner");
            auto *root = builder.call(
                Type::of<uint>(), op, {annotated, c});
            auto *ret = builder.return_(root);

            expect(xir_verify_module(&m).succeeded());
            auto info = reassociate_pass_run_on_function(f);
            expect(info.reassociated_inst_count == 0u);
            expect(ret->return_value() == root);
            expect(annotated->parent_block() == body);
            expect(annotated->find_metadata<CommentMD>() != nullptr);
            static_cast<void>(dce_pass_run_on_function(f));
            expect(annotated->parent_block() == body);
            expect(annotated->find_metadata<CommentMD>() != nullptr);
            expect(xir_verify_module(&m).succeeded());
        };
        run_case(ArithmeticOp::BINARY_ADD);
        run_case(ArithmeticOp::BINARY_MUL);
    };

    "reassociate_sub_keeps_annotated_lhs_subexpression"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *a = f->create_value_argument(Type::of<int>());
        auto *b_arg = f->create_value_argument(Type::of<int>());
        auto *c = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *annotated = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, b_arg});
        annotated->set_location("reassociate_metadata.cpp", 41);
        auto *sub = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_SUB, {annotated, c});
        auto *ret = builder.return_(sub);

        expect(xir_verify_module(&m).succeeded());
        auto info = reassociate_pass_run_on_function(f);
        expect(info.reassociated_inst_count == 1u);
        expect(ret->return_value() != sub);
        expect(annotated->parent_block() == body);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        static_cast<void>(dce_pass_run_on_function(f));
        expect(annotated->parent_block() == body);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };
}

// ---- cvp ----

void reg_cvp() {

    "cvp_equal_condition_propagates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *loaded = b.load(Type::of<int>(), local);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {loaded, val});
        auto *if_inst = b.if_(eq);
        auto *true_b = if_inst->create_true_block();
        auto *false_b = if_inst->create_false_block();
        auto *merge_b = if_inst->create_merge_block();
        b.set_insertion_point(true_b);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                           {loaded, m.create_constant_one(Type::of<int>())});
        b.br(merge_b);
        b.set_insertion_point(false_b);
        b.br(merge_b);
        b.set_insertion_point(merge_b);
        b.return_void();
        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(sum->operand(0) == val);
        expect(eq->operand(0) == loaded) << "the condition itself is not dominated by the taken block";
    };

    "cvp_no_if_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
    };

    "cvp_float_zero_does_not_lose_signed_zero"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *var = b.call(Type::of<float>(), ArithmeticOp::UNARY_MINUS, {x});
        auto *zero = m.create_constant_zero(Type::of<float>());
        auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {var, zero});
        auto *if_inst = b.if_(eq);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        auto *div = b.call(Type::of<float>(), ArithmeticOp::BINARY_DIV,
                           {m.create_constant_one(Type::of<float>()), var});
        b.return_(div);
        b.set_insertion_point(false_block);
        b.return_(zero);
        b.set_insertion_point(merge);
        b.unreachable_();
        auto info = cvp_pass_run_on_function(f);
        expect(info.replaced_inst_count == 0u);
        expect(div->operand(1) == var);
        expect(body->terminator() == if_inst);
        expect(if_inst->merge_block() == merge);
    };

    "cvp_does_not_propagate_condition_fact_into_selected_merge"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *sink = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_zero(Type::of<int>()));
        auto *loaded = b.load(Type::of<int>(), local);
        int32_t expected_value = 42;
        auto *expected =
            m.create_constant(Type::of<int>(), &expected_value);
        auto *eq = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {loaded, expected});
        auto *if_inst = b.if_(eq);
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        if_inst->set_true_target(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {loaded, m.create_constant_one(Type::of<int>())});
        b.store(sink, sum);
        b.return_void();

        auto info = cvp_pass_run_on_function(k);
        expect(!info.changed());
        expect(sum->operand(0) == loaded);
        expect(xir_verify_module(&m).succeeded());
    };

    "cvp_rejects_sibling_cross_entry_into_selected_arm"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *sink = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_zero(Type::of<int>()));
        auto *loaded = b.load(Type::of<int>(), local);
        int32_t expected_value = 7;
        auto *expected =
            m.create_constant(Type::of<int>(), &expected_value);
        auto *eq = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {loaded, expected});
        auto *if_inst = b.if_(eq);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(false_block);
        b.br(true_block);
        b.set_insertion_point(true_block);
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {loaded, m.create_constant_one(Type::of<int>())});
        b.store(sink, sum);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = cvp_pass_run_on_function(k);
        expect(!info.changed());
        expect(sum->operand(0) == loaded);
        expect(xir_verify_module(&m).succeeded());
    };

    "cvp_ignores_disconnected_owned_uses_outside_dom_tree"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *disconnected = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *loaded = b.load(Type::of<int>(), local);
        auto *one = m.create_constant_one(Type::of<int>());
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {loaded, one});
        auto *if_inst = b.if_(condition);
        auto *selected = if_inst->create_true_block();
        auto *other = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(selected);
        auto *selected_use = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {loaded, one});
        b.br(merge);
        b.set_insertion_point(other);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        b.set_insertion_point(disconnected);
        auto *disconnected_use = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {loaded, m.create_constant_zero(Type::of<int>())});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(selected_use->operand(0u) == one);
        expect(disconnected_use->operand(0u) == loaded);
        expect(disconnected_use->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "cvp_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        auto info = cvp_pass_run_on_module(&m);
        expect(info.replaced_inst_count == 0u);
    };
}

// ---- dead_arg_elim ----

void reg_dead_arg_elim() {

    "dead_arg_elim_unused_callable_arg_removed"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<void>());
        auto *unused = c->create_value_argument(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dead_arg_elim_pass_run_on_function(c);
        expect(info.removed_arg_count == 1u);
    };

    "dead_arg_elim_all_args_used_no_change"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *arg = c->create_value_argument(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_(arg);
        auto info = dead_arg_elim_pass_run_on_function(c);
        expect(info.removed_arg_count == 0u);
    };

    "dead_arg_elim_retains_annotated_unused_argument"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        auto *argument =
            callee->create_value_argument(Type::of<int>());
        argument->add_comment("preserve argument provenance");
        auto *callee_body = callee->create_body_block();
        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();
        builder.set_insertion_point(kernel_body);
        auto *call = builder.call(nullptr, callee, {one});
        builder.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = dead_arg_elim_pass_run_on_function(callee);
        expect(info.removed_arg_count == 0u);
        expect(callee->arguments().count_size() == 1u);
        expect(callee->arguments().front() == argument);
        expect(argument->find_metadata<CommentMD>() != nullptr);
        expect(call->argument_count() == 1u);
        expect(call->argument(0u) == one);
        expect(xir_verify_module(&m).succeeded());
    };

    "dead_arg_elim_removes_unannotated_siblings_around_annotated_argument"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        static_cast<void>(
            callee->create_value_argument(Type::of<int>()));
        auto *preserved =
            callee->create_value_argument(Type::of<int>());
        preserved->set_location("dead_arg_metadata.cpp", 23);
        static_cast<void>(
            callee->create_value_argument(Type::of<int>()));
        auto *callee_body = callee->create_body_block();
        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t two_value = 2;
        auto *two =
            m.create_constant(Type::of<int>(), &two_value);
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();
        builder.set_insertion_point(kernel_body);
        auto *call =
            builder.call(nullptr, callee, {zero, one, two});
        builder.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = dead_arg_elim_pass_run_on_function(callee);
        expect(info.removed_arg_count == 2u);
        expect(callee->arguments().count_size() == 1u);
        expect(callee->arguments().front() == preserved);
        expect(preserved->find_metadata<LocationMD>() != nullptr);
        expect(call->argument_count() == 1u);
        expect(call->argument(0u) == one);
        expect(xir_verify_module(&m).succeeded());
    };

    "dead_arg_elim_kernel_skipped"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dead_arg_elim_pass_run_on_function(k);
        expect(info.removed_arg_count == 0u);
    };

    "dead_arg_elim_bodyless_callable_signature_is_preserved"_test = [] {
        Module m;
        auto *declaration = m.create_callable(Type::of<void>());
        auto *argument =
            declaration->create_value_argument(Type::of<int>());

        auto info = dead_arg_elim_pass_run_on_function(declaration);

        expect(info.removed_arg_count == 0u);
        expect(declaration->arguments().count_size() == 1u);
        expect(argument->is_linked());
    };

    "dead_arg_elim_accepts_null_inputs"_test = [] {
        auto function_info = dead_arg_elim_pass_run_on_function(nullptr);
        auto module_info = dead_arg_elim_pass_run_on_module(nullptr);
        expect(function_info.removed_arg_count == 0u);
        expect(module_info.removed_arg_count == 0u);
    };

    "dead_arg_elim_ray_query_callback_abi_preserved"_test = [] {
        Module m;
        auto *query_type = Type::of<RayQueryAll>();
        auto make_callback = [&] {
            auto *callback = m.create_callable(nullptr);
            callback->create_reference_argument(query_type);
            callback->create_value_argument(Type::of<int>());
            auto *callback_body = callback->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(callback_body);
            b.return_void();
            return callback;
        };
        auto *surface_callback = make_callback();
        auto *procedural_callback = make_callback();

        auto *pipeline_function = m.create_callable(nullptr);
        auto *query = pipeline_function->create_reference_argument(query_type);
        auto *capture = pipeline_function->create_value_argument(Type::of<int>());
        auto *body = pipeline_function->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        std::array<Value *, 1u> captures{capture};
        b.ray_query_pipeline(
            query, surface_callback, procedural_callback,
            luisa::span<Value *const>{captures});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dead_arg_elim_pass_run_on_module(&m);
        expect(info.removed_arg_count == 0u);
        expect(surface_callback->arguments().count_size() == 2u);
        expect(procedural_callback->arguments().count_size() == 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "dead_arg_elim_rejects_function_value_used_as_unrelated_call_argument"_test = [] {
        Module m;
        auto *candidate = m.create_callable(Type::of<int>());
        candidate->create_value_argument(Type::of<int>());
        auto *candidate_body = candidate->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(candidate_body);
        b.return_(one);

        auto *consumer = m.create_callable(nullptr);
        consumer->create_value_argument(Type::of<int>());
        auto *consumer_body = consumer->create_body_block();
        b.set_insertion_point(consumer_body);
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        auto *unrelated_call = b.call(nullptr, consumer, {candidate});
        b.return_void();

        auto info = dead_arg_elim_pass_run_on_function(candidate);
        expect(info.removed_arg_count == 0u);
        expect(candidate->arguments().count_size() == 1u);
        expect(unrelated_call->callee() == consumer);
        expect(unrelated_call->argument_count() == 1u);
        expect(unrelated_call->argument(0u) == candidate);
    };

    "dead_arg_elim_rejects_malformed_call_arity_before_mutation"_test = [] {
        Module m;
        auto *candidate = m.create_callable(nullptr);
        candidate->create_value_argument(Type::of<int>());
        auto *candidate_body = candidate->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(candidate_body);
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        auto *malformed_call = b.call(nullptr, candidate, {});
        b.return_void();

        auto info = dead_arg_elim_pass_run_on_function(candidate);
        expect(info.removed_arg_count == 0u);
        expect(candidate->arguments().count_size() == 1u);
        expect(malformed_call->argument_count() == 0u);
        expect(malformed_call->is_linked());
    };

    "dead_arg_elim_rejects_callee_reused_as_call_argument_before_mutation"_test = [] {
        Module m;
        auto *candidate = m.create_callable(Type::of<int>());
        auto *unused = candidate->create_value_argument(Type::of<int>());
        auto *candidate_body = candidate->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(candidate_body);
        b.return_(one);

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        // Deliberately malformed: Function values are not legal ordinary call
        // arguments. The same CallInst therefore contributes two uses of
        // `candidate`, only one of which is the callee operand.
        auto *malformed_call =
            b.call(Type::of<int>(), candidate, {candidate});
        b.return_void();

        auto info = dead_arg_elim_pass_run_on_function(candidate);
        expect(info.removed_arg_count == 0u);
        expect(candidate->arguments().count_size() == 1u);
        expect(unused->is_linked());
        expect(malformed_call->callee() == candidate);
        expect(malformed_call->argument_count() == 1u);
        expect(malformed_call->argument(0u) == candidate);
        expect(malformed_call->is_linked());
    };

    "dead_arg_elim_updates_each_distinct_valid_call_site_once"_test = [] {
        Module m;
        auto *candidate = m.create_callable(nullptr);
        candidate->create_value_argument(Type::of<int>());
        auto *candidate_body = candidate->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(candidate_body);
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(kernel_body);
        auto *first = b.call(nullptr, candidate, {one});
        auto *second = b.call(nullptr, candidate, {one});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dead_arg_elim_pass_run_on_function(candidate);
        expect(info.removed_arg_count == 1u);
        expect(candidate->arguments().empty());
        expect(first->argument_count() == 0u);
        expect(second->argument_count() == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "dead_arg_elim_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            auto *c = m.create_callable(Type::of<void>());
            c->create_value_argument(Type::of<float>());
            auto *body = c->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        auto info = dead_arg_elim_pass_run_on_module(&m);
        expect(info.removed_arg_count == 2u);
    };
}

// ---- div_rem_pairs ----

void reg_div_rem_pairs() {

    "div_rem_pairs_div_and_mod_merged"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<int>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *div = b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        auto *mod = b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        mod->set_location("div_rem_pairs.cpp", 29);
        mod->add_comment("remainder replacement metadata");
        auto mod_locked = mod->lock();
        auto *ret = b.return_(mod);
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 1u);
        expect(mod_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *sub = static_cast<ArithmeticInst *>(ret->return_value());
        expect(sub->op() == ArithmeticOp::BINARY_SUB);
        expect(sub->operand(0) == a);
        expect(sub->operand(1)->isa<ArithmeticInst>());
        auto *mul = static_cast<ArithmeticInst *>(sub->operand(1));
        expect(mul->op() == ArithmeticOp::BINARY_MUL);
        expect(mul->operand(0) == div);
        expect(mul->operand(1) == bv);
        auto *location = sub->find_metadata<LocationMD>();
        expect(location != nullptr);
        expect(location != nullptr && location->line() == 29);
        expect(sub->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "div_rem_pairs_no_mod_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 0u);
    };

    "div_rem_pairs_mod_before_div_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 0u);
    };

    "div_rem_pairs_nested_remainders_preserve_current_operands"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dispatch_id = m.create_dispatch_id();
        uint32_t x_index_v = 0u;
        uint32_t outer_v = 64u;
        uint32_t inner_v = 8u;
        auto *x_index = m.create_constant(Type::of<uint32_t>(), &x_index_v);
        auto *outer = m.create_constant(Type::of<uint32_t>(), &outer_v);
        auto *inner = m.create_constant(Type::of<uint32_t>(), &inner_v);
        auto *x = b.call(Type::of<uint32_t>(), ArithmeticOp::EXTRACT, {dispatch_id, x_index});
        b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {x, outer});
        auto *rem = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {x, outer});
        b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {rem, inner});
        auto *nested_rem = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {rem, inner});
        b.return_(nested_rem);

        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 2u);
        auto mod_count = 0u;
        body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_MOD) {
                mod_count++;
            }
        });
        expect(mod_count == 0u);
        auto *ret = static_cast<ReturnInst *>(body->terminator());
        auto *nested_sub = static_cast<ArithmeticInst *>(ret->return_value());
        expect(nested_sub->op() == ArithmeticOp::BINARY_SUB);
        auto *outer_sub = static_cast<ArithmeticInst *>(nested_sub->operand(0));
        auto *nested_mul = static_cast<ArithmeticInst *>(nested_sub->operand(1));
        auto *nested_div = static_cast<ArithmeticInst *>(nested_mul->operand(0));
        expect(outer_sub->op() == ArithmeticOp::BINARY_SUB);
        expect(nested_mul->op() == ArithmeticOp::BINARY_MUL);
        expect(nested_div->op() == ArithmeticOp::BINARY_DIV);
        expect(nested_div->operand(0) == outer_sub);
        expect(nested_mul->operand(1) == inner);
    };

    "div_rem_pairs_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 10, b_v = 3;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
            b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
            b.return_void();
        }
        auto info = div_rem_pairs_pass_run_on_module(&m);
        expect(info.merged_pair_count == 2u);
    };

    "div_rem_pairs_null_inputs_are_noops"_test = [] {
        expect(div_rem_pairs_pass_run_on_function(nullptr)
                   .merged_pair_count == 0u);
        expect(div_rem_pairs_pass_run_on_module(nullptr)
                   .merged_pair_count == 0u);
    };
}

// ---- local_load_elimination ----

void reg_local_load_elimination() {

    "local_load_elim_duplicate_load_removed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        auto *ld1 = b.load(Type::of<int>(), alloca);
        auto *ld2 = b.load(Type::of<int>(), alloca);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld1, ld2});
        b.return_(add);
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 1u);
    };

    "local_load_elim_no_duplicate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
    };

    "local_load_elim_does_not_forward_reference_loads"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *p0 = k->create_reference_argument(Type::of<int>());
        auto *p1 = k->create_reference_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ld0 = b.load(Type::of<int>(), p0);
        b.store(p1, m.create_constant_one(Type::of<int>()));
        auto *ld1 = b.load(Type::of<int>(), p0);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld0, ld1});
        b.store(p1, sum);
        b.return_void();
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(ld0->is_linked());
        expect(ld1->is_linked());
    };

    "local_load_elim_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            b.store(alloca, m.create_constant_zero(Type::of<int>()));
            auto *ld1 = b.load(Type::of<int>(), alloca);
            auto *ld2 = b.load(Type::of<int>(), alloca);
            auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld1, ld2});
            b.return_(add);
        }
        auto info = local_load_elimination_pass_run_on_module(&m);
        expect(info.removed_load_count == 2u);
    };

    "local_load_elim_entry_backedge_does_not_forward_future_load"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *exit = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *first_load = b.load(Type::of<int>(), alloca);
        auto first_load_lock = first_load->lock();
        auto *increment = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                 {first_load, m.create_constant_one(Type::of<int>())});
        auto *store = b.store(alloca, increment);
        auto *future_load = b.load(Type::of<int>(), alloca);
        b.cond_br(m.create_undefined(Type::of<bool>()), body, exit);
        b.set_insertion_point(exit);
        b.return_(future_load);

        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(first_load->is_linked());
        expect(future_load->is_linked());
        expect(increment->operand(0u) == first_load_lock.get());
        expect(first_load->next() == increment);
        expect(increment->next() == store);
        expect(store->next() == future_load);
    };

    "local_load_elim_loop_fanout_keeps_analysis_storage_stable"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *header = k->create_basic_block();
        auto *exit = k->create_basic_block();
        std::array<BasicBlock *, 7u> branches{};
        std::array<BasicBlock *, 8u> latches{};
        for (auto &block : branches) {
            block = k->create_basic_block();
        }
        for (auto &block : latches) {
            block = k->create_basic_block();
        }

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        b.br(header);

        b.set_insertion_point(header);
        auto *value = b.load(Type::of<int>(), alloca);
        static_cast<void>(value);
        b.cond_br(
            m.create_undefined(Type::of<bool>()),
            branches.front(),
            exit);

        auto tree_block = [&](size_t index) noexcept {
            return index < branches.size() ?
                       branches[index] :
                       latches[index - branches.size()];
        };
        for (size_t i = 0u; i < branches.size(); ++i) {
            b.set_insertion_point(branches[i]);
            b.cond_br(
                m.create_undefined(Type::of<bool>()),
                tree_block(2u * i + 1u),
                tree_block(2u * i + 2u));
        }
        for (auto *latch : latches) {
            b.set_insertion_point(latch);
            b.br(header);
        }
        b.set_insertion_point(exit);
        b.return_void();

        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(value->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "local_load_elim_annotated_duplicate_is_retained"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        auto *first = b.load(Type::of<int>(), alloca);
        auto *annotated = b.load(Type::of<int>(), alloca);
        annotated->set_location("local_load.cpp", 23);
        auto *result = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {first, annotated});
        b.return_(result);

        auto info = local_load_elimination_pass_run_on_function(f);
        expect(info.removed_load_count == 0u);
        expect(annotated->is_linked());
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "local_load_elim_null_inputs_are_noops"_test = [] {
        expect(local_load_elimination_pass_run_on_function(nullptr)
                   .removed_load_count == 0u);
        PassReport report;
        expect(local_load_elimination_pass_run_on_module(nullptr, &report)
                   .removed_load_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- local_store_forward ----

void reg_local_store_forward() {

    "local_store_forward_load_after_store_forwarded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 1u);
    };

    "local_store_forward_no_store_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
    };

    "local_store_forward_nested_partial_store_blocks_uniform_forward"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner = Type::array(Type::of<float>(), 2u);
        auto *outer = Type::array(inner, 2u);
        auto *alloca = b.alloca_local(outer);
        float init_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto *init = m.create_constant(outer, init_data);
        b.store(alloca, init);
        uint32_t zero_value = 0u;
        uint32_t one_value = 1u;
        auto *zero = m.create_constant(Type::of<uint>(), &zero_value);
        auto *one = m.create_constant(Type::of<uint>(), &one_value);
        auto *row = b.gep(inner, alloca, {zero});
        auto *element = b.gep(Type::of<float>(), row, {one});
        b.store(element, m.create_constant_zero(Type::of<float>()));
        auto *load = b.load(outer, alloca);
        auto *ret = b.return_(load);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(ret->return_value() == load);
        expect(load->variable() == alloca);
    };

    "local_store_forward_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            int32_t val_v = 7;
            auto *val = m.create_constant(Type::of<int>(), &val_v);
            b.store(alloca, val);
            auto *ld = b.load(Type::of<int>(), alloca);
            b.return_(ld);
        }
        auto info = local_store_forward_pass_run_on_module(&m);
        expect(info.removed_load_count == 2u);
    };

    "local_store_forward_annotated_load_is_retained"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *value = m.create_constant_one(Type::of<int>());
        b.store(alloca, value);
        auto *load = b.load(Type::of<int>(), alloca);
        load->add_comment("load metadata needs a unique owner");
        b.return_(load);

        auto info = local_store_forward_pass_run_on_function(f);
        expect(info.removed_load_count == 0u);
        expect(load->is_linked());
        expect(load->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "local_store_forward_null_inputs_are_noops"_test = [] {
        expect(local_store_forward_pass_run_on_function(nullptr)
                   .removed_load_count == 0u);
        PassReport report;
        expect(local_store_forward_pass_run_on_module(nullptr, &report)
                   .removed_load_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- dead_store_elimination ----

void reg_dead_store_elimination() {

    "dse_overwritten_store_eliminated"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val1_v = 1, val2_v = 2;
        auto *val1 = m.create_constant(Type::of<int>(), &val1_v);
        auto *val2 = m.create_constant(Type::of<int>(), &val2_v);
        auto *store1 = b.store(alloca, val1);
        auto store1_locked = store1->lock();
        auto *store2 = b.store(alloca, val2);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = dead_store_elimination_pass_run_on_function(k);
        expect(info.eliminated_store_count == 1u);
        expect(store1_locked->use_list().empty());
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 1u);
        expect(store2->variable() == alloca);
        expect(store2->value() == val2);
        expect(ld->variable() == alloca);
    };

    "dse_no_dead_store_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = dead_store_elimination_pass_run_on_function(k);
        expect(info.eliminated_store_count == 0u);
    };

    "dse_retains_annotated_overwritten_store_in_one_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *first =
            b.store(alloca, m.create_constant_zero(Type::of<int>()));
        first->set_location("dse_metadata.cpp", 17);
        auto *second =
            b.store(alloca, m.create_constant_one(Type::of<int>()));
        b.load(Type::of<int>(), alloca);
        b.return_void();

        auto info = dead_store_elimination_pass_run_on_function(k);

        expect(info.eliminated_store_count == 0u);
        expect(first->is_linked());
        expect(second->is_linked());
        expect(first->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "dse_retains_annotated_overwritten_store_across_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *next = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *first =
            b.store(alloca, m.create_constant_zero(Type::of<int>()));
        first->add_comment("cross-block metadata owner");
        b.br(next);
        b.set_insertion_point(next);
        auto *second =
            b.store(alloca, m.create_constant_one(Type::of<int>()));
        b.load(Type::of<int>(), alloca);
        b.return_void();

        auto info = dead_store_elimination_pass_run_on_function(k);

        expect(info.eliminated_store_count == 0u);
        expect(first->is_linked());
        expect(second->is_linked());
        expect(first->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "dse_two_block_straight_line_cycle_terminates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *loop_block = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *store0 = b.store(alloca, m.create_constant_zero(Type::of<int>()));
        [[maybe_unused]] auto store0_lock = store0->lock();
        b.br(loop_block);
        b.set_insertion_point(loop_block);
        auto *store1 = b.store(alloca, m.create_constant_one(Type::of<int>()));
        b.br(body);

        auto info = dead_store_elimination_pass_run_on_function(k);

        expect(info.eliminated_store_count == 1u);
        expect(!store0->is_linked());
        expect(store1->is_linked());
        auto rerun = dead_store_elimination_pass_run_on_function(k);
        expect(rerun.eliminated_store_count == 0u);
        expect(store1->is_linked());
    };

    "dse_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            int32_t val1_v = 1, val2_v = 2;
            auto *val1 = m.create_constant(Type::of<int>(), &val1_v);
            auto *val2 = m.create_constant(Type::of<int>(), &val2_v);
            b.store(alloca, val1);
            b.store(alloca, val2);
            auto *ld = b.load(Type::of<int>(), alloca);
            b.return_(ld);
        }
        auto info = dead_store_elimination_pass_run_on_module(&m);
        expect(info.eliminated_store_count == 2u);
    };

    "dse_null_inputs_are_noops"_test = [] {
        expect(dead_store_elimination_pass_run_on_function(nullptr)
                   .eliminated_store_count == 0u);
        PassReport report;
        expect(dead_store_elimination_pass_run_on_module(nullptr, &report)
                   .eliminated_store_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- loop_rotation ----

void reg_loop_rotation() {

    "loop_rotation_rejects_structured_loop_without_mutation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        auto *phi = b.phi(Type::of<int>());
        int32_t bound_v = 4;
        auto *bound = m.create_constant(Type::of<int>(), &bound_v);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        b.cond_br(cond, loop_body, merge);

        b.set_insertion_point(loop_body);
        b.br(update);

        b.set_insertion_point(update);
        int32_t one_v = 1;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
        b.br(prepare);

        int32_t start_v = 0;
        auto *start = m.create_constant(Type::of<int>(), &start_v);
        phi->add_incoming(start, body);
        phi->add_incoming(inc, update);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = loop_rotation_pass_run_on_function(k);
        expect(info.rotated_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
        expect(loop->prepare_block() == prepare);
        expect(loop->merge_block() == merge);
    };

    "loop_rotation_no_loop_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = loop_rotation_pass_run_on_function(k);
        expect(info.rotated_loop_count == 0u);
        expect(info.succeeded());
    };

    "loop_rotation_module_runs_all_functions"_test = [] {
        Module m;
        for (int fn = 0; fn < 2; ++fn) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *merge = loop->create_merge_block();

            b.set_insertion_point(prepare);
            auto *phi = b.phi(Type::of<int>());
            int32_t bound_v = 3;
            auto *bound = m.create_constant(Type::of<int>(), &bound_v);
            auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
            b.cond_br(cond, loop_body, merge);

            b.set_insertion_point(loop_body);
            b.br(update);

            b.set_insertion_point(update);
            int32_t one_v = 1;
            auto *one = m.create_constant(Type::of<int>(), &one_v);
            auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
            b.br(prepare);

            int32_t start_v = 0;
            auto *start = m.create_constant(Type::of<int>(), &start_v);
            phi->add_incoming(start, body);
            phi->add_incoming(inc, update);

            b.set_insertion_point(merge);
            b.return_void();
        }
        auto info = loop_rotation_pass_run_on_module(&m);
        expect(info.rotated_loop_count == 0u);
        expect(info.structured_cfg_error_count == 2u);
    };
}

// ---- scalar_evolution ----

void reg_scalar_evolution() {

    "scev_argument_stride_and_rerun_are_current"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *stride_arg = k->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t bound_value = 8;
        auto *bound = m.create_constant(Type::of<int>(), &bound_value);
        b.set_insertion_point(prepare);
        auto *phi = b.phi(Type::of<int>(), {{zero, body}});
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        b.cond_br(cond, loop_body, merge);
        b.set_insertion_point(loop_body);
        auto *constant_sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, one});
        b.br(update);
        b.set_insertion_point(update);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, stride_arg});
        phi->add_incoming(inc, update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_(phi);

        auto first = scev_pass_run_on_function(k);
        expect(first.analyzed_loop_count == 1u);
        auto *phi_scev = scev_get_for_value(phi);
        expect(phi_scev != nullptr);
        expect(phi_scev->kind() == SCEV::Kind::ADD_REC);
        auto *add_rec = static_cast<const SCEVAddRec *>(phi_scev);
        expect(add_rec->stride()->kind() == SCEV::Kind::UNKNOWN);
        expect(static_cast<const SCEVUnknown *>(add_rec->stride())->value() == stride_arg);
        auto *sum_scev = scev_get_for_value(constant_sum);
        expect(sum_scev != nullptr);
        expect(sum_scev->kind() == SCEV::Kind::ADD);
        expect(static_cast<const SCEVAddExpr *>(sum_scev)->operands().size() == 2u);

        inc->set_operand(1u, one);
        auto second = scev_pass_run_on_function(k);
        expect(second.analyzed_loop_count == 1u);
        auto *updated = scev_get_for_value(phi);
        expect(updated != nullptr);
        expect(updated->kind() == SCEV::Kind::ADD_REC);
        auto *updated_rec = static_cast<const SCEVAddRec *>(updated);
        expect(updated_rec->stride()->kind() == SCEV::Kind::CONSTANT);
        expect(static_cast<const SCEVConstant *>(updated_rec->stride())->constant() == one);
    };
}

// ---- scalarizer ----

void reg_scalarizer() {

    "scalarizer_float3_add_scalarized"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        auto *f = m.create_callable(vec_t);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float a_data[3] = {1.0f, 2.0f, 3.0f};
        float b_data[3] = {4.0f, 5.0f, 6.0f};
        auto *a = m.create_constant(vec_t, a_data);
        auto *bv = m.create_constant(vec_t, b_data);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
        auto add_locked = add->lock();
        auto *ret = b.return_(add);
        auto info = scalarizer_pass_run_on_function(f);
        expect(info.scalarized_inst_count == 1u);
        expect(add_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::AGGREGATE);
        size_t scalar_add_count = 0u;
        size_t vector_add_count = 0u;
        f->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_ADD) {
                if (inst->type() == Type::of<float>()) { ++scalar_add_count; }
                if (inst->type() == vec_t) { ++vector_add_count; }
            }
        });
        expect(scalar_add_count == 3u);
        expect(vector_add_count == 0u);
        expect(block_local_defs_precede_uses(body));
    };

    "scalarizer_chained_vector_ops_preserve_ssa_order"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        auto *f = m.create_callable(vec_t);
        auto *x = f->create_value_argument(vec_t);
        auto *y = f->create_value_argument(vec_t);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {x, y});
        auto *mul = b.call(vec_t, ArithmeticOp::BINARY_MUL, {add, y});
        auto *ret = b.return_(mul);

        auto info = scalarizer_pass_run_on_function(f);

        expect(info.scalarized_inst_count == 2u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::AGGREGATE);
        expect(block_local_defs_precede_uses(body));

        size_t scalar_add_count = 0u;
        size_t scalar_mul_count = 0u;
        size_t vector_component_op_count = 0u;
        for (auto *inst : body->instructions()) {
            if (!inst->isa<ArithmeticInst>()) continue;
            auto *arith = static_cast<ArithmeticInst *>(inst);
            if (arith->op() == ArithmeticOp::BINARY_ADD) {
                if (arith->type() == Type::of<float>()) { ++scalar_add_count; }
                if (arith->type() == vec_t) { ++vector_component_op_count; }
            }
            if (arith->op() == ArithmeticOp::BINARY_MUL) {
                if (arith->type() == Type::of<float>()) { ++scalar_mul_count; }
                if (arith->type() == vec_t) { ++vector_component_op_count; }
            }
        }
        expect(scalar_add_count == 3u);
        expect(scalar_mul_count == 3u);
        expect(vector_component_op_count == 0u);
    };

    "scalarizer_no_vector_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = scalarizer_pass_run_on_function(k);
        expect(info.scalarized_inst_count == 0u);
    };

    "scalarizer_module_runs_all_functions"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            float a_data[3] = {1.0f, 2.0f, 3.0f};
            float b_data[3] = {4.0f, 5.0f, 6.0f};
            auto *a = m.create_constant(vec_t, a_data);
            auto *bv = m.create_constant(vec_t, b_data);
            auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_(add);
        }
        auto info = scalarizer_pass_run_on_module(&m);
        expect(info.scalarized_inst_count == 2u);
    };

    "scalarizer_dead_candidate_is_not_an_unreported_mutation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<float2>());
        auto *dead = b.call(
            Type::of<float2>(), ArithmeticOp::BINARY_ADD, {zero, zero});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = scalarizer_pass_run_on_function(k);

        expect(info.scalarized_inst_count == 0u);
        expect(dead->is_linked())
            << "a no-change result must not silently erase dead candidates";
        expect(xir_verify_module(&m).succeeded());
    };

    "scalarizer_mixed_live_and_dead_candidates_reports_exact_mutation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<float2>());
        auto *one = m.create_constant_one(Type::of<float2>());
        auto *dead = b.call(
            Type::of<float2>(), ArithmeticOp::BINARY_SUB, {zero, one});
        auto *live = b.call(
            Type::of<float2>(), ArithmeticOp::BINARY_ADD, {zero, one});
        auto *storage = b.alloca_local(Type::of<float2>());
        b.store(storage, live);
        b.return_void();
        auto live_lock = live->lock();
        expect(xir_verify_module(&m).succeeded());

        auto info = scalarizer_pass_run_on_function(k);

        expect(info.scalarized_inst_count == 1u);
        expect(dead->is_linked());
        expect(live_lock->use_list().empty());
        expect(xir_verify_module(&m).succeeded());
    };

    "scalarizer_annotated_vector_instruction_is_retained"_test = [] {
        Module m;
        auto *type = Type::of<float2>();
        auto *f = m.create_callable(type);
        auto *x = f->create_value_argument(type);
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(type);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *add = b.call(
            type, ArithmeticOp::BINARY_ADD, {x, one});
        add->add_comment("vector metadata has no single scalar lane owner");
        auto *ret = b.return_(add);

        auto info = scalarizer_pass_run_on_function(f);
        expect(info.scalarized_inst_count == 0u);
        expect(ret->return_value() == add);
        expect(add->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "scalarizer_null_inputs_are_noops"_test = [] {
        expect(scalarizer_pass_run_on_function(nullptr)
                   .scalarized_inst_count == 0u);
        PassReport report;
        expect(scalarizer_pass_run_on_module(nullptr, &report)
                   .scalarized_inst_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- phi_cleanup ----

void reg_phi_cleanup() {

    "phi_cleanup_trivial_phi_removed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *phi = b.phi(Type::of<int>(), {{val, body}});
        b.return_(phi);
        auto info = phi_cleanup_pass_run_on_function(k);
        expect(info.removed_phi_count == 1u);
    };

    "phi_cleanup_no_phi_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = phi_cleanup_pass_run_on_function(k);
        expect(info.removed_phi_count == 0u);
    };

    "phi_cleanup_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t val_v = 7;
            auto *val = m.create_constant(Type::of<int>(), &val_v);
            auto *phi = b.phi(Type::of<int>(), {{val, body}});
            b.return_(phi);
        }
        auto info = phi_cleanup_pass_run_on_module(&m);
        expect(info.removed_phi_count == 2u);
    };

    "phi_cleanup_zero_incoming_vector_phi_becomes_undef"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int2>());
        auto *entry = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *phi = b.phi(Type::of<int2>());
        auto *ret = b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info = phi_cleanup_pass_run_on_function(f);
        expect(info.removed_phi_count == 1u);
        expect(ret->return_value() != nullptr);
        expect(ret->return_value()->isa<Undefined>());
        expect(xir_verify_module(&m).succeeded());
    };

    "phi_cleanup_annotated_live_phi_is_retained"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        auto *merge = f->create_basic_block();
        auto *value = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(), {{value, entry}});
        phi->set_location("phi_cleanup.cpp", 31);
        auto *ret = b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info = phi_cleanup_pass_run_on_function(f);
        expect(info.removed_phi_count == 0u);
        expect(ret->return_value() == phi);
        expect(phi->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "phi_cleanup_null_inputs_are_noops"_test = [] {
        expect(phi_cleanup_pass_run_on_function(nullptr)
                   .removed_phi_count == 0u);
        PassReport report;
        expect(phi_cleanup_pass_run_on_module(nullptr, &report)
                   .removed_phi_count == 0u);
        expect(report.entries().size() == 1u);
    };
}

// ---- if_conversion ----

void reg_if_conversion() {

    "if_conversion_diamond_converted"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<int>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        auto *entry_branch =
            b.cond_br(cond, true_block, false_block);
        entry_branch->set_location("if_conversion.cpp", 10);

        b.set_insertion_point(true_block);
        int32_t t_v = 1;
        auto *t_val = m.create_constant(Type::of<int>(), &t_v);
        b.br(merge);

        b.set_insertion_point(false_block);
        int32_t f_v = 0;
        auto *f_val = m.create_constant(Type::of<int>(), &f_v);
        b.br(merge);

        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(), {{t_val, true_block}, {f_val, false_block}});
        b.return_(phi);

        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 1u);
        expect(info.replaced_phi_count == 1u);
        expect(info.succeeded());
        expect(body->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(body->terminator())->target_block() == merge);
        expect(phi->incoming_count() == 1u);
        expect(phi->incoming(0).block == body);
        expect(phi->incoming(0).value->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(phi->incoming(0).value)->op() == ArithmeticOp::SELECT);
        expect(count_reachable_blocks(k) == 2u);
        auto location_count = size_t{0u};
        for (auto *metadata : body->terminator()->metadata_list()) {
            location_count += metadata->isa<LocationMD>() ? 1u : 0u;
        }
        expect(location_count == 1u);
        auto verification = xir_verify_module(&m);
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown verifier failure" :
                    verification.errors.front().message.c_str());
    };

    "if_conversion_annotated_side_block_is_retained_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition =
            k->create_value_argument(Type::of<bool>());
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry =
            b.cond_br(condition, true_block, false_block);
        true_block->add_comment("metadata owner deleted by conversion");
        b.set_insertion_point(true_block);
        auto *true_exit = b.br(merge);
        b.set_insertion_point(false_block);
        auto *false_exit = b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto before = xir_to_text_translate(&m, true);

        auto info = if_conversion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(!info.changed());
        expect(before == after);
        expect(body->terminator() == entry);
        expect(true_block->terminator() == true_exit);
        expect(false_block->terminator() == false_exit);
        expect(xir_verify_module(&m).succeeded());
    };

    "if_conversion_annotated_arm_exit_is_retained_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *condition =
            k->create_value_argument(Type::of<bool>());
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *entry =
            b.cond_br(condition, true_block, false_block);
        b.set_insertion_point(true_block);
        auto *true_exit = b.br(merge);
        true_exit->add_comment("arm-exit metadata");
        b.set_insertion_point(false_block);
        auto *false_exit = b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto before = xir_to_text_translate(&m, true);

        auto info = if_conversion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(!info.changed());
        expect(before == after);
        expect(body->terminator() == entry);
        expect(true_block->terminator() == true_exit);
        expect(false_block->terminator() == false_exit);
        expect(xir_verify_module(&m).succeeded());
    };

    "if_conversion_no_diamond_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 0u);
    };

    "if_conversion_rejects_structured_if_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_undefined(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == if_inst);
        expect(if_inst->true_block() == true_block);
        expect(if_inst->false_block() == false_block);
        expect(if_inst->merge_block() == merge);
    };

    "if_conversion_module_rejection_is_atomic_across_functions"_test = [] {
        Module m;
        XIRBuilder b;

        BasicBlock *plain_body;
        auto *plain = make_kernel_with_body(m, plain_body);
        auto *condition = plain->create_value_argument(Type::of<bool>());
        auto *plain_true = plain->create_basic_block();
        auto *plain_false = plain->create_basic_block();
        auto *plain_merge = plain->create_basic_block();
        b.set_insertion_point(plain_body);
        auto *plain_entry =
            b.cond_br(condition, plain_true, plain_false);
        b.set_insertion_point(plain_true);
        auto *plain_true_exit = b.br(plain_merge);
        b.set_insertion_point(plain_false);
        auto *plain_false_exit = b.br(plain_merge);
        b.set_insertion_point(plain_merge);
        b.return_void();

        BasicBlock *structured_body;
        auto *structured =
            make_kernel_with_body(m, structured_body);
        b.set_insertion_point(structured_body);
        auto *structured_if =
            b.if_(m.create_constant_one(Type::of<bool>()));
        auto *structured_true =
            structured_if->create_true_block();
        auto *structured_false =
            structured_if->create_false_block();
        auto *structured_merge =
            structured_if->create_merge_block();
        b.set_insertion_point(structured_true);
        b.br(structured_merge);
        b.set_insertion_point(structured_false);
        b.br(structured_merge);
        b.set_insertion_point(structured_merge);
        b.return_void();

        auto info = if_conversion_pass_run_on_module(&m);
        expect(!info.succeeded());
        expect(info.structured_cfg_error_count == 1u);
        expect(!info.changed());
        expect(plain_body->terminator() == plain_entry);
        expect(plain_true->terminator() == plain_true_exit);
        expect(plain_false->terminator() == plain_false_exit);
        expect(structured_body->terminator() == structured_if);
    };

    "if_conversion_null_module_is_a_noop"_test = [] {
        auto info = if_conversion_pass_run_on_module(nullptr);
        expect(info.succeeded());
        expect(!info.changed());
    };

    "if_conversion_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *true_block = k->create_basic_block();
            auto *false_block = k->create_basic_block();
            auto *merge = k->create_basic_block();
            b.cond_br(cond, true_block, false_block);

            b.set_insertion_point(true_block);
            int32_t t_v = 1;
            auto *t_val = m.create_constant(Type::of<int>(), &t_v);
            b.br(merge);

            b.set_insertion_point(false_block);
            int32_t f_v = 0;
            auto *f_val = m.create_constant(Type::of<int>(), &f_v);
            b.br(merge);

            b.set_insertion_point(merge);
            auto *phi = b.phi(Type::of<int>(), {{t_val, true_block}, {f_val, false_block}});
            b.return_(phi);
        }
        auto info = if_conversion_pass_run_on_module(&m);
        expect(info.converted_diamond_count == 2u);
    };
}

// ---- reg2mem ----

void reg_reg2mem() {

    "reg2mem_lowers_phi"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        b.cond_br(m.create_undefined(Type::of<bool>()), true_block, false_block);
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(), {{one, true_block}, {two, false_block}});
        auto phi_locked = phi->lock();
        auto *final_add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
        b.return_(final_add);
        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_phi_count == 1u);
        expect(info.changed());
        auto audit = audit_reg2mem_spills_on_function(k);
        expect(audit.remaining_phi_spill_count == 1u);
        expect(audit.remaining_cross_block_spill_count == 0u);
        expect(!audit.succeeded());
        expect(phi_locked->use_list().empty());
        expect(count_reachable_insts(k, DerivedInstructionTag::PHI) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 3u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOAD) == 1u);
        expect(final_add->operand(0)->isa<LoadInst>());
        expect(final_add->operand(1) == one);
        for (auto instruction : body->instructions()) {
            if (instruction->isa<AllocaInst>()) {
                auto spill = instruction->find_metadata<Reg2MemSpillMD>();
                expect(spill != nullptr);
                if (spill != nullptr) {
                    expect(spill->kind() == Reg2MemSpillKind::PHI);
                }
            }
        }
    };

    "reg2mem_marks_cross_block_repair_spill"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<int>());
        auto *body = k->create_body_block();
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(m.create_undefined(Type::of<bool>()), true_block, false_block);
        auto *one = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(true_block);
        auto *value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                             {one, one});
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_(value);

        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_cross_block_value_count == 1u);
        expect(info.changed());
        auto audit = audit_reg2mem_spills_on_function(k);
        expect(audit.remaining_phi_spill_count == 0u);
        expect(audit.remaining_cross_block_spill_count == 1u);
        expect(!audit.succeeded());
        auto mem2reg = mem2reg_pass_run_on_function(k);
        expect(mem2reg.promoted_alloca_count == 1u);
        expect(audit_reg2mem_spills_on_function(k).succeeded());
        expect(count_reachable_insts(k, DerivedInstructionTag::PHI) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "reg2mem_alloca_hoisting_is_reported_and_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<int>());
        auto *value = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        auto *late_alloca = b.alloca_local(Type::of<int>());
        b.store(late_alloca, value);
        b.return_void();

        auto first = reg2mem_pass_run_on_function(k);
        expect(first.lowered_phi_count == 0u);
        expect(first.lowered_cross_block_value_count == 0u);
        expect(first.hoisted_alloca_count == 1u);
        expect(first.changed());
        expect(body->instructions().front() == late_alloca);

        auto second = reg2mem_pass_run_on_function(k);
        expect(second.hoisted_alloca_count == 0u);
        expect(!second.changed());
        expect(body->instructions().front() == late_alloca);
        expect(xir_verify_module(&m).succeeded());
    };

    "reg2mem_cross_block_rvalue_repair_preserves_phi_edge_uses"_test = [] {
        Module m;
        auto *k = m.create_callable(Type::of<int>());
        auto *body = k->create_body_block();
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(m.create_undefined(Type::of<bool>()), true_block, false_block);
        auto *one = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(true_block);
        auto *true_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                  {one, one});
        b.br(merge);
        b.set_insertion_point(false_block);
        auto *false_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB,
                                   {one, one});
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(),
                          {{true_value, true_block},
                           {false_value, false_block}});
        b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info =
            reg2mem_pass_repair_cross_block_rvalue_uses_on_function(k);
        expect(info.lowered_phi_count == 0u);
        expect(info.lowered_cross_block_value_count == 0u);
        expect(!info.changed());
        expect(count_reachable_insts(k, DerivedInstructionTag::PHI) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 0u);
        expect(audit_reg2mem_spills_on_function(k).succeeded());
        expect(xir_verify_module(&m).succeeded());
    };

    "reg2mem_audit_checks_all_owned_blocks_and_accepts_user_allocas"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *orphan = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.alloca_local(Type::of<int>());
        b.return_void();
        b.set_insertion_point(orphan);
        auto *spill = b.alloca_local(Type::of<float>());
        spill->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::CROSS_BLOCK);
        b.unreachable_();

        auto audit = audit_reg2mem_spills_on_function(k);
        expect(audit.remaining_phi_spill_count == 0u);
        expect(audit.remaining_cross_block_spill_count == 1u);
        expect(audit.remaining_spill_count() == 1u);
        expect(!audit.succeeded());
    };

    "reg2mem_audit_rejects_marker_on_invalid_owner"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *argument = k->create_value_argument(Type::of<int>());
        auto *constant = m.create_constant_one(Type::of<int>());
        auto *undefined = m.create_undefined(Type::of<float>());
        auto *special_register = m.create_thread_id();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *return_inst = b.return_void();
        static_cast<void>(m.create_metadata<Reg2MemSpillMD>());
        static_cast<void>(constant->create_metadata<Reg2MemSpillMD>());
        static_cast<void>(undefined->create_metadata<Reg2MemSpillMD>());
        static_cast<void>(special_register->create_metadata<Reg2MemSpillMD>());
        static_cast<void>(k->create_metadata<Reg2MemSpillMD>());
        static_cast<void>(argument->create_metadata<Reg2MemSpillMD>());
        static_cast<void>(body->create_metadata<Reg2MemSpillMD>());
        return_inst->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::PHI);

        auto function_audit = audit_reg2mem_spills_on_function(k);
        expect(function_audit.remaining_phi_spill_count == 0u);
        expect(function_audit.remaining_cross_block_spill_count == 0u);
        expect(function_audit.remaining_invalid_spill_count == 4u);
        expect(function_audit.remaining_spill_count() == 4u);
        expect(!function_audit.succeeded());

        auto module_audit = audit_reg2mem_spills_on_module(&m);
        expect(module_audit.remaining_phi_spill_count == 0u);
        expect(module_audit.remaining_cross_block_spill_count == 0u);
        expect(module_audit.remaining_invalid_spill_count == 8u);
        expect(module_audit.remaining_spill_count() == 8u);
        expect(!module_audit.succeeded());
    };

    "reg2mem_no_phi_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_phi_count == 0u);
    };

    "reg2mem_preserves_phi_metadata_on_reload_and_mem2reg_retains_it"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        auto *true_block = f->create_basic_block();
        auto *false_block = f->create_basic_block();
        auto *merge = f->create_basic_block();
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_undefined(Type::of<bool>()),
                  true_block, false_block);
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(
            Type::of<int>(),
            {{one, true_block}, {zero, false_block}});
        phi->set_location("reg2mem_metadata.cpp", 37);
        auto *ret = b.return_(phi);

        auto lowered = reg2mem_pass_run_on_function(f);
        expect(lowered.lowered_phi_count == 1u);
        expect(ret->return_value()->isa<LoadInst>());
        auto *reload = static_cast<LoadInst *>(ret->return_value());
        auto *location = reload->find_metadata<LocationMD>();
        expect(location != nullptr);
        if (location != nullptr) {
            expect(location->file() ==
                   luisa::filesystem::path{"reg2mem_metadata.cpp"});
            expect(location->line() == 37);
        }
        expect(count_reachable_insts(f, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(f, DerivedInstructionTag::STORE) == 3u);
        expect(xir_verify_module(&m).succeeded());

        auto promoted = mem2reg_pass_run_on_function(f);
        expect(!promoted.changed());
        expect(ret->return_value() == reload);
        expect(reload->is_linked());
        expect(reload->find_metadata<LocationMD>() != nullptr);
        expect(count_reachable_insts(f, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(f, DerivedInstructionTag::STORE) == 3u);
        expect(count_reachable_insts(f, DerivedInstructionTag::LOAD) == 1u);
        expect(xir_verify_module(&m).succeeded())
            << "mem2reg must not erase source metadata cloned from a Phi";
    };

    "reg2mem_mem2reg_roundtrip_recovers_phi_name"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        auto *true_block = f->create_basic_block();
        auto *false_block = f->create_basic_block();
        auto *merge = f->create_basic_block();
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_undefined(Type::of<bool>()),
                  true_block, false_block);
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(
            Type::of<int>(),
            {{one, true_block}, {zero, false_block}});
        phi->set_name("named_phi");
        auto *ret = b.return_(phi);

        auto lowered = reg2mem_pass_run_on_function(f);
        expect(lowered.lowered_phi_count == 1u);
        expect(ret->return_value()->isa<LoadInst>());
        auto *reload =
            static_cast<LoadInst *>(ret->return_value());
        auto reload_name = reload->name();
        expect(reload_name.has_value());
        if (reload_name.has_value()) {
            expect(*reload_name ==
                   luisa::string_view{"named_phi"});
        }

        auto promoted = mem2reg_pass_run_on_function(f);
        expect(promoted.promoted_alloca_count == 1u);
        expect(ret->return_value()->isa<PhiInst>());
        auto *recovered =
            static_cast<PhiInst *>(ret->return_value());
        auto recovered_name = recovered->name();
        expect(recovered_name.has_value());
        if (recovered_name.has_value()) {
            expect(*recovered_name ==
                   luisa::string_view{"named_phi"});
        }
        expect(audit_reg2mem_spills_on_function(f).succeeded());
        expect(xir_verify_module(&m).succeeded());
    };

    "reg2mem_null_and_bodyless_inputs_are_total"_test = [] {
        expect(!reg2mem_pass_run_on_function(nullptr).changed());
        expect(!reg2mem_pass_repair_cross_block_rvalue_uses_on_function(
                    nullptr)
                    .changed());
        expect(audit_reg2mem_spills_on_function(nullptr).succeeded());

        Module m;
        auto *declaration = m.create_callable(Type::of<int>());
        expect(!reg2mem_pass_run_on_function(declaration).changed());
        expect(audit_reg2mem_spills_on_function(declaration).succeeded());

        PassReport report;
        expect(audit_reg2mem_spills_on_module(nullptr, &report).succeeded());
        expect(report.entries().size() == 4u);
    };

    "reg2mem_lowers_phi_in_disconnected_owned_cfg_component"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        auto *orphan_entry = f->create_basic_block();
        auto *orphan_true = f->create_basic_block();
        auto *orphan_false = f->create_basic_block();
        auto *orphan_merge = f->create_basic_block();
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_(zero);
        b.set_insertion_point(orphan_entry);
        b.cond_br(m.create_undefined(Type::of<bool>()),
                  orphan_true, orphan_false);
        b.set_insertion_point(orphan_true);
        b.br(orphan_merge);
        b.set_insertion_point(orphan_false);
        b.br(orphan_merge);
        b.set_insertion_point(orphan_merge);
        auto *phi = b.phi(
            Type::of<int>(),
            {{one, orphan_true}, {zero, orphan_false}});
        b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info = reg2mem_pass_run_on_function(f);
        expect(info.lowered_phi_count == 1u);
        expect(info.changed());
        size_t owned_phi_count = 0u;
        for (auto *block : f->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                owned_phi_count += inst->isa<PhiInst>();
            }
        }
        expect(owned_phi_count == 0u);
        expect(xir_verify_module(&m, {.require_no_phi = true}).succeeded())
            << "reg2mem must lower PHIs in every block owned by the function, "
               "including disconnected but verifier-visible CFG components";
    };

    "reg2mem_lowers_phi_in_unreachable_structured_switch_merge"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *selector = f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1u);
        auto *default_block = switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        b.return_(one);
        b.set_insertion_point(default_block);
        b.return_(zero);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        b.return_(phi);

        expect(xir_verify_module(&m).succeeded());
        auto info = reg2mem_pass_run_on_module(&m);
        expect(info.lowered_phi_count == 1u);
        expect(info.changed());
        expect(xir_verify_module(&m, {.require_no_phi = true}).succeeded())
            << "a structured merge role remains verifier-visible even when "
               "no executable switch arm reaches it";
        expect(switch_inst->merge_block() == merge);
        expect(body->terminator() == switch_inst);
    };
}

// ---- sroa ----

void reg_sroa() {

    "sroa_decomposes_vector_alloca"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto struct_ty = Type::of<float2>();
        auto *alloca = b.alloca_local(struct_ty);
        alloca->set_name("vector_slot");
        alloca->set_location("sroa_metadata.cpp", 73);
        alloca->add_comment("first source comment");
        alloca->add_comment("second source comment");
        alloca->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::CROSS_BLOCK);
        float data[2] = {1.0f, 2.0f};
        auto *init = m.create_constant(struct_ty, data);
        b.store(alloca, init);
        auto *ld = b.load(struct_ty, alloca);
        b.return_(ld);
        auto info = sroa_pass_run_on_function(k, {.decompose_vectors = true});
        expect(info.decomposed_alloca_count == 1u);
        expect(info.inserted_alloca_count == 2u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 2u);
        auto audit = audit_reg2mem_spills_on_function(k);
        expect(audit.remaining_phi_spill_count == 0u);
        expect(audit.remaining_cross_block_spill_count == 2u);
        size_t replacement_index = 0u;
        for (auto *inst : body->instructions()) {
            if (!inst->isa<AllocaInst>()) { continue; }
            auto expected_name = luisa::format("vector_slot_{}", replacement_index++);
            expect(inst->name().has_value());
            expect(inst->name().value() == expected_name);
            auto *location = inst->find_metadata<LocationMD>();
            expect(location != nullptr);
            if (location != nullptr) {
                expect(location->file() == luisa::filesystem::path{"sroa_metadata.cpp"});
                expect(location->line() == 73);
            }
            auto comment_count = 0u;
            for (auto *metadata : inst->metadata_list()) {
                if (metadata->isa<CommentMD>()) { ++comment_count; }
            }
            expect(comment_count == 2u);
        }
        expect(replacement_index == 2u);
        auto *ret = static_cast<ReturnInst *>(body->terminator());
        expect(ret->return_value() != ld);
        expect(ret->return_value()->type() == struct_ty);
    };

    "sroa_no_struct_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = sroa_pass_run_on_function(k);
        expect(info.decomposed_alloca_count == 0u);
    };

    "sroa_aggressive_dynamic_top_level_index_rejected"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *index = f->create_value_argument(Type::of<uint>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *array_type = Type::array(Type::of<float>(), 4u);
        auto *alloca = b.alloca_local(array_type);
        auto *gep = b.gep(Type::of<float>(), alloca, {index});
        auto *load = b.load(Type::of<float>(), gep);
        auto *ret = b.return_(load);
        auto info = sroa_pass_run_on_function(f, {.aggressive = true});
        expect(info.decomposed_alloca_count == 0u);
        expect(count_reachable_insts(f, DerivedInstructionTag::ALLOCA) == 1u);
        expect(ret->return_value() == load);
        expect(load->variable() == gep);
        expect(gep->base() == alloca);
        expect(xir_verify_module(&m).succeeded());
    };

    "sroa_annotated_one_index_gep_is_rejected_atomically"_test = [] {
        auto run = [](uint32_t element, bool use_location) noexcept {
            Module m;
            auto *f = m.create_callable(Type::of<float>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *array_type = Type::array(Type::of<float>(), 2u);
            auto *alloca = b.alloca_local(array_type);
            auto *index =
                m.create_constant(Type::of<uint>(), &element);
            auto *gep =
                b.gep(Type::of<float>(), alloca, {index});
            if (use_location) {
                gep->set_location("sroa_scalar_gep.cpp", 29);
            } else {
                gep->add_comment(
                    "one-index GEP has no unique replacement owner");
            }
            auto *load = b.load(Type::of<float>(), gep);
            b.return_(load);
            expect(xir_verify_module(&m).succeeded());
            auto before = xir_to_text_translate(&m, true);

            auto info = sroa_pass_run_on_function(f);

            auto after = xir_to_text_translate(&m, true);
            expect(!info.changed());
            expect(before == after);
            expect(gep->is_linked());
            expect(load->variable() == gep);
            expect(xir_verify_module(&m).succeeded());
        };
        // Distinct elements and metadata kinds exercise the same many-source
        // GEP-to-one-element-alloca ambiguity.
        run(0u, true);
        run(1u, false);
    };

    "sroa_constant_outer_dynamic_inner_index_is_safe"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *inner_index = k->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner_type = Type::array(Type::of<float>(), 4u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *alloca = b.alloca_local(outer_type);
        uint32_t zero_value = 0u;
        auto *zero = m.create_constant(Type::of<uint>(), &zero_value);
        auto *gep = b.gep(Type::of<float>(), alloca, {zero, inner_index});
        gep->set_location("nested_gep.cpp", 19);
        gep->add_comment("preserve replacement GEP metadata");
        auto *load = b.load(Type::of<float>(), gep);
        b.return_(load);
        auto info = sroa_pass_run_on_function(k);
        expect(info.decomposed_alloca_count == 1u);
        expect(info.inserted_alloca_count == 2u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 2u);
        expect(load->variable()->isa<GEPInst>());
        auto *new_gep = static_cast<GEPInst *>(load->variable());
        expect(new_gep->base()->isa<AllocaInst>());
        expect(new_gep->index_count() == 1u);
        expect(new_gep->index(0) == inner_index);
        auto *location = new_gep->find_metadata<LocationMD>();
        expect(location != nullptr);
        if (location != nullptr) {
            expect(location->file() == luisa::filesystem::path{"nested_gep.cpp"});
            expect(location->line() == 19);
        }
        expect(new_gep->find_metadata<CommentMD>() != nullptr);
    };

    "sroa_constant_top_level_indices_accept_all_integer_widths"_test = [] {
        auto run = [](const Type *index_type, const void *index_data) noexcept {
            Module m;
            auto *k = m.create_callable(Type::of<float>());
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *array_type = Type::array(Type::of<float>(), 2u);
            auto *alloca = b.alloca_local(array_type);
            auto *index = m.create_constant(index_type, index_data);
            auto *gep = b.gep(Type::of<float>(), alloca, {index});
            auto *load = b.load(Type::of<float>(), gep);
            auto *ret = b.return_(load);
            expect(xir_verify_module(&m).succeeded());

            auto info = sroa_pass_run_on_function(k);
            expect(info.decomposed_alloca_count == 1u);
            expect(info.inserted_alloca_count == 2u);
            expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 2u);
            expect(ret->return_value() == load);
            expect(load->variable()->isa<AllocaInst>());
            expect(load->variable() != alloca);
            expect(xir_verify_module(&m).succeeded());
        };
        int8_t i8 = 1;
        uint8_t u8 = 1u;
        int16_t i16 = 1;
        uint16_t u16 = 1u;
        int32_t i32 = 1;
        uint32_t u32 = 1u;
        int64_t i64 = 1;
        uint64_t u64 = 1u;
        run(Type::of<int8_t>(), &i8);
        run(Type::of<uint8_t>(), &u8);
        run(Type::of<int16_t>(), &i16);
        run(Type::of<uint16_t>(), &u16);
        run(Type::of<int32_t>(), &i32);
        run(Type::of<uint32_t>(), &u32);
        run(Type::of<int64_t>(), &i64);
        run(Type::of<uint64_t>(), &u64);
    };
}

// ---- inline ----

void reg_inline() {

    "inline_callable_inlined"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.return_(val);

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *call = b.call(Type::of<int>(), callee, {});
        auto call_locked = call->lock();
        auto *ret = b.return_(call);

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(call_locked->use_list().empty());
        expect(ret->return_value() == val);
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 0u);
        expect(count_reachable_insts(caller, DerivedInstructionTag::BRANCH) == 0u);
    };

    "inline_single_use_large_callable_preserves_compiler_partition"_test = [] {
        Module m;
        XIRBuilder b;
        auto *callee = m.create_callable(Type::of<uint>());
        auto *argument =
            callee->create_value_argument(Type::of<uint>());
        b.set_insertion_point(callee->create_body_block());
        auto *one = m.create_constant_one(Type::of<uint>());
        Value *value = argument;
        for (auto i = 0u;
             i < default_inline_single_use_instruction_budget;
             ++i) {
            value = b.call(Type::of<uint>(),
                           ArithmeticOp::BINARY_ADD,
                           {value, one});
        }
        b.return_(value);

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_argument =
            kernel->create_value_argument(Type::of<uint>());
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        auto *call = b.call(Type::of<uint>(), callee,
                            {kernel_argument});
        b.store(storage, call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto ordinary = inline_pass_run_on_module(&m);
        expect(!ordinary.changed());
        expect(ordinary.skipped_costly_callable_count == 1u);
        expect(call->is_linked());
        expect(callee->parent_module() == &m);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 1u);

        // A legality-driven caller can still explicitly select the same site.
        std::array<CallInst *, 1u> selected_calls{call};
        auto selected = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected_calls});
        expect(selected.inlined_call_count == 1u);
        expect(selected.skipped_costly_callable_count == 0u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_pass_reuses_one_dense_layout_per_callee_version"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<uint>());
        auto *argument =
            callee->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(callee->create_body_block());
        auto *one = m.create_constant_one(Type::of<uint>());
        Value *value = argument;
        constexpr auto chain_length = 32u;
        for (auto i = 0u; i < chain_length; ++i) {
            value = b.call(Type::of<uint>(),
                           ArithmeticOp::BINARY_ADD,
                           {value, one});
        }
        b.return_(value);

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_argument =
            kernel->create_value_argument(Type::of<uint>());
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        constexpr auto call_count = 3u;
        for (auto i = 0u; i < call_count; ++i) {
            auto *call = b.call(Type::of<uint>(), callee,
                                {kernel_argument});
            b.store(storage, call);
        }
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == call_count);
        expect(info.removed_callable_count == 1u);
        expect(info.inline_pass_summary_function_count == 1u);
        expect(info.inline_pass_summary_instruction_scan_count ==
               chain_length + 1u);
        expect(info.inline_pass_clone_layout_function_count == 1u);
        expect(info.inline_pass_clone_layout_value_count ==
               chain_length + 3u);
        expect(info.inline_pass_dense_resolver_apply_count == call_count);
        expect(info.inline_pass_dense_resolver_fallback_count == 0u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_pass_rebuilds_layout_after_prior_caller_mutation"_test = [] {
        Module m;
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<uint>());

        auto *inner = m.create_callable(Type::of<uint>());
        auto *inner_argument =
            inner->create_value_argument(Type::of<uint>());
        b.set_insertion_point(inner->create_body_block());
        b.return_(b.call(Type::of<uint>(),
                         ArithmeticOp::BINARY_ADD,
                         {inner_argument, one}));

        auto *middle = m.create_callable(Type::of<uint>());
        auto *middle_argument =
            middle->create_value_argument(Type::of<uint>());
        b.set_insertion_point(middle->create_body_block());
        b.return_(b.call(Type::of<uint>(), inner,
                         {middle_argument}));

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_argument =
            kernel->create_value_argument(Type::of<uint>());
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        auto *call = b.call(Type::of<uint>(), middle,
                            {kernel_argument});
        b.store(storage, call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        // `inner` is processed first and mutates `middle`. The version for
        // `middle` must be summarized and numbered only after that mutation;
        // no layout may survive across the version boundary.
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 2u);
        expect(info.removed_callable_count == 2u);
        expect(info.inline_pass_summary_function_count == 2u);
        expect(info.inline_pass_summary_instruction_scan_count == 4u);
        expect(info.inline_pass_clone_layout_function_count == 2u);
        expect(info.inline_pass_clone_layout_value_count == 8u);
        expect(info.inline_pass_dense_resolver_apply_count == 2u);
        expect(info.inline_pass_dense_resolver_fallback_count == 0u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_pass_caches_caller_barriers_across_mutations"_test = [] {
        Module m;
        XIRBuilder b;

        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_condition =
            callee->create_value_argument(Type::of<bool>());
        auto *callee_entry = callee->create_body_block();
        auto *callee_left = callee->create_basic_block();
        auto *callee_right = callee->create_basic_block();
        b.set_insertion_point(callee_entry);
        b.cond_br(callee_condition, callee_left, callee_right);
        b.set_insertion_point(callee_left);
        b.return_(m.create_constant_one(Type::of<int>()));
        b.set_insertion_point(callee_right);
        b.return_(m.create_constant_zero(Type::of<int>()));

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *kernel_value =
            kernel->create_value_argument(Type::of<uint>());
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<uint>());
        Value *filler = kernel_value;
        constexpr auto filler_instruction_count = 64u;
        for (auto i = 0u; i < filler_instruction_count; ++i) {
            filler = b.call(Type::of<uint>(),
                            ArithmeticOp::BINARY_ADD,
                            {filler, one});
        }
        constexpr auto call_count = 3u;
        for (auto i = 0u; i < call_count; ++i) {
            auto *call = b.call(Type::of<int>(), callee,
                                {kernel_condition});
            b.store(storage, call);
        }
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        // The initial caller has one alloca, 64 arithmetic instructions,
        // three calls, three stores, and one return. Successful inlining
        // mutates it after the first query, but cannot introduce a barrier;
        // the remaining two queries must therefore hit the same cache entry.
        constexpr auto initial_caller_instruction_count =
            1u + filler_instruction_count + call_count + call_count + 1u;
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == call_count);
        expect(info.removed_callable_count == 1u);
        expect(info.inline_pass_summary_function_count == 1u);
        expect(info.inline_pass_summary_instruction_scan_count == 3u);
        expect(info.inline_pass_clone_layout_function_count == 1u);
        expect(info.inline_pass_clone_layout_value_count == 7u);
        expect(info.inline_pass_dense_resolver_apply_count == call_count);
        expect(info.inline_pass_dense_resolver_fallback_count == 0u);
        expect(info.inline_pass_caller_barrier_function_count == 1u);
        expect(info.inline_pass_caller_barrier_instruction_scan_count ==
               initial_caller_instruction_count);
        expect(info.inline_pass_caller_barrier_cache_hit_count ==
               call_count - 1u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_recursive_callable_is_skipped"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *self_call = b.call(Type::of<int>(), callee, {});
        b.return_(self_call);
        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *call = b.call(Type::of<int>(), callee, {});
        b.return_(call);
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.removed_callable_count == 0u);
        expect(info.skipped_recursive_callable_count == 1u);
        expect(call->is_linked());
        expect(self_call->is_linked());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
    };

    "inline_recursion_in_disconnected_owned_block_is_skipped"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        auto *entry = callee->create_body_block();
        auto *disconnected = callee->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.return_void();
        b.set_insertion_point(disconnected);
        auto *self_call = b.call(nullptr, callee, {});
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        auto *call = b.call(nullptr, callee, {});
        b.return_void();

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_recursive_callable_count == 1u);
        expect(call->is_linked());
        expect(self_call->is_linked());
        expect(callee->is_linked());
        expect(call->callee() == callee);
        expect(self_call->callee() == callee);
    };

    "inline_mutually_recursive_scc_is_skipped"_test = [] {
        Module m;
        auto *left = m.create_callable(nullptr);
        auto *right = m.create_callable(nullptr);
        XIRBuilder b;
        b.set_insertion_point(left->create_body_block());
        auto *left_call = b.call(nullptr, right, {});
        b.return_void();
        b.set_insertion_point(right->create_body_block());
        auto *right_call = b.call(nullptr, left, {});
        b.return_void();

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_recursive_callable_count == 2u);
        expect(info.recursion_analysis_function_count == 2u);
        expect(info.recursion_analysis_call_use_visit_count == 2u);
        expect(info.recursion_analysis_edge_count == 2u);
        expect(info.recursion_analysis_vertex_visit_count == 4u);
        expect(info.recursion_analysis_edge_visit_count == 4u);
        expect(left_call->is_linked());
        expect(right_call->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_recursion_graph_ignores_non_callee_function_use"_test = [] {
        Module m;
        auto *payload = m.create_callable(nullptr);
        auto *consumer = m.create_callable(nullptr);
        consumer->create_value_argument(Type::of<int>());
        auto *caller = m.create_callable(nullptr);
        XIRBuilder b;
        b.set_insertion_point(payload->create_body_block());
        b.return_void();
        b.set_insertion_point(consumer->create_body_block());
        b.return_void();
        b.set_insertion_point(caller->create_body_block());
        // Deliberately verifier-invalid: payload is an ordinary argument use,
        // not another call-graph edge. The recursion analysis must inspect
        // operand identity before assigning the use to caller -> payload.
        auto *call = b.call(nullptr, consumer, {payload});
        b.return_void();

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_recursive_callable_count == 0u);
        expect(info.rejected_malformed_call_count == 1u);
        expect(info.recursion_analysis_function_count == 3u);
        expect(info.recursion_analysis_call_use_visit_count == 2u);
        expect(info.recursion_analysis_edge_count == 1u);
        expect(info.recursion_analysis_vertex_visit_count == 6u);
        expect(info.recursion_analysis_edge_visit_count == 2u);
        expect(call->is_linked());
    };

    "inline_recursion_analysis_is_linear_in_sparse_call_graph"_test = [] {
        Module m;
        constexpr auto callable_count = 128u;
        luisa::vector<luisa::compute::xir::Function *> callables;
        luisa::vector<BasicBlock *> bodies;
        callables.reserve(callable_count);
        bodies.reserve(callable_count);
        for (auto i = 0u; i < callable_count; ++i) {
            auto *callable = m.create_callable(nullptr);
            callables.emplace_back(callable);
            bodies.emplace_back(callable->create_body_block());
        }
        XIRBuilder b;
        for (auto i = 0u; i < callable_count; ++i) {
            b.set_insertion_point(bodies[i]);
            if (i + 1u != callable_count) {
                b.call(nullptr, callables[i + 1u], {});
            }
            b.return_void();
        }

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        b.set_insertion_point(kernel_body);
        auto *selected_call = b.call(nullptr, callables.front(), {});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{selected_call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(info.skipped_recursive_callable_count == 0u);
        expect(info.recursion_analysis_function_count == callable_count);
        expect(info.recursion_analysis_call_use_visit_count ==
               callable_count);
        expect(info.recursion_analysis_edge_count == callable_count - 1u);
        expect(info.recursion_analysis_vertex_visit_count ==
               2u * callable_count);
        expect(info.recursion_analysis_edge_visit_count ==
               2u * (callable_count - 1u));
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_single_block_callee_preserves_structured_caller"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *one = m.create_constant_one(Type::of<int>());
        b.return_(one);
        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<int>(), callee, {});
        auto call_locked = call->lock();
        auto *if_inst = b.if_(m.create_undefined(Type::of<bool>()));
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        auto *true_ret = b.return_(call);
        b.set_insertion_point(false_block);
        b.return_(m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(merge);
        b.unreachable_();
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 1u);
        expect(info.skipped_structured_call_count == 0u);
        expect(call_locked->use_list().empty());
        expect(body->terminator() == if_inst);
        expect(if_inst->true_block() == true_block);
        expect(if_inst->false_block() == false_block);
        expect(if_inst->merge_block() == merge);
        expect(true_ret->return_value() == one);
        expect(count_reachable_insts(caller, DerivedInstructionTag::BRANCH) == 0u);
    };

    "inline_multiblock_callee_rejected_in_structured_caller"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<bool>());
        auto *entry = callee->create_body_block();
        auto *left = callee->create_basic_block();
        auto *right = callee->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_undefined(Type::of<bool>()), left, right);
        b.set_insertion_point(left);
        b.return_(m.create_constant_one(Type::of<bool>()));
        b.set_insertion_point(right);
        b.return_(m.create_constant_zero(Type::of<bool>()));
        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<bool>(), callee, {});
        auto *if_inst = b.if_(call);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_structured_call_count == 1u);
        expect(call->is_linked());
        expect(body->terminator() == if_inst);
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_callee_succeeds_after_destructure"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<bool>());
        auto *entry = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *callee_if = b.if_(m.create_undefined(Type::of<bool>()));
        auto *callee_true = callee_if->create_true_block();
        auto *callee_false = callee_if->create_false_block();
        auto *callee_merge = callee_if->create_merge_block();
        b.set_insertion_point(callee_true);
        b.br(callee_merge);
        b.set_insertion_point(callee_false);
        b.br(callee_merge);
        b.set_insertion_point(callee_merge);
        b.return_(m.create_constant_one(Type::of<bool>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<bool>(), callee, {});
        auto call_locked = call->lock();
        auto *if_inst = b.if_(call);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
        auto destructure = destructure_cfg_pass_run_on_module(&m);
        expect(destructure.succeeded());
        expect(destructure.destructured_if_count == 2u);
        expect(xir_verify_module(&m).succeeded());

        auto after = inline_pass_run_on_module(&m);
        expect(after.inlined_call_count == 1u);
        expect(after.skipped_structured_call_count == 0u);
        expect(after.removed_callable_count == 1u);
        expect(call_locked->use_list().empty());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_no_call_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
    };

    "inline_null_entry_points_are_total_and_report_zero"_test = [] {
        PassReport report;
        expect(!inline_pass_run_on_module(nullptr, &report).changed());
        expect(report.entries().size() == 32u);
        report.clear();
        expect(!inline_all_pass_run_on_module(nullptr, &report).changed());
        expect(report.entries().size() == 32u);
        report.clear();
        expect(!inline_call_sites_pass_run_on_module(
                    nullptr, luisa::span<CallInst *const>{}, {}, &report)
                    .changed());
        expect(report.entries().size() == 32u);
    };

    "inline_bodyless_callable_declaration_is_never_inlined"_test = [] {
        Module m;
        auto *declaration = m.create_callable(nullptr);
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *call = b.call(nullptr, declaration, {});
        auto *ret = b.return_void();

        auto ordinary = inline_pass_run_on_module(&m);
        expect(!ordinary.changed());
        expect(ordinary.skipped_declaration_call_count == 1u);
        expect(call->parent_block() == body);
        expect(body->terminator() == ret);
        expect(call->callee() == declaration);
        expect(declaration->parent_module() == &m);

        auto all = inline_all_pass_run_on_module(&m);
        expect(!all.changed());
        expect(all.skipped_declaration_call_count == 1u);
        expect(call->parent_block() == body);
        expect(body->terminator() == ret);

        std::array<CallInst *, 1u> selected_calls{call};
        auto selected = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected_calls});
        expect(!selected.changed());
        expect(selected.skipped_declaration_call_count == 1u);
        expect(selected.rejected_malformed_call_count == 0u);
        expect(call->parent_block() == body);
        expect(body->terminator() == ret);
        expect(call->callee() == declaration);
    };

    "inline_signature_constrained_callable_is_retained"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        static_cast<void>(
            callee->create_metadata<SignatureConstraintMD>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_(m.create_constant_one(Type::of<int>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call = b.call(Type::of<int>(), callee, {});
        b.store(storage, call);
        b.return_void();

        auto info = inline_pass_run_on_module(&m);
        expect(!info.changed());
        expect(info.skipped_constrained_call_count == 1u);
        expect(call->is_linked());
        expect(callee->is_linked());
        expect(count_reachable_insts(
                   caller, DerivedInstructionTag::CALL) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_unmappable_call_and_return_metadata_are_retained"_test = [] {
        auto run = [](bool annotate_call) noexcept {
            Module m;
            auto *callee = m.create_callable(Type::of<int>());
            auto *callee_body = callee->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(callee_body);
            auto *callee_return =
                b.return_(m.create_constant_one(Type::of<int>()));

            BasicBlock *body;
            auto *caller = make_kernel_with_body(m, body);
            b.set_insertion_point(body);
            auto *storage = b.alloca_local(Type::of<int>());
            auto *call = b.call(Type::of<int>(), callee, {});
            b.store(storage, call);
            b.return_void();
            if (annotate_call) {
                call->add_comment("call metadata owner");
            } else {
                callee_return->add_comment("return metadata owner");
            }
            auto before = xir_to_text_translate(&m, true);

            auto info = inline_pass_run_on_module(&m);
            auto after = xir_to_text_translate(&m, true);
            expect(!info.changed());
            expect(info.skipped_metadata_call_count == 1u);
            expect(before == after);
            expect(call->is_linked());
            expect(count_reachable_insts(
                       caller, DerivedInstructionTag::CALL) == 1u);
            expect(xir_verify_module(&m).succeeded());
        };
        run(true);
        run(false);
    };

    "inline_selected_consumes_only_diagnostic_call_metadata_atomically"_test = [] {
        {
            Module m;
            auto *callee = m.create_callable(Type::of<int>());
            auto *callee_body = callee->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(callee_body);
            b.return_(m.create_constant_one(Type::of<int>()));

            BasicBlock *body;
            auto *caller = make_kernel_with_body(m, body);
            b.set_insertion_point(body);
            auto *storage = b.alloca_local(Type::of<int>());
            auto *call = b.call(Type::of<int>(), callee, {});
            call->set_name("mandatory_inline_call");
            call->set_location("mandatory_inline.cpp", 41);
            call->add_comment("source-only call annotation");
            b.store(storage, call);
            b.return_void();

            std::array<CallInst *, 1u> selected_calls{call};
            auto info = inline_call_sites_pass_run_on_module(
                &m, luisa::span{selected_calls},
                {.consume_call_site_diagnostic_metadata = true});
            expect(eq(info.inlined_call_count, 1u));
            expect(eq(
                info.consumed_call_site_diagnostic_metadata_count, 3u));
            expect(eq(info.skipped_metadata_call_count, 0u));
            expect(eq(count_reachable_insts(
                          caller, DerivedInstructionTag::CALL),
                      0u));
            expect(xir_verify_module(&m).succeeded());
        }
        {
            Module m;
            auto *callee = m.create_callable(Type::of<int>());
            auto *callee_body = callee->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(callee_body);
            b.return_(m.create_constant_one(Type::of<int>()));

            BasicBlock *body;
            auto *caller = make_kernel_with_body(m, body);
            b.set_insertion_point(body);
            auto *first_storage = b.alloca_local(Type::of<int>());
            auto *second_storage = b.alloca_local(Type::of<int>());
            auto *diagnostic_call =
                b.call(Type::of<int>(), callee, {});
            diagnostic_call->add_comment("admissible only as a group");
            b.store(first_storage, diagnostic_call);
            auto *semantic_call =
                b.call(Type::of<int>(), callee, {});
            semantic_call->metadata_list().push_front(
                luisa::make_managed<Reg2MemSpillMD>(
                    Reg2MemSpillKind::CROSS_BLOCK));
            b.store(second_storage, semantic_call);
            b.return_void();
            auto before = xir_to_text_translate(&m, true);

            std::array<CallInst *, 2u> selected_calls{
                diagnostic_call, semantic_call};
            auto info = inline_call_sites_pass_run_on_module(
                &m, luisa::span{selected_calls},
                {.consume_call_site_diagnostic_metadata = true});
            auto after = xir_to_text_translate(&m, true);
            expect(!info.changed());
            expect(eq(info.inlined_call_count, 0u));
            expect(eq(
                info.consumed_call_site_diagnostic_metadata_count, 0u));
            expect(eq(info.skipped_metadata_call_count, 1u));
            expect(before == after);
            expect(eq(count_reachable_insts(
                          caller, DerivedInstructionTag::CALL),
                      2u));
            expect(xir_verify_module(&m).succeeded());
        }
    };

    "inline_single_block_with_block_metadata_is_rejected_without_mutation"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        callee_body->add_comment("single-block metadata owner");
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_(m.create_constant_one(Type::of<int>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call = b.call(Type::of<int>(), callee, {});
        b.store(storage, call);
        b.return_void();
        auto before = xir_to_text_translate(&m, true);

        auto info = inline_all_pass_run_on_module(&m);
        auto after = xir_to_text_translate(&m, true);
        expect(!info.changed());
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_metadata_call_count == 1u);
        expect(before == after);
        expect(call->is_linked());
        expect(callee->is_linked());
        auto *comment = callee_body->find_metadata<CommentMD>();
        expect(comment != nullptr);
        expect(comment->comment() == "single-block metadata owner");
        expect(count_reachable_insts(
                   caller, DerivedInstructionTag::CALL) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_clones_basic_block_metadata"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *condition =
            callee->create_value_argument(Type::of<bool>());
        auto *entry = callee->create_body_block();
        auto *left = callee->create_basic_block();
        auto *right = callee->create_basic_block();
        left->add_comment("cloned inline block metadata");
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.return_(m.create_constant_one(Type::of<int>()));
        b.set_insertion_point(right);
        b.return_(m.create_constant_zero(Type::of<int>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        auto *caller_condition =
            caller->create_value_argument(Type::of<bool>());
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call =
            b.call(Type::of<int>(), callee, {caller_condition});
        b.store(storage, call);
        b.return_void();

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 1u);
        expect(info.call_site_clone_layout_function_count == 1u);
        expect(info.call_site_clone_layout_value_count == 7u);
        expect(info.call_site_dense_resolver_apply_count == 1u);
        expect(info.call_site_dense_resolver_fallback_count == 0u);
        auto annotated_block_count = size_t{0u};
        for (auto *block : caller->basic_blocks()) {
            auto *comment = block->find_metadata<CommentMD>();
            if (comment != nullptr &&
                comment->comment() ==
                    "cloned inline block metadata") {
                annotated_block_count++;
            }
        }
        expect(annotated_block_count == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_calls_share_immutable_function_summary"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<uint>());
        auto *callee_argument =
            callee->create_value_argument(Type::of<uint>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *one = m.create_constant_one(Type::of<uint>());
        Value *value = callee_argument;
        constexpr auto chain_length = 64u;
        for (auto i = 0u; i < chain_length; ++i) {
            value = b.call(Type::of<uint>(),
                           ArithmeticOp::BINARY_ADD,
                           {value, one});
        }
        b.return_(value);

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        auto *caller_argument =
            caller->create_value_argument(Type::of<uint>());
        b.set_insertion_point(caller_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        constexpr auto call_count = 32u;
        luisa::vector<CallInst *> selected;
        selected.reserve(call_count);
        for (auto i = 0u; i < call_count; ++i) {
            auto *call = b.call(Type::of<uint>(), callee,
                                {caller_argument});
            selected.emplace_back(call);
            b.store(storage, call);
        }
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == call_count);
        expect(info.removed_callable_count == 1u);
        expect(info.call_site_summary_function_count == 1u);
        expect(info.call_site_summary_instruction_scan_count ==
               chain_length + 1u);
        expect(info.call_site_cached_apply_count == call_count);
        expect(info.call_site_revalidated_apply_count == 0u);
        expect(info.call_site_clone_layout_function_count == 1u);
        expect(info.call_site_clone_layout_value_count ==
               chain_length + 3u);
        expect(info.call_site_dense_resolver_apply_count == call_count);
        expect(info.call_site_dense_resolver_fallback_count == 0u);
        expect(count_reachable_insts(
                   caller, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_call_revalidates_a_mutated_callee"_test = [] {
        Module m;
        XIRBuilder b;

        auto *inner = m.create_callable(Type::of<uint>());
        auto *inner_argument =
            inner->create_value_argument(Type::of<uint>());
        auto *inner_body = inner->create_body_block();
        b.set_insertion_point(inner_body);
        auto *one = m.create_constant_one(Type::of<uint>());
        b.return_(b.call(Type::of<uint>(),
                         ArithmeticOp::BINARY_ADD,
                         {inner_argument, one}));

        auto *middle = m.create_callable(Type::of<uint>());
        auto *middle_argument =
            middle->create_value_argument(Type::of<uint>());
        auto *middle_body = middle->create_body_block();
        b.set_insertion_point(middle_body);
        auto *inner_call =
            b.call(Type::of<uint>(), inner, {middle_argument});
        b.return_(inner_call);

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_argument =
            kernel->create_value_argument(Type::of<uint>());
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        auto *middle_call =
            b.call(Type::of<uint>(), middle, {kernel_argument});
        b.store(storage, middle_call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        // Preflight sees the original definitions. Applying the first entry
        // mutates `middle`, so its cached summary must be invalidated before
        // the second entry uses `middle` as a callee.
        std::array<CallInst *, 2u> selected{
            inner_call, middle_call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 2u);
        expect(info.removed_callable_count == 2u);
        expect(info.call_site_summary_function_count == 2u);
        expect(info.call_site_cached_apply_count == 1u);
        expect(info.call_site_revalidated_apply_count == 1u);
        expect(info.call_site_clone_layout_function_count == 1u);
        expect(info.call_site_clone_layout_value_count == 4u);
        expect(info.call_site_dense_resolver_apply_count == 1u);
        expect(info.call_site_dense_resolver_fallback_count == 0u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_dense_resolver_preserves_unowned_value_fallback"_test = [] {
        Module m;
        XIRBuilder b;

        auto *foreign = m.create_callable(Type::of<uint>());
        auto *foreign_argument =
            foreign->create_value_argument(Type::of<uint>());
        b.set_insertion_point(foreign->create_body_block());
        b.return_(foreign_argument);

        auto *callee = m.create_callable(Type::of<uint>());
        b.set_insertion_point(callee->create_body_block());
        // Deliberately verifier-invalid input: the return references a local
        // value owned by another function. The historical resolver repairs
        // an unnumbered typed local to a module undefined value; dense
        // numbering must retain that total fallback rather than indexing it.
        b.return_(foreign_argument);

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        b.set_insertion_point(kernel_body);
        auto *storage = b.alloca_local(Type::of<uint>());
        auto *call = b.call(Type::of<uint>(), callee, {});
        auto *store = b.store(storage, call);
        b.return_void();

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(info.call_site_clone_layout_function_count == 1u);
        expect(info.call_site_clone_layout_value_count == 2u);
        expect(info.call_site_dense_resolver_apply_count == 1u);
        expect(info.call_site_dense_resolver_fallback_count == 1u);
        expect(store->value()->isa<Undefined>());
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_call_site_legalizes_derived_lvalue"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *ref = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.store(ref, m.create_constant_one(Type::of<int>()));
        b.return_(b.load(Type::of<int>(), ref));

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *array_type = Type::array(Type::of<int>(), 2u);
        auto *local = b.alloca_local(array_type);
        auto *element = b.gep(Type::of<int>(), local,
                              {m.create_constant_zero(Type::of<uint>())});
        auto *call = b.call(Type::of<int>(), callee, {element});
        auto call_lock = call->lock();
        b.store(element, call);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(call_lock->use_list().empty());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) ==
               0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_empty_set_preserves_call"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *ref = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_(b.load(Type::of<int>(), ref));

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *call = b.call(Type::of<int>(), callee, {local});
        b.store(local, call);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span<CallInst *const>{});
        expect(info.inlined_call_count == 0u);
        expect(call->is_linked());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) ==
               1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_call_site_specializes_storage_buffer"_test = [] {
        Module m;
        auto *buffer_type = Type::buffer(Type::of<uint>());
        auto *callee = m.create_callable(Type::of<void>());
        auto *callee_buffer = callee->create_resource_argument(buffer_type);
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *index = m.create_constant_zero(Type::of<uint>());
        auto *read = b.call(
            Type::of<uint>(), ResourceReadOp::BUFFER_READ,
            {callee_buffer, index});
        b.call(ResourceWriteOp::BUFFER_WRITE,
               {callee_buffer, index, read});
        b.return_void();

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        auto *buffer = caller->create_resource_argument(buffer_type);
        b.set_insertion_point(caller_body);
        auto *call = b.call(nullptr, callee, {buffer});
        auto call_lock = call->lock();
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(call_lock->use_list().empty());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) ==
               0u);
        auto specialized_read_count = size_t{0u};
        auto specialized_write_count = size_t{0u};
        ResourceReadInst *specialized_read = nullptr;
        ResourceWriteInst *specialized_write = nullptr;
        caller->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceReadInst>()) {
                auto *resource_read = static_cast<ResourceReadInst *>(inst);
                if (resource_read->op() == ResourceReadOp::BUFFER_READ) {
                    specialized_read = resource_read;
                    specialized_read_count++;
                    expect(resource_read->operand(0u) == buffer)
                        << "inlined buffer read must use the caller resource";
                }
            } else if (inst->isa<ResourceWriteInst>()) {
                auto *resource_write = static_cast<ResourceWriteInst *>(inst);
                if (resource_write->op() == ResourceWriteOp::BUFFER_WRITE) {
                    specialized_write = resource_write;
                    specialized_write_count++;
                    expect(resource_write->operand(0u) == buffer)
                        << "inlined buffer write must use the caller resource";
                }
            }
        });
        expect(specialized_read_count == 1u);
        expect(specialized_write_count == 1u);
        auto preserved_read_to_write = false;
        if (specialized_read != nullptr) {
            if (specialized_write != nullptr) {
                preserved_read_to_write =
                    specialized_write->operand(2u) == specialized_read;
            }
        }
        expect(preserved_read_to_write)
            << "specialization must preserve the read-to-write data flow";
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_selected_call_sites_preflight_is_atomic"_test = [] {
        Module m;
        XIRBuilder b;

        auto *simple = m.create_callable(Type::of<void>());
        auto *simple_body = simple->create_body_block();
        b.set_insertion_point(simple_body);
        b.return_void();

        auto *structured = m.create_callable(Type::of<void>());
        auto *condition =
            structured->create_value_argument(Type::of<bool>());
        auto *structured_body = structured->create_body_block();
        b.set_insertion_point(structured_body);
        auto *if_inst = b.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge_block = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge_block);
        b.set_insertion_point(false_block);
        b.br(merge_block);
        b.set_insertion_point(merge_block);
        b.return_void();

        BasicBlock *kernel_body;
        auto *kernel = make_kernel_with_body(m, kernel_body);
        auto *kernel_condition =
            kernel->create_value_argument(Type::of<bool>());
        b.set_insertion_point(kernel_body);
        auto *simple_call = b.call(nullptr, simple, {});
        auto *structured_call =
            b.call(nullptr, structured, {kernel_condition});
        b.return_void();

        std::array<CallInst *, 2u> selected{
            simple_call, structured_call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});
        expect(eq(info.inlined_call_count, 0u));
        expect(eq(info.skipped_structured_call_count, 1u));
        expect(simple_call->is_linked())
            << "a later preflight failure must preserve earlier selected calls";
        expect(structured_call->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_split_retargets_successor_phi"_test = [] {
        Module m;
        XIRBuilder b;

        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_condition =
            callee->create_value_argument(Type::of<bool>());
        auto *callee_entry = callee->create_body_block();
        auto *callee_true = callee->create_basic_block();
        auto *callee_false = callee->create_basic_block();
        b.set_insertion_point(callee_entry);
        b.cond_br(callee_condition, callee_true, callee_false);
        b.set_insertion_point(callee_true);
        b.return_(m.create_constant_one(Type::of<int>()));
        b.set_insertion_point(callee_false);
        b.return_(m.create_constant_zero(Type::of<int>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        auto *path_condition =
            caller->create_value_argument(Type::of<bool>());
        auto *call_condition =
            caller->create_value_argument(Type::of<bool>());
        auto *call_block = caller->create_basic_block();
        auto *other_block = caller->create_basic_block();
        auto *join = caller->create_basic_block();
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        b.cond_br(path_condition, call_block, other_block);
        b.set_insertion_point(call_block);
        auto *call = b.call(
            Type::of<int>(), callee, {call_condition});
        b.br(join);
        b.set_insertion_point(other_block);
        b.br(join);
        b.set_insertion_point(join);
        auto *phi = b.phi(
            Type::of<int>(),
            {{call, call_block},
             {m.create_constant_zero(Type::of<int>()), other_block}});
        b.store(storage, phi);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 1u);
        expect(phi->incoming_count() == 2u);
        auto saw_other = false;
        auto saw_continuation = false;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            saw_other |= incoming.block == other_block;
            saw_continuation |=
                incoming.block != other_block &&
                incoming.block != call_block;
        }
        expect(saw_other);
        expect(saw_continuation);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_split_retargets_phis_in_all_successors"_test = [] {
        Module m;
        XIRBuilder b;

        auto *callee = m.create_callable(nullptr);
        auto *callee_condition =
            callee->create_value_argument(Type::of<bool>());
        auto *callee_entry = callee->create_body_block();
        auto *callee_left = callee->create_basic_block();
        auto *callee_right = callee->create_basic_block();
        b.set_insertion_point(callee_entry);
        b.cond_br(callee_condition, callee_left, callee_right);
        b.set_insertion_point(callee_left);
        b.return_void();
        b.set_insertion_point(callee_right);
        b.return_void();

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        auto *callee_arg =
            caller->create_value_argument(Type::of<bool>());
        auto *branch_condition =
            caller->create_value_argument(Type::of<bool>());
        auto *left = caller->create_basic_block();
        auto *right = caller->create_basic_block();
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call = b.call(nullptr, callee, {callee_arg});
        b.cond_br(branch_condition, left, right);
        b.set_insertion_point(left);
        auto *left_phi = b.phi(
            Type::of<int>(),
            {{m.create_constant_zero(Type::of<int>()), body}});
        b.store(storage, left_phi);
        b.return_void();
        b.set_insertion_point(right);
        auto *right_phi = b.phi(
            Type::of<int>(),
            {{m.create_constant_one(Type::of<int>()), body}});
        b.store(storage, right_phi);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 1u);
        expect(left_phi->incoming(0u).block != body);
        expect(right_phi->incoming(0u).block != body);
        expect(left_phi->incoming(0u).block ==
               right_phi->incoming(0u).block);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_materializes_implicit_entry_phi_edge"_test = [] {
        Module m;
        XIRBuilder b;
        auto *callee = m.create_callable(Type::of<int>());
        auto *condition =
            callee->create_value_argument(Type::of<bool>());
        auto *entry = callee->create_body_block();
        auto *left = callee->create_basic_block();
        auto *right = callee->create_basic_block();
        b.set_insertion_point(entry);
        auto *entry_phi = b.phi(Type::of<int>());
        entry_phi->set_location("inline_phi.cpp", 29);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.return_(entry_phi);
        b.set_insertion_point(right);
        b.return_(entry_phi);

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        auto *caller_condition =
            caller->create_value_argument(Type::of<bool>());
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call = b.call(
            Type::of<int>(), callee, {caller_condition});
        b.store(storage, call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 1u);
        PhiInst *cloned_entry_phi = nullptr;
        caller->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) {
                cloned_entry_phi = static_cast<PhiInst *>(inst);
            }
        });
        expect(cloned_entry_phi != nullptr);
        if (cloned_entry_phi != nullptr) {
            expect(cloned_entry_phi->incoming_count() == 1u);
            expect(cloned_entry_phi->incoming(0u).value->isa<Undefined>());
            auto *location =
                cloned_entry_phi->find_metadata<LocationMD>();
            expect(location != nullptr);
            if (location != nullptr) {
                expect(location->file() ==
                       luisa::filesystem::path{"inline_phi.cpp"});
                expect(location->line() == 29);
            }
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_drops_disconnected_phi_predecessor"_test = [] {
        Module m;
        XIRBuilder b;
        auto *callee = m.create_callable(Type::of<int>());
        auto *entry = callee->create_body_block();
        auto *merge = callee->create_basic_block();
        auto *disconnected = callee->create_basic_block();
        auto *live_value = m.create_constant_one(Type::of<int>());
        auto *dead_value = m.create_constant_zero(Type::of<int>());
        b.set_insertion_point(entry);
        b.br(merge);
        b.set_insertion_point(disconnected);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(
            Type::of<int>(),
            {{live_value, entry}, {dead_value, disconnected}});
        b.return_(phi);

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *call = b.call(Type::of<int>(), callee, {});
        b.store(storage, call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 1u);
        expect(xir_verify_module(&m).succeeded());
        size_t cloned_phi_count = 0u;
        caller->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<PhiInst>()) { return; }
            ++cloned_phi_count;
            auto *cloned = static_cast<PhiInst *>(inst);
            expect(cloned->incoming_count() == 1u);
            expect(cloned->incoming(0u).value == live_value);
        });
        expect(cloned_phi_count == 1u);
    };

    "inline_multiblock_entry_phi_replaces_disconnected_edge_with_call_edge"_test = [] {
        Module m;
        XIRBuilder b;
        auto *callee = m.create_callable(Type::of<int2>());
        auto *entry = callee->create_body_block();
        auto *disconnected = callee->create_basic_block();
        b.set_insertion_point(entry);
        auto *entry_phi = b.phi(Type::of<int2>());
        b.return_(entry_phi);
        b.set_insertion_point(disconnected);
        auto *dead_value = m.create_constant_zero(Type::of<int2>());
        b.br(entry);
        entry_phi->add_incoming(dead_value, disconnected);

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *storage = b.alloca_local(Type::of<int2>());
        auto *call = b.call(Type::of<int2>(), callee, {});
        b.store(storage, call);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 1u);
        PhiInst *cloned_entry = nullptr;
        caller->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) {
                cloned_entry = static_cast<PhiInst *>(inst);
            }
        });
        expect(cloned_entry != nullptr);
        if (cloned_entry != nullptr) {
            expect(cloned_entry->incoming_count() == 1u);
            expect(cloned_entry->incoming(0u).value->isa<Undefined>());
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_single_block_coroutine_callee_is_a_hard_barrier"_test = [] {
        Module m;
        XIRBuilder b;
        auto *callee = m.create_callable(nullptr);
        auto *callee_body = callee->create_body_block();
        b.set_insertion_point(callee_body);
        auto *resume = b.coro_resume(7u, nullptr);
        b.return_void();

        BasicBlock *body;
        make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(nullptr, callee, {});
        b.return_void();

        std::array<CallInst *, 1u> selected{call};
        auto info = inline_call_sites_pass_run_on_module(
            &m, luisa::span{selected});

        expect(info.inlined_call_count == 0u);
        expect(info.skipped_structured_call_count == 1u);
        expect(call->is_linked());
        expect(resume->is_linked());
    };
}

// ---- unused_callable_removal ----

void reg_unused_callable_removal() {

    "unused_callable_removed"_test = [] {
        Module m;
        m.create_callable(Type::of<void>())->create_body_block();
        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 1u);
    };

    "unused_callable_used_callable_kept"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        callee->create_body_block();
        XIRBuilder b;
        int32_t ret_v = 42;
        b.set_insertion_point(callee->body_block());
        auto *val = m.create_constant(Type::of<int>(), &ret_v);
        b.return_(val);

        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<int>(), callee, {});
        b.return_(call);

        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 0u);
    };
}

// ---- trace_gep ----

void reg_trace_gep() {

    "trace_gep_cascaded_gep_traced"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto inner_ty = Type::array(Type::of<float>(), 2u);
        auto outer_ty = Type::array(inner_ty, 2u);
        auto *alloca = b.alloca_local(outer_ty);
        uint32_t idx0_v = 0u, idx1_v = 1u;
        auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
        auto *idx1 = m.create_constant(Type::of<uint>(), &idx1_v);
        auto *gep1 = b.gep(inner_ty, alloca, {idx0});
        auto *gep2 = b.gep(Type::of<float>(), gep1, {idx1});
        auto *val = b.load(Type::of<float>(), gep2);
        b.return_(val);
        auto info = trace_gep_pass_run_on_function(k);
        expect(info.traced_gep_count == 1u);
        expect(gep2->base() == alloca);
        expect(gep2->index_count() == 2u);
        expect(gep2->index(0) == idx0);
        expect(gep2->index(1) == idx1);
    };

    "trace_gep_does_not_cross_annotated_immediate_base"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner_type =
            Type::array(Type::of<float>(), 2u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *storage = b.alloca_local(outer_type);
        auto *zero =
            m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *annotated =
            b.gep(inner_type, storage, {zero});
        annotated->add_comment(
            "nested address provenance owner");
        auto *element =
            b.gep(Type::of<float>(), annotated, {one});
        auto *load = b.load(Type::of<float>(), element);
        b.return_(load);

        expect(xir_verify_module(&m).succeeded());
        auto info = trace_gep_pass_run_on_function(f);
        expect(info.traced_gep_count == 0u);
        expect(element->base() == annotated);
        expect(annotated->is_linked());
        expect(annotated->find_metadata<CommentMD>() != nullptr);
        static_cast<void>(dce_pass_run_on_function(f));
        expect(annotated->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "trace_gep_flattens_unannotated_prefix_only_to_metadata_boundary"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *level1 =
            Type::array(Type::of<int>(), 4u);
        auto *level2 = Type::array(level1, 4u);
        auto *level3 = Type::array(level2, 4u);
        auto *storage = b.alloca_local(level3);
        auto *zero =
            m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t two_value = 2u;
        auto *two =
            m.create_constant(Type::of<uint>(), &two_value);
        auto *boundary =
            b.gep(level2, storage, {zero});
        boundary->set_location("trace_gep_boundary.cpp", 19);
        auto *middle = b.gep(level1, boundary, {one});
        auto middle_lock = middle->lock();
        auto *element =
            b.gep(Type::of<int>(), middle, {two});
        auto *load = b.load(Type::of<int>(), element);
        b.return_(load);

        expect(xir_verify_module(&m).succeeded());
        auto info = trace_gep_pass_run_on_function(f);
        expect(info.traced_gep_count == 1u);
        expect(element->base() == boundary);
        expect(element->index_count() == 2u);
        expect(element->index(0u) == one);
        expect(element->index(1u) == two);
        expect(boundary->is_linked());
        expect(boundary->find_metadata<LocationMD>() != nullptr);
        static_cast<void>(dce_pass_run_on_function(f));
        expect(boundary->is_linked());
        expect(!static_cast<GEPInst *>(
                    middle_lock.get())
                    ->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "trace_gep_no_gep_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = trace_gep_pass_run_on_function(k);
        expect(info.traced_gep_count == 0u);
        expect(info.removed_noop_gep_count == 0u);
        expect(!info.changed());
    };

    "trace_gep_noop_removal_is_reported"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *noop = b.gep(Type::of<int>(), storage, {});
        auto noop_lock = noop->lock();
        auto *load = b.load(Type::of<int>(), noop);
        b.store(storage, load);
        b.return_void();
        expect(!xir_verify_module(&m).succeeded())
            << "zero-index GEP is pre-legalization input";

        auto info = trace_gep_pass_run_on_function(k);

        expect(info.traced_gep_count == 0u);
        expect(info.removed_noop_gep_count == 1u);
        expect(info.changed());
        expect(noop_lock->use_list().empty());
        expect(load->variable() == storage);
        expect(xir_verify_module(&m).succeeded());
    };

    "trace_gep_module_reports_each_noop_removal"_test = [] {
        Module m;
        for (auto i = 0u; i < 2u; ++i) {
            static_cast<void>(i);
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *storage = b.alloca_local(Type::of<uint>());
            auto *noop = b.gep(Type::of<uint>(), storage, {});
            b.store(storage, b.load(Type::of<uint>(), noop));
            b.return_void();
        }

        auto info = trace_gep_pass_run_on_module(&m);

        expect(info.traced_gep_count == 0u);
        expect(info.removed_noop_gep_count == 2u);
        expect(info.changed());
        expect(xir_verify_module(&m).succeeded());
    };

    "trace_gep_annotated_noop_keeps_unique_metadata_owner"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *noop = b.gep(Type::of<int>(), storage, {});
        noop->set_location("trace_gep_metadata.cpp", 7);
        auto *load = b.load(Type::of<int>(), noop);
        b.return_(load);

        auto info = trace_gep_pass_run_on_function(f);
        expect(!info.changed());
        expect(noop->is_linked());
        expect(load->variable() == noop);
        expect(noop->find_metadata<LocationMD>() != nullptr);
    };

    "trace_gep_null_inputs_are_noops"_test = [] {
        expect(!trace_gep_pass_run_on_function(nullptr).changed());
        expect(!trace_gep_pass_run_on_module(nullptr).changed());
    };
}

// ---- transpose_gep ----

void reg_transpose_gep() {

    "transpose_gep_load_gep_transposed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto struct_ty = Type::of<float2>();
        auto *alloca = b.alloca_local(struct_ty);
        float init_data[2] = {1.0f, 2.0f};
        auto *init = m.create_constant(struct_ty, init_data);
        b.store(alloca, init);
        uint32_t idx0_v = 0u;
        auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
        auto *gep = b.gep(Type::of<float>(), alloca, {idx0});
        auto *val = b.load(Type::of<float>(), gep);
        auto gep_locked = gep->lock();
        auto val_locked = val->lock();
        auto *ret = b.return_(val);
        auto info = transpose_gep_pass_run_on_function(k);
        expect(info.transposed_load_count == 1u);
        expect(info.removed_gep_count == 1u);
        expect(info.changed());
        expect(gep_locked->use_list().empty());
        expect(val_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *extract = static_cast<ArithmeticInst *>(ret->return_value());
        expect(extract->op() == ArithmeticOp::EXTRACT);
        expect(extract->operand(0)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(extract->operand(0))->variable() == alloca);
        expect(extract->operand(1) == idx0);
    };

    "transpose_gep_no_gep_load_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = transpose_gep_pass_run_on_function(k);
        expect(info.transposed_load_count == 0u);
        expect(info.transposed_store_count == 0u);
        expect(info.removed_gep_count == 0u);
        expect(!info.changed());
    };

    "transpose_gep_reports_internal_noop_canonicalization"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(Type::of<int>());
        auto *noop = b.gep(Type::of<int>(), storage, {});
        auto *load = b.load(Type::of<int>(), noop);
        b.store(storage, load);
        b.return_void();

        auto info = transpose_gep_pass_run_on_function(k);

        expect(info.traced_gep_count == 0u);
        expect(info.removed_noop_gep_count == 1u);
        expect(info.transposed_load_count == 0u);
        expect(info.removed_gep_count == 0u);
        expect(info.changed());
        expect(load->variable() == storage);
        expect(xir_verify_module(&m).succeeded());
    };

    "transpose_gep_reports_internal_nested_gep_canonicalization"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto inner_ty = Type::array(Type::of<float>(), 2u);
        auto outer_ty = Type::array(inner_ty, 32u);
        auto *storage = b.alloca_local(outer_ty);
        auto *dynamic_index = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {m.create_constant_zero(Type::of<uint>()),
             m.create_constant_one(Type::of<uint>())});
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *outer_gep = b.gep(inner_ty, storage, {dynamic_index});
        auto *inner_gep = b.gep(Type::of<float>(), outer_gep, {zero});
        static_cast<void>(b.load(Type::of<float>(), inner_gep));
        b.return_void();

        auto info = transpose_gep_pass_run_on_function(k);

        expect(info.traced_gep_count == 1u);
        expect(info.removed_noop_gep_count == 0u);
        expect(info.transposed_load_count == 0u)
            << "large dynamically indexed arrays must not be transposed";
        expect(info.removed_gep_count == 0u);
        expect(info.changed());
        expect(inner_gep->base() == storage);
        expect(inner_gep->index_count() == 2u);
    };

    "transpose_gep_module_runs_all_functions"_test = [] {
        Module m;
        auto struct_ty = Type::of<float2>();
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(struct_ty);
            float init_data[2] = {1.0f, 2.0f};
            auto *init = m.create_constant(struct_ty, init_data);
            b.store(alloca, init);
            uint32_t idx0_v = 0u;
            auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
            auto *gep = b.gep(Type::of<float>(), alloca, {idx0});
            auto *val = b.load(Type::of<float>(), gep);
            b.return_(val);
        }
        auto info = transpose_gep_pass_run_on_module(&m);
        expect(info.transposed_load_count == 2u);
        expect(info.removed_gep_count == 2u);
        expect(info.changed());
    };

    "transpose_gep_load_metadata_moves_to_extract"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        auto *type = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(type);
        b.store(storage, m.create_constant_zero(type));
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *element = b.gep(Type::of<float>(), storage, {zero});
        auto *load = b.load(Type::of<float>(), element);
        load->set_location("transpose_gep_metadata.cpp", 18);
        auto *ret = b.return_(load);

        auto info = transpose_gep_pass_run_on_function(f);
        expect(info.transposed_load_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *extract =
            static_cast<ArithmeticInst *>(ret->return_value());
        auto *location = extract->find_metadata<LocationMD>();
        expect(location != nullptr);
        expect(location != nullptr && location->line() == 18);
        expect(xir_verify_module(&m).succeeded());
    };

    "transpose_gep_annotated_address_is_left_unchanged"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        auto *type = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *storage = b.alloca_local(type);
        b.store(storage, m.create_constant_zero(type));
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *element = b.gep(Type::of<float>(), storage, {zero});
        element->add_comment("address metadata has multiple possible users");
        auto *load = b.load(Type::of<float>(), element);
        b.return_(load);

        auto info = transpose_gep_pass_run_on_function(f);
        expect(info.transposed_load_count == 0u);
        expect(info.removed_gep_count == 0u);
        expect(element->is_linked());
        expect(load->variable() == element);
        expect(element->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "transpose_gep_null_inputs_are_noops"_test = [] {
        expect(!transpose_gep_pass_run_on_function(nullptr).changed());
        expect(!transpose_gep_pass_run_on_module(nullptr).changed());
    };
}

// ---- mem2reg ----

void reg_mem2reg() {

    "mem2reg_promotes_simple_alloca"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        auto ld_locked = ld->lock();
        auto *ret = b.return_(ld);
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 1u);
        expect(info.removed_load_count == 1u);
        expect(info.removed_store_count == 1u);
        expect(ld_locked->use_list().empty());
        expect(ret->return_value() == val);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOAD) == 0u);
    };

    "mem2reg_no_alloca_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
    };

    "mem2reg_retains_annotated_ordinary_alloca"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        alloca->add_comment("user-visible storage owner");
        auto *one = m.create_constant_one(Type::of<int>());
        auto *store = b.store(alloca, one);
        auto *load = b.load(Type::of<int>(), alloca);
        auto *ret = b.return_(load);

        auto info = mem2reg_pass_run_on_function(f);
        expect(!info.changed());
        expect(alloca->is_linked());
        expect(store->is_linked());
        expect(load->is_linked());
        expect(ret->return_value() == load);
        expect(alloca->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "mem2reg_promotes_named_alloca_and_names_inserted_phi"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *condition =
            f->create_value_argument(Type::of<bool>());
        auto *entry = f->create_body_block();
        auto *true_block = f->create_basic_block();
        auto *false_block = f->create_basic_block();
        auto *merge = f->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *alloca = b.alloca_local(Type::of<int>());
        alloca->set_name("named_local");
        b.cond_br(condition, true_block, false_block);
        b.set_insertion_point(true_block);
        b.store(alloca, m.create_constant_one(Type::of<int>()));
        b.br(merge);
        b.set_insertion_point(false_block);
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        b.br(merge);
        b.set_insertion_point(merge);
        auto *load = b.load(Type::of<int>(), alloca);
        auto *ret = b.return_(load);

        auto info = mem2reg_pass_run_on_function(f);
        expect(info.promoted_alloca_count == 1u);
        expect(info.inserted_phi_count == 1u);
        expect(ret->return_value()->isa<PhiInst>());
        auto *phi = static_cast<PhiInst *>(ret->return_value());
        auto phi_name = phi->name();
        expect(phi_name.has_value());
        if (phi_name.has_value()) {
            expect(*phi_name ==
                   luisa::string_view{"named_local"});
        }
        expect(count_reachable_insts(
                   f, DerivedInstructionTag::ALLOCA) == 0u);
        expect(count_reachable_insts(
                   f, DerivedInstructionTag::LOAD) == 0u);
        expect(count_reachable_insts(
                   f, DerivedInstructionTag::STORE) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "mem2reg_retains_annotated_load_and_store_owners"_test = [] {
        Module m;
        for (auto annotate_load : {false, true}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            auto *one = m.create_constant_one(Type::of<int>());
            auto *store = b.store(alloca, one);
            auto *load = b.load(Type::of<int>(), alloca);
            auto *annotated =
                annotate_load ? static_cast<Instruction *>(load) :
                                static_cast<Instruction *>(store);
            annotated->set_location(
                annotate_load ? "mem2reg_load.cpp" : "mem2reg_store.cpp",
                annotate_load ? 11u : 19u);
            auto *ret = b.return_(load);

            auto info = mem2reg_pass_run_on_function(f);
            expect(!info.changed());
            expect(alloca->is_linked());
            expect(store->is_linked());
            expect(load->is_linked());
            expect(ret->return_value() == load);
            expect(annotated->find_metadata<LocationMD>() != nullptr);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "mem2reg_round_trips_reg2mem_loop_phi_aggregates"_test = [] {
        auto *struct_type = Type::from("struct<16,uint,float>");
        expect(struct_type != nullptr);
        if (struct_type == nullptr) { return; }
        for (auto *type : std::array<const Type *, 2u>{
                 Type::of<float2>(), struct_type}) {
            Module m;
            auto *f = m.create_callable(type);
            auto *initial = f->create_value_argument(type);
            auto *next = f->create_value_argument(type);
            auto *iterate = f->create_value_argument(Type::of<bool>());
            auto *entry = f->create_body_block();
            auto *header = f->create_basic_block();
            auto *latch = f->create_basic_block();
            auto *exit = f->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(entry);
            b.br(header);
            b.set_insertion_point(header);
            auto *phi = b.phi(type, {{initial, entry}, {next, latch}});
            b.cond_br(iterate, latch, exit);
            b.set_insertion_point(latch);
            b.br(header);
            b.set_insertion_point(exit);
            b.return_(phi);
            expect(xir_verify_module(&m).succeeded());

            auto reg2mem = reg2mem_pass_run_on_function(f);
            expect(reg2mem.lowered_phi_count == 1u);
            auto spilled = audit_reg2mem_spills_on_function(f);
            expect(spilled.remaining_phi_spill_count == 1u);
            expect(spilled.remaining_cross_block_spill_count == 0u);
            expect(count_reachable_insts(f, DerivedInstructionTag::PHI) == 0u);
            expect(count_reachable_insts(f, DerivedInstructionTag::ALLOCA) == 1u);
            expect(count_reachable_insts(f, DerivedInstructionTag::STORE) == 3u)
                << "reg2mem must materialize undef, entry, and latch definitions";

            auto mem2reg = mem2reg_pass_run_on_function(f);
            expect(mem2reg.promoted_alloca_count == 1u);
            auto recovered = audit_reg2mem_spills_on_function(f);
            expect(recovered.succeeded());
            expect(count_reachable_insts(f, DerivedInstructionTag::PHI) == 1u);
            expect(count_reachable_insts(f, DerivedInstructionTag::ALLOCA) == 0u);
            expect(count_reachable_insts(f, DerivedInstructionTag::STORE) == 0u);
            expect(count_reachable_insts(f, DerivedInstructionTag::LOAD) == 0u);
            expect(xir_verify_module(&m).succeeded())
                << "reg2mem/mem2reg must preserve valid aggregate SSA";
        }
    };

    "mem2reg_retains_alloca_with_unreachable_load_store_users"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *dead = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        b.set_insertion_point(dead);
        auto *dead_load = b.load(Type::of<int>(), alloca);
        auto *dead_store = b.store(alloca, dead_load);
        b.unreachable_();
        [[maybe_unused]] auto alloca_lock = alloca->lock();
        [[maybe_unused]] auto ld_lock = ld->lock();
        [[maybe_unused]] auto dead_load_lock = dead_load->lock();
        [[maybe_unused]] auto dead_store_lock = dead_store->lock();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
        expect(info.removed_load_count == 0u);
        expect(info.removed_store_count == 0u);
        expect(alloca->is_linked());
        expect(ld->is_linked());
        expect(dead_load->is_linked());
        expect(dead_store->is_linked());
        expect(dead_store->value() == dead_load);
    };

    "mem2reg_retains_alloca_with_unreachable_load_used_by_owned_instruction"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *dead = k->definition()->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        b.store(alloca, one);
        auto *live_load = b.load(Type::of<int>(), alloca);
        b.return_(live_load);

        b.set_insertion_point(dead);
        auto *dead_load = b.load(Type::of<int>(), alloca);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {dead_load, one});
        b.return_(sum);

        [[maybe_unused]] auto alloca_lock = alloca->lock();
        [[maybe_unused]] auto live_load_lock = live_load->lock();
        [[maybe_unused]] auto dead_load_lock = dead_load->lock();
        [[maybe_unused]] auto sum_lock = sum->lock();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
        expect(info.removed_load_count == 0u);
        expect(info.removed_store_count == 0u);
        expect(alloca->is_linked());
        expect(live_load->is_linked());
        expect(dead_load->is_linked());
        expect(sum->is_linked());
        expect(sum->operand(0u) == dead_load);
        expect(!dead_load->use_list().empty());
    };
}

// ---- promote_ref_arg ----

void reg_promote_ref_arg() {

    "promote_ref_arg_rewrites_signature_body_and_call_site"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *ref_arg = c->create_reference_argument(Type::of<int>());
        ref_arg->set_location("promote_ref_arg.cpp", 41);
        auto *callee_body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *callee_load = b.load(Type::of<int>(), ref_arg);
        b.return_(callee_load);

        BasicBlock *caller_body;
        auto *k = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_one(Type::of<int>()));
        auto *call = b.call(Type::of<int>(), c, {local});
        b.return_(call);

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 1u);
        expect(c->arguments().count_size() == 1u);
        expect(!c->arguments().front()->is_reference());
        auto *location =
            c->arguments().front()->find_metadata<LocationMD>();
        expect(location != nullptr);
        if (location != nullptr) {
            expect(location->file() ==
                   luisa::filesystem::path{"promote_ref_arg.cpp"});
            expect(location->line() == 41);
        }
        expect(count_reachable_insts(c, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(c, DerivedInstructionTag::STORE) == 1u);
        expect(call->argument(0)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(call->argument(0))->variable() == local);
    };

    "promote_ref_arg_no_ref_args_no_change"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<void>());
        c->create_body_block();
        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 0u);
    };

    "promote_ref_arg_null_and_bodyless_inputs_are_noops"_test = [] {
        PassReport report;
        auto null_info =
            promote_ref_arg_pass_run_on_module(nullptr, &report);
        expect(!null_info.changed());
        expect(report.entries().size() == 1u);

        Module m;
        auto *declaration = m.create_callable(Type::of<int>());
        auto *ref =
            declaration->create_reference_argument(Type::of<int>());
        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(!info.changed());
        expect(declaration->body_block() == nullptr);
        expect(declaration->arguments().count_size() == 1u);
        expect(declaration->arguments().front() == ref);
        expect(ref->is_reference());
    };

    "promote_ref_arg_writable_alias_blocks_snapshot"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *read_ref = c->create_reference_argument(Type::of<int>());
        auto *write_ref = c->create_reference_argument(Type::of<int>());
        auto *callee_body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        b.store(write_ref, two);
        auto *loaded_after_store = b.load(Type::of<int>(), read_ref);
        b.return_(loaded_after_store);

        BasicBlock *caller_body;
        auto *k = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_one(Type::of<int>()));
        auto *call = b.call(Type::of<int>(), c, {local, local});
        b.return_(call);

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 0u);
        expect(c->arguments().count_size() == 2u);
        expect(c->arguments().front()->is_reference());
        expect(c->arguments().back()->is_reference());
        expect(call->argument(0u) == local);
        expect(call->argument(1u) == local);
        expect(loaded_after_store->variable() == read_ref);
    };

    "promote_ref_arg_rejects_shared_memory_snapshot"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *ref = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *loaded = b.load(Type::of<int>(), ref);
        b.return_(loaded);

        BasicBlock *caller_body;
        auto *kernel = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *shared = b.alloca_shared(Type::of<int>());
        auto *sink = b.alloca_local(Type::of<int>());
        auto *call = b.call(Type::of<int>(), callee, {shared});
        b.store(sink, call);
        b.return_void();

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(!info.changed());
        expect(callee->arguments().count_size() == 1u);
        expect(callee->arguments().front()->is_reference());
        expect(call->argument(0u) == shared);
        expect(loaded->variable() == ref);
        expect(xir_verify_module(&m).succeeded());
    };

    "promote_ref_arg_mixed_local_and_shared_calls_reject_atomically"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *ref = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *loaded = b.load(Type::of<int>(), ref);
        b.return_(loaded);

        BasicBlock *caller_body;
        auto *kernel = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *shared = b.alloca_shared(Type::of<int>());
        auto *sink = b.alloca_local(Type::of<int>());
        auto *local_call =
            b.call(Type::of<int>(), callee, {local});
        auto *shared_call =
            b.call(Type::of<int>(), callee, {shared});
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {local_call, shared_call});
        b.store(sink, sum);
        b.return_void();

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(!info.changed());
        expect(callee->arguments().count_size() == 1u);
        expect(callee->arguments().front()->is_reference());
        expect(local_call->argument(0u) == local);
        expect(shared_call->argument(0u) == shared);
        expect(loaded->variable() == ref);
        expect(xir_verify_module(&m).succeeded());
    };

    "promote_ref_arg_rejects_function_value_used_by_unrelated_call"_test = [] {
        Module m;
        auto *candidate = m.create_callable(Type::of<int>());
        auto *candidate_ref =
            candidate->create_reference_argument(Type::of<int>());
        auto *candidate_body = candidate->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(candidate_body);
        b.return_(b.load(Type::of<int>(), candidate_ref));

        auto *consumer = m.create_callable(nullptr);
        auto *consumer_ref =
            consumer->create_reference_argument(Type::of<int>());
        consumer->create_value_argument(Type::of<int>());
        auto *consumer_body = consumer->create_body_block();
        b.set_insertion_point(consumer_body);
        // Keep the unrelated consumer itself ineligible for reference
        // promotion so the module-level count isolates `candidate`.
        b.store(consumer_ref, m.create_constant_one(Type::of<int>()));
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        auto *local = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_one(Type::of<int>()));
        // Deliberately verifier-invalid: `candidate` is an ordinary value
        // argument of an unrelated call. The local argument at index zero
        // made the old call-site preflight accidentally accept this use.
        auto *unrelated = b.call(nullptr, consumer, {local, candidate});
        b.return_void();

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 0u);
        expect(candidate->arguments().count_size() == 1u);
        expect(candidate->arguments().front() == candidate_ref);
        expect(candidate_ref->is_reference());
        expect(unrelated->callee() == consumer);
        expect(unrelated->argument_count() == 2u);
        expect(unrelated->argument(0u) == local);
        expect(unrelated->argument(1u) == candidate);
    };

    "promote_ref_arg_updates_each_distinct_valid_call_site"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *ref = callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_(b.load(Type::of<int>(), ref));

        auto *kernel = m.create_kernel();
        auto *kernel_body = kernel->create_body_block();
        b.set_insertion_point(kernel_body);
        auto *first_local = b.alloca_local(Type::of<int>());
        auto *second_local = b.alloca_local(Type::of<int>());
        auto *first =
            b.call(Type::of<int>(), callee, {first_local});
        auto *second =
            b.call(Type::of<int>(), callee, {second_local});
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {first, second});
        auto *sink = b.alloca_local(Type::of<int>());
        b.store(sink, sum);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 1u);
        expect(!callee->arguments().front()->is_reference());
        expect(first->argument(0u)->isa<LoadInst>());
        expect(second->argument(0u)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(first->argument(0u))->variable() ==
               first_local);
        expect(static_cast<LoadInst *>(second->argument(0u))->variable() ==
               second_local);
        expect(xir_verify_module(&m).succeeded());
    };
}

// ---- outline ----

void reg_outline() {

    "outline_no_outline_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = outline_pass_run_on_module(&m);
        expect(info.outlined_func_count == 0u);
        expect(info.unsupported_outline_count == 0u);
        expect(info.succeeded());
        expect(xir_verify_module(&m).succeeded());
    };

    "outline_instruction_reports_unsupported_without_mutation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outline = b.outline();
        auto *entry = outline->create_target_block();
        auto *merge = outline->create_merge_block();
        b.set_insertion_point(entry);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = outline_pass_run_on_module(&m);
        expect(info.outlined_func_count == 0u);
        expect(info.unsupported_outline_count == 1u);
        expect(!info.succeeded());
        expect(body->terminator() == outline);
        expect(outline->target_block() == entry);
        expect(outline->merge_block() == merge);
        expect(xir_verify_module(&m).succeeded());
    };

    "outline_null_inputs_are_noops"_test = [] {
        expect(outline_pass_run_on_function(nullptr, nullptr).succeeded());
        expect(outline_pass_run_on_module(nullptr).succeeded());
    };
}

// ---- autodiff ----

void reg_autodiff() {

    "autodiff_options_run_both_modes_by_default"_test = [] {
        AutodiffOptions options;
        expect(options.run_forward);
        expect(options.run_backward);
    };

    "autodiff_scope_metadata_moves_to_entry_branch_in_both_modes"_test = [] {
        auto run_case = [](bool forward) noexcept {
            Module m;
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            auto *x =
                k->create_argument(Type::of<float>(), false);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *scope = forward ?
                              b.forward_autodiff_scope(1u) :
                              b.autodiff_scope();
            scope->set_location(
                forward ? "forward_scope.cpp" :
                          "reverse_scope.cpp",
                forward ? 11u : 29u);
            auto *merge = scope->create_merge_block();
            auto *entry = scope->create_entry_block();
            b.set_insertion_point(entry);
            auto *one =
                m.create_constant_one(Type::of<float>());
            if (forward) {
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_PROPAGATE_GRADIENT,
                    {x, one});
                auto *index =
                    m.create_constant_zero(
                        Type::of<uint32_t>());
                static_cast<void>(b.call(
                    Type::of<float>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_OUTPUT_GRADIENT,
                    {x, index}));
            } else {
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_REQUIRES_GRADIENT,
                    {x});
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_GRADIENT_MARKER,
                    {x, one});
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_BACKWARD,
                    {});
            }
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();

            expect(xir_verify_module(&m).succeeded());
            auto info = autodiff_pass_run_on_function(k);
            expect(info.transformed_scope_count == 1u);
            expect(body->terminator()->isa<BranchInst>());
            auto *location =
                body->terminator()->find_metadata<LocationMD>();
            expect(location != nullptr);
            if (location != nullptr) {
                expect(location->line() ==
                       (forward ? 11u : 29u));
            }
            expect(xir_verify_module(&m).succeeded());
        };
        run_case(true);
        run_case(false);
    };

    "autodiff_run_forward_false_leaves_forward_scope_unlowered"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {x, idx});
        static_cast<void>(gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k, {.run_forward = false});
        expect(info.transformed_scope_count == 0u);
        expect(scope->is_forward());
        expect(scope->n_forward_grads() == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 2u);
    };

    "autodiff_run_backward_false_leaves_scope_unlowered"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k, {.run_backward = false});
        expect(info.transformed_scope_count == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 3u);
    };

    "autodiff_forward_propagates_scalar_duals"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dx_out = b.alloca_local(Type::of<float>());
        auto *dy_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, one_f, zero_f});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {y, zero_f, one_f});
        auto *xy = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, y});
        auto *sx = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xy, sx});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {z, idx0});
        auto *dy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {z, idx1});
        auto dx_lock = dx->lock();
        auto dy_lock = dy->lock();
        b.store(dx_out, dx);
        b.store(dy_out, dy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(dx_lock->use_list().empty());
        expect(dy_lock->use_list().empty());
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_forward_handles_binary_mod"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dx_out = b.alloca_local(Type::of<float>());
        auto *dy_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, one_f, zero_f});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {y, zero_f, one_f});
        auto *r = b.call(Type::of<float>(), ArithmeticOp::BINARY_MOD, {x, y});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {r, idx0});
        auto *dy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {r, idx1});
        b.store(dx_out, dx);
        b.store(dy_out, dy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t mod_count = 0u;
        size_t trunc_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::BINARY_MOD) { mod_count++; }
                if (op == ArithmeticOp::TRUNC) { trunc_count++; }
            }
        });
        expect(mod_count >= 1u);
        expect(trunc_count >= 1u);
    };

    "autodiff_forward_propagates_mutable_cfg_state"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *flag = k->create_argument(Type::of<bool>(), false);
        auto *tag = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        auto *y = b.alloca_local(Type::of<float>());
        b.store(y, x);
        auto *if_inst = b.if_(flag);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {t0, t0});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {f0});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        auto *sw = b.switch_(tag);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::COS, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *out = b.load(Type::of<float>(), y);
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *gout = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {out, idx});
        b.store(grad_out, gout);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) == 1u);
        size_t sin_count = 0u;
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::SIN) { sin_count++; }
                if (op == ArithmeticOp::COS) { cos_count++; }
            }
        });
        expect(sin_count >= 2u);
        expect(cos_count >= 2u);
    };

    "autodiff_forward_propagates_structured_loop_state"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {s, x});
        b.store(y, sum);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        uint32_t grad_index = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &grad_index);
        auto *gout = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {out, idx});
        b.store(grad_out, gout);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 1u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_forward_handles_matrix_linalg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto vector_type = Type::of<float2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *vec = k->create_argument(vector_type, false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *vector_grad_out = b.alloca_local(vector_type);
        auto *scalar_grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_m = m.create_constant_zero(matrix_type);
        auto *one_m = m.create_constant_one(matrix_type);
        auto *zero_v = m.create_constant_zero(vector_type);
        auto *one_v = m.create_constant_one(vector_type);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {mat, one_m, zero_m});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {vec, zero_v, one_v});
        auto *mv = b.call(vector_type, ArithmeticOp::MATRIX_LINALG_MUL, {mat, vec});
        auto *outer = b.call(matrix_type, ArithmeticOp::OUTER_PRODUCT, {mv, vec});
        auto *inv = b.call(matrix_type, ArithmeticOp::MATRIX_INVERSE, {mat});
        auto *det = b.call(Type::of<float>(), ArithmeticOp::MATRIX_DETERMINANT, {mat});
        auto *combined = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {outer, inv});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dm = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {combined, idx0});
        auto *dv = b.call(vector_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {mv, idx1});
        auto *dd = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {det, idx0});
        b.store(matrix_grad_out, dm);
        b.store(vector_grad_out, dv);
        b.store(scalar_grad_out, dd);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t matmul_count = 0u;
        size_t determinant_count = 0u;
        size_t inverse_count = 0u;
        size_t outer_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                switch (static_cast<ArithmeticInst *>(inst)->op()) {
                    case ArithmeticOp::MATRIX_LINALG_MUL: matmul_count++; break;
                    case ArithmeticOp::MATRIX_DETERMINANT: determinant_count++; break;
                    case ArithmeticOp::MATRIX_INVERSE: inverse_count++; break;
                    case ArithmeticOp::OUTER_PRODUCT: outer_count++; break;
                    default: break;
                }
            }
        });
        expect(matmul_count >= 5u);
        expect(determinant_count >= 1u);
        expect(inverse_count >= 2u);
        expect(outer_count >= 3u);
    };

    "autodiff_forward_handles_matrix_scalar_components"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_m = m.create_constant_zero(matrix_type);
        auto *one_m = m.create_constant_one(matrix_type);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {mat, one_m, zero_m});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {scalar, zero_f, one_f});
        auto *mul = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_MUL, {mat, scalar});
        auto *div = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_DIV, {scalar, mat});
        auto *sum = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {mul, div});
        uint32_t one = 1u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &one);
        auto *ds = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {sum, idx});
        b.store(matrix_grad_out, ds);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t aggregate_count = 0u;
        size_t matrix_div_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::AGGREGATE) { aggregate_count++; }
                if (op == ArithmeticOp::MATRIX_COMP_DIV) { matrix_div_count++; }
            }
        });
        expect(aggregate_count >= 1u);
        expect(matrix_div_count >= 2u);
    };

    "autodiff_forward_projects_static_cast_aggregate_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto double_vector_type = Type::of<double2>();
        auto matrix_type = Type::of<float2x2>();
        auto *v = k->create_argument(vector_type, false);
        auto *s = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *vector_grad_out = b.alloca_local(double_vector_type);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {v, m.create_constant_one(vector_type)});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {s, m.create_constant_one(Type::of<float>())});
        auto *vd = b.cast_(double_vector_type, xir::CastOp::STATIC_CAST, v);
        auto *sm = b.call(matrix_type, ArithmeticOp::AGGREGATE, {b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s}), b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s})});
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *dvd = b.call(double_vector_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {vd, idx});
        auto *dsm = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {sm, idx});
        b.store(vector_grad_out, dvd);
        b.store(matrix_grad_out, dsm);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t float_to_double_scalar_cast_count = 0u;
        size_t aggregate_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<double>() &&
                    cast->value()->type() == Type::of<float>()) {
                    float_to_double_scalar_cast_count++;
                }
            } else if (inst->isa<ArithmeticInst>() &&
                       static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::AGGREGATE) {
                aggregate_count++;
            }
        });
        expect(float_to_double_scalar_cast_count >= 2u);
        expect(aggregate_count >= 4u);
    };

    "autodiff_reverse_projects_matrix_scalar_component_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {mat});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {scalar});
        auto *prod = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_MUL, {mat, scalar});
        auto *quot = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_DIV, {scalar, mat});
        auto *sum = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {prod, quot});
        auto *col0 = b.call(Type::of<float2>(), ArithmeticOp::EXTRACT, {sum, m.create_constant_zero(Type::of<uint32_t>())});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {col0});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gs = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {scalar});
        b.store(grad_out, gs);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t reduce_sum_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::REDUCE_SUM) {
                reduce_sum_count++;
            }
        });
        expect(reduce_sum_count >= 2u);
    };

    "autodiff_reverse_projects_vector_scalar_binary_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float3>();
        auto *vector = k->create_argument(vector_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {vector});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {scalar});
        auto *product = b.call(vector_type, ArithmeticOp::BINARY_MUL, {scalar, vector});
        auto *quotient = b.call(vector_type, ArithmeticOp::BINARY_DIV, {vector, scalar});
        auto *sum = b.call(vector_type, ArithmeticOp::BINARY_ADD, {product, quotient});
        auto *biased = b.call(vector_type, ArithmeticOp::BINARY_ADD, {sum, scalar});
        auto *output = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {biased});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *grad = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {scalar});
        b.store(grad_out, grad);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(k);

        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t reduce_sum_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::REDUCE_SUM) {
                reduce_sum_count++;
                expect(inst->type() == Type::of<float>());
                expect(inst->operand_count() == 1u);
                expect(inst->operand(0u)->type() == vector_type);
            }
        });
        expect(reduce_sum_count == 4u);
    };

    "autodiff_reverse_propagates_static_cast_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *xd = static_cast<Value *>(b.static_cast_(Type::of<double>(), x));
        auto *yd = b.call(Type::of<double>(), ArithmeticOp::BINARY_MUL, {xd, xd});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {yd, m.create_constant_one(Type::of<double>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        bool has_forward_cast = false;
        bool has_backward_cast = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<double>() &&
                    cast->value()->type() == Type::of<float>()) {
                    has_forward_cast = true;
                }
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->value()->type() == Type::of<double>()) {
                    has_backward_cast = true;
                }
            }
        });
        expect(has_forward_cast);
        expect(has_backward_cast);
    };

    "autodiff_reverse_projects_vector_static_cast_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto double_vector_type = Type::of<double2>();
        auto *x = k->create_argument(vector_type, false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(vector_type);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *xd = b.cast_(double_vector_type, xir::CastOp::STATIC_CAST, x);
        auto *yd = b.call(Type::of<double>(), ArithmeticOp::REDUCE_SUM, {xd});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {yd, m.create_constant_one(Type::of<double>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(vector_type, AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t forward_vector_cast_count = 0u;
        size_t backward_scalar_cast_count = 0u;
        size_t insert_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == double_vector_type &&
                    cast->value()->type() == vector_type) {
                    forward_vector_cast_count++;
                }
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->value()->type() == Type::of<double>()) {
                    backward_scalar_cast_count++;
                }
            } else if (inst->isa<ArithmeticInst>() &&
                       static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::INSERT) {
                insert_count++;
            }
        });
        expect(forward_vector_cast_count == 1u);
        expect(backward_scalar_cast_count >= 2u);
        expect(insert_count >= 2u);
    };

    "autodiff_reverse_insert_zeroes_overwritten_base_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        auto *z = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *gx_out = b.alloca_local(Type::of<float>());
        auto *gy_out = b.alloca_local(Type::of<float>());
        auto *gz_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {y});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {z});
        auto *base = b.call(vector_type, ArithmeticOp::AGGREGATE, {x, y});
        auto *updated = b.call(vector_type, ArithmeticOp::INSERT, {base, z, m.create_constant_zero(Type::of<uint32_t>())});
        auto *loss = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {updated});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {loss, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        auto *gy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {y});
        auto *gz = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {z});
        b.store(gx_out, gx);
        b.store(gy_out, gy);
        b.store(gz_out, gz);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        bool zeroes_overwritten_slot = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arith = static_cast<ArithmeticInst *>(inst);
                if (arith->op() == ArithmeticOp::INSERT &&
                    arith->type() == vector_type &&
                    arith->operand_count() == 3u &&
                    !arith->operand(0)->isa<Constant>() &&
                    arith->operand(1)->isa<Constant>() &&
                    static_cast<Constant *>(arith->operand(1))->type() == Type::of<float>() &&
                    static_cast<Constant *>(arith->operand(1))->as<float>() == 0.0f) {
                    zeroes_overwritten_slot = true;
                }
            }
        });
        expect(zeroes_overwritten_slot);
    };

    "autodiff_snapshots_mutable_cfg_selectors"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *flag_arg = k->create_argument(Type::of<bool>(), false);
        auto *tag_arg = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *flag = b.alloca_local(Type::of<bool>());
        auto *tag = b.alloca_local(Type::of<int>());
        b.store(y, x);
        b.store(flag, flag_arg);
        b.store(tag, tag_arg);
        auto *cond = b.load(Type::of<bool>(), flag);
        auto *forward_if = b.if_(cond);
        auto *if_merge = forward_if->create_merge_block();
        auto *if_true = forward_if->create_true_block();
        auto *if_false = forward_if->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {t0, t0});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {f0});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.store(flag, m.create_constant_zero(Type::of<bool>()));
        auto *selector = b.load(Type::of<int>(), tag);
        auto *forward_switch = b.switch_(selector);
        auto *sw_merge = forward_switch->create_merge_block();
        auto *sw_default = forward_switch->create_default_block();
        auto *sw_case = forward_switch->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::COS, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        int32_t zero = 0;
        b.store(tag, m.create_constant(Type::of<int>(), &zero));
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        IfInst *backward_if = nullptr;
        SwitchInst *backward_switch = nullptr;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<IfInst>() && inst != forward_if) {
                backward_if = static_cast<IfInst *>(inst);
            } else if (inst->isa<SwitchInst>() && inst != forward_switch) {
                backward_switch = static_cast<SwitchInst *>(inst);
            }
        });
        expect(backward_if != nullptr);
        expect(backward_switch != nullptr);
        auto *if_snapshot_store = find_store_before(forward_if, nullptr, cond);
        expect(if_snapshot_store != nullptr);
        auto *backward_if_condition = backward_if == nullptr ? nullptr : backward_if->condition();
        expect(backward_if_condition != nullptr && backward_if_condition->isa<LoadInst>());
        auto *backward_if_load = backward_if_condition != nullptr && backward_if_condition->isa<LoadInst>() ?
                                     static_cast<LoadInst *>(backward_if_condition) :
                                     nullptr;
        expect(backward_if_load != nullptr && if_snapshot_store != nullptr &&
               backward_if_load->variable() == if_snapshot_store->variable());
        expect(backward_if_load != nullptr && backward_if_load->variable() != flag);
        auto *switch_snapshot_store = find_store_before(forward_switch, nullptr, selector);
        expect(switch_snapshot_store != nullptr);
        auto *backward_switch_value = backward_switch == nullptr ? nullptr : backward_switch->value();
        expect(backward_switch_value != nullptr && backward_switch_value->isa<LoadInst>());
        auto *backward_switch_load = backward_switch_value != nullptr && backward_switch_value->isa<LoadInst>() ?
                                         static_cast<LoadInst *>(backward_switch_value) :
                                         nullptr;
        expect(backward_switch_load != nullptr && switch_snapshot_store != nullptr &&
               backward_switch_load->variable() == switch_snapshot_store->variable());
        expect(backward_switch_load != nullptr && backward_switch_load->variable() != tag);
        size_t cross_block_spill_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<AllocaInst>()) { return; }
            auto *spill = inst->find_metadata<Reg2MemSpillMD>();
            if (spill == nullptr ||
                spill->kind() != Reg2MemSpillKind::CROSS_BLOCK) {
                return;
            }
            cross_block_spill_count++;
            auto has_store = false;
            auto has_load = false;
            for (auto &&use : inst->use_list()) {
                if (auto *user = use->user(); user != nullptr) {
                    has_store |= user->isa<StoreInst>();
                    has_load |= user->isa<LoadInst>();
                }
            }
            expect(has_store);
            expect(has_load);
        });
        expect(cross_block_spill_count > 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::PHI) == 0u);
        auto spill_audit = audit_reg2mem_spills_on_function(k);
        expect(spill_audit.remaining_phi_spill_count == 0u);
        expect(spill_audit.remaining_cross_block_spill_count ==
               cross_block_spill_count);
        expect(spill_audit.remaining_invalid_spill_count == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_preserves_native_switch_in_backward"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *tag = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        b.store(y, x);
        auto *sw = b.switch_(tag);
        auto *sw_merge = sw->create_merge_block();
        auto *default_block = sw->create_default_block();
        auto *case_block = sw->create_case_block(7);
        b.set_insertion_point(default_block);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(case_block);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {c0, c0});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) == 2u);
    };

    "autodiff_handles_native_pow_int"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        int32_t three = 3;
        auto *exp = m.create_constant(Type::of<int>(), &three);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *pow = b.call(Type::of<float>(), ArithmeticOp::POW_INT, {x, exp});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {pow, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t pow_int_count = 0u;
        bool has_exponent_cast = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::POW_INT) {
                pow_int_count++;
            } else if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->operand(0)->type() == Type::of<int>()) {
                    has_exponent_cast = true;
                }
            }
        });
        expect(pow_int_count == 2u);
        expect(has_exponent_cast);
    };

    "autodiff_handles_native_smoothstep"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *edge0 = k->create_argument(Type::of<float>(), false);
        auto *edge1 = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {edge0});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {edge1});
        auto *smooth = b.call(Type::of<float>(), ArithmeticOp::SMOOTHSTEP, {edge0, edge1, x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {smooth, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t smoothstep_count = 0u;
        size_t saturate_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arith = static_cast<ArithmeticInst *>(inst);
                if (arith->op() == ArithmeticOp::SMOOTHSTEP) {
                    smoothstep_count++;
                } else if (arith->op() == ArithmeticOp::SATURATE) {
                    saturate_count++;
                }
            }
        });
        expect(smoothstep_count == 1u);
        expect(saturate_count >= 1u);
    };

    "autodiff_accumulate_gradient_marks_reverse_root"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        auto *loss = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {s, s});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT, {loss, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_reverse_bounded_dynamic_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *n = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, n});
        prepare->add_comment(
            "bounded autodiff prepare provenance");
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {s, x});
        b.store(y, sum);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 2u);
        size_t prepare_metadata_count = 0u;
        for (auto *block : k->definition()->basic_blocks()) {
            if (auto *comment =
                    block->find_metadata<CommentMD>();
                comment != nullptr &&
                comment->comment() ==
                    "bounded autodiff prepare provenance") {
                ++prepare_metadata_count;
            }
        }
        // Original prepare + 64 guarded evaluations + overflow check.
        expect(prepare_metadata_count >= 66u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_reverse_bounded_dynamic_loop_with_nested_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *n = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        int32_t zero = 0;
        int32_t one = 1;
        int32_t two = 2;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *two_c = m.create_constant(Type::of<int>(), &two);
        b.store(y, x);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, n});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *iv_body = b.load(Type::of<int>(), i);
        auto *parity = b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {iv_body, two_c});
        auto *branch_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {parity, zero_c});
        auto *if_inst = b.if_(branch_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *yt0 = b.load(Type::of<float>(), y);
        auto *yt1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {yt0});
        auto *yt2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yt1, x});
        b.store(y, yt2);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::COS, {yf0});
        auto *yf2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {yf1, x});
        b.store(y, yf2);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        bool all_terminated = true;
        k->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            all_terminated &= block->is_terminated();
        });
        expect(all_terminated);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 2u);
    };

    "autodiff_inlines_callable_before_reverse_pass"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<float>());
        auto *callee_arg = callee->create_argument(Type::of<float>(), false);
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *sin_x = b.call(Type::of<float>(), ArithmeticOp::SIN, {callee_arg});
        auto *mul = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {sin_x, callee_arg});
        b.return_(mul);
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *call = b.call(Type::of<float>(), callee, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {call, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(count_reachable_insts(k, DerivedInstructionTag::CALL) == 1u);
        auto inline_info = inline_all_pass_run_on_module(&m);
        expect(inline_info.inlined_call_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CALL) == 0u);
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_before_reverse_pass"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *four_c = m.create_constant(Type::of<int>(), &four);
        b.store(i, zero_c);
        auto *loop = b.loop();
        loop->add_comment("autodiff loop provenance");
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        loop_body->add_comment(
            "autodiff body block provenance");
        update->add_comment(
            "autodiff update block provenance");
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, four_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        b.store(y, s);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        auto *loop_comment =
            entry->terminator()->find_metadata<CommentMD>();
        expect(loop_comment != nullptr);
        if (loop_comment != nullptr) {
            expect(loop_comment->comment() ==
                   "autodiff loop provenance");
        }
        size_t cloned_body_metadata_count = 0u;
        size_t cloned_update_metadata_count = 0u;
        for (auto *block : k->definition()->basic_blocks()) {
            if (auto *comment =
                    block->find_metadata<CommentMD>()) {
                cloned_body_metadata_count +=
                    comment->comment() ==
                    "autodiff body block provenance";
                cloned_update_metadata_count +=
                    comment->comment() ==
                    "autodiff update block provenance";
            }
        }
        expect(cloned_body_metadata_count >= 2u);
        expect(cloned_update_metadata_count >= 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_fixed_trip_rejects_body_write_to_induction_storage"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *value = b.alloca_local(Type::of<float>());
        auto *index = b.alloca_local(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t trip_count = 4;
        auto *bound = m.create_constant(Type::of<int>(), &trip_count);
        b.store(value, x);
        b.store(index, zero);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), index);
        auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *old_value = b.load(Type::of<float>(), value);
        b.store(value, b.call(Type::of<float>(), ArithmeticOp::SIN, {old_value}));
        // This extra write makes the apparent canonical update advance twice
        // per iteration. Treating the loop as four fixed trips is unsound.
        auto *body_index = b.load(Type::of<int>(), index);
        b.store(index, b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {body_index, one}));
        b.br(update);
        b.set_insertion_point(update);
        auto *update_index = b.load(Type::of<int>(), index);
        b.store(index, b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {update_index, one}));
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *output = b.load(Type::of<float>(), value);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        // The conservative dynamic lowering retains all 64 guarded copies.
        // The old unsound fixed-trip proof emitted only four copies.
        size_t sin_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::SIN) {
                sin_count++;
            }
        });
        expect(sin_count == 64u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_fixed_trip_rejects_effectful_prepare_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *value = b.alloca_local(Type::of<float>());
        auto *index = b.alloca_local(Type::of<int>());
        auto *prepare_count = b.alloca_local(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t trip_count = 4;
        auto *bound = m.create_constant(Type::of<int>(), &trip_count);
        b.store(value, x);
        b.store(index, zero);
        b.store(prepare_count, zero);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *old_count = b.load(Type::of<int>(), prepare_count);
        b.store(prepare_count,
                b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {old_count, one}));
        auto *iv = b.load(Type::of<int>(), index);
        auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *old_value = b.load(Type::of<float>(), value);
        b.store(value, b.call(Type::of<float>(), ArithmeticOp::SIN, {old_value}));
        b.br(update);
        b.set_insertion_point(update);
        auto *update_index = b.load(Type::of<int>(), index);
        b.store(index, b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {update_index, one}));
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *loaded_output = b.load(Type::of<float>(), value);
        auto *observed_prepare_count =
            b.static_cast_(Type::of<float>(), b.load(Type::of<int>(), prepare_count));
        auto *output = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                              {loaded_output, observed_prepare_count});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        size_t prepare_store_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (auto *store = inst->isa<StoreInst>() ?
                                  static_cast<StoreInst *>(inst) :
                                  nullptr;
                store != nullptr && store->variable() == prepare_count) {
                prepare_store_count++;
            }
        });
        // One initializer plus 64 guarded checks and the overflow check. The
        // original fixed lowering emitted only one initializer plus four
        // checks, omitting the required final false-condition evaluation.
        expect(prepare_store_count == 66u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_fixed_trip_analysis_honors_narrow_integer_wrapping"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *value = b.alloca_local(Type::of<float>());
        auto *index = b.alloca_local(Type::of<int8_t>());
        int8_t start_value = 126;
        int8_t bound_value = 0;
        int8_t step_value = 1;
        auto *start = m.create_constant(Type::of<int8_t>(), &start_value);
        auto *bound = m.create_constant(Type::of<int8_t>(), &bound_value);
        auto *step = m.create_constant(Type::of<int8_t>(), &step_value);
        b.store(value, x);
        b.store(index, start);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *current_index = b.load(Type::of<int8_t>(), index);
        auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {current_index, bound});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *current_value = b.load(Type::of<float>(), value);
        auto *next_value = b.call(Type::of<float>(), ArithmeticOp::SIN,
                                  {b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {current_value, x})});
        b.store(value, next_value);
        b.br(update);
        b.set_insertion_point(update);
        auto *old_index = b.load(Type::of<int8_t>(), index);
        auto *next_index = b.call(Type::of<int8_t>(), ArithmeticOp::BINARY_ADD, {old_index, step});
        b.store(index, next_index);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *output = b.load(Type::of<float>(), value);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(k);

        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        size_t sin_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::SIN) {
                sin_count++;
            }
        });
        expect(sin_count == 2u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_explicit_step_xor_condition"_test = [] {
        auto run_case = [](int32_t start_v, int32_t bound_v, int32_t step_v) {
            Module m;
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            auto *x = k->create_argument(Type::of<float>(), false);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *scope = b.autodiff_scope();
            auto *merge = scope->create_merge_block();
            auto *entry = scope->create_entry_block();
            b.set_insertion_point(entry);
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
            auto *y = b.alloca_local(Type::of<float>());
            auto *i = b.alloca_local(Type::of<int>());
            auto *step = b.alloca_local(Type::of<int>());
            int32_t zero_v = 0;
            auto *start = m.create_constant(Type::of<int>(), &start_v);
            auto *bound = m.create_constant(Type::of<int>(), &bound_v);
            auto *step_c = m.create_constant(Type::of<int>(), &step_v);
            auto *zero = m.create_constant(Type::of<int>(), &zero_v);
            b.store(y, x);
            b.store(i, start);
            b.store(step, step_c);
            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *loop_merge = loop->create_merge_block();
            b.set_insertion_point(prepare);
            auto *iv = b.load(Type::of<int>(), i);
            auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
            auto *cond_step = b.load(Type::of<int>(), step);
            auto *neg_step = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {cond_step, zero});
            auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_BIT_XOR, {cmp, neg_step});
            b.cond_br(cond, loop_body, loop_merge);
            b.set_insertion_point(loop_body);
            auto *yv = b.load(Type::of<float>(), y);
            auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yv, x});
            auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {sum});
            b.store(y, s);
            b.br(update);
            b.set_insertion_point(update);
            auto *iv_next_base = b.load(Type::of<int>(), i);
            auto *step_update = b.load(Type::of<int>(), step);
            auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, step_update});
            b.store(i, iv_next);
            b.br(prepare);
            b.set_insertion_point(loop_merge);
            auto *out = b.load(Type::of<float>(), y);
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();
            auto info = autodiff_pass_run_on_function(k);
            expect(info.transformed_scope_count == 1u);
            expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
            expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
            expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        };
        run_case(0, 6, 2);
        run_case(3, 0, -1);
    };

    "autodiff_unrolls_fixed_trip_loop_with_update_state_store"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        float scale_v = 0.5f;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *scale = m.create_constant(Type::of<float>(), &scale_v);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        b.store(y, s);
        b.br(update);
        b.set_insertion_point(update);
        auto *yu = b.load(Type::of<float>(), y);
        auto *y_next = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yu, scale});
        b.store(y, y_next);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_nested_cfg_before_dce"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        float half = 0.5f;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *half_c = m.create_constant(Type::of<float>(), &half);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *if_y = b.load(Type::of<float>(), y);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {if_y, half_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {t0, x});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {f0, x});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        auto scalarizer_info = scalarizer_pass_run_on_function(k);
        static_cast<void>(scalarizer_info);
        auto sroa_info = sroa_pass_run_on_function(k);
        static_cast<void>(sroa_info);
        auto dce_info = dce_pass_run_on_function(k);
        static_cast<void>(dce_info);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) >= 6u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 6u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_vector_state_before_dce"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *float2_t = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *p = b.alloca_local(float2_t);
        auto *v = b.alloca_local(float2_t);
        auto *i = b.alloca_local(Type::of<int>());
        uint32_t ix = 0u;
        uint32_t iy = 1u;
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        float c025 = 0.25f;
        float c05 = 0.5f;
        auto *ix_c = m.create_constant(Type::of<uint32_t>(), &ix);
        auto *iy_c = m.create_constant(Type::of<uint32_t>(), &iy);
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *four_c = m.create_constant(Type::of<int>(), &four);
        auto *c025_c = m.create_constant(Type::of<float>(), &c025);
        auto *c05_c = m.create_constant(Type::of<float>(), &c05);
        auto *x2 = b.call(float2_t, ArithmeticOp::AGGREGATE, {x, x});
        auto *v0 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {x2, c025_c});
        b.store(p, x2);
        b.store(v, v0);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, four_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(2);
        b.set_insertion_point(sw_default);
        auto *pd0 = b.load(float2_t, p);
        auto *vd0 = b.load(float2_t, v);
        auto *pd1 = b.call(float2_t, ArithmeticOp::BINARY_ADD, {pd0, vd0});
        b.store(p, pd1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *pc0 = b.load(float2_t, p);
        auto *vc0 = b.load(float2_t, v);
        auto *vc1 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {vc0, pc0});
        b.store(v, vc1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *pl = b.load(float2_t, p);
        auto *px = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {pl, ix_c});
        auto *py = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {pl, iy_c});
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {px, py});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *vt0 = b.load(float2_t, v);
        auto *vt1 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {vt0, c05_c});
        b.store(v, vt1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *pf0 = b.load(float2_t, p);
        auto *pf1 = b.call(float2_t, ArithmeticOp::BINARY_ADD, {pf0, x2});
        b.store(p, pf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out_p = b.load(float2_t, p);
        auto *out_v = b.load(float2_t, v);
        auto *out_px = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_p, ix_c});
        auto *out_py = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_p, iy_c});
        auto *out_vx = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_v, ix_c});
        auto *sum0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {out_px, out_py});
        auto *out = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {sum0, out_vx});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        auto scalarizer_info = scalarizer_pass_run_on_function(k);
        static_cast<void>(scalarizer_info);
        auto sroa_info = sroa_pass_run_on_function(k);
        static_cast<void>(sroa_info);
        auto dce_info = dce_pass_run_on_function(k);
        static_cast<void>(dce_info);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) >= 8u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 8u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_continue_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t two = 2;
        int32_t five = 5;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *two_c = m.create_constant(Type::of<int>(), &two);
        auto *five_c = m.create_constant(Type::of<int>(), &five);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, five_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *if_iv = b.load(Type::of<int>(), i);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {if_iv, two_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.continue_(update);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {yf0});
        b.store(y, yf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        auto *ym0 = b.load(Type::of<float>(), y);
        auto *ym1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {ym0, x});
        b.store(y, ym1);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_break_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        int32_t six = 6;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *six_c = m.create_constant(Type::of<int>(), &six);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, six_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv0 = b.load(Type::of<float>(), y);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yv0, x});
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {sum});
        b.store(y, s);
        auto *if_iv = b.load(Type::of<int>(), i);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {if_iv, three_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.break_(loop_merge);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {yf0, x});
        b.store(y, yf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
    };

    "autodiff_early_exit_normalization_preserves_false_loop_prepare"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *scope_merge = scope->create_merge_block();
        auto *scope_entry = scope->create_entry_block();
        b.set_insertion_point(scope_entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t trip_count = 3;
        auto *trip_count_value = m.create_constant(Type::of<int>(), &trip_count);
        b.store(y, x);
        b.store(i, zero);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {iv, trip_count_value});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *value = b.load(Type::of<float>(), y);
        auto *next_value = b.call(Type::of<float>(), ArithmeticOp::SIN, {value});
        b.store(y, next_value);
        auto *break_iv = b.load(Type::of<int>(), i);
        auto *break_condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {break_iv, one});
        auto *break_if = b.if_(break_condition);
        auto *break_block = break_if->create_true_block();
        auto *continue_block = break_if->create_false_block();
        auto *break_merge = break_if->create_merge_block();
        b.set_insertion_point(break_block);
        b.break_(loop_merge);
        b.set_insertion_point(continue_block);
        b.br(break_merge);
        b.set_insertion_point(break_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *update_iv = b.load(Type::of<int>(), i);
        auto *next_iv = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {update_iv, one});
        b.store(i, next_iv);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(scope_merge);

        b.set_insertion_point(scope_merge);
        auto *false_loop = b.loop();
        auto *false_prepare = false_loop->create_prepare_block();
        auto *false_body = false_loop->create_body_block();
        auto *false_update = false_loop->create_update_block();
        auto *false_merge = false_loop->create_merge_block();
        auto *false_value = m.create_constant_zero(Type::of<bool>());
        b.set_insertion_point(false_prepare);
        auto *false_phi = b.phi(Type::of<bool>());
        false_phi->add_incoming(false_value, scope_merge);
        false_phi->add_incoming(false_phi, false_update);
        b.cond_br(false_phi, false_body, false_merge);
        b.set_insertion_point(false_body);
        static_cast<void>(b.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT, {false_value}));
        b.br(false_update);
        b.set_insertion_point(false_update);
        b.br(false_prepare);
        b.set_insertion_point(false_merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);

        LoopInst *remaining_loop = nullptr;
        size_t remaining_loop_count = 0u;
        for (auto *block : k->basic_blocks()) {
            if (block->is_terminated() && block->terminator()->isa<LoopInst>()) {
                remaining_loop = static_cast<LoopInst *>(block->terminator());
                remaining_loop_count++;
            }
        }
        expect(remaining_loop_count == 1u);
        if (remaining_loop != nullptr) {
            auto *remaining_prepare = remaining_loop->prepare_block();
            expect(remaining_prepare != nullptr);
            if (remaining_prepare != nullptr) {
                expect(remaining_prepare->terminator()->isa<ConditionalBranchInst>());
                if (remaining_prepare->terminator()->isa<ConditionalBranchInst>()) {
                    auto *branch = static_cast<ConditionalBranchInst *>(remaining_prepare->terminator());
                    expect(branch->condition()->isa<Constant>());
                    if (branch->condition()->isa<Constant>()) {
                        expect(!static_cast<Constant *>(branch->condition())->as<bool>());
                    }
                    expect(branch->true_block() == remaining_loop->body_block());
                    expect(branch->false_block() == remaining_loop->merge_block());
                }
            }
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_unrolls_fixed_trip_loop_with_switch_early_exit_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        int32_t six = 6;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *six_c = m.create_constant(Type::of<int>(), &six);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, six_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_continue = sw->create_case_block(1);
        auto *sw_break = sw->create_case_block(4);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {d0, x});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_continue);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.continue_(update);
        b.set_insertion_point(sw_break);
        auto *b0 = b.load(Type::of<float>(), y);
        auto *b1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {b0});
        b.store(y, b1);
        b.break_(loop_merge);
        b.set_insertion_point(sw_merge);
        auto *m0 = b.load(Type::of<float>(), y);
        auto *m1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {m0, x});
        b.store(y, m1);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_fixed_trip_preserves_final_prepare_ssa_values"_test = [] {
        auto run_case = [](int32_t start_value,
                           int32_t bound_value) noexcept {
            Module m;
            BasicBlock *body;
            auto *kernel = make_kernel_with_body(m, body);
            auto *x =
                kernel->create_argument(Type::of<float>(), false);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *scope = b.autodiff_scope();
            auto *scope_merge = scope->create_merge_block();
            auto *scope_entry = scope->create_entry_block();
            b.set_insertion_point(scope_entry);
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
                {x});
            auto *index = b.alloca_local(Type::of<int32_t>());
            auto *start = m.create_constant(
                Type::of<int32_t>(), &start_value);
            auto *bound = m.create_constant(
                Type::of<int32_t>(), &bound_value);
            auto one_value = int32_t{1};
            auto *one = m.create_constant(
                Type::of<int32_t>(), &one_value);
            b.store(index, start);
            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *loop_merge = loop->create_merge_block();
            b.set_insertion_point(prepare);
            auto *current = b.load(Type::of<int32_t>(), index);
            auto *condition = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                {current, bound});
            b.cond_br(condition, loop_body, loop_merge);
            b.set_insertion_point(loop_body);
            b.br(update);
            b.set_insertion_point(update);
            auto *old_index =
                b.load(Type::of<int32_t>(), index);
            auto *next_index = b.call(
                Type::of<int32_t>(),
                ArithmeticOp::BINARY_ADD,
                {old_index, one});
            b.store(index, next_index);
            b.br(prepare);
            b.set_insertion_point(loop_merge);
            auto *zero_f =
                m.create_constant_zero(Type::of<float>());
            auto *one_f =
                m.create_constant_one(Type::of<float>());
            // The original loop evaluates prepare once more on exit. This use
            // therefore observes the final false condition, including for a
            // zero-trip loop.
            auto *scale = b.call(
                Type::of<float>(), ArithmeticOp::SELECT,
                {zero_f, one_f, condition});
            auto *out = b.call(
                Type::of<float>(), ArithmeticOp::BINARY_MUL,
                {x, scale});
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
                {out, one_f});
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
            b.br(scope_merge);
            b.set_insertion_point(scope_merge);
            b.return_void();
            expect(xir_verify_module(&m).succeeded());

            auto info =
                autodiff_pass_run_on_function(kernel);

            expect(info.transformed_scope_count == 1u);
            expect(count_reachable_insts(
                       kernel,
                       DerivedInstructionTag::LOOP) == 0u);
            expect(scale->operand(2u)->isa<LoadInst>());
            if (scale->operand(2u)->isa<LoadInst>()) {
                expect(static_cast<LoadInst *>(
                           scale->operand(2u))
                           ->parent_block() == loop_merge);
            }
            expect(xir_verify_module(&m).succeeded());
        };
        run_case(0, 2);
        run_case(2, 2);
    };

    "autodiff_dynamic_loop_preserves_final_prepare_ssa_value"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *x = kernel->create_argument(
            Type::of<float>(), false);
        auto *bound = kernel->create_argument(
            Type::of<int32_t>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *scope_merge = scope->create_merge_block();
        auto *scope_entry = scope->create_entry_block();
        b.set_insertion_point(scope_entry);
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
            {x});
        auto *index = b.alloca_local(Type::of<int32_t>());
        auto *zero_i =
            m.create_constant_zero(Type::of<int32_t>());
        auto *one_i =
            m.create_constant_one(Type::of<int32_t>());
        b.store(index, zero_i);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *current = b.load(Type::of<int32_t>(), index);
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {current, bound});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        auto *old_index =
            b.load(Type::of<int32_t>(), index);
        auto *next_index = b.call(
            Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
            {old_index, one_i});
        b.store(index, next_index);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *zero_f =
            m.create_constant_zero(Type::of<float>());
        auto *one_f =
            m.create_constant_one(Type::of<float>());
        auto *scale = b.call(
            Type::of<float>(), ArithmeticOp::SELECT,
            {zero_f, one_f, condition});
        auto *out = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_MUL,
            {x, scale});
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
            {out, one_f});
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(scope_merge);
        b.set_insertion_point(scope_merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = autodiff_pass_run_on_function(kernel);

        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::LOOP) == 0u);
        expect(scale->operand(2u)->isa<LoadInst>());
        if (scale->operand(2u)->isa<LoadInst>()) {
            expect(static_cast<LoadInst *>(
                       scale->operand(2u))
                       ->parent_block() == loop_merge);
        }
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_loop_clone_is_independent_of_structural_block_order"_test = [] {
        Module m;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(m, body);
        auto *x = kernel->create_argument(
            Type::of<float>(), false);
        auto *take_value = kernel->create_argument(
            Type::of<bool>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *scope_merge = scope->create_merge_block();
        auto *scope_entry = scope->create_entry_block();
        b.set_insertion_point(scope_entry);
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
            {x});
        auto *result = b.alloca_local(Type::of<float>());
        auto *index = b.alloca_local(Type::of<int32_t>());
        auto *zero_i =
            m.create_constant_zero(Type::of<int32_t>());
        auto *one_i =
            m.create_constant_one(Type::of<int32_t>());
        b.store(index, zero_i);
        b.store(
            result,
            m.create_constant_zero(Type::of<float>()));
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *current = b.load(Type::of<int32_t>(), index);
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {current, one_i});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *branch = b.if_(take_value);
        auto *branch_true = branch->create_true_block();
        auto *branch_false = branch->create_false_block();
        auto *branch_merge = branch->create_merge_block();
        b.set_insertion_point(branch_true);
        auto *value = b.call(
            Type::of<float>(), ArithmeticOp::SIN, {x});
        b.br(branch_merge);
        b.set_insertion_point(branch_false);
        b.unreachable_("the sibling arm does not enter the merge");
        b.set_insertion_point(branch_merge);
        // branch_true is the only executable predecessor, so `value`
        // dominates this use even though a naïve structural DFS may visit the
        // merge block before branch_true.
        b.store(result, value);
        b.br(update);
        b.set_insertion_point(update);
        auto *old_index =
            b.load(Type::of<int32_t>(), index);
        auto *next_index = b.call(
            Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
            {old_index, one_i});
        b.store(index, next_index);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), result);
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
            {out, m.create_constant_one(Type::of<float>())});
        b.call(
            Type::of<void>(),
            AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(scope_merge);
        b.set_insertion_point(scope_merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = autodiff_pass_run_on_function(kernel);

        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(
                   kernel, DerivedInstructionTag::LOOP) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "autodiff_loop_expansion_lowers_nested_selection_phi"_test = [] {
        auto run_case = [](bool dynamic_bound) noexcept {
            Module m;
            BasicBlock *body;
            auto *kernel = make_kernel_with_body(m, body);
            auto *x =
                kernel->create_argument(Type::of<float>(), false);
            auto *choose_sine =
                kernel->create_argument(Type::of<bool>(), false);
            auto *dynamic =
                dynamic_bound ?
                    kernel->create_argument(
                        Type::of<int32_t>(), false) :
                    nullptr;
            auto *zero_i =
                m.create_constant_zero(Type::of<int32_t>());
            auto *one_i =
                m.create_constant_one(Type::of<int32_t>());
            auto two_value = int32_t{2};
            auto *two_i = m.create_constant(
                Type::of<int32_t>(), &two_value);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *scope = b.autodiff_scope();
            auto *scope_merge = scope->create_merge_block();
            auto *scope_entry = scope->create_entry_block();
            b.set_insertion_point(scope_entry);
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
                {x});
            auto *index =
                b.alloca_local(Type::of<int32_t>());
            auto *result =
                b.alloca_local(Type::of<float>());
            b.store(index, zero_i);
            b.store(
                result,
                m.create_constant_zero(Type::of<float>()));

            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *loop_merge = loop->create_merge_block();
            b.set_insertion_point(prepare);
            auto *current =
                b.load(Type::of<int32_t>(), index);
            auto *condition = b.call(
                Type::of<bool>(),
                ArithmeticOp::BINARY_LESS,
                {current,
                 dynamic_bound ?
                     static_cast<Value *>(dynamic) :
                     static_cast<Value *>(two_i)});
            b.cond_br(condition, loop_body, loop_merge);

            b.set_insertion_point(loop_body);
            auto *selection = b.if_(choose_sine);
            auto *sine_block =
                selection->create_true_block();
            auto *cosine_block =
                selection->create_false_block();
            auto *selection_merge =
                selection->create_merge_block();
            b.set_insertion_point(sine_block);
            auto *sine = b.call(
                Type::of<float>(),
                ArithmeticOp::SIN, {x});
            b.br(selection_merge);
            b.set_insertion_point(cosine_block);
            auto *cosine = b.call(
                Type::of<float>(),
                ArithmeticOp::COS, {x});
            b.br(selection_merge);
            b.set_insertion_point(selection_merge);
            auto *selected = b.phi(
                Type::of<float>(),
                {{sine, sine_block},
                 {cosine, cosine_block}});
            b.store(result, selected);
            b.br(update);

            b.set_insertion_point(update);
            auto *old_index =
                b.load(Type::of<int32_t>(), index);
            auto *next_index = b.call(
                Type::of<int32_t>(),
                ArithmeticOp::BINARY_ADD,
                {old_index, one_i});
            b.store(index, next_index);
            b.br(prepare);
            b.set_insertion_point(loop_merge);
            auto *out =
                b.load(Type::of<float>(), result);
            auto *one_f =
                m.create_constant_one(Type::of<float>());
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
                {out, one_f});
            b.call(
                Type::of<void>(),
                AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
            b.br(scope_merge);
            b.set_insertion_point(scope_merge);
            b.return_void();
            expect(xir_verify_module(&m).succeeded());

            auto info =
                autodiff_pass_run_on_function(kernel);

            expect(info.transformed_scope_count == 1u);
            expect(count_reachable_insts(
                       kernel,
                       DerivedInstructionTag::LOOP) == 0u);
            expect(count_reachable_insts(
                       kernel,
                       DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
            expect(count_reachable_insts(
                       kernel,
                       DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
            expect(xir_verify_module(&m).succeeded());
        };
        run_case(false);
        run_case(true);
    };

    "autodiff_slots_are_reinitialized_at_nested_scope_entry"_test = [] {
        auto run_case = [](bool forward) noexcept {
            Module m;
            BasicBlock *body;
            auto *kernel = make_kernel_with_body(m, body);
            auto *x = kernel->create_argument(
                Type::of<float>(), false);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *index = b.alloca_local(Type::of<int32_t>());
            auto *zero_i =
                m.create_constant_zero(Type::of<int32_t>());
            auto *one_i =
                m.create_constant_one(Type::of<int32_t>());
            auto two_value = int32_t{2};
            auto *two_i = m.create_constant(
                Type::of<int32_t>(), &two_value);
            b.store(index, zero_i);
            auto *outer = b.loop();
            auto *outer_prepare =
                outer->create_prepare_block();
            auto *outer_body = outer->create_body_block();
            auto *outer_update =
                outer->create_update_block();
            auto *outer_merge =
                outer->create_merge_block();
            b.set_insertion_point(outer_prepare);
            auto *current =
                b.load(Type::of<int32_t>(), index);
            auto *condition = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                {current, two_i});
            b.cond_br(
                condition, outer_body, outer_merge);
            b.set_insertion_point(outer_body);
            auto *scope =
                forward ?
                    static_cast<AutodiffScopeInst *>(
                        b.forward_autodiff_scope(1u)) :
                    b.autodiff_scope();
            auto *scope_entry =
                scope->create_entry_block();
            auto *scope_merge =
                scope->create_merge_block();
            b.set_insertion_point(scope_entry);
            auto *one_f =
                m.create_constant_one(Type::of<float>());
            if (forward) {
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_PROPAGATE_GRADIENT,
                    {x, one_f});
                auto *value = b.call(
                    Type::of<float>(),
                    ArithmeticOp::SIN, {x});
                auto *gradient_index =
                    m.create_constant_zero(
                        Type::of<uint32_t>());
                static_cast<void>(b.call(
                    Type::of<float>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_OUTPUT_GRADIENT,
                    {value, gradient_index}));
            } else {
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_REQUIRES_GRADIENT,
                    {x});
                auto *value = b.call(
                    Type::of<float>(),
                    ArithmeticOp::SIN, {x});
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_GRADIENT_MARKER,
                    {value, one_f});
                b.call(
                    Type::of<void>(),
                    AutodiffIntrinsicOp::
                        AUTODIFF_BACKWARD,
                    {});
            }
            b.br(scope_merge);
            b.set_insertion_point(scope_merge);
            b.br(outer_update);
            b.set_insertion_point(outer_update);
            auto *old_index =
                b.load(Type::of<int32_t>(), index);
            auto *next_index = b.call(
                Type::of<int32_t>(),
                ArithmeticOp::BINARY_ADD,
                {old_index, one_i});
            b.store(index, next_index);
            b.br(outer_prepare);
            b.set_insertion_point(outer_merge);
            b.return_void();
            expect(xir_verify_module(&m).succeeded());
            luisa::unordered_set<const Instruction *>
                original_instructions;
            for (auto *block : kernel->basic_blocks()) {
                for (auto *instruction :
                     block->instructions()) {
                    original_instructions.emplace(
                        instruction);
                }
            }

            auto info =
                autodiff_pass_run_on_function(kernel);

            expect(info.transformed_scope_count == 1u);
            auto saw_scope_local_reset = false;
            for (auto *instruction :
                 outer_body->instructions()) {
                if (original_instructions.contains(
                        instruction) ||
                    !instruction->isa<StoreInst>()) {
                    continue;
                }
                auto *store =
                    static_cast<StoreInst *>(
                        instruction);
                saw_scope_local_reset |=
                    store->variable() != nullptr &&
                    store->variable()->isa<AllocaInst>() &&
                    store->value() != nullptr &&
                    store->value()->isa<Constant>() &&
                    static_cast<Constant *>(
                        store->value())
                        ->type()
                        ->is_float() &&
                    static_cast<Constant *>(
                        store->value())
                            ->as<float>() == 0.0f;
            }
            expect(saw_scope_local_reset)
                << "autodiff temporary state must be reset "
                   "on every dynamic entry to a nested scope";
            expect(xir_verify_module(&m).succeeded());
        };
        run_case(false);
        run_case(true);
    };
}

// Regression tests

void reg_regression() {

    "regression_vec3_sub_self_produces_vector_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<int3>();
        auto *x = m.create_constant_zero(vec_t);
        auto *sub = b.call(vec_t, ArithmeticOp::BINARY_SUB, {x, x});
        auto sub_locked = sub->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(sub_locked->use_list().empty());
    };

    "regression_int3_add_zero_preserves_vector_type"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<int3>();
        auto *zero = m.create_constant_zero(vec_t);
        int x_data[3] = {1, 2, 3};
        auto *x = m.create_constant(vec_t, x_data);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {x, zero});
        auto add_locked = add->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(add_locked->use_list().empty());
    };

    "regression_float3_mul_one_preserves_vector_type"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<float3>();
        float one_data[3] = {1.0f, 1.0f, 1.0f};
        float x_data[3] = {2.0f, 3.0f, 4.0f};
        auto *one = m.create_constant(vec_t, one_data);
        auto *x = m.create_constant(vec_t, x_data);
        auto *mul = b.call(vec_t, ArithmeticOp::BINARY_MUL, {x, one});
        auto mul_locked = mul->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(mul_locked->use_list().empty());
    };

    "regression_float_add_zero_skipped_for_nan_safety"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "regression_float_mul_zero_skipped_for_nan_safety"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 2.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "regression_constfold_int_add_value_correct"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto add_locked = add->lock();
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(add_locked->use_list().empty());
        bool found_7 = false;
        for (auto c : m.constant_list()) {
            if (c->type() == Type::of<int>()) {
                int32_t v = *static_cast<const int32_t *>(c->data());
                if (v == 7) found_7 = true;
            }
        }
        expect(found_7);
    };

    "regression_constfold_int_unary_minus_int_min_value_correct"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = std::numeric_limits<int32_t>::min();
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *neg = b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x});
        auto neg_locked = neg->lock();
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(neg_locked->use_list().empty());
        bool found = false;
        for (auto c : m.constant_list()) {
            if (c->type() == Type::of<int>()) {
                int32_t v = *static_cast<const int32_t *>(c->data());
                if (v == std::numeric_limits<int32_t>::min()) found = true;
            }
        }
        expect(found);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_pass_entry_totality();
    reg_algebraic_simplify();
    reg_const_fold();
    reg_dce();
    reg_gvn();
    reg_sccp();
    reg_simplify_libcalls();
    reg_reassociate();
    reg_cvp();
    reg_dead_arg_elim();
    reg_div_rem_pairs();
    reg_local_load_elimination();
    reg_local_store_forward();
    reg_dead_store_elimination();
    reg_loop_rotation();
    reg_scalar_evolution();
    reg_scalarizer();
    reg_phi_cleanup();
    reg_if_conversion();
    reg_reg2mem();
    reg_sroa();
    reg_inline();
    reg_unused_callable_removal();
    reg_trace_gep();
    reg_transpose_gep();
    reg_mem2reg();
    reg_promote_ref_arg();
    reg_outline();
    reg_autodiff();
    reg_regression();
    return 0;
}
