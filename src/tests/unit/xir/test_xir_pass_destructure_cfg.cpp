// Test for lowering structured XIR control flow to explicit CFG edges.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

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

[[nodiscard]] size_t count_owned_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *block : def->basic_blocks()) { ++n; }
    return n;
}

[[nodiscard]] size_t count_isa_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->isa<BranchInst>()) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_isa_cond_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->isa<ConditionalBranchInst>()) { ++n; }
    });
    return n;
}

}// namespace

void reg_destructure_cfg() {

    "destructure_empty_function"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_if_count == 0u);
        expect(info.destructured_loop_count == 0u);
        expect(info.destructured_simple_loop_count == 0u);
        expect(info.destructured_break_count == 0u);
        expect(info.destructured_continue_count == 0u);
    };

    "destructure_single_if"_test = [] {
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
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_if_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
        expect(count_isa_cond_branch(def) >= 1u);
    };

    "destructure_simple_loop_with_break"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        auto *lbody = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        b.set_insertion_point(lbody);
        b.break_(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_simple_loop_count == 1u);
        expect(info.destructured_break_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 0u);
    };

    "destructure_loop_with_continue"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prep = loop->create_prepare_block();
        auto *lbody = loop->create_body_block();
        auto *upd = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prep);
        b.br(lbody);
        b.set_insertion_point(lbody);
        b.continue_(upd);
        b.set_insertion_point(upd);
        b.break_(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_loop_count == 1u);
        expect(info.destructured_continue_count == 1u);
        expect(info.destructured_break_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::CONTINUE) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 0u);
    };

    "destructure_nested_if"_test = [] {
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
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_if_count == 2u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::IF) == 0u);
    };

    "destructure_switch_preserved"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *val = m.create_constant_zero(Type::of<int>());
        auto *sw = b.switch_(val);
        auto *def_block = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        auto *c0 = sw->create_case_block(0);
        auto *c1 = sw->create_case_block(1);
        b.set_insertion_point(c0);
        b.br(merge);
        b.set_insertion_point(c1);
        b.br(merge);
        b.set_insertion_point(def_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::SWITCH) == 1u);
        expect(info.destructured_if_count == 0u);
    };

    "destructure_ray_query_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *rq_obj = b.alloca_local(Type::of<int>());
        auto *rq = b.ray_query_loop();
        auto *disp = rq->create_dispatch_block();
        auto *merge = rq->create_merge_block();
        b.set_insertion_point(disp);
        auto *dispatch_inst = b.ray_query_dispatch(rq_obj);
        auto *on_surf = dispatch_inst->create_on_surface_candidate_block();
        auto *on_proc = dispatch_inst->create_on_procedural_candidate_block();
        dispatch_inst->set_exit_block(merge);
        b.set_insertion_point(on_surf);
        b.br(disp);
        b.set_insertion_point(on_proc);
        b.br(disp);
        b.set_insertion_point(merge);
        auto *exit_phi = b.phi(Type::of<int>());
        exit_phi->add_incoming(m.create_constant_one(Type::of<int>()), disp);
        b.return_void();
        auto lower_info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(lower_info.lowered_ray_query_loop_count == 1u);
        expect(lower_info.error_count == 0u);
        expect(lower_info.succeeded());
        expect(disp->is_terminated());
        expect(disp->terminator()->isa<UnreachableInst>());
        auto *lowered_loop = static_cast<LoopInst *>(body->terminator());
        auto *prepare_term = lowered_loop->prepare_block()->terminator();
        expect(prepare_term->isa<ConditionalBranchInst>());
        auto *prepare_branch = static_cast<ConditionalBranchInst *>(prepare_term);
        expect(prepare_branch->true_block() == lowered_loop->body_block());
        expect(prepare_branch->false_block() == lowered_loop->merge_block());
        expect(exit_phi->incoming_count() == 1u);
        expect(exit_phi->incoming(0u).block == lowered_loop->prepare_block());
        auto info = destructure_cfg_pass_run_on_function(k);
        (void)info;
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RAY_QUERY_LOOP) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::RAY_QUERY_DISPATCH) == 0u);
        expect(count_terminator_kind(def, DerivedInstructionTag::LOOP) == 0u);
    };

    "ray_query_to_loop_rejects_dispatch_phi_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *rq_obj = b.alloca_local(Type::of<int>());
        auto *rq = b.ray_query_loop();
        auto *dispatch = rq->create_dispatch_block();
        auto *merge = rq->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_phi = b.phi(Type::of<int>());
        dispatch_phi->add_incoming(m.create_constant_zero(Type::of<int>()), body);
        auto *dispatch_inst = b.ray_query_dispatch(rq_obj);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
        dispatch_phi->add_incoming(m.create_constant_one(Type::of<int>()), surface);
        dispatch_phi->add_incoming(m.create_constant_one(Type::of<int>()), procedural);
        b.set_insertion_point(surface);
        b.br(dispatch);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(body->terminator() == rq);
        expect(dispatch->terminator() == dispatch_inst);
        expect(dispatch_phi->is_linked());
        expect(count_owned_blocks(k->definition()) == block_count);
    };

    "ray_query_to_loop_rejects_handler_entry_phi_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *rq_obj = b.alloca_local(Type::of<int>());
        auto *rq = b.ray_query_loop();
        auto *dispatch = rq->create_dispatch_block();
        auto *merge = rq->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(rq_obj);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        auto *handler_phi = b.phi(Type::of<int>());
        handler_phi->add_incoming(m.create_constant_zero(Type::of<int>()), dispatch);
        b.br(dispatch);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(body->terminator() == rq);
        expect(dispatch->terminator() == dispatch_inst);
        expect(handler_phi->is_linked());
        expect(count_owned_blocks(k->definition()) == block_count);
    };

    "ray_query_to_loop_rejects_same_handler_entry_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *rq_obj = b.alloca_local(Type::of<int>());
        auto *rq = b.ray_query_loop();
        auto *dispatch = rq->create_dispatch_block();
        auto *merge = rq->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(rq_obj);
        dispatch_inst->set_exit_block(merge);
        auto *shared_handler = k->create_basic_block();
        dispatch_inst->set_on_surface_candidate_block(shared_handler);
        dispatch_inst->set_on_procedural_candidate_block(shared_handler);
        b.set_insertion_point(shared_handler);
        auto *handler_exit = b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_owned_blocks(k->definition()) == block_count);
        expect(body->terminator() == rq);
        expect(dispatch->terminator() == dispatch_inst);
        expect(dispatch_inst->on_surface_candidate_block() == shared_handler);
        expect(dispatch_inst->on_procedural_candidate_block() == shared_handler);
        expect(shared_handler->terminator() == handler_exit);
        expect(static_cast<BranchInst *>(handler_exit)->target_block() == dispatch);
    };

    "ray_query_to_loop_rejects_dispatch_as_handler_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<int>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        dispatch_inst->set_on_surface_candidate_block(dispatch);
        auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(procedural);
        auto *procedural_exit = b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(count_owned_blocks(k->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(dispatch->terminator() == dispatch_inst);
        expect(procedural->terminator() == procedural_exit);
    };

    "ray_query_to_loop_rejects_null_handler_branch_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<int>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
        b.set_insertion_point(surface);
        auto *null_branch = b.br(nullptr);
        b.set_insertion_point(procedural);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(count_owned_blocks(k->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(surface->terminator() == null_branch);
        expect(static_cast<BranchInst *>(surface->terminator())->target_block() == nullptr);
    };

    "ray_query_to_loop_rejects_later_invalid_loop_function_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *query0 = b.alloca_local(Type::of<int>());
        auto *loop0 = b.ray_query_loop();
        auto *dispatch0 = loop0->create_dispatch_block();
        auto *merge0 = loop0->create_merge_block();
        b.set_insertion_point(dispatch0);
        auto *dispatch_inst0 = b.ray_query_dispatch(query0);
        dispatch_inst0->set_exit_block(merge0);
        auto *surface0 = dispatch_inst0->create_on_surface_candidate_block();
        auto *procedural0 = dispatch_inst0->create_on_procedural_candidate_block();
        b.set_insertion_point(surface0);
        auto *surface_exit0 = b.br(dispatch0);
        b.set_insertion_point(procedural0);
        auto *procedural_exit0 = b.br(dispatch0);

        b.set_insertion_point(merge0);
        auto *query1 = b.alloca_local(Type::of<int>());
        auto *loop1 = b.ray_query_loop();
        auto *dispatch1 = loop1->create_dispatch_block();
        auto *merge1 = loop1->create_merge_block();
        b.set_insertion_point(dispatch1);
        auto *dispatch_phi = b.phi(Type::of<int>());
        dispatch_phi->add_incoming(m.create_constant_zero(Type::of<int>()), merge0);
        auto *dispatch_inst1 = b.ray_query_dispatch(query1);
        dispatch_inst1->set_exit_block(merge1);
        auto *surface1 = dispatch_inst1->create_on_surface_candidate_block();
        auto *procedural1 = dispatch_inst1->create_on_procedural_candidate_block();
        dispatch_phi->add_incoming(m.create_constant_one(Type::of<int>()), surface1);
        dispatch_phi->add_incoming(m.create_constant_one(Type::of<int>()), procedural1);
        b.set_insertion_point(surface1);
        b.br(dispatch1);
        b.set_insertion_point(procedural1);
        b.br(dispatch1);
        b.set_insertion_point(merge1);
        b.return_void();
        auto block_count = count_owned_blocks(k->definition());

        auto info = lower_ray_query_loop_to_loop_pass_run_on_function(k);
        expect(info.lowered_ray_query_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_owned_blocks(k->definition()) == block_count);
        expect(body->terminator() == loop0);
        expect(dispatch0->terminator() == dispatch_inst0);
        expect(surface0->terminator() == surface_exit0);
        expect(procedural0->terminator() == procedural_exit0);
        expect(merge0->terminator() == loop1);
        expect(dispatch1->terminator() == dispatch_inst1);
        expect(dispatch_phi->is_linked());
    };

    "destructure_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        for (size_t i = 0; i < kFns; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sl = b.simple_loop();
            auto *lbody = sl->create_body_block();
            auto *merge = sl->create_merge_block();
            b.set_insertion_point(lbody);
            b.break_(merge);
            b.set_insertion_point(merge);
            b.return_void();
        }
        auto info = destructure_cfg_pass_run_on_module(&m);
        expect(info.destructured_simple_loop_count == kFns);
        expect(info.destructured_break_count == kFns);
        for (auto f : m.function_list()) {
            auto *def = f->definition();
            if (def == nullptr) { continue; }
            expect(count_terminator_kind(def, DerivedInstructionTag::SIMPLE_LOOP) == 0u);
            expect(count_terminator_kind(def, DerivedInstructionTag::BREAK) == 0u);
        }
    };

    "destructure_external_function_skipped"_test = [] {
        Module m;
        auto *ext = m.create_external_function(Type::of<void>());
        auto info = destructure_cfg_pass_run_on_function(ext);
        expect(info.destructured_if_count == 0u);
        expect(info.destructured_loop_count == 0u);
    };

    "destructure_disconnected_owned_structured_region"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto *disconnected = k->create_basic_block();
        b.set_insertion_point(disconnected);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(if_true);
        b.br(merge);
        b.set_insertion_point(if_false);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_if_count == 1u);
        expect(disconnected->terminator()->isa<ConditionalBranchInst>());
        for (auto *block : k->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                expect(!inst->isa<IfInst>());
            }
        }
    };

    "destructure_patches_disconnected_unterminated_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto *disconnected = k->create_basic_block();

        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.leaked_block_count == 1u);
        expect(!info.succeeded());
        expect(disconnected->terminator()->isa<UnreachableInst>());
    };

    "destructure_malformed_if_is_not_counted_or_retried"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));

        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_if_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(body->terminator() == if_inst);
    };

    "destructure_malformed_construct_rejects_function_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *valid_header = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *malformed = b.if_(m.create_constant_one(Type::of<bool>()));
        b.set_insertion_point(valid_header);
        auto *valid_loop = b.simple_loop();
        auto *valid_body = valid_loop->create_body_block();
        auto *valid_merge = valid_loop->create_merge_block();
        b.set_insertion_point(valid_body);
        b.break_(valid_merge);
        b.set_insertion_point(valid_merge);
        b.return_void();

        auto info = destructure_cfg_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.destructured_simple_loop_count == 0u);
        expect(info.destructured_break_count == 0u);
        expect(body->terminator() == malformed);
        expect(valid_header->terminator() == valid_loop);
    };

    "destructure_if_with_foreign_merge_rejects_function_atomically"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *foreign = m.create_callable(nullptr);
        auto *foreign_body = foreign->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(foreign_body);
        b.return_void();
        b.set_insertion_point(body);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *owned_merge = if_inst->create_merge_block();
        if_inst->set_merge_block(foreign_body);
        b.set_insertion_point(true_block);
        b.br(owned_merge);
        b.set_insertion_point(false_block);
        b.br(owned_merge);
        b.set_insertion_point(owned_merge);
        b.return_void();

        auto info = destructure_cfg_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.error_count == 1u);
        expect(info.destructured_if_count == 0u);
        expect(body->terminator() == if_inst);
        expect(if_inst->merge_block() == foreign_body);
    };

    "destructure_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sl = b.simple_loop();
        auto *lbody = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        b.set_insertion_point(lbody);
        b.break_(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto first = destructure_cfg_pass_run_on_function(k);
        auto second = destructure_cfg_pass_run_on_function(k);
        expect(first.destructured_simple_loop_count == 1u);
        expect(first.destructured_break_count == 1u);
        expect(second.destructured_simple_loop_count == 0u);
        expect(second.destructured_break_count == 0u);
    };

    "destructure_empty_module_runs_cleanly"_test = [] {
        Module m;
        auto info = destructure_cfg_pass_run_on_module(&m);
        expect(info.destructured_if_count == 0u);
        expect(info.destructured_loop_count == 0u);
    };

    "destructure_if_branch_targets_preserved"_test = [] {
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
        auto *parent = body;
        (void)destructure_cfg_pass_run_on_function(k);
        auto *new_term = parent->terminator();
        expect(new_term->isa<ConditionalBranchInst>());
        auto *cbr = static_cast<ConditionalBranchInst *>(new_term);
        expect(cbr->true_block() == t);
        expect(cbr->false_block() == f);
    };

    "spill_early_return_single_return_noop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_early_return_count == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "spill_early_return_void_two_returns"_test = [] {
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
        b.return_void();
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = destructure_cfg_pass_run_on_function(k);
        expect(info.destructured_early_return_count == 2u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };

    "spill_early_return_non_void_two_returns"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *v0 = m.create_constant_zero(Type::of<int>());
        auto *v1 = m.create_constant_one(Type::of<int>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.return_(v0);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_(v1);
        auto info = destructure_cfg_pass_run_on_function(c);
        expect(info.destructured_early_return_count == 2u);
        auto *def = c->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_destructure_cfg();
    return 0;
}
