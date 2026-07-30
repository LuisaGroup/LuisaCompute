// Test for coroutine state materialization and verifier-preserving failure paths.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/verifier.h>

#include <initializer_list>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] CallableFunction *make_post_split_callable(Module &m, Value *&frame_arg_out, BasicBlock *&body_out) noexcept {
    auto *cf = m.create_callable(nullptr);
    luisa::vector<const Type *> fields;
    fields.reserve(CoroFrameDesc::reserved_field_count);
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; ++i) {
        fields.emplace_back(Type::of<uint>());
    }
    frame_arg_out = cf->create_reference_argument(Type::structure(fields));
    body_out = cf->create_body_block();
    return cf;
}

[[nodiscard]] const Type *make_frame_type(
    std::initializer_list<const Type *> user_fields = {}) noexcept {
    luisa::vector<const Type *> fields;
    fields.reserve(CoroFrameDesc::reserved_field_count +
                   user_fields.size());
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; ++i) {
        fields.emplace_back(Type::of<uint>());
    }
    fields.insert(fields.end(), user_fields.begin(), user_fields.end());
    return Type::structure(fields);
}

[[nodiscard]] size_t count_inst_tag(Module &m, DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (f->definition() == nullptr) { continue; }
        auto *def = static_cast<FunctionDefinition *>(f);
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == tag) { n++; }
        });
    }
    return n;
}

}// namespace

void reg_coro_materialize() {

    "no_register_materializes_resume_without_frame_traffic"_test = [] {
        // given: callable with frame arg, no registers
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_resume(1u, frame_arg);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: nothing registered, no stores/loads for user vars
        expect(info.register_count == 0u);
        expect(info.frame_field_count == CoroFrameDesc::reserved_field_count);
        expect(info.load_inserted_count == 0u);
        expect(info.store_inserted_count == 0u);
        expect(info.succeeded());
        expect(info.callable_count == 1u);
        expect(info.resume_lowered_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 0u);
    };

    "suspend_replaced_with_store_and_return"_test = [] {
        // given: callable with CoroSuspendInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_suspend(7u, "chk", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: no CoroSuspendInst left
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 0u);
        expect(info.suspend_lowered_count == 1u);
        expect(info.callable_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::RETURN) == 1u);
    };

    "resume_replaced"_test = [] {
        // given: callable with CoroResumeInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_resume(1u, frame_arg);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: no CoroResumeInst left
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 0u);
        expect(info.resume_lowered_count == 1u);
    };

    "resume_metadata_moves_to_the_continuation_entry_block"_test = [] {
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *callable =
            make_post_split_callable(m, frame_arg, body);
        static_cast<void>(callable);
        body->set_name("continuation_entry");
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *resume = b.coro_resume(1u, frame_arg);
        resume->set_name("resume_boundary");
        resume->add_comment("resume provenance");
        b.return_void();

        auto info = coro_materialize_pass_run_on_module(&m);

        expect(info.succeeded());
        expect(info.resume_lowered_count == 1u);
        expect(!resume->is_linked());
        expect(body->name().has_value());
        if (body->name()) {
            expect(*body->name() == "resume_boundary");
        }
        expect(body->metadata_list().count_size() == 3u);
        expect(xir_verify_module(&m).succeeded());
    };

    "split_aware_materialize_rejects_ambiguous_resume_boundary_atomically"_test = [] {
        Module m;
        auto *frame_type = make_frame_type();
        auto *entry_callable = m.create_callable(nullptr);
        auto *entry_frame =
            entry_callable->create_reference_argument(frame_type);
        auto *entry = entry_callable->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *entry_return = b.return_void();

        auto *resume_callable = m.create_callable(nullptr);
        auto *resume_frame =
            resume_callable->create_reference_argument(frame_type);
        auto *resume_body = resume_callable->create_body_block();
        b.set_insertion_point(resume_body);
        auto *first = b.coro_resume(1u, resume_frame);
        auto *second = b.coro_resume(1u, resume_frame);
        auto *resume_return = b.return_void();

        CoroCfgDistillResult cfg;
        cfg.scopes.resize(2u);
        cfg.scopes[0u].trigger_token = 0u;
        cfg.scopes[1u].trigger_token = 1u;
        CoroSplitInfo split;
        split.subroutines.emplace_back(
            CoroSplitInfo::Subroutine{
                .scope_index = 0u,
                .trigger_token = 0u,
                .callable = entry_callable,
                .frame_argument = entry_frame});
        split.subroutines.emplace_back(
            CoroSplitInfo::Subroutine{
                .scope_index = 1u,
                .trigger_token = 1u,
                .callable = resume_callable,
                .frame_argument = resume_frame});

        expect(xir_verify_module(&m).succeeded());
        auto info =
            coro_materialize_pass_run_on_module_with_cfg(
                &m, cfg, split);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(first->is_linked());
        expect(second->is_linked());
        expect(entry->terminator() == entry_return);
        expect(resume_body->terminator() == resume_return);
        expect(count_inst_tag(
                   m, DerivedInstructionTag::STORE) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };

    "terminate_lowered"_test = [] {
        // given: callable with CoroTerminateInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_resume(7u, frame_arg);
        b.coro_terminate();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: CoroTerminateInst is replaced by terminal-token store + return
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_TERMINATE) == 0u);
        expect(info.terminal_lowered_count == 1u);
        expect(info.resume_lowered_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::RETURN) == 1u);
    };

    "suspend_and_terminate_both_lowered"_test = [] {
        // given: callable with both CoroSuspendInst and CoroTerminateInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);
        auto *suspend_block = cf->create_basic_block();
        auto *terminate_block = cf->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(m.create_constant_one(Type::of<bool>()), suspend_block, terminate_block);
        b.set_insertion_point(suspend_block);
        b.coro_suspend(11u, "suspend-branch", frame_arg);
        b.set_insertion_point(terminate_block);
        b.coro_terminate();

        // when
        CoroSplitInfo split;
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 0u,
            .trigger_token = 0u,
            .callable = cf,
            .frame_argument = frame_arg,
        });
        CoroCfgDistillResult cfg;
        cfg.scopes.emplace_back();
        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        // then: both independent exit kinds were actually present and lowered
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_TERMINATE) == 0u);
        expect(info.suspend_lowered_count == 1u);
        expect(info.terminal_lowered_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::RETURN) == 2u);
    };

    "token_store_has_user_registers_empty"_test = [] {
        // given: post-split callable with token store but no user registers
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_resume(42u, frame_arg);

        // simulate post-split token store
        auto token_field = 6u;
        auto *field_token = m.create_constant(Type::of<uint32_t>(), &token_field);
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_token});
        auto token_val = static_cast<uint32_t>(42u);
        auto *tok_c = m.create_constant(Type::of<uint32_t>(), &token_val);
        b.store(gep0, tok_c);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: no user registers
        expect(info.register_count == 0u);
        expect(info.frame_field_count == CoroFrameDesc::reserved_field_count);
        expect(info.callable_count == 1u);
        expect(info.resume_lowered_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 1u);
    };

    "structured_rejection_leaves_all_materialize_candidates_unchanged"_test = [] {
        Module m;
        XIRBuilder b;

        // The first callable is valid plain CFG and would otherwise mutate.
        Value *plain_frame;
        BasicBlock *plain_body;
        auto *plain = make_post_split_callable(m, plain_frame, plain_body);
        b.set_insertion_point(plain_body);
        auto *plain_suspend = b.coro_suspend(31u, "plain", plain_frame);

        // The second callable contains a structured If around a suspend.
        Value *structured_frame;
        BasicBlock *structured_body;
        auto *structured = make_post_split_callable(m, structured_frame, structured_body);
        b.set_insertion_point(structured_body);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge_block = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        auto *structured_suspend = b.coro_suspend(37u, "structured", structured_frame);
        b.set_insertion_point(false_block);
        b.br(merge_block);
        b.set_insertion_point(merge_block);
        b.return_void();

        auto legacy_info = coro_materialize_pass_run_on_module(&m);

        expect(!legacy_info.succeeded());
        expect(legacy_info.structured_cfg_error_count == 1u);
        expect(legacy_info.callable_count == 0u);
        expect(plain_body->terminator() == plain_suspend);
        expect(structured_body->terminator() == if_inst);
        expect(true_block->terminator() == structured_suspend);
        expect(if_inst->merge_block() == merge_block);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 2u);

        // Production uses the split-aware public entry; it must be atomic too.
        CoroCfgDistillResult cfg;
        cfg.scopes.resize(2u);
        cfg.scopes[1u].trigger_token = 1u;
        CoroSplitInfo split;
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 0u,
            .callable = plain,
            .frame_argument = plain_frame,
        });
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 1u,
            .trigger_token = 1u,
            .callable = structured,
            .frame_argument = structured_frame,
        });
        auto split_aware_info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        expect(!split_aware_info.succeeded());
        expect(split_aware_info.structured_cfg_error_count == 1u);
        expect(split_aware_info.callable_count == 0u);
        expect(plain_body->terminator() == plain_suspend);
        expect(structured_body->terminator() == if_inst);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 2u);
    };

    "invalid_split_metadata_is_fully_preflighted_and_atomic"_test = [] {
        Module m;
        Module foreign_module;
        XIRBuilder b;

        Value *valid_frame;
        BasicBlock *valid_body;
        auto *valid = make_post_split_callable(m, valid_frame, valid_body);
        b.set_insertion_point(valid_body);
        auto *valid_suspend = b.coro_suspend(41u, "valid", valid_frame);

        // This structured callable deliberately reuses scope zero. Projection
        // before validation would overwrite/hide one of these two requests.
        Value *structured_frame;
        BasicBlock *structured_body;
        auto *structured = make_post_split_callable(m, structured_frame, structured_body);
        b.set_insertion_point(structured_body);
        auto *structured_if = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *structured_true = structured_if->create_true_block();
        auto *structured_false = structured_if->create_false_block();
        auto *structured_merge = structured_if->create_merge_block();
        b.set_insertion_point(structured_true);
        auto *structured_suspend = b.coro_suspend(43u, "structured", structured_frame);
        b.set_insertion_point(structured_false);
        b.br(structured_merge);
        b.set_insertion_point(structured_merge);
        b.return_void();

        Value *mismatched_frame;
        BasicBlock *mismatched_body;
        auto *mismatched = make_post_split_callable(m, mismatched_frame, mismatched_body);
        b.set_insertion_point(mismatched_body);
        auto *mismatched_suspend = b.coro_suspend(47u, "mismatched-frame", mismatched_frame);

        Value *foreign_frame;
        BasicBlock *foreign_body;
        auto *foreign = make_post_split_callable(foreign_module, foreign_frame, foreign_body);
        b.set_insertion_point(foreign_body);
        auto *foreign_suspend = b.coro_suspend(53u, "foreign", foreign_frame);

        CoroCfgDistillResult cfg;
        cfg.scopes.resize(4u);// scope 3 is deliberately missing
        cfg.scopes[1u].trigger_token = 1u;
        cfg.scopes[2u].trigger_token = 2u;
        cfg.scopes[3u].trigger_token = 3u;
        CoroSplitInfo split;
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 0u,
            .callable = valid,
            .frame_argument = valid_frame,
        });
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 0u,// duplicate, and must still be structured-preflighted
            .callable = structured,
            .frame_argument = structured_frame,
        });
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 1u,
            .trigger_token = 1u,
            .callable = mismatched,
            .frame_argument = valid_frame,// belongs to a different callable
        });
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 2u,
            .trigger_token = 2u,
            .callable = foreign,
            .frame_argument = foreign_frame,
        });
        split.subroutines.emplace_back(CoroSplitInfo::Subroutine{
            .scope_index = 4u,// out of range, null callable/frame
            .callable = nullptr,
            .frame_argument = nullptr,
        });

        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count >= 5u);
        expect(info.structured_cfg_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(valid_body->terminator() == valid_suspend);
        expect(structured_body->terminator() == structured_if);
        expect(structured_true->terminator() == structured_suspend);
        expect(mismatched_body->terminator() == mismatched_suspend);
        expect(foreign_body->terminator() == foreign_suspend);
        expect(count_inst_tag(m, DerivedInstructionTag::STORE) == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 3u);
        expect(count_inst_tag(foreign_module, DerivedInstructionTag::CORO_SUSPEND) == 1u);
    };

    "duplicate_cfg_trigger_tokens_are_rejected_before_materialization"_test = [] {
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *callable = make_post_split_callable(m, frame_arg, body);
        static_cast<void>(callable);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *resume = b.coro_resume(1u, frame_arg);
        b.return_void();
        CoroCfgDistillResult cfg;
        cfg.scopes.resize(3u);
        cfg.scopes[0u].trigger_token = 0u;
        cfg.scopes[1u].trigger_token = 1u;
        cfg.scopes[2u].trigger_token = 1u;

        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 1u);
        expect(resume->parent_block() == body);
    };

    "wide_integer_token_field_index_is_decoded_by_type"_test = [] {
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *callable =
            make_post_split_callable(m, frame_arg, body);
        static_cast<void>(callable);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *resume = b.coro_resume(17u, frame_arg);
        uint64_t token_field = 6u;
        auto *field_index =
            m.create_constant(Type::of<uint64_t>(), &token_field);
        auto *token_ptr =
            b.gep(Type::of<uint32_t>(), frame_arg, {field_index});
        uint32_t token = 17u;
        b.store(token_ptr,
                m.create_constant(Type::of<uint32_t>(), &token));
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = coro_materialize_pass_run_on_module(&m);

        expect(info.succeeded());
        expect(info.resume_lowered_count == 1u);
        expect(!resume->is_linked());
        expect(count_inst_tag(
                   m, DerivedInstructionTag::CORO_RESUME) == 0u);
        expect(count_inst_tag(
                   m, DerivedInstructionTag::STORE) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "legacy_materialize_rejects_frame_layout_mismatch_atomically"_test = [] {
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *callable =
            make_post_split_callable(m, frame_arg, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<float>());
        state->set_name("state");
        auto *resume = b.coro_resume(19u, frame_arg);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = coro_materialize_pass_run_on_module(&m);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(resume->is_linked());
        expect(resume->parent_block() == body);
        expect(state->is_linked());
        expect(count_inst_tag(
                   m, DerivedInstructionTag::CORO_RESUME) == 1u);
        expect(xir_verify_module(&m).succeeded());
        static_cast<void>(callable);
    };

    "legacy_materialize_rejects_cross_callable_register_type_collision"_test = [] {
        Module m;
        XIRBuilder b;
        auto add_callable = [&](const Type *state_type) noexcept {
            auto *callable = m.create_callable(nullptr);
            auto *frame = callable->create_reference_argument(
                make_frame_type({state_type}));
            auto *body = callable->create_body_block();
            b.set_insertion_point(body);
            auto *state = b.alloca_local(state_type);
            state->set_name("state");
            auto *resume = b.coro_resume(23u, frame);
            b.return_void();
            return resume;
        };
        auto *float_resume = add_callable(Type::of<float>());
        auto *int_resume = add_callable(Type::of<int>());
        expect(xir_verify_module(&m).succeeded());

        auto info = coro_materialize_pass_run_on_module(&m);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(float_resume->is_linked());
        expect(int_resume->is_linked());
        expect(count_inst_tag(
                   m, DerivedInstructionTag::CORO_RESUME) == 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "materialize_rejects_mixed_frame_operands_atomically"_test = [] {
        Module m;
        auto *callable = m.create_callable(nullptr);
        auto *frame_type = make_frame_type();
        auto *first_frame =
            callable->create_reference_argument(frame_type);
        auto *second_frame =
            callable->create_reference_argument(frame_type);
        auto *entry = callable->create_body_block();
        auto *left = callable->create_basic_block();
        auto *right = callable->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_constant_one(Type::of<bool>()),
                  left, right);
        b.set_insertion_point(left);
        auto *left_suspend =
            b.coro_suspend(29u, "left", first_frame);
        b.set_insertion_point(right);
        auto *right_suspend =
            b.coro_suspend(31u, "right", second_frame);
        expect(xir_verify_module(&m).succeeded());

        auto info = coro_materialize_pass_run_on_module(&m);

        expect(!info.succeeded());
        expect(info.invalid_input_error_count == 1u);
        expect(info.callable_count == 0u);
        expect(left->terminator() == left_suspend);
        expect(right->terminator() == right_suspend);
        expect(count_inst_tag(
                   m, DerivedInstructionTag::CORO_SUSPEND) == 2u);
        expect(count_inst_tag(
                   m, DerivedInstructionTag::STORE) == 0u);
        expect(xir_verify_module(&m).succeeded());
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_materialize();
    return 0;
}
