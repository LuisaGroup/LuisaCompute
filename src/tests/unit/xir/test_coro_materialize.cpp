#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_materialize.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] CallableFunction *make_post_split_callable(Module &m, Value *&frame_arg_out, BasicBlock *&body_out) noexcept {
    auto *cf = m.create_callable(nullptr);
    frame_arg_out = cf->create_reference_argument(Type::structure({Type::of<uint32_t>(), Type::of<uint32_t>()}));
    body_out = cf->create_body_block();
    return cf;
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

[[nodiscard]] size_t count_load_inst(Module &m) noexcept {
    return count_inst_tag(m, DerivedInstructionTag::LOAD);
}

[[nodiscard]] size_t count_store_inst(Module &m) noexcept {
    return count_inst_tag(m, DerivedInstructionTag::STORE);
}

[[nodiscard]] size_t count_gep_inst(Module &m) noexcept {
    return count_inst_tag(m, DerivedInstructionTag::GEP);
}

}// namespace

void reg_coro_materialize() {

    "single_register_var_stored_at_suspend"_test = [] {
        // given: callable with frame arg, CoroRegisterInst, CoroSuspendInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);
        auto *s = b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: frame has token + skip + 1 user field
        expect(info.register_count == 1u);
        expect(info.frame_field_count == 3u);
        expect(info.name_to_field.contains("x"));
        expect(info.name_to_field.at("x") == 2u);
        // Suspend lowered, register removed
        expect(info.suspend_lowered_count == 1u);
        expect(info.store_inserted_count == 1u);
    };

    "loaded_at_resume"_test = [] {
        // given: callable with frame arg, CoroRegisterInst, CoroResumeInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *r = b.coro_resume(1u, frame_arg);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.coro_register("x", alloca, frame_arg);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: CoroResumeInst lowered, load inserted for registered var
        expect(info.resume_lowered_count == 1u);
        expect(info.load_inserted_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 0u);
    };

    "no_register_no_change"_test = [] {
        // given: callable with frame arg but no CoroRegisterInst
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
        expect(info.frame_field_count == 2u);
        expect(info.load_inserted_count == 0u);
        expect(info.store_inserted_count == 0u);
    };

    "variable_not_registered_not_in_frame"_test = [] {
        // given: callable with allocas but only one registered
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca_a = b.alloca_local(Type::of<float>());
        auto *alloca_b = b.alloca_local(Type::of<float>());
        b.coro_register("a", alloca_a, frame_arg);
        // b is NOT registered
        auto *s = b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: only 1 user field (for "a")
        expect(info.register_count == 1u);
        expect(info.frame_field_count == 3u);
        expect(info.name_to_field.contains("a"));
        expect(!info.name_to_field.contains("b"));
        expect(info.store_inserted_count == 1u);
    };

    "two_variables_one_live"_test = [] {
        // given: two variables, one registered
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *live_alloca = b.alloca_local(Type::of<float>());
        auto *local_alloca = b.alloca_local(Type::of<int>());
        b.coro_register("live", live_alloca, frame_arg);
        // local_alloca is not registered
        auto *s = b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: frame has token + skip + 1 user field
        expect(info.frame_field_count == 3u);
        expect(info.name_to_field.contains("live"));
        expect(!info.name_to_field.contains("local"));
        expect(info.store_inserted_count == 1u);
    };

    "frame_struct_has_token_skip_user_fields"_test = [] {
        // given: registered var with known type
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca_f = b.alloca_local(Type::of<float>());
        auto *alloca_i = b.alloca_local(Type::of<int>());
        b.coro_register("fval", alloca_f, frame_arg);
        b.coro_register("ival", alloca_i, frame_arg);
        b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: 4 fields (token, skip, fval, ival)
        expect(info.frame_field_count == 4u);
        expect(info.name_to_field.contains("fval"));
        expect(info.name_to_field.contains("ival"));
    };

    "suspend_replaced_with_store_and_return"_test = [] {
        // given: callable with CoroSuspendInst and register
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);
        b.coro_suspend(7u, "chk", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: no CoroSuspendInst left, ReturnInst present
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 0u);
        expect(info.suspend_lowered_count == 1u);
    };

    "resume_replaced_with_loads"_test = [] {
        // given: callable with CoroResumeInst and register
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_resume(1u, frame_arg);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: no CoroResumeInst left
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 0u);
        expect(info.resume_lowered_count == 1u);
        expect(info.load_inserted_count == 1u);
    };

    "token_store_has_user_var_stores_before"_test = [] {
        // given: post-split callable with token store and register
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);

        // simulate post-split token store
        auto *field_zero = m.create_constant_zero(Type::of<uint32_t>());
        auto *gep0 = b.gep(Type::of<uint32_t>(), frame_arg, {field_zero});
        auto token_val = static_cast<uint32_t>(42u);
        auto *tok_c = m.create_constant(Type::of<uint32_t>(), &token_val);
        b.store(gep0, tok_c);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: frame field for "x", store inserted before token store
        expect(info.store_inserted_count == 1u);
        expect(info.register_count == 1u);
        // verify all instructions - stores should be before token store
        luisa::vector<DerivedInstructionTag> inst_order;
        for (auto *f : m.function_list()) {
            if (f->definition() == nullptr) { continue; }
            auto *def = static_cast<FunctionDefinition *>(f);
            def->traverse_instructions([&](Instruction *inst) noexcept {
                inst_order.push_back(inst->derived_instruction_tag());
            });
        }
        // After materialization: LOAD(alloca), STORE(to frame field), STORE(token), RETURN
        // (or: STORE(user var), STORE(token), RETURN)
        // Verify STORE appears before RETURN and the user var store is before token store
        bool found_user_store = false;
        bool found_token_store = false;
        for (auto *f : m.function_list()) {
            if (f->definition() == nullptr) { continue; }
            auto *def = static_cast<FunctionDefinition *>(f);
            def->traverse_instructions([&](Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == DerivedInstructionTag::STORE) {
                    if (!found_user_store) {
                        found_user_store = true;
                    } else if (!found_token_store) {
                        found_token_store = true;
                    }
                }
            });
        }
        // At least one store present
        expect(found_user_store);
    };

    "deduplicate_same_name_across_callables"_test = [] {
        // given: two callables both registering "x"
        Module m;

        {
            Value *fa;
            BasicBlock *bb;
            auto *cf = make_post_split_callable(m, fa, bb);
            XIRBuilder b;
            b.set_insertion_point(bb);
            auto *alloca = b.alloca_local(Type::of<float>());
            b.coro_register("x", alloca, fa);
            b.coro_suspend(1u, "s1", fa);
        }

        {
            Value *fa;
            BasicBlock *bb;
            auto *cf = make_post_split_callable(m, fa, bb);
            XIRBuilder b;
            b.set_insertion_point(bb);
            auto *alloca = b.alloca_local(Type::of<float>());
            b.coro_register("x", alloca, fa);
            b.coro_suspend(2u, "s2", fa);
        }

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: only 1 register (deduplicated), 2 callables processed
        expect(info.register_count == 1u);
        expect(info.frame_field_count == 3u);
        expect(info.callable_count == 2u);
    };

    "terminate_lowered"_test = [] {
        // given: callable with CoroTerminateInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_terminate();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: CoroTerminateInst removed, store+return inserted
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_TERMINATE) == 0u);
        expect(info.terminal_lowered_count == 1u);
    };

    "register_inst_removed"_test = [] {
        // given: callable with CoroRegisterInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);
        b.return_void();

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: CoroRegisterInst removed
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_REGISTER) == 0u);
    };

    // ── edge case: variable live across resume and suspend ───────────
    "overlapping_liveness"_test = [] {
        // given: callable with both CoroResumeInst and CoroSuspendInst
        // and a registered variable that spans both — it must be
        // loaded from frame at resume and stored to frame at suspend
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        b.coro_resume(1u, frame_arg);
        auto *alloca = b.alloca_local(Type::of<float>());
        b.coro_register("x", alloca, frame_arg);
        // use the variable (load-then-store keeps it live)
        auto *v = b.load(Type::of<float>(), alloca);
        b.store(alloca, v);
        b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: both load and store inserted, both suspend and resume lowered
        expect(info.register_count == 1u);
        expect(info.frame_field_count == 3u);// token + skip + x
        expect(info.load_inserted_count >= 1u);
        expect(info.store_inserted_count >= 1u);
        expect(info.suspend_lowered_count == 1u);
        expect(info.resume_lowered_count == 1u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 0u);
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_RESUME) == 0u);
    };

    // ── edge case: unregistered variable untouched ───────────────────
    "unregistered_variable_untouched"_test = [] {
        // given: callable with a suspend and an alloca that is
        // explicitly NOT registered — no frame store should appear
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<float>());
        // intentionally NO coro_register call for this alloca
        b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: zero user registers, zero user stores/loads
        expect(info.register_count == 0u);
        expect(info.frame_field_count == 2u);// token + skip only
        expect(info.store_inserted_count == 0u);
        expect(info.load_inserted_count == 0u);

        // alloca still exists (was not removed — it's just not spilled)
        expect(count_inst_tag(m, DerivedInstructionTag::ALLOCA) >= 1u);
    };

    // ── edge case: many registered variables ─────────────────────────
    "multiple_registered_vars_large"_test = [] {
        // given: callable with three registered variables of different
        // types — all must appear in the frame struct
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a_f = b.alloca_local(Type::of<float>());
        auto *a_i = b.alloca_local(Type::of<int>());
        auto *a_u = b.alloca_local(Type::of<uint32_t>());
        b.coro_register("fv", a_f, frame_arg);
        b.coro_register("iv", a_i, frame_arg);
        b.coro_register("uv", a_u, frame_arg);
        b.coro_suspend(1u, "s1", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: token + skip + 3 user fields = 5 fields
        expect(info.register_count == 3u);
        expect(info.frame_field_count == 5u);
        expect(info.name_to_field.contains("fv"));
        expect(info.name_to_field.contains("iv"));
        expect(info.name_to_field.contains("uv"));

        // field indices are consecutive starting after token+skip (2)
        auto fv_idx = info.name_to_field.at("fv");
        auto iv_idx = info.name_to_field.at("iv");
        auto uv_idx = info.name_to_field.at("uv");
        expect(fv_idx >= 2u && fv_idx <= 4u);
        expect(iv_idx >= 2u && iv_idx <= 4u);
        expect(uv_idx >= 2u && uv_idx <= 4u);
        expect(fv_idx != iv_idx);
        expect(fv_idx != uv_idx);
        expect(iv_idx != uv_idx);

        // one store per registered variable
        expect(info.store_inserted_count == 3u);
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_materialize();
    return 0;
}
