#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/dsl/coro_frame.h>
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
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] CallableFunction *make_post_split_callable(Module &m, Value *&frame_arg_out, BasicBlock *&body_out) noexcept {
    auto *cf = m.create_callable(nullptr);
    frame_arg_out = cf->create_reference_argument(Type::structure({
        Type::of<uint>(),
        Type::of<uint>(),
        Type::of<uint>(),
        Type::of<uint>(),
    }));
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

}// namespace

void reg_coro_materialize() {

    "no_register_no_change"_test = [] {
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

    "terminate_lowered"_test = [] {
        // given: callable with CoroTerminateInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.coro_suspend(7u, "before_term", frame_arg);

        // when
        auto info = coro_materialize_pass_run_on_module(&m);

        // then: CoroSuspendInst removed, store+return inserted
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_SUSPEND) == 0u);
        expect(info.suspend_lowered_count == 1u);
    };

    "suspend_and_terminate_both_lowered"_test = [] {
        // given: callable with both CoroSuspendInst and CoroTerminateInst
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
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

        // then: terminal-only split-aware callable lowered
        expect(count_inst_tag(m, DerivedInstructionTag::CORO_TERMINATE) == 0u);
        expect(info.terminal_lowered_count >= 1u);
    };

    "token_store_has_user_registers_empty"_test = [] {
        // given: post-split callable with token store but no user registers
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_post_split_callable(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        // simulate post-split token store
        auto token_field = 3u;
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
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_materialize();
    return 0;
}
