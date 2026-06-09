#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_reg2mem.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] CallableFunction *make_continuation(Module &m, Value *&frame_arg_out, BasicBlock *&body_out) noexcept {
    auto *cf = m.create_callable(nullptr);
    frame_arg_out = cf->create_reference_argument(Type::structure({Type::of<uint32_t>()}));
    body_out = cf->create_body_block();
    return cf;
}

[[nodiscard]] size_t count_phi(Module &m) noexcept {
    size_t n = 0u;
    for (auto *f : m.function_list()) {
        if (f->definition() == nullptr) { continue; }
        auto *def = static_cast<FunctionDefinition *>(f);
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::PHI) { n++; }
        });
    }
    return n;
}

}// namespace

void reg_coro_reg2mem() {

    "single_callable_phi_lowered"_test = [] {
        // given: a callable with frame arg and phi nodes from branches
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_continuation(m, frame_arg, body);

        auto *true_bb = cf->create_basic_block();
        auto *false_bb = cf->create_basic_block();
        auto *merge_bb = cf->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, true_bb, false_bb);

        b.set_insertion_point(true_bb);
        auto *val_one = m.create_constant_one(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(false_bb);
        auto *val_zero = m.create_constant_zero(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *phi = b.phi(Type::of<float>());
        phi->add_incoming(val_one, true_bb);
        phi->add_incoming(val_zero, false_bb);
        b.return_void();

        expect(count_phi(m) == 1u);

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then
        expect(info.callable_count == 1u);
        expect(count_phi(m) == 0u);
    };

    "no_frame_arg_no_processing"_test = [] {
        // given: a callable WITHOUT a frame arg but with phi nodes
        Module m;
        auto *cf = m.create_callable(nullptr);
        auto *body = cf->create_body_block();

        auto *true_bb = cf->create_basic_block();
        auto *false_bb = cf->create_basic_block();
        auto *merge_bb = cf->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, true_bb, false_bb);

        b.set_insertion_point(true_bb);
        auto *val_one = m.create_constant_one(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(false_bb);
        auto *val_zero = m.create_constant_zero(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *phi = b.phi(Type::of<float>());
        phi->add_incoming(val_one, true_bb);
        phi->add_incoming(val_zero, false_bb);
        b.return_void();

        expect(count_phi(m) == 1u);

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then: not processed - no frame arg
        expect(info.callable_count == 0u);
        expect(info.lowered_phi_count == 0u);
        // phi still present (not a coroutine continuation)
        expect(count_phi(m) == 1u);
    };

    "kernel_not_processed"_test = [] {
        // given: a kernel with phi nodes (not a callable, not a continuation)
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();

        auto *true_bb = k->create_basic_block();
        auto *false_bb = k->create_basic_block();
        auto *merge_bb = k->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, true_bb, false_bb);

        b.set_insertion_point(true_bb);
        auto *val_one = m.create_constant_one(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(false_bb);
        auto *val_zero = m.create_constant_zero(Type::of<float>());
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *phi = b.phi(Type::of<float>());
        phi->add_incoming(val_one, true_bb);
        phi->add_incoming(val_zero, false_bb);
        b.return_void();

        expect(count_phi(m) == 1u);

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then: not processed - not a callable
        expect(info.callable_count == 0u);
        expect(info.lowered_phi_count == 0u);
        expect(count_phi(m) == 1u);
    };

    "multiple_continuations_all_processed"_test = [] {
        // given: two callables with frame args and phi nodes
        Module m;
        Value *frame_arg1;
        BasicBlock *body1;
        auto *cf1 = make_continuation(m, frame_arg1, body1);

        auto *t1 = cf1->create_basic_block();
        auto *f1 = cf1->create_basic_block();
        auto *m1 = cf1->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(body1);
        b.cond_br(m.create_constant_one(Type::of<bool>()), t1, f1);
        b.set_insertion_point(t1);
        b.br(m1);
        b.set_insertion_point(f1);
        b.br(m1);
        b.set_insertion_point(m1);
        auto *phi1 = b.phi(Type::of<float>());
        phi1->add_incoming(m.create_constant_one(Type::of<float>()), t1);
        phi1->add_incoming(m.create_constant_zero(Type::of<float>()), f1);
        b.return_void();

        Value *frame_arg2;
        BasicBlock *body2;
        auto *cf2 = make_continuation(m, frame_arg2, body2);

        auto *t2 = cf2->create_basic_block();
        auto *f2 = cf2->create_basic_block();
        auto *m2 = cf2->create_basic_block();

        b.set_insertion_point(body2);
        b.cond_br(m.create_constant_one(Type::of<bool>()), t2, f2);
        b.set_insertion_point(t2);
        b.br(m2);
        b.set_insertion_point(f2);
        b.br(m2);
        b.set_insertion_point(m2);
        auto *phi2 = b.phi(Type::of<int>());
        phi2->add_incoming(m.create_constant_one(Type::of<int>()), t2);
        phi2->add_incoming(m.create_constant_zero(Type::of<int>()), f2);
        b.return_void();

        expect(count_phi(m) == 2u);

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then
        expect(info.callable_count == 2u);
        expect(count_phi(m) == 0u);
    };

    "no_callables_zero_count"_test = [] {
        // given: module with no callables
        Module m;

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then
        expect(info.callable_count == 0u);
        expect(info.lowered_phi_count == 0u);
        expect(info.lowered_cross_block_value_count == 0u);
    };

    "callable_with_frame_arg_no_phi"_test = [] {
        // given: a callable with frame arg but no phi nodes
        Module m;
        Value *frame_arg;
        BasicBlock *body;
        auto *cf = make_continuation(m, frame_arg, body);

        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        expect(count_phi(m) == 0u);

        // when
        auto info = coro_reg2mem_pass_run_on_module(&m);

        // then: processed but no phi lowered
        expect(info.callable_count == 1u);
        expect(info.lowered_phi_count == 0u);
        expect(count_phi(m) == 0u);
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_reg2mem();
    return 0;
}
