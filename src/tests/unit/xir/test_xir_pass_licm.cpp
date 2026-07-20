#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/passes/licm.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

[[nodiscard]] size_t count_instructions_in_block(BasicBlock *bb) noexcept {
    size_t n = 0u;
    for (auto *inst : bb->instructions()) {
        (void)inst;
        n++;
    }
    return n;
}

[[nodiscard]] bool appears_before(BasicBlock *block,
                                  Instruction *first,
                                  Instruction *second) noexcept {
    auto saw_first = false;
    for (auto *inst : block->instructions()) {
        if (inst == first) { saw_first = true; }
        if (inst == second) { return saw_first; }
    }
    return false;
}

}// namespace

void reg_licm() {

    "licm_hoist_pure_arith"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *inv_add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c0, c1});
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 1u);
        expect(inv_add->parent_block() == prep);
    };

    "licm_no_hoist_store"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *c0 = m.create_constant_zero(Type::of<int>());
        b.store(alloca, c0);
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 0u);
    };

    "licm_no_hoist_load"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;

        b.set_insertion_point(body);
        // Install the local and its initialization before the LoopInst
        // terminator so this remains a valid structured entry block.
        auto *ext_alloca = b.alloca_local(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        b.store(ext_alloca, one);
        auto *loop = b.loop();
        auto *prep = loop->create_prepare_block();
        auto *lbody = loop->create_body_block();
        auto *upd = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prep);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *ld = b.load(Type::of<int>(), ext_alloca);
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        // Load is not pure (reads memory), so not hoisted
        expect(info.hoisted_count == 0u);
        expect(ld->parent_block() == lbody);
    };

    "licm_no_speculative_global_read"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *buffer = k->create_resource_argument(Type::buffer(Type::of<int>()));
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prep = loop->create_prepare_block();
        auto *lbody = loop->create_body_block();
        auto *upd = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prep);
        auto *false_const = m.create_constant_zero(Type::of<bool>());
        b.cond_br(false_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *read = b.call(Type::of<int>(), ResourceReadOp::BUFFER_READ, {buffer, zero});
        b.br(upd);

        b.set_insertion_point(upd);
        b.br(prep);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 0u);
        expect(read->parent_block() == lbody);
    };

    "licm_no_speculative_undefined_arithmetic"_test = [] {
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
        b.cond_br(m.create_constant_zero(Type::of<bool>()), lbody, merge);

        int32_t one_value = 1;
        int32_t shift_value = 32;
        auto *one = m.create_constant(Type::of<int>(), &one_value);
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *shift = m.create_constant(Type::of<int>(), &shift_value);
        b.set_insertion_point(lbody);
        auto *div = b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {one, zero});
        auto *shl = b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {one, shift});
        b.br(upd);

        b.set_insertion_point(upd);
        b.br(prep);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 0u);
        expect(div->parent_block() == lbody);
        expect(shl->parent_block() == lbody);
    };

    "licm_fixed_point_chained_invariant"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *c2 = [&]() { int32_t v = 2; return m.create_constant(Type::of<int>(), &v); }();
        auto *a = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c0, c1});
        auto *b_inst = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, c2});
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 2u);
        expect(a->parent_block() == prep);
        expect(b_inst->parent_block() == prep);
    };

    "licm_cross_block_invariants_preserve_def_use_order"_test = [] {
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

        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(prep);
        b.cond_br(true_const, lbody, merge);

        // The producer is in the body and its user is in update. The pass used
        // to collect these blocks from an unordered_set, which commonly visits
        // update first and emits the consumer before the producer in prepare.
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(lbody);
        auto *producer = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                {zero, one});
        b.br(upd);

        b.set_insertion_point(upd);
        auto *consumer = b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL,
                                {producer, one});
        b.br(prep);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);

        expect(info.hoisted_count == 2u);
        expect(producer->parent_block() == prep);
        expect(consumer->parent_block() == prep);
        expect(consumer->operand(0u) == producer);
        expect(appears_before(prep, producer, consumer));
        expect(appears_before(prep, consumer, prep->terminator()));
    };

    "licm_no_hoist_loop_variant_operand"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *c0 = m.create_constant_zero(Type::of<int>());
        b.store(alloca, c0);
        auto *ld = b.load(Type::of<int>(), alloca);
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *variant_use = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld, c1});
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        // variant_use depends on ld which reads memory in the loop body
        // ld is not invariant, so variant_use should not be hoisted
        expect(info.hoisted_count == 0u);
    };

    "licm_hoist_resource_query"_test = [] {
        // RESOURCE_QUERY is pure, should be hoisted
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *buffer = k->create_resource_argument(Type::buffer(Type::of<int>()));
        XIRBuilder b;

        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prep = loop->create_prepare_block();
        auto *lbody = loop->create_body_block();
        auto *upd = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prep);
        auto *false_const = m.create_constant_zero(Type::of<bool>());
        b.cond_br(false_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *size_query = b.call(Type::of<uint>(), ResourceQueryOp::BUFFER_SIZE, {buffer});
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(false_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 1u);
        expect(size_query->parent_block() == prep);
        expect(appears_before(prep, size_query, prep->terminator()));
    };

    "licm_empty_module"_test = [] {
        Module m;
        auto info = licm_pass_run_on_module(&m);
        expect(info.hoisted_count == 0u);
    };

    "licm_no_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 0u);
    };

    "licm_idempotent"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c0, c1});
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto first = licm_pass_run_on_function(k);
        auto second = licm_pass_run_on_function(k);
        expect(first.hoisted_count == 1u);
        expect(second.hoisted_count == 0u);
    };

    "licm_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 3u;
        for (size_t i = 0; i < kFns; ++i) {
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
            auto *true_const = m.create_constant_one(Type::of<bool>());
            b.cond_br(true_const, lbody, merge);

            b.set_insertion_point(lbody);
            auto *c0 = m.create_constant_zero(Type::of<int>());
            auto *c1 = m.create_constant_one(Type::of<int>());
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c0, c1});
            b.br(upd);

            b.set_insertion_point(upd);
            b.cond_br(true_const, prep, merge);

            b.set_insertion_point(merge);
            b.return_void();
        }
        auto info = licm_pass_run_on_module(&m);
        expect(info.hoisted_count == kFns);
    };

    "licm_no_hoist_terminal"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        // Nothing to hoist - terminators are skipped
        expect(info.hoisted_count == 0u);
    };

    "licm_no_hoist_alloca"_test = [] {
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
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, lbody, merge);

        b.set_insertion_point(lbody);
        b.alloca_local(Type::of<int>());
        b.br(upd);

        b.set_insertion_point(upd);
        b.cond_br(true_const, prep, merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = licm_pass_run_on_function(k);
        expect(info.hoisted_count == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_licm();
    return 0;
}
