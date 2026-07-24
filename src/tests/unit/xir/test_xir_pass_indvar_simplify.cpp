#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

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

struct LoopFixture {
    Module m;
    BasicBlock *body;
    KernelFunction *k;
    LoopInst *loop;
    BasicBlock *prep;
    BasicBlock *lbody;
    BasicBlock *upd;
    BasicBlock *merge;
    XIRBuilder b;

    LoopFixture() {
        k = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        loop = b.loop();
        prep = loop->create_prepare_block();
        lbody = loop->create_body_block();
        upd = loop->create_update_block();
        merge = loop->create_merge_block();
    }
};

}// namespace

void reg_indvar_simplify() {

    "indvar_remove_dead_iv"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;

        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *iv = b.phi(Type::of<int>(), {{c0, fix.body}});
        auto *prep_true = m.create_constant_one(Type::of<bool>());
        b.cond_br(prep_true, fix.lbody, fix.merge);

        b.set_insertion_point(fix.lbody);
        b.br(fix.upd);

        b.set_insertion_point(fix.upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        auto iv_locked = iv->lock();
        auto inc_locked = inc->lock();
        iv->add_incoming(inc, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 1u);
        expect(iv_locked->use_list().empty());
        expect(inc_locked->use_list().empty());
        size_t phi_count = 0u;
        size_t add_count = 0u;
        fix.k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) { ++phi_count; }
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_ADD) {
                ++add_count;
            }
        });
        expect(phi_count == 0u);
        expect(add_count == 0u);
    };

    "indvar_increment_with_external_user_is_kept"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;
        b.set_insertion_point(fix.body->instructions().head_sentinel());
        auto *sink = b.alloca_local(Type::of<int>());
        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *iv = b.phi(Type::of<int>(), {{c0, fix.body}});
        b.cond_br(m.create_constant_one(Type::of<bool>()), fix.lbody, fix.merge);
        b.set_insertion_point(fix.lbody);
        b.br(fix.upd);
        b.set_insertion_point(fix.upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        iv->add_incoming(inc, fix.upd);
        auto *store = b.store(sink, inc);
        b.cond_br(m.create_constant_one(Type::of<bool>()), fix.prep, fix.merge);
        b.set_insertion_point(fix.merge);
        b.return_void();
        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 0u);
        expect(iv->is_linked());
        expect(inc->is_linked());
        expect(store->value() == inc);
        expect(iv->incoming_count() == 2u);
    };

    "indvar_keep_used_iv"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;

        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *iv = b.phi(Type::of<int>(), {{c0, fix.body}});
        auto *prep_true = m.create_constant_one(Type::of<bool>());
        b.cond_br(prep_true, fix.lbody, fix.merge);

        b.set_insertion_point(fix.lbody);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, iv);
        b.br(fix.upd);

        b.set_insertion_point(fix.upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        iv->add_incoming(inc, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 0u);
    };

    "indvar_keep_iv_used_in_compare"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;

        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *c10 = [&]() { int32_t v = 10; return m.create_constant(Type::of<int>(), &v); }();
        auto *iv = b.phi(Type::of<int>(), {{c0, fix.body}});
        auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, c10});
        b.cond_br(cmp, fix.lbody, fix.merge);

        b.set_insertion_point(fix.lbody);
        b.br(fix.upd);

        b.set_insertion_point(fix.upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        iv->add_incoming(inc, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 0u);
    };

    "indvar_not_iv_noop"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;

        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *phi = b.phi(Type::of<int>(), {{c0, fix.body}});
        auto *prep_true = m.create_constant_one(Type::of<bool>());
        b.cond_br(prep_true, fix.lbody, fix.merge);

        b.set_insertion_point(fix.lbody);
        b.br(fix.upd);

        b.set_insertion_point(fix.upd);
        phi->add_incoming(c1, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 0u);
    };

    "indvar_empty_module"_test = [] {
        Module m;
        auto info = indvar_simplify_pass_run_on_module(&m);
        expect(info.removed_dead_iv_count == 0u);
    };

    "indvar_no_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(k);
        expect(info.removed_dead_iv_count == 0u);
    };

    "indvar_idempotent"_test = [] {
        LoopFixture fix;
        auto &m = fix.m;
        auto &b = fix.b;

        b.set_insertion_point(fix.prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *iv = b.phi(Type::of<int>(), {{c0, fix.body}});
        auto *prep_true = m.create_constant_one(Type::of<bool>());
        b.cond_br(prep_true, fix.lbody, fix.merge);

        b.set_insertion_point(fix.lbody);
        b.br(fix.upd);

        b.set_insertion_point(fix.upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        iv->add_incoming(inc, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto first = indvar_simplify_pass_run_on_function(fix.k);
        auto second = indvar_simplify_pass_run_on_function(fix.k);
        expect(first.removed_dead_iv_count == 1u);
        expect(second.removed_dead_iv_count == 0u);
    };

    "indvar_module_runs_all_functions"_test = [] {
        Module m;
        constexpr size_t kFns = 2u;
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
            auto *c0 = m.create_constant_zero(Type::of<int>());
            auto *c1 = m.create_constant_one(Type::of<int>());
            auto *iv = b.phi(Type::of<int>(), {{c0, body}});
            auto *true_const = m.create_constant_one(Type::of<bool>());
            b.cond_br(true_const, lbody, merge);

            b.set_insertion_point(lbody);
            b.br(upd);

            b.set_insertion_point(upd);
            auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
            iv->add_incoming(inc, upd);
            b.cond_br(true_const, prep, merge);

            b.set_insertion_point(merge);
            b.return_void();
        }
        auto info = indvar_simplify_pass_run_on_module(&m);
        expect(info.removed_dead_iv_count == kFns);
    };
}

namespace {

struct PlainCountedLoop {
    KernelFunction *kernel;
    ResourceArgument *buffer;
    BasicBlock *entry;
    BasicBlock *header;
    BasicBlock *latch;
    BasicBlock *exit;
    PhiInst *iv;
};

// entry -> header { iv = phi(entry: 0, latch: next); cond = iv < bound;
//                   cond_br(cond, latch, exit) }
// latch { ...custom body...; next = iv + 1; br header }
[[nodiscard]] PlainCountedLoop make_plain_counted_loop(
    Module &m, uint32_t bound_value, Value *stride_override = nullptr) noexcept {
    PlainCountedLoop loop;
    loop.kernel = m.create_kernel();
    loop.buffer = loop.kernel->create_resource_argument(Type::buffer(Type::of<float>()));
    auto *def = loop.kernel->definition();
    loop.entry = loop.kernel->create_body_block();
    loop.header = def->create_basic_block();
    loop.latch = def->create_basic_block();
    loop.exit = def->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
    XIRBuilder b;
    b.set_insertion_point(loop.entry);
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                        {loop.iv, bound});
    b.cond_br(cond, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    return loop;
}

void finish_plain_counted_loop(Module &m, PlainCountedLoop &loop,
                               XIRBuilder &b, Value *stride_value) noexcept {
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, stride_value});
    b.br(loop.header);
    b.set_insertion_point(loop.exit);
    b.return_void();
    loop.iv->add_incoming(zero, loop.entry);
    loop.iv->add_incoming(next, loop.latch);
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

[[nodiscard]] size_t count_phis(BasicBlock *block) noexcept {
    auto count = 0u;
    block->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { count++; }
    });
    return count;
}

}// namespace

void reg_indvar_strength_reduction() {

    "scaled_iv_buffer_index_is_strength_reduced"_test = [] {
        Module m;
        auto loop = make_plain_counted_loop(m, 16u);
        auto *c1 = m.create_constant_one(Type::of<uint>());
        uint32_t four_value = 4u;
        auto *c4 = m.create_constant(Type::of<uint>(), &four_value);
        XIRBuilder b;
        b.set_insertion_point(loop.latch);
        auto *scaled = b.call(Type::of<uint>(), ArithmeticOp::BINARY_MUL,
                              {loop.iv, c4});
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                            {loop.buffer, scaled});
        static_cast<void>(read);
        finish_plain_counted_loop(m, loop, b, c1);

        auto info = indvar_simplify_pass_run_on_function(loop.kernel->definition());
        expect(info.simplified_iv_count == 1u);
        // a new accumulator phi appears next to the induction phi
        expect(count_phis(loop.header) == 2u);
        // the buffer read now uses the accumulator, not the multiply
        expect(read->operand(1u) != scaled);
        PhiInst *acc = nullptr;
        loop.header->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>() && inst != loop.iv) {
                acc = static_cast<PhiInst *>(inst);
            }
        });
        expect(acc != nullptr);
        expect(read->operand(1u) == acc);
        expect(acc->incoming_count() == 2u);
        expect_module_valid(m);
    };

    "scaled_iv_plus_base_is_strength_reduced_together"_test = [] {
        Module m;
        auto loop = make_plain_counted_loop(m, 16u);
        auto *c1 = m.create_constant_one(Type::of<uint>());
        uint32_t four_value = 4u;
        auto *c4 = m.create_constant(Type::of<uint>(), &four_value);
        auto *base = loop.kernel->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(loop.latch);
        auto *scaled = b.call(Type::of<uint>(), ArithmeticOp::BINARY_MUL,
                              {loop.iv, c4});
        auto *offset = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                              {scaled, base});
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                            {loop.buffer, offset});
        static_cast<void>(read);
        finish_plain_counted_loop(m, loop, b, c1);

        auto info = indvar_simplify_pass_run_on_function(loop.kernel->definition());
        expect(info.simplified_iv_count == 1u);
        // the whole add chain collapses into the accumulator
        expect(read->operand(1u) != offset);
        expect(read->operand(1u)->isa<PhiInst>());
        expect_module_valid(m);
    };

    "non_constant_stride_is_not_strength_reduced"_test = [] {
        Module m2;
        auto loop = make_plain_counted_loop(m2, 16u);
        auto *stride_arg = loop.kernel->create_value_argument(Type::of<uint>());
        uint32_t four_value = 4u;
        auto *c4 = m2.create_constant(Type::of<uint>(), &four_value);
        XIRBuilder b;
        b.set_insertion_point(loop.latch);
        auto *scaled = b.call(Type::of<uint>(), ArithmeticOp::BINARY_MUL,
                              {loop.iv, c4});
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                            {loop.buffer, scaled});
        static_cast<void>(read);
        finish_plain_counted_loop(m2, loop, b, stride_arg);

        auto info = indvar_simplify_pass_run_on_function(loop.kernel->definition());
        expect(info.simplified_iv_count == 0u);
        expect(read->operand(1u) == scaled);
        expect_module_valid(m2);
    };

    "multiple_scaled_uses_get_independent_accumulators"_test = [] {
        Module m;
        auto loop = make_plain_counted_loop(m, 16u);
        auto *c1 = m.create_constant_one(Type::of<uint>());
        uint32_t two_value = 2u;
        uint32_t four_value = 4u;
        auto *c2 = m.create_constant(Type::of<uint>(), &two_value);
        auto *c4 = m.create_constant(Type::of<uint>(), &four_value);
        XIRBuilder b;
        b.set_insertion_point(loop.latch);
        auto *scaled2 = b.call(Type::of<uint>(), ArithmeticOp::BINARY_MUL,
                               {loop.iv, c2});
        auto *scaled4 = b.call(Type::of<uint>(), ArithmeticOp::BINARY_MUL,
                               {loop.iv, c4});
        auto *read2 = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                             {loop.buffer, scaled2});
        auto *read4 = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                             {loop.buffer, scaled4});
        static_cast<void>(read2);
        static_cast<void>(read4);
        finish_plain_counted_loop(m, loop, b, c1);

        auto info = indvar_simplify_pass_run_on_function(loop.kernel->definition());
        expect(info.simplified_iv_count == 2u);
        expect(count_phis(loop.header) == 3u);
        expect(read2->operand(1u) != scaled2);
        expect(read4->operand(1u) != scaled4);
        expect_module_valid(m);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_indvar_simplify();
    reg_indvar_strength_reduction();
    return 0;
}
