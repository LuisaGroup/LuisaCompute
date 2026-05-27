#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/indvar_simplify.h>

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
        iv->add_incoming(inc, fix.upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, fix.prep, fix.merge);

        b.set_insertion_point(fix.merge);
        b.return_void();

        auto info = indvar_simplify_pass_run_on_function(fix.k);
        expect(info.removed_dead_iv_count == 1u);
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

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_indvar_simplify();
    return 0;
}
