#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/sccp.h>

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

}// namespace

void reg_sccp() {

    "sccp_loop_phi_undefined_init"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);

        int zero = 0, one = 1;
        auto *c0 = m.create_constant(Type::of<int>(), &zero);
        auto *c1 = m.create_constant(Type::of<int>(), &one);

        auto *header = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *latch = k->create_basic_block();
        auto *exit_bb = k->create_basic_block();

        int ten = 10;
        auto *c10 = m.create_constant(Type::of<int>(), &ten);

        XIRBuilder b;

        b.set_insertion_point(body);
        b.br(header);

        b.set_insertion_point(header);
        auto *phi_acc = b.phi(Type::of<int>());
        auto *phi_i = b.phi(Type::of<int>());
        auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi_i, c10});
        b.cond_br(cmp, loop_body, exit_bb);

        b.set_insertion_point(loop_body);
        auto *add_result = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_acc, c1});
        b.br(latch);

        b.set_insertion_point(latch);
        auto *i_inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_i, c1});
        b.br(header);

        b.set_insertion_point(exit_bb);
        b.return_(phi_acc);

        phi_acc->add_incoming(m.create_undefined(Type::of<int>()), body);
        phi_acc->add_incoming(add_result, latch);
        phi_i->add_incoming(c0, body);
        phi_i->add_incoming(i_inc, latch);

        auto info = sccp_pass_run_on_function(k);
        auto *ret = static_cast<ReturnInst *>(exit_bb->terminator());
        expect(info.folded_inst_count == 0u);
        expect(info.removed_branch_count == 0u);
        expect(ret->return_value() == phi_acc);
    };

    "sccp_arithmetic_bottom_operand"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        int one = 1;
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        auto *undef = m.create_undefined(Type::of<int>());
        auto *add_l = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {undef, c1});
        auto *add_r = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c1, undef});
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add_l, add_r});

        auto *ret = b.return_(sum);

        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(info.removed_branch_count == 0u);
        expect(ret->return_value() == sum);
    };

    "sccp_constant_branch_prunes_phi"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *entry = k->create_body_block();
        auto *true_bb = k->create_basic_block();
        auto *false_bb = k->create_basic_block();
        auto *merge_bb = k->create_basic_block();

        int one = 1, two = 2;
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        auto *c2 = m.create_constant(Type::of<int>(), &two);
        auto *ct = m.create_constant_one(Type::of<bool>());

        XIRBuilder b;

        b.set_insertion_point(entry);
        b.cond_br(ct, true_bb, false_bb);

        b.set_insertion_point(true_bb);
        b.br(merge_bb);

        b.set_insertion_point(false_bb);
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(c1, true_bb);
        phi->add_incoming(c2, false_bb);
        auto *ret = b.return_(phi);

        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(info.removed_branch_count == 1u);
        expect(ret->return_value() == c1);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_sccp();
    return 0;
}
