#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

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

    "sccp_constant_indexed_branch_prunes_phi"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *entry = k->create_body_block();
        auto *case_bb = k->create_basic_block();
        auto *default_bb = k->create_basic_block();
        auto *merge_bb = k->create_basic_block();

        uint32_t selector = 7u;
        int one = 1;
        int two = 2;
        auto *selector_constant =
            m.create_constant(Type::of<uint32_t>(), &selector);
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        auto *c2 = m.create_constant(Type::of<int>(), &two);

        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *indexed_branch =
            b.indexed_branch(selector_constant);
        indexed_branch->set_default_block(default_bb);
        indexed_branch->add_case(7u, case_bb);

        b.set_insertion_point(case_bb);
        b.br(merge_bb);
        b.set_insertion_point(default_bb);
        b.br(merge_bb);
        b.set_insertion_point(merge_bb);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(c1, case_bb);
        phi->add_incoming(c2, default_bb);
        auto *ret = b.return_(phi);

        auto info = sccp_pass_run_on_function(k);

        expect(info.folded_inst_count == 1u);
        expect(info.removed_branch_count == 1u);
        expect(entry->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(entry->terminator())
                   ->target_block() == case_bb);
        expect(ret->return_value() == c1);
    };

    "sccp_preserves_constant_false_structured_loop_prepare"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *prepare_branch = b.cond_br(
            m.create_constant_zero(Type::of<bool>()), body, merge);
        b.set_insertion_point(body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto before = xir_to_text_translate(&m, true);
        auto info = sccp_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.removed_branch_count == 0u);
        expect(prepare->terminator() == prepare_branch);
        expect(before == after);
        expect(xir_verify_module(&m).succeeded());
    };

    "sccp_pow_int_decodes_signed_narrow_exponent"_test = [] {
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

        auto info = sccp_pass_run_on_function(f);

        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.5f);
    };

    "sccp_pow_int_decodes_unsigned_64_bit_exponent"_test = [] {
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

        auto info = sccp_pass_run_on_function(f);

        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 1.0f);
    };

    "sccp_shift_count_uses_its_declared_integer_width"_test = [] {
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
        add_case(ArithmeticOp::BINARY_SHIFT_LEFT, uint64_t{1} << 32u);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, -(int64_t{1} << 32u));

        expect(xir_verify_module(&m).succeeded());
        auto info = sccp_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        for (auto *ret : returns) {
            expect(ret->return_value()->isa<ArithmeticInst>());
        }
        expect(xir_verify_module(&m).succeeded());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_sccp();
    return 0;
}
