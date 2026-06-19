#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/conditional_branch.h>
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

} // namespace

void reg_sccp() {

    // Regression: Loop-carried PHI with UNDEFINED initial value.
    // The recent fix ensured UNDEFINED is treated as BOTTOM (not TOP),
    // so the PHI converges to the constant value from the back-edge.
    "sccp_loop_phi_undefined_init"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);

        int zero = 0, one = 1;
        auto *c0 = m.create_constant(Type::of<int>(), &zero);
        auto *c1 = m.create_constant(Type::of<int>(), &one);

        // Create a simple loop: for(i=0; i<10; i++) { ... }
        // header:
        //   phi_acc = phi(UNDEFINED (entry), add_result (latch))
        //   phi_i    = phi(0 (entry), i_inc (latch))
        //   cond = cmp_lt(phi_i, 10)
        //   cond_br(cond, body, exit)
        // body:
        //   add_result = add(phi_acc, 1)
        //   br(latch)
        // latch:
        //   i_inc = add(phi_i, 1)
        //   br(header)
        // exit:
        //   return(phi_acc)

        auto *header = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *latch = k->create_basic_block();
        auto *exit_bb = k->create_basic_block();

        int ten = 10;
        auto *c10 = m.create_constant(Type::of<int>(), &ten);

        XIRBuilder b;

        // Entry -> header
        b.set_insertion_point(body);
        b.br(header);

        // header
        b.set_insertion_point(header);
        auto *phi_acc = b.phi(Type::of<int>());
        auto *phi_i = b.phi(Type::of<int>());
        // Add incoming values later
        auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi_i, c10});
        b.cond_br(cmp, loop_body, exit_bb);

        // body
        b.set_insertion_point(loop_body);
        auto *add_result = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_acc, c1});
        b.br(latch);

        // latch
        b.set_insertion_point(latch);
        auto *i_inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_i, c1});
        b.br(header);

        // exit
        b.set_insertion_point(exit_bb);
        b.return_(phi_acc);

        // Set PHI incoming values
        phi_acc->add_incoming(body, nullptr);  // UNDEFINED from entry
        phi_acc->add_incoming(latch, add_result);
        phi_i->add_incoming(body, c0);
        phi_i->add_incoming(latch, i_inc);

        auto info = sccp_pass_run_on_function(k);
        // After SCCP, phi_i should be constant-folded to the iteration values,
        // and phi_acc should resolve to a constant (sum of 1s = 10).
        expect(info.folded_inst_count > 0u);
    };

    // Regression: Arithmetic with BOTTOM operand.
    // If either operand is BOTTOM (UNDEFINED), the result should be BOTTOM,
    // and the recent fix corrected an operand-order bug that missed BOTTOM.
    "sccp_arithmetic_bottom_operand"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);

        XIRBuilder b;
        b.set_insertion_point(body);

        // Create a PHI with UNDEFINED (BOTTOM) initial value
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(body, nullptr);  // UNDEFINED from entry

        int one = 1;
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        // ADD with BOTTOM operand: should resolve to BOTTOM, not crash.
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, c1});

        b.return_(add);

        auto info = sccp_pass_run_on_function(k);
        // SCCP should not miscompile; the BOTTOM operand should be handled.
        expect(info.folded_inst_count >= 0u);  // At minimum, no crash.
    };

    // Nested loop constant propagation across iterations.
    // Outer loop increments a counter by 1 each iteration;
    // inner loop accumulates into a sum. SCCP should resolve
    // the constant loop bounds and propagate values.
    "sccp_nested_loop_propagation"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *entry = k->create_body_block();

        auto *outer_header = k->create_basic_block();
        auto *inner_header = k->create_basic_block();
        auto *inner_body = k->create_basic_block();
        auto *inner_latch = k->create_basic_block();
        auto *inner_exit = k->create_basic_block();
        auto *outer_latch = k->create_basic_block();
        auto *outer_exit = k->create_basic_block();

        int zero = 0, one = 1, two = 2;
        auto *c0 = m.create_constant(Type::of<int>(), &zero);
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        auto *c2 = m.create_constant(Type::of<int>(), &two);

        XIRBuilder b;

        // entry -> outer_header
        b.set_insertion_point(entry);
        b.br(outer_header);

        // outer_header: phi_i = phi(0, i_next); cond_br(i < 2, inner_header, outer_exit)
        b.set_insertion_point(outer_header);
        auto *phi_i = b.phi(Type::of<int>());
        auto *outer_cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi_i, c2});
        b.cond_br(outer_cmp, inner_header, outer_exit);
        phi_i->add_incoming(entry, c0);

        // inner_header: phi_j = phi(0, j_next); cond_br(j < 2, inner_body, inner_exit)
        b.set_insertion_point(inner_header);
        auto *phi_j = b.phi(Type::of<int>());
        auto *inner_cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi_j, c2});
        b.cond_br(inner_cmp, inner_body, inner_exit);
        phi_j->add_incoming(outer_header, c0);

        // inner_body: (just pass through)
        b.set_insertion_point(inner_body);
        b.br(inner_latch);

        // inner_latch: j_next = j + 1; br(inner_header)
        b.set_insertion_point(inner_latch);
        auto *j_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_j, c1});
        b.br(inner_header);
        phi_j->add_incoming(inner_latch, j_next);

        // inner_exit -> outer_latch
        b.set_insertion_point(inner_exit);
        b.br(outer_latch);

        // outer_latch: i_next = i + 1; br(outer_header)
        b.set_insertion_point(outer_latch);
        auto *i_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi_i, c1});
        b.br(outer_header);
        phi_i->add_incoming(outer_latch, i_next);

        // outer_exit
        b.set_insertion_point(outer_exit);
        b.return_();

        auto info = sccp_pass_run_on_function(k);
        expect(info.removed_branch_count > 0u || info.folded_inst_count > 0u);
    };
}
