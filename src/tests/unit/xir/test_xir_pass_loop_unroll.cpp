// Test for the XIR loop-unrolling pass.
// This test covers:
// - full unroll of a constant-trip-count loop
// - phi elimination after unrolling
// - trip count above the maximum rejected
// - unroll_pure_only rejecting loops with buffer writes
// - unroll_pure_only accepting side-effect-free loops
// - nested loops with a variable outer bound (inner unrolled, outer kept)
// - variable trip count rejected

#include "ut/ut.hpp"

#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/xir/passes/dom_tree.h>

#include "../../../../src/xir/passes/natural_loop.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct CountedLoop {
    KernelFunction *kernel;
    BasicBlock *entry;
    BasicBlock *header;
    BasicBlock *latch;
    BasicBlock *exit;
    PhiInst *iv;
    PhiInst *acc;
    PhiInst *exit_phi;
};

// entry -> header { iv = phi(entry: 0, latch: next);
//                   acc = phi(entry: 0.0f, latch: acc_next);
//                   cond = iv < bound; cond_br(cond, latch, exit) }
// latch { [optional buffer write]; next = iv + 1; acc_next = acc + iv;
//         br header }
// exit { result = phi(header: acc); return }
[[nodiscard]] CountedLoop make_counted_loop(Module &m, uint32_t bound_value,
                                            bool with_buffer_write) noexcept {
    CountedLoop loop;
    loop.kernel = m.create_kernel();
    auto *def = loop.kernel->definition();
    loop.entry = loop.kernel->create_body_block();
    loop.header = def->create_basic_block();
    loop.latch = def->create_basic_block();
    loop.exit = def->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *zero_f = m.create_constant_zero(Type::of<float>());
    auto *one = m.create_constant_one(Type::of<uint>());
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
    ResourceArgument *buffer = nullptr;
    if (with_buffer_write) {
        buffer = loop.kernel->create_resource_argument(Type::buffer(Type::of<float>()));
    }

    XIRBuilder b;
    b.set_insertion_point(loop.entry);
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    loop.acc = b.phi(Type::of<float>());
    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                        {loop.iv, bound});
    b.cond_br(cond, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    if (with_buffer_write) {
        auto *val = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                           {loop.acc, loop.acc});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, loop.iv, val});
    }
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, one});
    auto *acc_next = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                            {loop.acc, loop.acc});
    b.br(loop.header);
    b.set_insertion_point(loop.exit);
    loop.exit_phi = b.phi(Type::of<float>());
    b.return_void();
    loop.iv->add_incoming(zero, loop.entry);
    loop.iv->add_incoming(next, loop.latch);
    loop.acc->add_incoming(zero_f, loop.entry);
    loop.acc->add_incoming(acc_next, loop.latch);
    loop.exit_phi->add_incoming(loop.acc, loop.header);
    return loop;
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

[[nodiscard]] size_t count_loops(FunctionDefinition *def) noexcept {
    auto dom_tree = compute_dom_tree(def);
    return discover_natural_loops(def, dom_tree).size();
}

[[nodiscard]] size_t count_phi_in_block(BasicBlock *block) noexcept {
    auto count = 0u;
    block->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { count++; }
    });
    return count;
}

}// namespace

void reg_loop_unroll() {

    "constant_trip_count_loop_is_fully_unrolled"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 4u, false);
        LoopUnrollOptions options{.max_trip_count = 256u};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 1u);
        expect(info.succeeded());
        // the loop is gone: no back-edges remain
        expect(count_loops(loop.kernel->definition()) == 0u);
        // four body copies plus entry and exit
        auto block_count = 0u;
        for (auto *block : loop.kernel->definition()->basic_blocks()) {
            static_cast<void>(block);
            block_count++;
        }
        expect(block_count == 6u);
        // the exit phi is fed by the last iteration's latch
        expect(loop.exit_phi->incoming_count() == 1u);
        expect_module_valid(m);
    };

    "trip_count_two_eliminates_header_phis"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 2u, false);
        auto info = loop_unroll_pass_run_on_function(loop.kernel, {});
        expect(info.unrolled_loop_count == 1u);
        // original header block is removed from the function
        auto header_still_present = false;
        for (auto *block : loop.kernel->definition()->basic_blocks()) {
            if (block == loop.header) { header_still_present = true; }
        }
        expect(!header_still_present);
        expect(count_loops(loop.kernel->definition()) == 0u);
        expect_module_valid(m);
    };

    "trip_count_above_max_is_not_unrolled"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 300u, false);
        LoopUnrollOptions options{.max_trip_count = 256u};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 0u);
        expect(count_loops(loop.kernel->definition()) == 1u);
        expect_module_valid(m);
    };

    "pure_only_rejects_loop_with_buffer_write"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 4u, true);
        LoopUnrollOptions options{.unroll_pure_only = true};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 0u);
        expect(count_loops(loop.kernel->definition()) == 1u);
        expect_module_valid(m);
    };

    "pure_only_unrolls_side_effect_free_loop"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 4u, false);
        LoopUnrollOptions options{.unroll_pure_only = true};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 1u);
        expect(count_loops(loop.kernel->definition()) == 0u);
        expect_module_valid(m);
    };

    "buffer_write_loop_unrolls_when_side_effects_allowed"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 4u, true);
        LoopUnrollOptions options{.unroll_pure_only = false};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 1u);
        expect(count_loops(loop.kernel->definition()) == 0u);
        // four cloned buffer writes remain
        auto write_count = 0u;
        loop.kernel->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(write_count == 4u);
        expect_module_valid(m);
    };

    "nested_loop_unrolls_inner_and_keeps_variable_outer"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *bound_arg = kernel->create_value_argument(Type::of<uint>());
        auto *entry = kernel->create_body_block();
        auto *outer_header = def->create_basic_block();
        auto *inner_preheader = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_latch = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t inner_bound_value = 2u;
        auto *inner_bound = m.create_constant(Type::of<uint>(), &inner_bound_value);

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        auto *outer_iv = b.phi(Type::of<uint>());
        auto *outer_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                                  {outer_iv, bound_arg});
        b.cond_br(outer_cond, inner_preheader, exit);
        b.set_insertion_point(inner_preheader);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        auto *inner_iv = b.phi(Type::of<uint>());
        auto *inner_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                                  {inner_iv, inner_bound});
        b.cond_br(inner_cond, inner_latch, outer_latch);
        b.set_insertion_point(inner_latch);
        auto *inner_next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                                  {inner_iv, one});
        b.br(inner_header);
        b.set_insertion_point(outer_latch);
        auto *outer_next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                                  {outer_iv, one});
        b.br(outer_header);
        b.set_insertion_point(exit);
        b.return_void();
        outer_iv->add_incoming(zero, entry);
        outer_iv->add_incoming(outer_next, outer_latch);
        inner_iv->add_incoming(zero, inner_preheader);
        inner_iv->add_incoming(inner_next, inner_latch);

        auto info = loop_unroll_pass_run_on_function(kernel, {});
        // the inner loop (constant trip 2) is unrolled; the outer loop has a
        // variable bound and remains
        expect(info.unrolled_loop_count == 1u);
        expect(count_loops(def) == 1u);
        expect_module_valid(m);
    };

    "loop_with_buffer_read_and_write_unrolls"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *in_buf = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *out_buf = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 4u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        auto two_value = 2.0f;
        auto *two_f = m.create_constant(Type::of<float>(), &two_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(cond, latch, exit);
        b.set_insertion_point(latch);
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {in_buf, iv});
        auto *scaled = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {read, two_f});
        b.call(ResourceWriteOp::BUFFER_WRITE, {out_buf, iv, scaled});
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto info = loop_unroll_pass_run_on_function(kernel, {});
        expect(info.unrolled_loop_count == 1u);
        expect(count_loops(def) == 0u);
        auto read_count = 0u;
        auto write_count = 0u;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceReadInst>()) { read_count++; }
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(read_count == 4u);
        expect(write_count == 4u);
        expect_module_valid(m);
    };

    "variable_trip_count_is_not_unrolled"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *bound_arg = kernel->create_value_argument(Type::of<uint>());
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound_arg});
        b.cond_br(cond, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto info = loop_unroll_pass_run_on_function(kernel, {});
        expect(info.unrolled_loop_count == 0u);
        expect(count_loops(def) == 1u);
        expect_module_valid(m);
    };

    "partial_unroll_peels_first_iterations_and_keeps_loop"_test = [] {
        Module m;
        // trip count 16 exceeds max_trip_count 8, so full unrolling rejects
        // the loop; peeling with factor 4 emits 4 straight-line copies and
        // keeps the remaining loop (trip count 16 - 4 = 12).
        auto loop = make_counted_loop(m, 16u, false);
        LoopUnrollOptions options{
            .max_trip_count = 8u,
            .partial_unroll_factor = 4u};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 0u);
        expect(info.partially_unrolled_loop_count == 1u);
        expect(info.succeeded());
        // the remaining loop survives: exactly one back-edge remains
        expect(count_loops(loop.kernel->definition()) == 1u);
        // entry + 4 peeled body copies + original header/latch + exit
        auto block_count = 0u;
        for (auto *block : loop.kernel->definition()->basic_blocks()) {
            static_cast<void>(block);
            block_count++;
        }
        expect(block_count == 8u);
        // the induction phi's entry edge now comes from the last peeled
        // latch (not the entry block) and carries the peeled recurrence
        // value rather than the constant start
        expect(loop.iv->incoming_count() == 2u);
        for (auto i = 0u; i < loop.iv->incoming_count(); ++i) {
            auto incoming = loop.iv->incoming(i);
            expect(incoming.block != loop.entry);
        }
        // the exit phi is still fed by the surviving header edge
        expect(loop.exit_phi->incoming_count() == 1u);
        expect(loop.exit_phi->incoming(0u).block == loop.header);
        expect_module_valid(m);
    };

    "partial_unroll_factor_covering_trip_count_is_rejected"_test = [] {
        Module m;
        // trip count 4 with factor 8: peeling must keep at least one
        // iteration in the loop, and full unrolling owns small loops, so
        // nothing is transformed here (max_trip_count 2 rejects full
        // unrolling, and the factor covers the whole trip count).
        auto loop = make_counted_loop(m, 4u, false);
        LoopUnrollOptions options{
            .max_trip_count = 2u,
            .partial_unroll_factor = 8u};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 0u);
        expect(info.partially_unrolled_loop_count == 0u);
        expect(count_loops(loop.kernel->definition()) == 1u);
        expect_module_valid(m);
    };

    "partial_unroll_disabled_by_default"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 300u, false);
        LoopUnrollOptions options{.max_trip_count = 256u};
        auto info = loop_unroll_pass_run_on_function(loop.kernel, options);
        expect(info.unrolled_loop_count == 0u);
        expect(info.partially_unrolled_loop_count == 0u);
        expect(count_loops(loop.kernel->definition()) == 1u);
        expect_module_valid(m);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_loop_unroll();
    return 0;
}
