// Test for the XIR loop-rotation pass.
// This test covers:
// - top-checked while loop rotated to a bottom-checked loop with a guard
// - nested loops (inner and outer both rotated)
// - already bottom-checked loop left unchanged
// - multi-exit loop rejected
// - single-block (empty body) loop rotated

#include "ut/ut.hpp"

#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/translators/xir2text.h>

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
    ArithmeticInst *condition;
};

// entry -> header { iv = phi(entry: 0, latch: next); cond = iv < bound;
//                   cond_br(cond, latch, exit) }
// latch { next = iv + 1; br header }
// exit { return }
[[nodiscard]] CountedLoop make_counted_loop(Module &m, uint32_t bound_value) noexcept {
    CountedLoop loop;
    loop.kernel = m.create_kernel();
    auto *def = loop.kernel->definition();
    loop.entry = loop.kernel->create_body_block();
    loop.header = def->create_basic_block();
    loop.latch = def->create_basic_block();
    loop.exit = def->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *one = m.create_constant_one(Type::of<uint>());
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);

    XIRBuilder b;
    b.set_insertion_point(loop.entry);
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    loop.condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                            {loop.iv, bound});
    b.cond_br(loop.condition, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, one});
    b.br(loop.header);
    b.set_insertion_point(loop.exit);
    b.return_void();
    loop.iv->add_incoming(zero, loop.entry);
    loop.iv->add_incoming(next, loop.latch);
    return loop;
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

}// namespace

void reg_loop_rotation() {

    "top_checked_loop_is_rotated_with_guard"_test = [] {
        Module m;
        auto loop = make_counted_loop(m, 64u);
        auto info = loop_rotation_pass_run_on_function(loop.kernel->definition());
        expect(info.rotated_loop_count == 1u);
        expect(info.succeeded());

        // the header now unconditionally enters the body
        auto *header_terminator = loop.header->terminator();
        expect(header_terminator->isa<BranchInst>());
        expect(static_cast<BranchInst *>(header_terminator)->target_block() == loop.latch);
        // the latch performs the bottom check
        auto *latch_terminator = loop.latch->terminator();
        expect(latch_terminator->isa<ConditionalBranchInst>());
        auto *latch_branch = static_cast<ConditionalBranchInst *>(latch_terminator);
        expect(latch_branch->true_block() == loop.header);
        expect(latch_branch->false_block() == loop.exit);
        // the entry now branches to a guard, which checks the start value
        auto *entry_terminator = loop.entry->terminator();
        expect(entry_terminator->isa<BranchInst>());
        auto *guard = static_cast<BranchInst *>(entry_terminator)->target_block();
        expect(guard != loop.header);
        auto *guard_terminator = guard->terminator();
        expect(guard_terminator->isa<ConditionalBranchInst>());
        auto *guard_branch = static_cast<ConditionalBranchInst *>(guard_terminator);
        expect(guard_branch->true_block() == loop.header);
        expect(guard_branch->false_block() == loop.exit);
        // the header phi enters from the guard
        expect(loop.iv->incoming_count() == 2u);
        auto saw_guard = false;
        auto saw_latch = false;
        for (auto i = 0u; i < loop.iv->incoming_count(); ++i) {
            saw_guard = saw_guard || loop.iv->incoming(i).block == guard;
            saw_latch = saw_latch || loop.iv->incoming(i).block == loop.latch;
        }
        expect(saw_guard && saw_latch);
        expect_module_valid(m);
    };

    "nested_loops_rotate_innermost_only"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *outer_header = def->create_basic_block();

        // Build a proper nest in `def`:
        // entry -> outer_header { cond_br(c1, inner_preheader, exit) }
        // inner_preheader { br inner_header }
        // inner_header { cond_br(c2, inner_body, outer_latch) }
        // inner_body { br inner_header }
        // outer_latch { br outer_header }
        auto *inner_preheader = def->create_basic_block();
        auto *inner_header = def->create_basic_block();
        auto *inner_body = def->create_basic_block();
        auto *outer_latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *c = m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(outer_header);
        b.set_insertion_point(outer_header);
        b.cond_br(c, inner_preheader, exit);
        b.set_insertion_point(inner_preheader);
        b.br(inner_header);
        b.set_insertion_point(inner_header);
        b.cond_br(c, inner_body, outer_latch);
        b.set_insertion_point(inner_body);
        b.br(inner_header);
        b.set_insertion_point(outer_latch);
        b.br(outer_header);
        b.set_insertion_point(exit);
        b.return_void();

        auto info = loop_rotation_pass_run_on_function(def);
        // only the innermost loop is rotated; the outer loop is left in
        // top-checked form (see pass comment about the LoopInst path)
        expect(info.rotated_loop_count == 1u);
        // the outer header keeps its conditional top check
        expect(outer_header->terminator()->isa<ConditionalBranchInst>());
        // the inner header now enters its body unconditionally
        expect(inner_header->terminator()->isa<BranchInst>());
        expect_module_valid(m);
    };

    "already_bottom_checked_loop_is_unchanged"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *check = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *c = m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        // the header is the body and unconditionally falls into the check
        b.br(check);
        b.set_insertion_point(check);
        // the bottom check lives at the end of the body
        b.cond_br(c, header, exit);
        b.set_insertion_point(exit);
        b.return_void();

        auto info = loop_rotation_pass_run_on_function(def);
        // This is already a do-while: the header has no separate top check
        // block, so rotation must not apply twice.
        expect(info.rotated_loop_count == 0u);
        expect_module_valid(m);
    };

    "multi_exit_loop_is_rejected"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *body = def->create_basic_block();
        auto *exit1 = def->create_basic_block();
        auto *exit2 = def->create_basic_block();
        auto *c = m.create_constant_one(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        b.cond_br(c, body, exit1);
        b.set_insertion_point(body);
        b.cond_br(c, header, exit2);
        b.set_insertion_point(exit1);
        b.return_void();
        b.set_insertion_point(exit2);
        b.return_void();

        auto info = loop_rotation_pass_run_on_function(def);
        expect(info.rotated_loop_count == 0u);
        expect_module_valid(m);
    };

    "loop_rotation_rejects_observable_header_store_without_mutation"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 8u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *slot = b.alloca_local(Type::of<uint>());
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        b.store(slot, iv);
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_rotation_pass_run_on_function(def);
        auto after = xir_to_text_translate(&m, true);
        expect(info.rotated_loop_count == 0u);
        expect(before == after);
        expect_module_valid(m);
    };

    "loop_rotation_rejects_opaque_header_call_without_mutation"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        auto *callee_body = callee->create_body_block();
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 8u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.return_void();
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        static_cast<void>(b.call(nullptr, callee, {}));
        auto *condition = b.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(condition, latch, exit);
        b.set_insertion_point(latch);
        auto *next = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_rotation_pass_run_on_function(def);
        auto after = xir_to_text_translate(&m, true);
        expect(info.rotated_loop_count == 0u);
        expect(before == after);
        expect_module_valid(m);
    };

    "loop_rotation_rejects_ambiguous_control_metadata_atomically"_test = [] {
        auto run = [](bool annotate_condition) noexcept {
            Module m;
            auto loop = make_counted_loop(m, 8u);
            if (annotate_condition) {
                loop.condition->add_comment(
                    "condition metadata cannot be duplicated");
            } else {
                loop.header->terminator()->add_comment(
                    "branch metadata has no unique rotated owner");
            }
            auto before = xir_to_text_translate(&m, true);
            auto info = loop_rotation_pass_run_on_function(
                loop.kernel->definition());
            auto after = xir_to_text_translate(&m, true);
            expect(!info.changed());
            expect(before == after);
            expect_module_valid(m);
        };
        run(false);
        run(true);
    };

    "inverted_condition_loop_preserves_arm_order"_test = [] {
        // while-style loop expressed as cond_br(iv >= bound, exit, body):
        // the body is the FALSE target, so the guard and latch branches must
        // preserve that order (regression: inverted order skipped the loop).
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 16u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER_EQUAL,
                            {iv, bound});
        b.cond_br(cond, exit, latch);
        b.set_insertion_point(latch);
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        b.br(header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, latch);

        auto info = loop_rotation_pass_run_on_function(def);
        expect(info.rotated_loop_count == 1u);
        // the guard must branch to the header when the condition is FALSE
        auto *guard = static_cast<BranchInst *>(entry->terminator())->target_block();
        auto *guard_branch = static_cast<ConditionalBranchInst *>(guard->terminator());
        expect(guard_branch->true_block() == exit);
        expect(guard_branch->false_block() == header);
        // the latch must mirror the same arm order
        auto *latch_branch = static_cast<ConditionalBranchInst *>(latch->terminator());
        expect(latch_branch->true_block() == exit);
        expect(latch_branch->false_block() == header);
        expect_module_valid(m);
    };

    "restructure_round_trips_rotated_loop"_test = [] {
        // Build an already-rotated (bottom-checked) loop:
        // entry -> guard { cond_br(c0, exit, header) }
        // header { iv = phi(guard: 0, latch: next); br latch }
        // latch { next = iv + 1; cond_br(next >= bound, exit, header) }
        // exit { return }
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *guard = def->create_basic_block();
        auto *header = def->create_basic_block();
        auto *latch = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 16u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        auto *false_const = m.create_constant_zero(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(guard);
        b.set_insertion_point(guard);
        b.cond_br(false_const, exit, header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        b.br(latch);
        b.set_insertion_point(latch);
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER_EQUAL,
                            {next, bound});
        b.cond_br(cond, exit, header);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, guard);
        iv->add_incoming(next, latch);

        static_cast<void>(reg2mem_pass_run_on_function(kernel));
        auto info = restructure_cfg_pass_run_on_function(kernel);
        LUISA_INFO("restructure: loops={} ifs={} irreducible={} unstructured={}",
                   info.restructured_loop_count, info.restructured_if_count,
                   info.irreducible_region_count, info.unstructured_branch_count);
        auto text = xir_to_text_translate(&m, true);
        LUISA_INFO("restructured module:\n{}", text);
        expect(info.irreducible_region_count == 0u);
        expect_module_valid(m);
        // the exit condition must survive as a break inside the loop
        auto break_count = 0u;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<BreakInst>()) { break_count++; }
        });
        expect(break_count == 1u);
    };

    "single_block_loop_is_rotated"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *def = kernel->definition();
        auto *entry = kernel->create_body_block();
        auto *header = def->create_basic_block();
        auto *exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 8u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *iv = b.phi(Type::of<uint>());
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv, one});
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
        b.cond_br(cond, header, exit);
        b.set_insertion_point(exit);
        b.return_void();
        iv->add_incoming(zero, entry);
        iv->add_incoming(next, header);

        auto info = loop_rotation_pass_run_on_function(def);
        expect(info.rotated_loop_count == 1u);
        // the header still self-loops, but the entry now passes through a guard
        auto *guard = static_cast<BranchInst *>(entry->terminator())->target_block();
        expect(guard != header);
        expect(guard->terminator()->isa<ConditionalBranchInst>());
        expect(header->terminator()->isa<ConditionalBranchInst>());
        expect_module_valid(m);
    };

    "loop_rotation_module_rejection_is_atomic_across_functions"_test = [] {
        Module m;
        auto plain = make_counted_loop(m, 64u);

        auto *structured_kernel = m.create_kernel();
        auto *parent = structured_kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(parent);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.br(body);
        b.set_insertion_point(body);
        b.br(update);
        b.set_insertion_point(update);
        b.break_(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect_module_valid(m);

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_rotation_pass_run_on_module(&m);
        auto after = xir_to_text_translate(&m, true);
        expect(!info.succeeded());
        expect(!info.changed());
        expect(info.structured_cfg_error_count == 1u);
        expect(info.rotated_loop_count == 0u);
        expect(before == after);
        expect(plain.header->terminator()->isa<ConditionalBranchInst>());
        expect_module_valid(m);
    };

    "loop_rotation_null_module_is_a_noop"_test = [] {
        auto info = loop_rotation_pass_run_on_module(nullptr);
        expect(info.succeeded());
        expect(!info.changed());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_loop_rotation();
    return 0;
}
