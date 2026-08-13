// Test for the conservative XIR loop-fusion contract.

#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

#include <limits>

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

struct CountedLoopFixture {
    Module &m;
    BasicBlock *body{nullptr};
    KernelFunction *k{nullptr};
    LoopInst *loop{nullptr};
    BasicBlock *prep{nullptr};
    BasicBlock *lbody{nullptr};
    BasicBlock *upd{nullptr};
    BasicBlock *merge{nullptr};
    PhiInst *iv{nullptr};
    XIRBuilder b;

    CountedLoopFixture(Module &module_, BasicBlock *parent, int32_t bound, int32_t step = 1) noexcept
        : m(module_) {
        b.set_insertion_point(parent);
        loop = b.loop();
        prep = loop->create_prepare_block();
        lbody = loop->create_body_block();
        upd = loop->create_update_block();
        merge = loop->create_merge_block();

        b.set_insertion_point(prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        iv = b.phi(Type::of<int>(), {{c0, parent}});
        auto *bound_const = [&]() {
            int32_t v = bound;
            return m.create_constant(Type::of<int>(), &v);
        }();
        auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound_const});
        b.cond_br(cmp, lbody, merge);

        b.set_insertion_point(upd);
        auto *step_const = [&]() {
            int32_t v = step;
            return m.create_constant(Type::of<int>(), &v);
        }();
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, step_const});
        iv->add_incoming(inc, upd);
        b.br(prep);
    }
};

[[nodiscard]] Value *gep_array_element(XIRBuilder &b, Module &m, Value *array_alloca,
                                       Value *index, const Type *elem_type) noexcept {
    return b.gep(elem_type, array_alloca, {index});
}

}// namespace

void reg_loop_fusion() {

    "loop_fusion_rejects_structured_independent_loops"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 8;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop1(m, body, n);
        b.set_insertion_point(loop1.lbody);
        auto *gep_a = gep_array_element(b, m, arr_a, loop1.iv, Type::of<float>());
        auto *val = m.create_constant_zero(Type::of<float>());
        b.store(gep_a, val);
        b.br(loop1.upd);

        b.set_insertion_point(loop1.merge);
        CountedLoopFixture loop2(m, loop1.merge, n);
        b.set_insertion_point(loop2.lbody);
        auto *gep_b = gep_array_element(b, m, arr_b, loop2.iv, Type::of<float>());
        b.store(gep_b, val);
        b.br(loop2.upd);

        b.set_insertion_point(loop2.merge);
        b.return_void();

        auto *loop1_term = body->terminator();
        auto *loop2_term = loop1.merge->terminator();
        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop1_term);
        expect(loop1.merge->terminator() == loop2_term);
        expect(loop1.loop->merge_block() == loop1.merge);
        expect(loop2.loop->merge_block() == loop2.merge);
    };

    "loop_fusion_rejects_structured_dependent_loops"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 8;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop1(m, body, n);
        b.set_insertion_point(loop1.lbody);
        auto *gep_a = gep_array_element(b, m, arr_a, loop1.iv, Type::of<float>());
        auto *val = m.create_constant_zero(Type::of<float>());
        b.store(gep_a, val);
        b.br(loop1.upd);

        b.set_insertion_point(loop1.merge);
        CountedLoopFixture loop2(m, loop1.merge, n);
        b.set_insertion_point(loop2.lbody);
        auto *gep_a2 = gep_array_element(b, m, arr_a, loop2.iv, Type::of<float>());
        auto *ld = b.load(Type::of<float>(), gep_a2);
        auto *gep_b = gep_array_element(b, m, arr_b, loop2.iv, Type::of<float>());
        b.store(gep_b, ld);
        b.br(loop2.upd);

        b.set_insertion_point(loop2.merge);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop1.loop);
        expect(loop1.merge->terminator() == loop2.loop);
    };

    "loop_fusion_empty_module"_test = [] {
        Module m;
        auto info = loop_fusion_pass_run_on_module(&m);
        expect(info.fused_loop_count == 0u);
        expect(info.succeeded());
    };

    "loop_fusion_no_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.succeeded());
    };
}

namespace {

struct PlainCountedLoop {
    BasicBlock *preheader;
    BasicBlock *header;
    BasicBlock *latch;
    BasicBlock *exit;
    PhiInst *iv;
};

// Build one counted loop whose body writes a constant to `buffer` at `iv`.
// Returns the blocks so tests can chain another loop at `exit`.
// The loop's exit block is created fresh; it becomes the next loop's
// preheader when chaining loops, or the function exit for the last one.
[[nodiscard]] PlainCountedLoop build_plain_counted_loop(
    Module &m, XIRBuilder &b, BasicBlock *preheader,
    uint32_t bound_value, ResourceArgument *buffer,
    bool write_constant_body,
    uint32_t stride_value = 1u) noexcept {
    PlainCountedLoop loop;
    loop.preheader = preheader;
    auto *defn = preheader->parent_function();
    loop.header = defn->create_basic_block();
    loop.latch = defn->create_basic_block();
    loop.exit = defn->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *stride = m.create_constant(
        Type::of<uint>(), &stride_value);
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
    auto *one_f = m.create_constant_one(Type::of<float>());

    b.set_insertion_point(preheader);
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                        {loop.iv, bound});
    b.cond_br(cond, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    if (write_constant_body) {
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, loop.iv, one_f});
    }
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, stride});
    b.br(loop.header);
    loop.iv->add_incoming(zero, preheader);
    loop.iv->add_incoming(next, loop.latch);
    return loop;
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

[[nodiscard]] size_t count_cond_br(FunctionDefinition *def) noexcept {
    auto count = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ConditionalBranchInst>()) { count++; }
    });
    return count;
}

}// namespace

void reg_loop_fusion_plain_cfg() {

    "plain_resource_loops_without_noalias_contract_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 8u, buf_b, true);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        // Distinct resource arguments may be overlapping runtime views.
        expect(count_cond_br(k->definition()) == 2u);
        // both buffer writes are still present
        auto write_count = 0u;
        k->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(write_count == 2u);
        expect_module_valid(m);
    };

    "plain_loops_with_different_trip_counts_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 16u, buf_b, true);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(count_cond_br(k->definition()) == 2u);
        expect_module_valid(m);
    };

    "plain_loops_with_write_read_dependence_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        // second loop READS the buffer the first loop writes
        auto *def = k->definition();
        auto *preheader2 = loop1.exit;
        auto *header2 = def->create_basic_block();
        auto *latch2 = def->create_basic_block();
        auto *exit2 = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 8u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        b.set_insertion_point(preheader2);
        b.br(header2);
        b.set_insertion_point(header2);
        auto *iv2 = b.phi(Type::of<uint>());
        auto *cond2 = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv2, bound});
        b.cond_br(cond2, latch2, exit2);
        b.set_insertion_point(latch2);
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buf_a, iv2});
        static_cast<void>(read);
        auto *next2 = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv2, one});
        b.br(header2);
        iv2->add_incoming(zero, preheader2);
        iv2->add_incoming(next2, latch2);
        b.set_insertion_point(exit2);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect_module_valid(m);
    };

    "plain_three_resource_loops_remain_separate_without_noalias"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_c = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 8u, buf_b, true);
        auto loop3 = build_plain_counted_loop(m, b, loop2.exit, 8u, buf_c, true);
        b.set_insertion_point(loop3.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(count_cond_br(k->definition()) == 3u);
        auto write_count = 0u;
        k->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(write_count == 3u);
        expect_module_valid(m);
    };

    "plain_loops_on_distinct_local_allocas_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *a = b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto *c = b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *value = m.create_constant_one(Type::of<uint>());
        b.set_insertion_point(loop1.latch->terminator()->prev());
        auto *a_element = b.gep(Type::of<uint>(), a, {loop1.iv});
        b.store(a_element, value);
        b.set_insertion_point(loop2.latch->terminator()->prev());
        auto *c_element = b.gep(Type::of<uint>(), c, {loop2.iv});
        b.store(c_element, value);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 1u);
        expect(count_cond_br(k->definition()) == 1u);
        expect_module_valid(m);
    };

    "plain_loop_fusion_rejects_unmapped_deleted_metadata"_test = [] {
        auto run = [](bool annotate_block) noexcept {
            Module m;
            BasicBlock *entry;
            auto *k = make_kernel_with_body(m, entry);
            XIRBuilder b;
            b.set_insertion_point(entry);
            auto *a = b.alloca_local(
                Type::array(Type::of<uint>(), 8u));
            auto *c = b.alloca_local(
                Type::array(Type::of<uint>(), 8u));
            auto loop1 = build_plain_counted_loop(
                m, b, entry, 8u, nullptr, false);
            auto loop2 = build_plain_counted_loop(
                m, b, loop1.exit, 8u, nullptr, false);
            auto *one = m.create_constant_one(Type::of<uint>());
            b.set_insertion_point(loop1.latch->terminator()->prev());
            b.store(b.gep(Type::of<uint>(), a, {loop1.iv}), one);
            b.set_insertion_point(loop2.latch->terminator()->prev());
            b.store(b.gep(Type::of<uint>(), c, {loop2.iv}), one);
            b.set_insertion_point(loop2.exit);
            b.return_void();
            if (annotate_block) {
                loop2.header->add_comment(
                    "deleted second header metadata");
            } else {
                static_cast<Instruction *>(
                    static_cast<ConditionalBranchInst *>(
                        loop2.header->terminator())
                        ->condition())
                    ->add_comment("deleted second comparison metadata");
            }
            auto before = xir_to_text_translate(&m, true);

            auto info = loop_fusion_pass_run_on_function(k);
            auto after = xir_to_text_translate(&m, true);
            expect(!info.changed());
            expect(info.succeeded());
            expect(before == after);
            expect(count_cond_br(k->definition()) == 2u);
            expect_module_valid(m);
        };
        run(false);
        run(true);
    };

    "plain_loop_fusion_retargets_second_body_entry_phi"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *a = b.alloca_local(
            Type::array(Type::of<uint>(), 8u));
        auto *c = b.alloca_local(
            Type::array(Type::of<uint>(), 8u));
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *one = m.create_constant_one(Type::of<uint>());
        b.set_insertion_point(loop1.latch->terminator()->prev());
        auto *a_element =
            b.gep(Type::of<uint>(), a, {loop1.iv});
        b.store(a_element, one);

        b.set_insertion_point(
            loop2.latch->instructions().head_sentinel());
        auto *body_phi = b.phi(
            Type::of<uint>(), {{loop2.iv, loop2.header}});
        b.set_insertion_point(loop2.latch->terminator()->prev());
        auto *c_element =
            b.gep(Type::of<uint>(), c, {loop2.iv});
        b.store(c_element, body_phi);
        b.set_insertion_point(loop2.exit);
        b.return_void();
        expect_module_valid(m);

        auto info = loop_fusion_pass_run_on_function(k);

        expect(info.fused_loop_count == 1u);
        expect(body_phi->incoming_count() == 1u);
        expect(body_phi->incoming(0u).block == loop1.latch);
        expect(body_phi->incoming(0u).value == loop1.iv);
        expect(count_cond_br(k->definition()) == 1u);
        expect_module_valid(m);
    };

    "plain_loops_with_different_predicates_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *second_branch = static_cast<ConditionalBranchInst *>(
            loop2.header->terminator());
        static_cast<ArithmeticInst *>(second_branch->condition())
            ->set_op(ArithmeticOp::BINARY_LESS_EQUAL);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect(count_cond_br(k->definition()) == 2u);
        expect_module_valid(m);
    };

    "plain_loops_with_inverted_continuation_polarity_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *second_branch = static_cast<ConditionalBranchInst *>(
            loop2.header->terminator());
        second_branch->set_true_target(loop2.exit);
        second_branch->set_false_target(loop2.latch);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect_module_valid(m);
    };

    "plain_loop_fusion_rejects_second_header_value_used_by_body"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *array = b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *one = m.create_constant_one(Type::of<uint>());
        auto *condition = static_cast<ConditionalBranchInst *>(
                              loop2.header->terminator())
                              ->condition();
        b.set_insertion_point(static_cast<Instruction *>(condition)->prev());
        auto *header_value = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {loop2.iv, one});
        b.set_insertion_point(loop2.latch->terminator()->prev());
        auto *element = b.gep(Type::of<uint>(), array, {loop2.iv});
        b.store(element, header_value);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect_module_valid(m);
    };

    "plain_loop_fusion_rejects_wrapping_nonterminating_loops"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        auto maximum = std::numeric_limits<uint32_t>::max();
        auto loop1 = build_plain_counted_loop(
            m, b, entry, maximum, nullptr, false, 2u);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, maximum, nullptr, false, 2u);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        // The first scalar loop cycles through even IVs forever and never
        // reaches UINT32_MAX. Fusion would incorrectly start executing the
        // second loop's body, so termination must be proven first.
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect(count_cond_br(k->definition()) == 2u);
        expect_module_valid(m);
    };

    "plain_loop_fusion_requires_formalized_trip_count_predicate"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        for (auto *header : {loop1.header, loop2.header}) {
            auto *branch = static_cast<ConditionalBranchInst *>(
                header->terminator());
            static_cast<ArithmeticInst *>(branch->condition())
                ->set_op(ArithmeticOp::BINARY_NOT_EQUAL);
        }
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect_module_valid(m);
    };

    "plain_loop_fusion_rejects_shared_preheader_bypass"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *def = k->definition();
        auto *first_preheader = def->create_basic_block();
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(
            m, b, first_preheader, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        b.set_insertion_point(entry);
        b.cond_br(
            m.create_undefined(Type::of<bool>()),
            first_preheader, loop1.exit);
        b.set_insertion_point(loop2.exit);
        b.return_void();
        expect_module_valid(m);

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        // loop1.exit is also loop2's preheader, but an external branch can
        // bypass loop1 and reach it directly. It therefore cannot be deleted.
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect(count_cond_br(k->definition()) == 3u);
        expect_module_valid(m);
    };

    "loop_fusion_module_rejection_is_atomic_across_functions"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *plain = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *a = b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto *c = b.alloca_local(Type::array(Type::of<uint>(), 8u));
        auto loop1 = build_plain_counted_loop(
            m, b, entry, 8u, nullptr, false);
        auto loop2 = build_plain_counted_loop(
            m, b, loop1.exit, 8u, nullptr, false);
        auto *one = m.create_constant_one(Type::of<uint>());
        b.set_insertion_point(loop1.latch->terminator()->prev());
        b.store(b.gep(Type::of<uint>(), a, {loop1.iv}), one);
        b.set_insertion_point(loop2.latch->terminator()->prev());
        b.store(b.gep(Type::of<uint>(), c, {loop2.iv}), one);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        BasicBlock *structured_parent;
        auto *structured =
            make_kernel_with_body(m, structured_parent);
        CountedLoopFixture structured_loop(m, structured_parent, 8);
        structured_loop.b.set_insertion_point(structured_loop.lbody);
        structured_loop.b.br(structured_loop.upd);
        structured_loop.b.set_insertion_point(structured_loop.merge);
        structured_loop.b.return_void();
        static_cast<void>(structured);
        expect_module_valid(m);

        auto before = xir_to_text_translate(&m, true);
        auto info = loop_fusion_pass_run_on_module(&m);
        auto after = xir_to_text_translate(&m, true);
        expect(!info.succeeded());
        expect(!info.changed());
        expect(info.structured_cfg_error_count == 1u);
        expect(info.fused_loop_count == 0u);
        expect(before == after);
        expect(count_cond_br(plain->definition()) == 2u);
        expect_module_valid(m);
    };

    "loop_fusion_null_module_is_a_noop"_test = [] {
        auto info = loop_fusion_pass_run_on_module(nullptr);
        expect(info.succeeded());
        expect(!info.changed());
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_loop_fusion();
    reg_loop_fusion_plain_cfg();
    return 0;
}
