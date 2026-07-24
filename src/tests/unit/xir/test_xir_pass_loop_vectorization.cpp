// Test for the conservative XIR loop-vectorization contract.

#include "ut/ut.hpp"
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
#include <luisa/xir/passes/loop_vectorization.h>
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

[[nodiscard]] Value *gep_array_element(XIRBuilder &b, Module &m, Value *array_alloca,
                                       Value *index, const Type *elem_type) noexcept {
    return b.gep(elem_type, array_alloca, {index});
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

}// namespace

void reg_loop_vectorization() {

    "loop_vectorization_rejects_structured_array_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 12;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_c = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop(m, body, n);
        b.set_insertion_point(loop.lbody);
        auto *gep_b = gep_array_element(b, m, arr_b, loop.iv, Type::of<float>());
        auto *gep_c = gep_array_element(b, m, arr_c, loop.iv, Type::of<float>());
        auto *gep_a = gep_array_element(b, m, arr_a, loop.iv, Type::of<float>());
        auto *lb = b.load(Type::of<float>(), gep_b);
        auto *lc = b.load(Type::of<float>(), gep_c);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {lb, lc});
        b.store(gep_a, sum);
        b.br(loop.upd);

        b.set_insertion_point(loop.merge);
        b.return_void();

        auto *original_loop = body->terminator();
        auto info = loop_vectorization_pass_run_on_function(k);
        expect(info.vectorized_loop_count == 0u);
        expect(info.created_vector_inst_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == original_loop);
        expect(loop.loop->prepare_block() == loop.prep);
        expect(loop.loop->body_block() == loop.lbody);
        expect(loop.loop->update_block() == loop.upd);
        expect(loop.loop->merge_block() == loop.merge);
    };

    "loop_vectorization_rejects_structured_non_unit_stride"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 12;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_c = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop(m, body, n, /*step=*/2);
        b.set_insertion_point(loop.lbody);
        auto *gep_b = gep_array_element(b, m, arr_b, loop.iv, Type::of<float>());
        auto *gep_c = gep_array_element(b, m, arr_c, loop.iv, Type::of<float>());
        auto *gep_a = gep_array_element(b, m, arr_a, loop.iv, Type::of<float>());
        auto *lb = b.load(Type::of<float>(), gep_b);
        auto *lc = b.load(Type::of<float>(), gep_c);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {lb, lc});
        b.store(gep_a, sum);
        b.br(loop.upd);

        b.set_insertion_point(loop.merge);
        b.return_void();

        auto info = loop_vectorization_pass_run_on_function(k);
        expect(info.vectorized_loop_count == 0u);
        expect(info.created_vector_inst_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
    };

    "loop_vectorization_rejects_structured_empty_body"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);

        CountedLoopFixture loop(m, body, 8);
        loop.b.set_insertion_point(loop.lbody);
        loop.b.br(loop.upd);

        loop.b.set_insertion_point(loop.merge);
        loop.b.return_void();

        auto info = loop_vectorization_pass_run_on_function(k);
        expect(info.vectorized_loop_count == 0u);
        expect(info.created_vector_inst_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
    };

    "loop_vectorization_empty_module"_test = [] {
        Module m;
        auto info = loop_vectorization_pass_run_on_module(&m);
        expect(info.vectorized_loop_count == 0u);
        expect(info.created_vector_inst_count == 0u);
        expect(info.succeeded());
    };

    "loop_vectorization_no_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = loop_vectorization_pass_run_on_function(k);
        expect(info.vectorized_loop_count == 0u);
        expect(info.created_vector_inst_count == 0u);
        expect(info.succeeded());
    };
}

namespace {

struct VectorizableLoop {
    KernelFunction *kernel;
    BasicBlock *entry;
    BasicBlock *header;
    BasicBlock *latch;
    BasicBlock *exit;
    PhiInst *iv;
    AllocaInst *arr_a;
    AllocaInst *arr_b;
    AllocaInst *arr_c;
};

// entry -> header { iv = phi(0, next); cond = iv < bound;
//                   cond_br(cond, latch, exit) }
// latch { c[iv] = a[iv] + b[iv]; next = iv + stride; br header }
[[nodiscard]] VectorizableLoop make_vectorizable_loop(
    Module &m, uint32_t bound_value, uint32_t stride_value,
    bool second_phi = false, bool alias_store = false) noexcept {
    VectorizableLoop loop;
    loop.kernel = make_kernel_with_body(m, loop.entry);
    auto *def = loop.kernel->definition();
    loop.header = def->create_basic_block();
    loop.latch = def->create_basic_block();
    loop.exit = def->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *stride = m.create_constant(Type::of<uint>(), &stride_value);
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);

    XIRBuilder b;
    b.set_insertion_point(loop.entry);
    loop.arr_a = b.alloca_local(Type::array(Type::of<float>(), 64u));
    loop.arr_b = b.alloca_local(Type::array(Type::of<float>(), 64u));
    loop.arr_c = b.alloca_local(Type::array(Type::of<float>(), 64u));
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    PhiInst *extra = nullptr;
    if (second_phi) {
        extra = b.phi(Type::of<float>());
    }
    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                        {loop.iv, bound});
    b.cond_br(cond, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    auto *pa = gep_array_element(b, m, loop.arr_a, loop.iv, Type::of<float>());
    auto *va = b.load(Type::of<float>(), pa);
    auto *pb = gep_array_element(b, m, loop.arr_b, loop.iv, Type::of<float>());
    auto *vb = b.load(Type::of<float>(), pb);
    auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {va, vb});
    auto *store_base = alias_store ? loop.arr_a : loop.arr_c;
    auto *pc = gep_array_element(b, m, store_base, loop.iv, Type::of<float>());
    b.store(pc, sum);
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, stride});
    b.br(loop.header);
    b.set_insertion_point(loop.exit);
    b.return_void();
    loop.iv->add_incoming(zero, loop.entry);
    loop.iv->add_incoming(next, loop.latch);
    if (extra != nullptr) {
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        extra->add_incoming(zero_f, loop.entry);
        extra->add_incoming(sum, loop.latch);
    }
    return loop;
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

[[nodiscard]] size_t count_vector_aggregates(FunctionDefinition *def) noexcept {
    auto count = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ArithmeticInst>() &&
            static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::AGGREGATE &&
            inst->type() != nullptr && inst->type()->is_vector()) {
            count++;
        }
    });
    return count;
}

}// namespace

void reg_loop_vectorization_plain_cfg() {

    "plain_elementwise_add_is_vectorized_four_wide"_test = [] {
        Module m;
        auto loop = make_vectorizable_loop(m, 16u, 1u);
        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 1u);
        expect(info.created_vector_inst_count > 0u);
        // the induction now steps by the vector factor
        auto found_step = false;
        for (auto i = 0u; i < loop.iv->incoming_count(); ++i) {
            auto incoming = loop.iv->incoming(i);
            if (incoming.value->isa<ArithmeticInst>()) {
                auto *add = static_cast<ArithmeticInst *>(incoming.value);
                if (add->op() == ArithmeticOp::BINARY_ADD) {
                    for (auto j = 0u; j < 2u; ++j) {
                        if (add->operand(j)->isa<Constant>()) {
                            auto *c = static_cast<Constant *>(add->operand(j));
                            if (c->as<uint32_t>() == 4u) { found_step = true; }
                        }
                    }
                }
            }
        }
        expect(found_step);
        // vector aggregates exist (packed lanes)
        expect(count_vector_aggregates(loop.kernel->definition()) > 0u);
        expect_module_valid(m);
    };

    "plain_trip_count_not_multiple_of_vf_vectorized_with_peeled_remainder"_test = [] {
        Module m;
        // trip count 10 = 2 vector iterations (8) + 2 peeled scalar ones
        auto loop = make_vectorizable_loop(m, 10u, 1u);
        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 1u);
        expect(info.created_vector_inst_count > 0u);
        // the loop bound is tightened to 8 (largest multiple of VF <= 10)
        auto *header_branch = static_cast<ConditionalBranchInst *>(
            loop.header->terminator());
        auto *condition = static_cast<ArithmeticInst *>(header_branch->condition());
        auto found_tightened_bound = false;
        for (auto i = 0u; i < condition->operand_count(); ++i) {
            auto *operand = condition->operand(i);
            if (operand->isa<Constant>() &&
                static_cast<Constant *>(operand)->as<uint32_t>() == 8u) {
                found_tightened_bound = true;
            }
        }
        expect(found_tightened_bound);
        // entry + header + latch + 2 peel blocks + exit
        auto block_count = 0u;
        for (auto *block : loop.kernel->definition()->basic_blocks()) {
            static_cast<void>(block);
            block_count++;
        }
        expect(block_count == 6u);
        // the header no longer exits directly to the exit block
        expect(header_branch->true_block() != loop.exit &&
               header_branch->false_block() != loop.exit);
        expect_module_valid(m);
    };

    "plain_non_reduction_second_phi_rejected"_test = [] {
        Module m;
        // the extra phi's latch incoming is the body value itself (not an
        // accumulator combine), so it is not a valid reduction pattern
        auto loop = make_vectorizable_loop(m, 16u, 1u, true);
        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 0u);
        expect_module_valid(m);
    };

    "plain_non_unit_stride_rejected"_test = [] {
        Module m;
        auto loop = make_vectorizable_loop(m, 16u, 2u);
        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 0u);
        expect_module_valid(m);
    };

    "plain_store_load_alias_rejected"_test = [] {
        Module m;
        auto loop = make_vectorizable_loop(m, 16u, 1u, false, true);
        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 0u);
        expect_module_valid(m);
    };

    "plain_sum_reduction_is_vectorized_with_horizontal_fold"_test = [] {
        Module m;
        // acc = phi(0.0f, acc + (a[iv] + b[iv])); no stores in the body
        VectorizableLoop loop;
        loop.kernel = make_kernel_with_body(m, loop.entry);
        auto *def = loop.kernel->definition();
        loop.header = def->create_basic_block();
        loop.latch = def->create_basic_block();
        loop.exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        uint32_t bound_value = 16u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        uint32_t one_value = 1u;
        auto *one = m.create_constant(Type::of<uint>(), &one_value);

        XIRBuilder b;
        b.set_insertion_point(loop.entry);
        loop.arr_a = b.alloca_local(Type::array(Type::of<float>(), 64u));
        loop.arr_b = b.alloca_local(Type::array(Type::of<float>(), 64u));
        b.br(loop.header);
        b.set_insertion_point(loop.header);
        loop.iv = b.phi(Type::of<uint>());
        auto *acc = b.phi(Type::of<float>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                            {loop.iv, bound});
        b.cond_br(cond, loop.latch, loop.exit);
        b.set_insertion_point(loop.latch);
        auto *pa = gep_array_element(b, m, loop.arr_a, loop.iv, Type::of<float>());
        auto *va = b.load(Type::of<float>(), pa);
        auto *pb = gep_array_element(b, m, loop.arr_b, loop.iv, Type::of<float>());
        auto *vb = b.load(Type::of<float>(), pb);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {va, vb});
        auto *acc_next = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                                {acc, sum});
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                            {loop.iv, one});
        b.br(loop.header);
        b.set_insertion_point(loop.exit);
        auto *result = b.phi(Type::of<float>());
        b.return_void();
        loop.iv->add_incoming(zero, loop.entry);
        loop.iv->add_incoming(next, loop.latch);
        acc->add_incoming(zero_f, loop.entry);
        acc->add_incoming(acc_next, loop.latch);
        result->add_incoming(acc, loop.header);

        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 1u);
        expect(info.created_vector_inst_count > 0u);
        // the accumulator survives as a scalar phi fed by a folded combine
        expect(acc->incoming_count() == 2u);
        auto found_folded_combine = false;
        for (auto i = 0u; i < acc->incoming_count(); ++i) {
            auto incoming = acc->incoming(i);
            if (incoming.block == loop.latch &&
                incoming.value->isa<ArithmeticInst>()) {
                auto *combine = static_cast<ArithmeticInst *>(incoming.value);
                if (combine->op() == ArithmeticOp::BINARY_ADD &&
                    combine->operand(0u) == acc) {
                    found_folded_combine = true;
                }
            }
        }
        expect(found_folded_combine);
        // the original per-iteration combine is gone
        auto combine_still_present = false;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst == static_cast<Instruction *>(acc_next)) {
                combine_still_present = true;
            }
        });
        expect(!combine_still_present);
        expect_module_valid(m);
    };

    "plain_reduction_with_remainder_is_rejected"_test = [] {
        Module m;
        // trip count 10 (not a multiple of VF) + reduction: peeling the
        // trailing iterations would need accumulator threading, so the
        // loop is left scalar
        VectorizableLoop loop;
        loop.kernel = make_kernel_with_body(m, loop.entry);
        auto *def = loop.kernel->definition();
        loop.header = def->create_basic_block();
        loop.latch = def->create_basic_block();
        loop.exit = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        uint32_t bound_value = 10u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        uint32_t one_value = 1u;
        auto *one = m.create_constant(Type::of<uint>(), &one_value);

        XIRBuilder b;
        b.set_insertion_point(loop.entry);
        loop.arr_a = b.alloca_local(Type::array(Type::of<float>(), 64u));
        loop.arr_b = b.alloca_local(Type::array(Type::of<float>(), 64u));
        b.br(loop.header);
        b.set_insertion_point(loop.header);
        loop.iv = b.phi(Type::of<uint>());
        auto *acc = b.phi(Type::of<float>());
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                            {loop.iv, bound});
        b.cond_br(cond, loop.latch, loop.exit);
        b.set_insertion_point(loop.latch);
        auto *pa = gep_array_element(b, m, loop.arr_a, loop.iv, Type::of<float>());
        auto *va = b.load(Type::of<float>(), pa);
        auto *pb = gep_array_element(b, m, loop.arr_b, loop.iv, Type::of<float>());
        auto *vb = b.load(Type::of<float>(), pb);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {va, vb});
        auto *acc_next = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                                {acc, sum});
        auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                            {loop.iv, one});
        b.br(loop.header);
        b.set_insertion_point(loop.exit);
        b.return_void();
        loop.iv->add_incoming(zero, loop.entry);
        loop.iv->add_incoming(next, loop.latch);
        acc->add_incoming(zero_f, loop.entry);
        acc->add_incoming(acc_next, loop.latch);

        auto info = loop_vectorization_pass_run_on_function(loop.kernel);
        expect(info.vectorized_loop_count == 0u);
        expect_module_valid(m);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_loop_vectorization();
    reg_loop_vectorization_plain_cfg();
    return 0;
}
