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

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_loop_vectorization();
    return 0;
}
