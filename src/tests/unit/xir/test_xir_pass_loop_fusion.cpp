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
#include <luisa/xir/passes/loop_fusion.h>

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

    "loop_fusion_two_independent_loops"_test = [] {
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

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 1u);
    };

    "loop_fusion_flow_dependence_not_fused"_test = [] {
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
    };

    "loop_fusion_empty_module"_test = [] {
        Module m;
        auto info = loop_fusion_pass_run_on_module(&m);
        expect(info.fused_loop_count == 0u);
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
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_loop_fusion();
    return 0;
}
