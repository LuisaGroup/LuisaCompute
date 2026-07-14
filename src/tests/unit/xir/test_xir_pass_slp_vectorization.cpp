#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/slp_vectorization.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

namespace {

[[nodiscard]] Value *gep_array_element(XIRBuilder &b, Module &m, Value *array_alloca,
                                       int32_t index, const Type *elem_type) noexcept {
    auto *c = m.create_constant(Type::of<int>(), &index);
    return b.gep(elem_type, array_alloca, {c});
}

[[nodiscard]] Constant *float_const(Module &m, float v) noexcept {
    return m.create_constant(Type::of<float>(), &v);
}

}// namespace

void reg_slp_vectorization() {

    "slp_vectorization_consecutive_stores"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *x = b.alloca_local(Type::of<float>());
        b.store(x, m.create_constant_zero(Type::of<float>()));
        auto *xv = b.load(Type::of<float>(), x);

        // Build values first, then emit stores consecutively.
        auto *v0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 1.0f)});
        auto *v1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 2.0f)});
        auto *v2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 3.0f)});
        auto *v3 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 4.0f)});
        auto *p0 = gep_array_element(b, m, arr, 0, Type::of<float>());
        auto *p1 = gep_array_element(b, m, arr, 1, Type::of<float>());
        auto *p2 = gep_array_element(b, m, arr, 2, Type::of<float>());
        auto *p3 = gep_array_element(b, m, arr, 3, Type::of<float>());
        b.store(p0, v0);
        b.store(p1, v1);
        b.store(p2, v2);
        b.store(p3, v3);

        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 1u);
        expect(info.vectorized_inst_count >= 2u);
    };

    "slp_vectorization_non_isomorphic_rejected"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *x = b.alloca_local(Type::of<float>());
        b.store(x, m.create_constant_zero(Type::of<float>()));
        auto *xv = b.load(Type::of<float>(), x);

        // Mixed operations: add, sub, mul, add -> not isomorphic.
        b.store(gep_array_element(b, m, arr, 0, Type::of<float>()),
                b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 1.0f)}));
        b.store(gep_array_element(b, m, arr, 1, Type::of<float>()),
                b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {xv, float_const(m, 2.0f)}));
        b.store(gep_array_element(b, m, arr, 2, Type::of<float>()),
                b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {xv, float_const(m, 3.0f)}));
        b.store(gep_array_element(b, m, arr, 3, Type::of<float>()),
                b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 4.0f)}));

        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
    };

    "slp_vectorization_empty_module"_test = [] {
        Module m;
        auto info = slp_vectorization_pass_run_on_module(&m);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
    };

    "slp_vectorization_no_store"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_slp_vectorization();
    return 0;
}
