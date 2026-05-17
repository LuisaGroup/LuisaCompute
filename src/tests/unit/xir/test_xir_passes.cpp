#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/ast/type_registry.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

static KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

// ---- algebraic_simplify: integer identities ----

void reg_algebraic_simplify() {

    "algsimpl_int_add_zero_rhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 7;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_add_zero_lhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 5;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_sub_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 3;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_sub_self"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = 9;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {x, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_one_rhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 4;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_one_lhs"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 4;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {one, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_mul_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 99;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_div_one"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t one_v = 1, x_v = 8;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_div_zero_numerator"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 5;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {zero, x});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_bitor_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 42;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_OR, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_bitxor_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 13;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_XOR, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_int_shift_left_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t zero_v = 0, x_v = 7;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_float_add_zero_not_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.5f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_float_mul_one_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float one_v = 1.0f, x_v = 3.14f;
        auto *one = m.create_constant(Type::of<float>(), &one_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, one});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_float_mul_zero_not_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 2.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_no_simplification"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 3; ++i) {
            BasicBlock *body;
            make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t zero_v = 0, x_v = i + 1;
            auto *zero = m.create_constant(Type::of<int>(), &zero_v);
            auto *x = m.create_constant(Type::of<int>(), &x_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, zero});
            b.return_void();
        }
        auto info = algebraic_simplify_pass_run_on_module(&m);
        expect(info.simplified_inst_count == 3u);
    };
}

// ---- const_fold ----

void reg_const_fold() {

    "constfold_int_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_sub"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_mul"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 6, b_v = 7;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MUL, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_div"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 20, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_div_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 5, b_v = 0;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_uint_div_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t a_v = 5u, b_v = 0u;
        auto *a = m.create_constant(Type::of<uint>(), &a_v);
        auto *bv = m.create_constant(Type::of<uint>(), &b_v);
        b.call(Type::of<uint>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_mod_by_zero_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 7, b_v = 0;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_shift_left"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_shift_left_overflow_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 32;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_shift_right_overflow_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 8, b_v = 33;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_RIGHT, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_int_unary_minus"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = 5;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_unary_minus_int_min"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = std::numeric_limits<int32_t>::min();
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_float_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float a_v = 1.5f, b_v = 2.5f;
        auto *a = m.create_constant(Type::of<float>(), &a_v);
        auto *bv = m.create_constant(Type::of<float>(), &b_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_float_sqrt"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 4.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::SQRT, {x});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_comparison_less"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_non_const_operand_not_folded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *undef = m.create_undefined(Type::of<int>());
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, undef});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "constfold_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 4; ++i) {
            BasicBlock *body;
            make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = i + 1, b_v = i + 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_void();
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 4u);
    };

    "constfold_uint_unary_minus"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_v = 3u;
        auto *x = m.create_constant(Type::of<uint>(), &x_v);
        b.call(Type::of<uint>(), ArithmeticOp::UNARY_MINUS, {x});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_int_bitand"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 0xFF, b_v = 0x0F;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_BIT_AND, {a, bv});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };

    "constfold_float_clamp"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.5f, lo_v = 0.0f, hi_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
    };
}

// ---- loop_unroll ----

void reg_loop_unroll() {

    "loop_unroll_no_loops"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.unrolled_loop_count == 0u);
    };

    "loop_unroll_non_analyzable_loop_not_unrolled"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.br(merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.unrolled_loop_count == 0u);
    };

    "loop_unroll_counted_loop_4_trips"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        auto *phi = b.phi(Type::of<int>());
        int32_t bound_v = 4;
        auto *bound = m.create_constant(Type::of<int>(), &bound_v);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        b.cond_br(cond, loop_body, merge);

        b.set_insertion_point(loop_body);
        b.br(update);

        b.set_insertion_point(update);
        int32_t one_v = 1;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
        b.br(prepare);

        int32_t start_v = 0;
        auto *start = m.create_constant(Type::of<int>(), &start_v);
        phi->add_incoming(start, body);
        phi->add_incoming(inc, update);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.unrolled_loop_count == 1u);
    };

    "loop_unroll_trip_count_exceeds_max_not_unrolled"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        auto *phi = b.phi(Type::of<int>());
        int32_t bound_v = 100;
        auto *bound = m.create_constant(Type::of<int>(), &bound_v);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        b.cond_br(cond, loop_body, merge);

        b.set_insertion_point(loop_body);
        b.br(update);

        b.set_insertion_point(update);
        int32_t one_v = 1;
        auto *one = m.create_constant(Type::of<int>(), &one_v);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
        b.br(prepare);

        int32_t start_v = 0;
        auto *start = m.create_constant(Type::of<int>(), &start_v);
        phi->add_incoming(start, body);
        phi->add_incoming(inc, update);

        b.set_insertion_point(merge);
        b.return_void();

        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.unrolled_loop_count == 0u);
    };

    "loop_unroll_module_runs_all_functions"_test = [] {
        Module m;
        for (int fn = 0; fn < 2; ++fn) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *merge = loop->create_merge_block();

            b.set_insertion_point(prepare);
            auto *phi = b.phi(Type::of<int>());
            int32_t bound_v = 3;
            auto *bound = m.create_constant(Type::of<int>(), &bound_v);
            auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
            b.cond_br(cond, loop_body, merge);

            b.set_insertion_point(loop_body);
            b.br(update);

            b.set_insertion_point(update);
            int32_t one_v = 1;
            auto *one = m.create_constant(Type::of<int>(), &one_v);
            auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
            b.br(prepare);

            int32_t start_v = 0;
            auto *start = m.create_constant(Type::of<int>(), &start_v);
            phi->add_incoming(start, body);
            phi->add_incoming(inc, update);

            b.set_insertion_point(merge);
            b.return_void();
        }
        auto info = loop_unroll_pass_run_on_module(&m);
        expect(info.unrolled_loop_count == 2u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_algebraic_simplify();
    reg_const_fold();
    reg_loop_unroll();
    return 0;
}
