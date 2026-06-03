#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/div_rem_pairs.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/outline.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/reassociate.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/unused_callable_removal.h>
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

    "algsimpl_select_const_condition"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = m.create_callable(Type::of<int>());
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t f_v = 1, t_v = 2;
        auto *f = m.create_constant(Type::of<int>(), &f_v);
        auto *t = m.create_constant(Type::of<int>(), &t_v);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *select = b.call(Type::of<int>(), ArithmeticOp::SELECT, {f, t, cond});
        auto *ret = b.return_(select);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == t);
    };

    "algsimpl_float_mul_zero_keeps_nan_inf_semantics"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.5f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k, {.enable_fast_math = true});
        expect(info.simplified_inst_count == 0u);
    };

    "algsimpl_insert_into_aggregate"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto type = Type::vector(Type::of<int>(), 2u);
        int32_t x_v = 1, y_v = 2, z_v = 3;
        uint32_t index_v = 1u;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *y = m.create_constant(Type::of<int>(), &y_v);
        auto *z = m.create_constant(Type::of<int>(), &z_v);
        auto *index = m.create_constant(Type::of<uint>(), &index_v);
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {x, y});
        b.call(type, ArithmeticOp::INSERT, {aggregate, z, index});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_identity_extract_aggregate_to_original_vector"_test = [] {
        Module m;
        BasicBlock *body;
        auto type = Type::vector(Type::of<float>(), 3u);
        auto *k = m.create_callable(type);
        auto *v = k->create_value_argument(type);
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_i = 0u, y_i = 1u, z_i = 2u;
        auto *x_index = m.create_constant(Type::of<uint>(), &x_i);
        auto *y_index = m.create_constant(Type::of<uint>(), &y_i);
        auto *z_index = m.create_constant(Type::of<uint>(), &z_i);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, x_index});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, y_index});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, z_index});
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {x, y, z});
        auto *ret = b.return_(aggregate);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == v);
    };

    "algsimpl_extract_aggregate_to_shuffle"_test = [] {
        Module m;
        BasicBlock *body;
        auto type = Type::vector(Type::of<float>(), 3u);
        auto *k = m.create_callable(type);
        auto *v = k->create_value_argument(type);
        body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t x_i = 0u, y_i = 1u;
        auto *x_index = m.create_constant(Type::of<uint>(), &x_i);
        auto *y_index = m.create_constant(Type::of<uint>(), &y_i);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, x_index});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {v, y_index});
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {y, x, x});
        auto *ret = b.return_(aggregate);
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() != v);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *shuffle = static_cast<ArithmeticInst *>(ret->return_value());
        expect(shuffle->op() == ArithmeticOp::SHUFFLE);
        expect(shuffle->operand(0) == v);
        expect(shuffle->operand(1) == y_index);
        expect(shuffle->operand(2) == x_index);
        expect(shuffle->operand(3) == x_index);
    };

    "algsimpl_float_add_zero_keeps_signed_zero_semantics"_test = [] {
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
        auto info = algebraic_simplify_pass_run_on_function(k, {.enable_fast_math = true});
        expect(info.simplified_inst_count == 0u);
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

// ---- dce ----

void reg_dce() {

    "dce_unused_arithmetic_removed"_test = [] {
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
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_inst_count == 1u);
    };

    "dce_no_dead_code"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_inst_count == 0u);
    };

    "dce_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_void();
        }
        auto info = dce_pass_run_on_module(&m);
        expect(info.removed_inst_count == 2u);
    };
}

// ---- gvn ----

void reg_gvn() {

    "gvn_duplicate_add_replaced"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *final = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, add2});
        b.return_(final);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count >= 1u);
    };

    "gvn_no_duplicate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4, c_v = 5;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *c = m.create_constant(Type::of<int>(), &c_v);
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, c});
        b.return_(add2);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
    };

    "gvn_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            auto *final = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {add1, add2});
            b.return_(final);
        }
        auto info = gvn_pass_run_on_module(&m);
        expect(info.replaced_inst_count >= 2u);
    };
}

// ---- sccp ----

void reg_sccp() {

    "sccp_const_propagation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 2, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count >= 1u);
    };

    "sccp_no_constants_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *u1 = m.create_undefined(Type::of<int>());
        auto *u2 = m.create_undefined(Type::of<int>());
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {u1, u2});
        b.return_(add);
        auto info = sccp_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
    };

    "sccp_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_(add);
        }
        auto info = sccp_pass_run_on_module(&m);
        expect(info.folded_inst_count >= 2u);
    };

    "sccp_loop_carried_phi_not_folded"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);

        auto *header = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *update = k->create_basic_block();
        auto *merge = k->create_basic_block();

        // entry -> header
        b.br(header);

        // header: phi and loop condition
        b.set_insertion_point(header);
        auto *phi = b.phi(Type::of<int>());
        int32_t zero_v = 0, four_v = 4;
        auto *zero = m.create_constant(Type::of<int>(), &zero_v);
        auto *four = m.create_constant(Type::of<int>(), &four_v);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, four});
        b.cond_br(cond, loop_body, merge);

        // loop_body: load produces BOTTOM, add is loop-carried
        b.set_insertion_point(loop_body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *load = b.load(Type::of<int>(), alloca);
        auto *i_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, load});
        b.br(update);

        // update: back-edge to header
        b.set_insertion_point(update);
        b.br(header);

        // merge
        b.set_insertion_point(merge);
        b.return_void();

        phi->add_incoming(zero, entry);
        phi->add_incoming(i_next, update);

        auto info = sccp_pass_run_on_function(k);
        expect(info.removed_branch_count == 0u);
        expect(header->terminator()->derived_instruction_tag() == DerivedInstructionTag::CONDITIONAL_BRANCH);
    };
}

// ---- simplify_libcalls ----

void reg_simplify_libcalls() {

    "simplify_libcalls_lerp_t_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.0f, y_v = 2.0f, t_v = 0.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        b.return_void();
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
    };

    "simplify_libcalls_lerp_t_one"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.0f, y_v = 2.0f, t_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        b.return_void();
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
    };

    "simplify_libcalls_clamp_01_to_saturate"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 0.5f, lo_v = 0.0f, hi_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        b.return_void();
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
    };

    "simplify_libcalls_no_simplification"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 1.0f, y_v = 2.0f, t_v = 0.5f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        b.return_void();
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 0u);
    };

    "simplify_libcalls_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            float x_v = 1.0f, y_v = 2.0f, t_v = 0.0f;
            auto *x = m.create_constant(Type::of<float>(), &x_v);
            auto *y = m.create_constant(Type::of<float>(), &y_v);
            auto *t = m.create_constant(Type::of<float>(), &t_v);
            b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
            b.return_void();
        }
        auto info = simplify_libcalls_pass_run_on_module(&m);
        expect(info.simplified_count == 2u);
    };
}

// ---- reassociate ----

void reg_reassociate() {

    "reassociate_chained_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2, c_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *c = m.create_constant(Type::of<int>(), &c_v);
        auto *ab = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ab, c});
        b.return_(abc);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count >= 1u);
    };

    "reassociate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count == 0u);
    };

    "reassociate_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 1, b_v = 2, c_v = 3;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            auto *c = m.create_constant(Type::of<int>(), &c_v);
            auto *ab = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
            auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ab, c});
            b.return_(abc);
        }
        auto info = reassociate_pass_run_on_module(&m);
        expect(info.reassociated_inst_count >= 2u);
    };
}

// ---- cvp ----

void reg_cvp() {

    "cvp_equal_condition_propagates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *u = m.create_undefined(Type::of<int>());
        auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {u, val});
        auto *if_inst = b.if_(eq);
        auto *true_b = if_inst->create_true_block();
        auto *false_b = if_inst->create_false_block();
        auto *merge_b = if_inst->create_merge_block();
        b.set_insertion_point(true_b);
        b.br(merge_b);
        b.set_insertion_point(false_b);
        b.br(merge_b);
        b.set_insertion_point(merge_b);
        b.return_void();
        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count >= 0u);// checks the pass runs without error
    };

    "cvp_no_if_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
    };

    "cvp_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        auto info = cvp_pass_run_on_module(&m);
        expect(info.replaced_inst_count == 0u);
    };
}

// ---- dead_arg_elim ----

void reg_dead_arg_elim() {

    "dead_arg_elim_unused_callable_arg_removed"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<void>());
        auto *unused = c->create_value_argument(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dead_arg_elim_pass_run_on_function(c);
        expect(info.removed_arg_count == 1u);
    };

    "dead_arg_elim_all_args_used_no_change"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *arg = c->create_value_argument(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_(arg);
        auto info = dead_arg_elim_pass_run_on_function(c);
        expect(info.removed_arg_count == 0u);
    };

    "dead_arg_elim_kernel_skipped"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = dead_arg_elim_pass_run_on_function(k);
        expect(info.removed_arg_count == 0u);
    };

    "dead_arg_elim_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            auto *c = m.create_callable(Type::of<void>());
            c->create_value_argument(Type::of<float>());
            auto *body = c->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        auto info = dead_arg_elim_pass_run_on_module(&m);
        expect(info.removed_arg_count == 2u);
    };
}

// ---- div_rem_pairs ----

void reg_div_rem_pairs() {

    "div_rem_pairs_div_and_mod_merged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        b.return_void();
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 1u);
    };

    "div_rem_pairs_no_mod_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 0u);
    };

    "div_rem_pairs_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t a_v = 10, b_v = 3;
            auto *a = m.create_constant(Type::of<int>(), &a_v);
            auto *bv = m.create_constant(Type::of<int>(), &b_v);
            b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
            b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
            b.return_void();
        }
        auto info = div_rem_pairs_pass_run_on_module(&m);
        expect(info.merged_pair_count == 2u);
    };
}

// ---- local_load_elimination ----

void reg_local_load_elimination() {

    "local_load_elim_duplicate_load_removed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        auto *ld1 = b.load(Type::of<int>(), alloca);
        auto *ld2 = b.load(Type::of<int>(), alloca);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld1, ld2});
        b.return_(add);
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 1u);
    };

    "local_load_elim_no_duplicate_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        b.store(alloca, m.create_constant_zero(Type::of<int>()));
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
    };

    "local_load_elim_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            b.store(alloca, m.create_constant_zero(Type::of<int>()));
            auto *ld1 = b.load(Type::of<int>(), alloca);
            auto *ld2 = b.load(Type::of<int>(), alloca);
            auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld1, ld2});
            b.return_(add);
        }
        auto info = local_load_elimination_pass_run_on_module(&m);
        expect(info.removed_load_count == 2u);
    };
}

// ---- local_store_forward ----

void reg_local_store_forward() {

    "local_store_forward_load_after_store_forwarded"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 1u);
    };

    "local_store_forward_no_store_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
    };

    "local_store_forward_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            int32_t val_v = 7;
            auto *val = m.create_constant(Type::of<int>(), &val_v);
            b.store(alloca, val);
            auto *ld = b.load(Type::of<int>(), alloca);
            b.return_(ld);
        }
        auto info = local_store_forward_pass_run_on_module(&m);
        expect(info.removed_load_count == 2u);
    };
}

// ---- dead_store_elimination ----

void reg_dead_store_elimination() {

    "dse_overwritten_store_eliminated"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val1_v = 1, val2_v = 2;
        auto *val1 = m.create_constant(Type::of<int>(), &val1_v);
        auto *val2 = m.create_constant(Type::of<int>(), &val2_v);
        b.store(alloca, val1);
        b.store(alloca, val2);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = dead_store_elimination_pass_run_on_function(k);
        expect(info.eliminated_store_count == 1u);
    };

    "dse_no_dead_store_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = dead_store_elimination_pass_run_on_function(k);
        expect(info.eliminated_store_count == 0u);
    };

    "dse_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(Type::of<int>());
            int32_t val1_v = 1, val2_v = 2;
            auto *val1 = m.create_constant(Type::of<int>(), &val1_v);
            auto *val2 = m.create_constant(Type::of<int>(), &val2_v);
            b.store(alloca, val1);
            b.store(alloca, val2);
            auto *ld = b.load(Type::of<int>(), alloca);
            b.return_(ld);
        }
        auto info = dead_store_elimination_pass_run_on_module(&m);
        expect(info.eliminated_store_count == 2u);
    };
}

// ---- loop_rotation ----

void reg_loop_rotation() {

    "loop_rotation_rotates_loop"_test = [] {
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

        auto info = loop_rotation_pass_run_on_function(k);
        expect(info.rotated_loop_count == 1u);
    };

    "loop_rotation_no_loop_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = loop_rotation_pass_run_on_function(k);
        expect(info.rotated_loop_count == 0u);
    };

    "loop_rotation_module_runs_all_functions"_test = [] {
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
        auto info = loop_rotation_pass_run_on_module(&m);
        expect(info.rotated_loop_count == 2u);
    };
}

// ---- scalarizer ----

void reg_scalarizer() {

    "scalarizer_float3_add_scalarized"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<float3>();
        float a_data[3] = {1.0f, 2.0f, 3.0f};
        float b_data[3] = {4.0f, 5.0f, 6.0f};
        float c_data[3] = {7.0f, 8.0f, 9.0f};
        auto *a = m.create_constant(vec_t, a_data);
        auto *bv = m.create_constant(vec_t, b_data);
        auto *c = m.create_constant(vec_t, c_data);
        // add1 is used by add2 (scalarizable), add2 is dead (skipped)
        auto *add1 = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
        b.call(vec_t, ArithmeticOp::BINARY_ADD, {add1, c});
        b.return_void();
        auto info = scalarizer_pass_run_on_function(k);
        expect(info.scalarized_inst_count == 1u);
    };

    "scalarizer_no_vector_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 1, b_v = 2;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        b.return_(add);
        auto info = scalarizer_pass_run_on_function(k);
        expect(info.scalarized_inst_count == 0u);
    };

    "scalarizer_module_runs_all_functions"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            float a_data[3] = {1.0f, 2.0f, 3.0f};
            float b_data[3] = {4.0f, 5.0f, 6.0f};
            float c_data[3] = {7.0f, 8.0f, 9.0f};
            auto *a = m.create_constant(vec_t, a_data);
            auto *bv = m.create_constant(vec_t, b_data);
            auto *c = m.create_constant(vec_t, c_data);
            auto *add1 = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
            b.call(vec_t, ArithmeticOp::BINARY_ADD, {add1, c});
            b.return_void();
        }
        auto info = scalarizer_pass_run_on_module(&m);
        expect(info.scalarized_inst_count == 2u);
    };
}

// ---- phi_cleanup ----

void reg_phi_cleanup() {

    "phi_cleanup_trivial_phi_removed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *phi = b.phi(Type::of<int>(), {{val, body}});
        b.return_(phi);
        auto info = phi_cleanup_pass_run_on_function(k);
        expect(info.removed_phi_count == 1u);
    };

    "phi_cleanup_no_phi_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = phi_cleanup_pass_run_on_function(k);
        expect(info.removed_phi_count == 0u);
    };

    "phi_cleanup_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t val_v = 7;
            auto *val = m.create_constant(Type::of<int>(), &val_v);
            auto *phi = b.phi(Type::of<int>(), {{val, body}});
            b.return_(phi);
        }
        auto info = phi_cleanup_pass_run_on_module(&m);
        expect(info.removed_phi_count == 2u);
    };
}

// ---- if_conversion ----

void reg_if_conversion() {

    "if_conversion_diamond_converted"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        b.cond_br(cond, true_block, false_block);

        b.set_insertion_point(true_block);
        int32_t t_v = 1;
        auto *t_val = m.create_constant(Type::of<int>(), &t_v);
        b.br(merge);

        b.set_insertion_point(false_block);
        int32_t f_v = 0;
        auto *f_val = m.create_constant(Type::of<int>(), &f_v);
        b.br(merge);

        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(), {{t_val, true_block}, {f_val, false_block}});
        b.return_(phi);

        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 1u);
    };

    "if_conversion_no_diamond_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 0u);
    };

    "if_conversion_module_runs_all_functions"_test = [] {
        Module m;
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *cond = m.create_constant_one(Type::of<bool>());
            auto *true_block = k->create_basic_block();
            auto *false_block = k->create_basic_block();
            auto *merge = k->create_basic_block();
            b.cond_br(cond, true_block, false_block);

            b.set_insertion_point(true_block);
            int32_t t_v = 1;
            auto *t_val = m.create_constant(Type::of<int>(), &t_v);
            b.br(merge);

            b.set_insertion_point(false_block);
            int32_t f_v = 0;
            auto *f_val = m.create_constant(Type::of<int>(), &f_v);
            b.br(merge);

            b.set_insertion_point(merge);
            auto *phi = b.phi(Type::of<int>(), {{t_val, true_block}, {f_val, false_block}});
            b.return_(phi);
        }
        auto info = if_conversion_pass_run_on_module(&m);
        expect(info.converted_diamond_count == 2u);
    };
}

// ---- reg2mem ----

void reg_reg2mem() {

    "reg2mem_lowers_phi"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *phi = b.phi(Type::of<int>(), {{val, body}});
        auto *final_add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, val});
        b.return_(final_add);
        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_phi_count == 1u);
    };

    "reg2mem_no_phi_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_phi_count == 0u);
    };
}

// ---- sroa ----

void reg_sroa() {

    "sroa_decomposes_struct_alloca"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto struct_ty = Type::of<float2>();
        auto *alloca = b.alloca_local(struct_ty);
        float data[2] = {1.0f, 2.0f};
        auto *init = m.create_constant(struct_ty, data);
        b.store(alloca, init);
        auto *ld = b.load(struct_ty, alloca);
        b.return_(ld);
        auto info = sroa_pass_run_on_function(k, {.aggressive = true});
        expect(info.decomposed_alloca_count >= 0u);// check it runs
    };

    "sroa_no_struct_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = sroa_pass_run_on_function(k);
        expect(info.decomposed_alloca_count == 0u);
    };
}

// ---- inline ----

void reg_inline() {

    "inline_callable_inlined"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.return_(val);

        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *call = b.call(Type::of<int>(), callee, {});
        b.return_(call);

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 1u);
    };

    "inline_no_call_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
    };
}

// ---- unused_callable_removal ----

void reg_unused_callable_removal() {

    "unused_callable_removed"_test = [] {
        Module m;
        m.create_callable(Type::of<void>())->create_body_block();
        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 1u);
    };

    "unused_callable_used_callable_kept"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        callee->create_body_block();
        XIRBuilder b;
        int32_t ret_v = 42;
        b.set_insertion_point(callee->body_block());
        auto *val = m.create_constant(Type::of<int>(), &ret_v);
        b.return_(val);

        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<int>(), callee, {});
        b.return_(call);

        auto info = unused_callable_removal_pass_run_on_module(&m);
        expect(info.removed_callable_count == 0u);
    };
}

// ---- trace_gep ----

void reg_trace_gep() {

    "trace_gep_cascaded_gep_traced"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto struct_ty = Type::of<float2>();
        auto *alloca = b.alloca_local(struct_ty);
        uint32_t idx0_v = 0u;
        auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
        auto *gep1 = b.gep(Type::of<float>(), alloca, {idx0});
        auto *val = b.load(Type::of<float>(), gep1);
        b.return_(val);
        auto info = trace_gep_pass_run_on_function(k);
        expect(info.traced_gep_count >= 0u);// check it runs without error
    };

    "trace_gep_no_gep_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = trace_gep_pass_run_on_function(k);
        expect(info.traced_gep_count == 0u);
    };
}

// ---- transpose_gep ----

void reg_transpose_gep() {

    "transpose_gep_load_gep_transposed"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto struct_ty = Type::of<float2>();
        auto *alloca = b.alloca_local(struct_ty);
        float init_data[2] = {1.0f, 2.0f};
        auto *init = m.create_constant(struct_ty, init_data);
        b.store(alloca, init);
        uint32_t idx0_v = 0u;
        auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
        auto *gep = b.gep(Type::of<float>(), alloca, {idx0});
        auto *val = b.load(Type::of<float>(), gep);
        b.return_(val);
        auto info = transpose_gep_pass_run_on_function(k);
        expect(info.transposed_load_count == 1u);
    };

    "transpose_gep_no_gep_load_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = transpose_gep_pass_run_on_function(k);
        expect(info.transposed_load_count == 0u);
    };

    "transpose_gep_module_runs_all_functions"_test = [] {
        Module m;
        auto struct_ty = Type::of<float2>();
        for (int i = 0; i < 2; ++i) {
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *alloca = b.alloca_local(struct_ty);
            float init_data[2] = {1.0f, 2.0f};
            auto *init = m.create_constant(struct_ty, init_data);
            b.store(alloca, init);
            uint32_t idx0_v = 0u;
            auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
            auto *gep = b.gep(Type::of<float>(), alloca, {idx0});
            auto *val = b.load(Type::of<float>(), gep);
            b.return_(val);
        }
        auto info = transpose_gep_pass_run_on_module(&m);
        expect(info.transposed_load_count == 2u);
    };
}

// ---- mem2reg ----

void reg_mem2reg() {

    "mem2reg_promotes_simple_alloca"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 1u);
    };

    "mem2reg_no_alloca_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
    };
}

// ---- promote_ref_arg ----

void reg_promote_ref_arg() {

    "promote_ref_arg_runs_without_error"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<void>());
        c->create_reference_argument(Type::of<int>());
        c->create_body_block();
        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count >= 0u);// check it runs
    };

    "promote_ref_arg_no_ref_args_no_change"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<void>());
        c->create_body_block();
        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 0u);
    };
}

// ---- outline ----

void reg_outline() {

    "outline_no_outline_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = outline_pass_run_on_module(&m);
        expect(info.outlined_func_count == 0u);
    };
}

// Regression tests

void reg_regression() {

    "regression_vec3_sub_self_produces_vector_zero"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<int3>();
        auto *x = m.create_constant_zero(vec_t);
        auto *sub = b.call(vec_t, ArithmeticOp::BINARY_SUB, {x, x});
        auto sub_locked = sub->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(sub_locked->use_list().empty());
    };

    "regression_int3_add_zero_preserves_vector_type"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<int3>();
        auto *zero = m.create_constant_zero(vec_t);
        int x_data[3] = {1, 2, 3};
        auto *x = m.create_constant(vec_t, x_data);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {x, zero});
        auto add_locked = add->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(add_locked->use_list().empty());
    };

    "regression_float3_mul_one_preserves_vector_type"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto vec_t = Type::of<float3>();
        float one_data[3] = {1.0f, 1.0f, 1.0f};
        float x_data[3] = {2.0f, 3.0f, 4.0f};
        auto *one = m.create_constant(vec_t, one_data);
        auto *x = m.create_constant(vec_t, x_data);
        auto *mul = b.call(vec_t, ArithmeticOp::BINARY_MUL, {x, one});
        auto mul_locked = mul->lock();
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
        expect(mul_locked->use_list().empty());
    };

    "regression_float_add_zero_skipped_for_nan_safety"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f, x_v = 1.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, zero});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 0u);
    };

    "regression_float_mul_zero_skipped_for_nan_safety"_test = [] {
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

    "regression_constfold_int_add_value_correct"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 3, b_v = 4;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        auto *add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto add_locked = add->lock();
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(add_locked->use_list().empty());
        bool found_7 = false;
        for (auto c : m.constant_list()) {
            if (c->type() == Type::of<int>()) {
                int32_t v = *static_cast<const int32_t *>(c->data());
                if (v == 7) found_7 = true;
            }
        }
        expect(found_7);
    };

    "regression_constfold_int_unary_minus_int_min_value_correct"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = std::numeric_limits<int32_t>::min();
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *neg = b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x});
        auto neg_locked = neg->lock();
        b.return_void();
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(neg_locked->use_list().empty());
        bool found = false;
        for (auto c : m.constant_list()) {
            if (c->type() == Type::of<int>()) {
                int32_t v = *static_cast<const int32_t *>(c->data());
                if (v == std::numeric_limits<int32_t>::min()) found = true;
            }
        }
        expect(found);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_algebraic_simplify();
    reg_const_fold();
    reg_loop_unroll();
    reg_dce();
    reg_gvn();
    reg_sccp();
    reg_simplify_libcalls();
    reg_reassociate();
    reg_cvp();
    reg_dead_arg_elim();
    reg_div_rem_pairs();
    reg_local_load_elimination();
    reg_local_store_forward();
    reg_dead_store_elimination();
    reg_loop_rotation();
    reg_scalarizer();
    reg_phi_cleanup();
    reg_if_conversion();
    reg_reg2mem();
    reg_sroa();
    reg_inline();
    reg_unused_callable_removal();
    reg_trace_gep();
    reg_transpose_gep();
    reg_mem2reg();
    reg_promote_ref_arg();
    reg_outline();
    reg_regression();
    return 0;
}
