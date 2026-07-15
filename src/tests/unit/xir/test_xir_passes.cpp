#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/autodiff.h>
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
#include <luisa/xir/instructions/return.h>

#include <limits>

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

static size_t count_reachable_insts(FunctionDefinition *f, DerivedInstructionTag tag) noexcept {
    size_t count = 0u;
    f->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == tag) { count++; }
    });
    return count;
}

static size_t count_reachable_blocks(FunctionDefinition *f) noexcept {
    size_t count = 0u;
    f->traverse_basic_blocks([&](BasicBlock *) noexcept { count++; });
    return count;
}

static StoreInst *find_store_before(Instruction *before, Value *variable, Value *value) noexcept {
    auto *block = before == nullptr ? nullptr : before->parent_block();
    if (block == nullptr) { return nullptr; }
    for (auto *inst : block->instructions()) {
        if (inst == before) { break; }
        if (inst->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(inst);
            if ((variable == nullptr || store->variable() == variable) &&
                (value == nullptr || store->value() == value)) {
                return store;
            }
        }
    }
    return nullptr;
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

    "algsimpl_float_sub_self_requires_fast_math"_test = [] {
        {
            Module m;
            auto *f = m.create_callable(Type::of<float>());
            auto *x = f->create_value_argument(Type::of<float>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, x});
            auto *ret = b.return_(sub);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 0u);
            expect(ret->return_value() == sub);
        }
        {
            Module m;
            auto *f = m.create_callable(Type::of<float>());
            auto *x = f->create_value_argument(Type::of<float>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, x});
            auto *ret = b.return_(sub);
            auto info = algebraic_simplify_pass_run_on_function(f, {.enable_fast_math = true});
            expect(info.simplified_inst_count == 1u);
            expect(ret->return_value()->isa<Constant>());
            expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.0f);
        }
    };

    "algsimpl_float_vector_sub_self_requires_fast_math"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float2>());
        auto *x = f->create_value_argument(Type::of<float2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sub = b.call(Type::of<float2>(), ArithmeticOp::BINARY_SUB, {x, x});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
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
        auto *shift = b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_LEFT, {a, bv});
        auto *ret = b.return_(shift);
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == shift);
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
        auto *shift = b.call(Type::of<int>(), ArithmeticOp::BINARY_SHIFT_RIGHT, {a, bv});
        auto *ret = b.return_(shift);
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == shift);
    };

    "constfold_int_unary_minus"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = 5;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x}));
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == -x_v);
    };

    "constfold_int_unary_minus_int_min"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_v = std::numeric_limits<int32_t>::min();
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::UNARY_MINUS, {x}));
        auto info = const_fold_pass_run_on_function(k);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == x_v);
    };

    "constfold_signed_overflow_wraps_without_ub"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        auto add_case = [&](ArithmeticOp op, int32_t lhs, int32_t rhs) noexcept {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        };
        add_case(ArithmeticOp::BINARY_ADD, std::numeric_limits<int32_t>::max(), 1);
        add_case(ArithmeticOp::BINARY_SUB, std::numeric_limits<int32_t>::min(), 1);
        add_case(ArithmeticOp::BINARY_MUL, std::numeric_limits<int32_t>::max(), 2);
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 3u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::max());
        expect(static_cast<Constant *>(returns[2]->return_value())->as<int32_t>() == -2);
    };

    "constfold_int_min_div_mod_negative_one_not_folded"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_DIV, ArithmeticOp::BINARY_MOD}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = std::numeric_limits<int32_t>::min();
            int32_t rhs = -1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        expect(returns[0]->return_value()->isa<ArithmeticInst>());
        expect(returns[1]->return_value()->isa<ArithmeticInst>());
    };

    "constfold_signed_shifts_are_defined"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_SHIFT_LEFT, ArithmeticOp::BINARY_SHIFT_RIGHT}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = op == ArithmeticOp::BINARY_SHIFT_LEFT ? -1 : -4;
            int32_t rhs = 1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 2u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == -2);
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == -2);
    };

    "constfold_signed_shift_boundaries"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        auto add_case = [&](ArithmeticOp op, int32_t lhs, int32_t rhs) noexcept {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        };
        add_case(ArithmeticOp::BINARY_SHIFT_LEFT, 1, 31);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, std::numeric_limits<int32_t>::min(), 31);
        add_case(ArithmeticOp::BINARY_SHIFT_RIGHT, std::numeric_limits<int32_t>::min(), 0);
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 3u);
        expect(static_cast<Constant *>(returns[0]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
        expect(static_cast<Constant *>(returns[1]->return_value())->as<int32_t>() == -1);
        expect(static_cast<Constant *>(returns[2]->return_value())->as<int32_t>() == std::numeric_limits<int32_t>::min());
    };

    "constfold_negative_shift_counts_are_not_folded"_test = [] {
        Module m;
        luisa::vector<ReturnInst *> returns;
        for (auto op : {ArithmeticOp::BINARY_SHIFT_LEFT, ArithmeticOp::BINARY_SHIFT_RIGHT}) {
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            int32_t lhs = 1, rhs = -1;
            auto *a = m.create_constant(Type::of<int>(), &lhs);
            auto *bv = m.create_constant(Type::of<int>(), &rhs);
            returns.emplace_back(b.return_(b.call(Type::of<int>(), op, {a, bv})));
        }
        auto info = const_fold_pass_run_on_module(&m);
        expect(info.folded_inst_count == 0u);
        expect(returns[0]->return_value()->isa<ArithmeticInst>());
        expect(returns[1]->return_value()->isa<ArithmeticInst>());
    };

    "constfold_abs_int_min_wraps_without_ub"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t value = std::numeric_limits<int32_t>::min();
        auto *v = m.create_constant(Type::of<int>(), &value);
        auto *ret = b.return_(b.call(Type::of<int>(), ArithmeticOp::ABS, {v}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(static_cast<Constant *>(ret->return_value())->as<int32_t>() == value);
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

        LoopUnrollOptions options;
        options.max_trip_count = 16u;
        auto info = loop_unroll_pass_run_on_function(k, options);
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

    "dce_repairs_phi_after_unreachable_predecessor_removed"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *entry = f->create_body_block();
        auto *live = f->create_basic_block();
        auto *dead = f->create_basic_block();
        auto *merge = f->create_basic_block();

        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(live);

        b.set_insertion_point(live);
        int32_t live_v = 1;
        auto *live_c = m.create_constant(Type::of<int>(), &live_v);
        auto *live_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {live_c, live_c});
        b.br(merge);

        b.set_insertion_point(dead);
        int32_t dead_v = 2;
        auto *dead_c = m.create_constant(Type::of<int>(), &dead_v);
        auto *dead_value = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {dead_c, dead_c});
        b.br(merge);

        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(live_value, live);
        phi->add_incoming(dead_value, dead);
        b.return_(phi);

        auto info = dce_pass_run_on_function(f);
        expect(info.removed_block_count >= 1u);
        expect(phi->incoming_count() == 1u);
        expect(phi->incoming(0u).block == live);
        expect(phi->incoming(0u).value == live_value);
    };

    "dce_unreachable_block_cleanup_is_idempotent"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *dead = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(dead);
        b.unreachable_();
        auto first = dce_pass_run_on_function(k);
        auto second = dce_pass_run_on_function(k);
        expect(first.removed_inst_count == 0u);
        expect(second.removed_inst_count == 0u);
        expect(second.removed_block_count == 0u);
    };

    "dce_exec_reachability_preserves_structured_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *true_c = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(body);
        auto *if_inst = b.if_(true_c);
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        auto *if_merge = if_inst->create_merge_block();
        b.set_insertion_point(if_true);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        int32_t selector_v = 1;
        auto *selector = m.create_constant(Type::of<int>(), &selector_v);
        auto *sw = b.switch_(selector);
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        auto *sw_merge = sw->create_merge_block();
        b.set_insertion_point(sw_default);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(true_c, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 1u);
        expect(count_reachable_blocks(k) == 10u);
        expect(body->terminator()->isa<IfInst>());
        expect(if_true->terminator()->isa<BranchInst>());
        expect(if_false->terminator()->isa<UnreachableInst>());
        expect(if_merge->terminator()->isa<SwitchInst>());
        expect(sw_case->terminator()->isa<BranchInst>());
        expect(sw_default->terminator()->isa<UnreachableInst>());
        expect(sw_merge->terminator()->isa<LoopInst>());
        auto *result_loop = static_cast<LoopInst *>(sw_merge->terminator());
        expect(result_loop->prepare_block() == prepare);
        expect(result_loop->body_block() == loop_body);
        expect(result_loop->update_block() == update);
        expect(result_loop->merge_block() == nullptr);
    };

    "dce_constant_cond_br_becomes_taken_branch"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *taken = k->create_basic_block();
        auto *dead = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_constant_one(Type::of<bool>()), taken, dead);
        b.set_insertion_point(taken);
        b.return_void();
        b.set_insertion_point(dead);
        b.return_void();

        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 1u);
        expect(entry->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(entry->terminator())->target_block() == taken);
        expect(count_reachable_blocks(k) == 2u);
    };

    "dce_constant_if_preserves_taken_break_in_loop_scope"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        auto *if_inst = b.if_(m.create_constant_one(Type::of<bool>()));
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.break_(merge);
        b.set_insertion_point(if_false);
        b.continue_(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop_body->terminator()->isa<IfInst>());
        expect(if_true->terminator()->isa<BreakInst>());
        expect(static_cast<BreakInst *>(if_true->terminator())->target_block() == merge);
        expect(if_false->terminator()->isa<UnreachableInst>());
        expect(loop->body_block() == loop_body);
        expect(loop->update_block() == update);
        expect(update->terminator()->isa<UnreachableInst>());
        expect(loop->merge_block() == merge);
    };

    "dce_constant_switch_preserves_taken_continue_in_loop_scope"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        b.set_insertion_point(prepare);
        b.br(loop_body);
        b.set_insertion_point(loop_body);
        int32_t selector_value = 1;
        auto *selector = m.create_constant(Type::of<int>(), &selector_value);
        auto *switch_inst = b.switch_(selector);
        auto *switch_case = switch_inst->create_case_block(1);
        auto *switch_default = switch_inst->create_default_block();
        b.set_insertion_point(switch_case);
        b.continue_(update);
        b.set_insertion_point(switch_default);
        b.break_(merge);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop_body->terminator()->isa<SwitchInst>());
        expect(switch_case->terminator()->isa<ContinueInst>());
        expect(static_cast<ContinueInst *>(switch_case->terminator())->target_block() == update);
        expect(switch_default->terminator()->isa<UnreachableInst>());
        expect(loop->body_block() == loop_body);
        expect(loop->update_block() == update);
        expect(loop->merge_block() == nullptr);
    };

    "dce_loop_preserves_dead_body_and_update_shells"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(m.create_constant_zero(Type::of<bool>()), loop_body, merge);
        b.set_insertion_point(loop_body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(loop->prepare_block() == prepare);
        expect(prepare->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(prepare->terminator())->target_block() == merge);
        expect(loop->body_block() == loop_body);
        expect(loop_body->terminator()->isa<UnreachableInst>());
        expect(loop->update_block() == update);
        expect(update->terminator()->isa<UnreachableInst>());
        expect(loop->merge_block() == merge);
        expect(count_reachable_blocks(k) == 3u);
    };

    "dce_clears_unreachable_if_merge_but_keeps_executable_unreachable"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *condition = k->create_value_argument(Type::of<bool>());
        auto *entry = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *if_inst = b.if_(condition);
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(if_true);
        b.unreachable_("executable unreachable");
        b.set_insertion_point(if_false);
        b.return_void();
        b.set_insertion_point(merge);
        b.return_void();

        (void)dce_pass_run_on_function(k);
        expect(if_inst->merge_block() == nullptr);
        expect(if_true->terminator()->isa<UnreachableInst>());
        expect(static_cast<UnreachableInst *>(if_true->terminator())->message() == "executable unreachable");
        expect(count_reachable_blocks(k) == 3u);
    };

    "dce_keeps_reachable_self_loop_and_removes_disconnected_cycle"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *dead_a = k->create_basic_block();
        auto *dead_b = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(entry);
        b.set_insertion_point(dead_a);
        b.br(dead_b);
        b.set_insertion_point(dead_b);
        b.br(dead_a);

        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 2u);
        expect(entry->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(entry->terminator())->target_block() == entry);
        expect(count_reachable_blocks(k) == 1u);
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
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t}));
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
        expect(ret->return_value() == x);
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
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t}));
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
        expect(ret->return_value() == y);
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
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi}));
        auto info = simplify_libcalls_pass_run_on_function(k);
        expect(info.simplified_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::SATURATE);
        expect(static_cast<ArithmeticInst *>(ret->return_value())->operand(0) == x);
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

    "simplify_libcalls_step_zero_edge_keeps_sign_dependent_result"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<float>());
        auto *step = b.call(Type::of<float>(), ArithmeticOp::STEP, {zero, x});
        auto *ret = b.return_(step);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == step);
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
        auto *local = b.alloca_local(Type::of<int>());
        auto *loaded = b.load(Type::of<int>(), local);
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {loaded, val});
        auto *if_inst = b.if_(eq);
        auto *true_b = if_inst->create_true_block();
        auto *false_b = if_inst->create_false_block();
        auto *merge_b = if_inst->create_merge_block();
        b.set_insertion_point(true_b);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                           {loaded, m.create_constant_one(Type::of<int>())});
        b.br(merge_b);
        b.set_insertion_point(false_b);
        b.br(merge_b);
        b.set_insertion_point(merge_b);
        b.return_void();
        auto info = cvp_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(sum->operand(0) == val);
        expect(eq->operand(0) == loaded) << "the condition itself is not dominated by the taken block";
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

    "div_rem_pairs_mod_before_div_no_change"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t a_v = 10, b_v = 3;
        auto *a = m.create_constant(Type::of<int>(), &a_v);
        auto *bv = m.create_constant(Type::of<int>(), &b_v);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        b.return_void();
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 0u);
    };

    "div_rem_pairs_nested_remainders_preserve_current_operands"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dispatch_id = m.create_dispatch_id();
        uint32_t x_index_v = 0u;
        uint32_t outer_v = 64u;
        uint32_t inner_v = 8u;
        auto *x_index = m.create_constant(Type::of<uint32_t>(), &x_index_v);
        auto *outer = m.create_constant(Type::of<uint32_t>(), &outer_v);
        auto *inner = m.create_constant(Type::of<uint32_t>(), &inner_v);
        auto *x = b.call(Type::of<uint32_t>(), ArithmeticOp::EXTRACT, {dispatch_id, x_index});
        b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {x, outer});
        auto *rem = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {x, outer});
        b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {rem, inner});
        auto *nested_rem = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {rem, inner});
        b.return_(nested_rem);

        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 2u);
        auto mod_count = 0u;
        body->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_MOD) {
                mod_count++;
            }
        });
        expect(mod_count == 0u);
        auto *ret = static_cast<ReturnInst *>(body->terminator());
        auto *nested_sub = static_cast<ArithmeticInst *>(ret->return_value());
        expect(nested_sub->op() == ArithmeticOp::BINARY_SUB);
        auto *outer_sub = static_cast<ArithmeticInst *>(nested_sub->operand(0));
        auto *nested_mul = static_cast<ArithmeticInst *>(nested_sub->operand(1));
        auto *nested_div = static_cast<ArithmeticInst *>(nested_mul->operand(0));
        expect(outer_sub->op() == ArithmeticOp::BINARY_SUB);
        expect(nested_mul->op() == ArithmeticOp::BINARY_MUL);
        expect(nested_div->op() == ArithmeticOp::BINARY_DIV);
        expect(nested_div->operand(0) == outer_sub);
        expect(nested_mul->operand(1) == inner);
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

    "local_load_elim_does_not_forward_reference_loads"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *p0 = k->create_reference_argument(Type::of<int>());
        auto *p1 = k->create_reference_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ld0 = b.load(Type::of<int>(), p0);
        b.store(p1, m.create_constant_one(Type::of<int>()));
        auto *ld1 = b.load(Type::of<int>(), p0);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ld0, ld1});
        b.store(p1, sum);
        b.return_void();
        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(ld0->is_linked());
        expect(ld1->is_linked());
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

    "sroa_decomposes_vector_alloca"_test = [] {
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
        auto info = sroa_pass_run_on_function(k, {.decompose_vectors = true});
        expect(info.decomposed_alloca_count == 1u);
        expect(info.inserted_alloca_count == 2u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 2u);
        auto *ret = static_cast<ReturnInst *>(body->terminator());
        expect(ret->return_value() != ld);
        expect(ret->return_value()->type() == struct_ty);
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
        auto inner_ty = Type::array(Type::of<float>(), 2u);
        auto outer_ty = Type::array(inner_ty, 2u);
        auto *alloca = b.alloca_local(outer_ty);
        uint32_t idx0_v = 0u, idx1_v = 1u;
        auto *idx0 = m.create_constant(Type::of<uint>(), &idx0_v);
        auto *idx1 = m.create_constant(Type::of<uint>(), &idx1_v);
        auto *gep1 = b.gep(inner_ty, alloca, {idx0});
        auto *gep2 = b.gep(Type::of<float>(), gep1, {idx1});
        auto *val = b.load(Type::of<float>(), gep2);
        b.return_(val);
        auto info = trace_gep_pass_run_on_function(k);
        expect(info.traced_gep_count == 1u);
        expect(gep2->base() == alloca);
        expect(gep2->index_count() == 2u);
        expect(gep2->index(0) == idx0);
        expect(gep2->index(1) == idx1);
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

    "mem2reg_ignores_unreachable_alloca_users"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *dead = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        int32_t val_v = 42;
        auto *val = m.create_constant(Type::of<int>(), &val_v);
        b.store(alloca, val);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        b.set_insertion_point(dead);
        auto *dead_load = b.load(Type::of<int>(), alloca);
        b.store(alloca, dead_load);
        b.unreachable_();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 1u);
        expect(info.removed_load_count >= 2u);
        expect(info.removed_store_count >= 2u);
    };
}

// ---- promote_ref_arg ----

void reg_promote_ref_arg() {

    "promote_ref_arg_rewrites_signature_body_and_call_site"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *ref_arg = c->create_reference_argument(Type::of<int>());
        auto *callee_body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *callee_load = b.load(Type::of<int>(), ref_arg);
        b.return_(callee_load);

        BasicBlock *caller_body;
        auto *k = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_one(Type::of<int>()));
        auto *call = b.call(Type::of<int>(), c, {local});
        b.return_(call);

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 1u);
        expect(c->arguments().count_size() == 1u);
        expect(!c->arguments().front()->is_reference());
        expect(count_reachable_insts(c, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(c, DerivedInstructionTag::STORE) == 1u);
        expect(call->argument(0)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(call->argument(0))->variable() == local);
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

// ---- autodiff ----

void reg_autodiff() {

    "autodiff_options_run_both_modes_by_default"_test = [] {
        AutodiffOptions options;
        expect(options.run_forward);
        expect(options.run_backward);
    };

    "autodiff_run_forward_false_leaves_forward_scope_unlowered"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {x, idx});
        static_cast<void>(gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k, {.run_forward = false});
        expect(info.transformed_scope_count == 0u);
        expect(scope->is_forward());
        expect(scope->n_forward_grads() == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 2u);
    };

    "autodiff_run_backward_false_leaves_scope_unlowered"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k, {.run_backward = false});
        expect(info.transformed_scope_count == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 3u);
    };

    "autodiff_forward_propagates_scalar_duals"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dx_out = b.alloca_local(Type::of<float>());
        auto *dy_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, one_f, zero_f});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {y, zero_f, one_f});
        auto *xy = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {x, y});
        auto *sx = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xy, sx});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {z, idx0});
        auto *dy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {z, idx1});
        auto dx_lock = dx->lock();
        auto dy_lock = dy->lock();
        b.store(dx_out, dx);
        b.store(dy_out, dy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(dx_lock->use_list().empty());
        expect(dy_lock->use_list().empty());
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_forward_handles_binary_mod"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *dx_out = b.alloca_local(Type::of<float>());
        auto *dy_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, one_f, zero_f});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {y, zero_f, one_f});
        auto *r = b.call(Type::of<float>(), ArithmeticOp::BINARY_MOD, {x, y});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {r, idx0});
        auto *dy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {r, idx1});
        b.store(dx_out, dx);
        b.store(dy_out, dy);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t mod_count = 0u;
        size_t trunc_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::BINARY_MOD) { mod_count++; }
                if (op == ArithmeticOp::TRUNC) { trunc_count++; }
            }
        });
        expect(mod_count >= 1u);
        expect(trunc_count >= 1u);
    };

    "autodiff_forward_propagates_mutable_cfg_state"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *flag = k->create_argument(Type::of<bool>(), false);
        auto *tag = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        auto *y = b.alloca_local(Type::of<float>());
        b.store(y, x);
        auto *if_inst = b.if_(flag);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {t0, t0});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {f0});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        auto *sw = b.switch_(tag);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::COS, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *out = b.load(Type::of<float>(), y);
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *gout = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {out, idx});
        b.store(grad_out, gout);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) == 1u);
        size_t sin_count = 0u;
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::SIN) { sin_count++; }
                if (op == ArithmeticOp::COS) { cos_count++; }
            }
        });
        expect(sin_count >= 2u);
        expect(cos_count >= 2u);
    };

    "autodiff_forward_propagates_structured_loop_state"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {x, m.create_constant_one(Type::of<float>())});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {s, x});
        b.store(y, sum);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        uint32_t grad_index = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &grad_index);
        auto *gout = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {out, idx});
        b.store(grad_out, gout);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 1u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_forward_handles_matrix_linalg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto vector_type = Type::of<float2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *vec = k->create_argument(vector_type, false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *vector_grad_out = b.alloca_local(vector_type);
        auto *scalar_grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_m = m.create_constant_zero(matrix_type);
        auto *one_m = m.create_constant_one(matrix_type);
        auto *zero_v = m.create_constant_zero(vector_type);
        auto *one_v = m.create_constant_one(vector_type);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {mat, one_m, zero_m});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {vec, zero_v, one_v});
        auto *mv = b.call(vector_type, ArithmeticOp::MATRIX_LINALG_MUL, {mat, vec});
        auto *outer = b.call(matrix_type, ArithmeticOp::OUTER_PRODUCT, {mv, vec});
        auto *inv = b.call(matrix_type, ArithmeticOp::MATRIX_INVERSE, {mat});
        auto *det = b.call(Type::of<float>(), ArithmeticOp::MATRIX_DETERMINANT, {mat});
        auto *combined = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {outer, inv});
        uint32_t zero = 0u;
        uint32_t one = 1u;
        auto *idx0 = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *idx1 = m.create_constant(Type::of<uint32_t>(), &one);
        auto *dm = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {combined, idx0});
        auto *dv = b.call(vector_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {mv, idx1});
        auto *dd = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {det, idx0});
        b.store(matrix_grad_out, dm);
        b.store(vector_grad_out, dv);
        b.store(scalar_grad_out, dd);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t matmul_count = 0u;
        size_t determinant_count = 0u;
        size_t inverse_count = 0u;
        size_t outer_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                switch (static_cast<ArithmeticInst *>(inst)->op()) {
                    case ArithmeticOp::MATRIX_LINALG_MUL: matmul_count++; break;
                    case ArithmeticOp::MATRIX_DETERMINANT: determinant_count++; break;
                    case ArithmeticOp::MATRIX_INVERSE: inverse_count++; break;
                    case ArithmeticOp::OUTER_PRODUCT: outer_count++; break;
                    default: break;
                }
            }
        });
        expect(matmul_count >= 5u);
        expect(determinant_count >= 1u);
        expect(inverse_count >= 2u);
        expect(outer_count >= 3u);
    };

    "autodiff_forward_handles_matrix_scalar_components"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *scope = b.forward_autodiff_scope(2u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        auto *zero_m = m.create_constant_zero(matrix_type);
        auto *one_m = m.create_constant_one(matrix_type);
        auto *zero_f = m.create_constant_zero(Type::of<float>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {mat, one_m, zero_m});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {scalar, zero_f, one_f});
        auto *mul = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_MUL, {mat, scalar});
        auto *div = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_DIV, {scalar, mat});
        auto *sum = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {mul, div});
        uint32_t one = 1u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &one);
        auto *ds = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {sum, idx});
        b.store(matrix_grad_out, ds);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t aggregate_count = 0u;
        size_t matrix_div_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto op = static_cast<ArithmeticInst *>(inst)->op();
                if (op == ArithmeticOp::AGGREGATE) { aggregate_count++; }
                if (op == ArithmeticOp::MATRIX_COMP_DIV) { matrix_div_count++; }
            }
        });
        expect(aggregate_count >= 1u);
        expect(matrix_div_count >= 2u);
    };

    "autodiff_forward_projects_static_cast_aggregate_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto double_vector_type = Type::of<double2>();
        auto matrix_type = Type::of<float2x2>();
        auto *v = k->create_argument(vector_type, false);
        auto *s = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *vector_grad_out = b.alloca_local(double_vector_type);
        auto *matrix_grad_out = b.alloca_local(matrix_type);
        auto *scope = b.forward_autodiff_scope(1u);
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {v, m.create_constant_one(vector_type)});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {s, m.create_constant_one(Type::of<float>())});
        auto *vd = b.cast_(double_vector_type, CastOp::STATIC_CAST, v);
        auto *sm = b.call(matrix_type, ArithmeticOp::AGGREGATE, {b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s}),
                                                                 b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s})});
        uint32_t zero = 0u;
        auto *idx = m.create_constant(Type::of<uint32_t>(), &zero);
        auto *dvd = b.call(double_vector_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {vd, idx});
        auto *dsm = b.call(matrix_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {sm, idx});
        b.store(vector_grad_out, dvd);
        b.store(matrix_grad_out, dsm);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t float_to_double_scalar_cast_count = 0u;
        size_t aggregate_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == Type::of<double>() &&
                    cast->value()->type() == Type::of<float>()) {
                    float_to_double_scalar_cast_count++;
                }
            } else if (inst->isa<ArithmeticInst>() &&
                       static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::AGGREGATE) {
                aggregate_count++;
            }
        });
        expect(float_to_double_scalar_cast_count >= 2u);
        expect(aggregate_count >= 4u);
    };

    "autodiff_reverse_projects_matrix_scalar_component_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto matrix_type = Type::of<float2x2>();
        auto *mat = k->create_argument(matrix_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {mat});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {scalar});
        auto *prod = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_MUL, {mat, scalar});
        auto *quot = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_DIV, {scalar, mat});
        auto *sum = b.call(matrix_type, ArithmeticOp::MATRIX_COMP_ADD, {prod, quot});
        auto *col0 = b.call(Type::of<float2>(), ArithmeticOp::EXTRACT, {sum, m.create_constant_zero(Type::of<uint32_t>())});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {col0});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {y, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gs = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {scalar});
        b.store(grad_out, gs);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t reduce_sum_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::REDUCE_SUM) {
                reduce_sum_count++;
            }
        });
        expect(reduce_sum_count >= 2u);
    };

    "autodiff_reverse_propagates_static_cast_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *xd = static_cast<Value *>(b.static_cast_(Type::of<double>(), x));
        auto *yd = b.call(Type::of<double>(), ArithmeticOp::BINARY_MUL, {xd, xd});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {yd, m.create_constant_one(Type::of<double>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        bool has_forward_cast = false;
        bool has_backward_cast = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == Type::of<double>() &&
                    cast->value()->type() == Type::of<float>()) {
                    has_forward_cast = true;
                }
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->value()->type() == Type::of<double>()) {
                    has_backward_cast = true;
                }
            }
        });
        expect(has_forward_cast);
        expect(has_backward_cast);
    };

    "autodiff_reverse_projects_vector_static_cast_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto double_vector_type = Type::of<double2>();
        auto *x = k->create_argument(vector_type, false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(vector_type);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *xd = b.cast_(double_vector_type, CastOp::STATIC_CAST, x);
        auto *yd = b.call(Type::of<double>(), ArithmeticOp::REDUCE_SUM, {xd});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {yd, m.create_constant_one(Type::of<double>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(vector_type, AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t forward_vector_cast_count = 0u;
        size_t backward_scalar_cast_count = 0u;
        size_t insert_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == double_vector_type &&
                    cast->value()->type() == vector_type) {
                    forward_vector_cast_count++;
                }
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->value()->type() == Type::of<double>()) {
                    backward_scalar_cast_count++;
                }
            } else if (inst->isa<ArithmeticInst>() &&
                       static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::INSERT) {
                insert_count++;
            }
        });
        expect(forward_vector_cast_count == 1u);
        expect(backward_scalar_cast_count >= 2u);
        expect(insert_count >= 2u);
    };

    "autodiff_reverse_insert_zeroes_overwritten_base_gradient"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float2>();
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *y = k->create_argument(Type::of<float>(), false);
        auto *z = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *gx_out = b.alloca_local(Type::of<float>());
        auto *gy_out = b.alloca_local(Type::of<float>());
        auto *gz_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {y});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {z});
        auto *base = b.call(vector_type, ArithmeticOp::AGGREGATE, {x, y});
        auto *updated = b.call(vector_type, ArithmeticOp::INSERT, {base, z, m.create_constant_zero(Type::of<uint32_t>())});
        auto *loss = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {updated});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {loss, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        auto *gy = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {y});
        auto *gz = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {z});
        b.store(gx_out, gx);
        b.store(gy_out, gy);
        b.store(gz_out, gz);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        bool zeroes_overwritten_slot = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arith = static_cast<ArithmeticInst *>(inst);
                if (arith->op() == ArithmeticOp::INSERT &&
                    arith->type() == vector_type &&
                    arith->operand_count() == 3u &&
                    !arith->operand(0)->isa<Constant>() &&
                    arith->operand(1)->isa<Constant>() &&
                    static_cast<Constant *>(arith->operand(1))->type() == Type::of<float>() &&
                    static_cast<Constant *>(arith->operand(1))->as<float>() == 0.0f) {
                    zeroes_overwritten_slot = true;
                }
            }
        });
        expect(zeroes_overwritten_slot);
    };

    "autodiff_snapshots_mutable_cfg_selectors"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *flag_arg = k->create_argument(Type::of<bool>(), false);
        auto *tag_arg = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *flag = b.alloca_local(Type::of<bool>());
        auto *tag = b.alloca_local(Type::of<int>());
        b.store(y, x);
        b.store(flag, flag_arg);
        b.store(tag, tag_arg);
        auto *cond = b.load(Type::of<bool>(), flag);
        auto *forward_if = b.if_(cond);
        auto *if_merge = forward_if->create_merge_block();
        auto *if_true = forward_if->create_true_block();
        auto *if_false = forward_if->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {t0, t0});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {f0});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.store(flag, m.create_constant_zero(Type::of<bool>()));
        auto *selector = b.load(Type::of<int>(), tag);
        auto *forward_switch = b.switch_(selector);
        auto *sw_merge = forward_switch->create_merge_block();
        auto *sw_default = forward_switch->create_default_block();
        auto *sw_case = forward_switch->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::COS, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        int32_t zero = 0;
        b.store(tag, m.create_constant(Type::of<int>(), &zero));
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        IfInst *backward_if = nullptr;
        SwitchInst *backward_switch = nullptr;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<IfInst>() && inst != forward_if) {
                backward_if = static_cast<IfInst *>(inst);
            } else if (inst->isa<SwitchInst>() && inst != forward_switch) {
                backward_switch = static_cast<SwitchInst *>(inst);
            }
        });
        expect(backward_if != nullptr);
        expect(backward_switch != nullptr);
        auto *if_snapshot_store = find_store_before(forward_if, nullptr, cond);
        expect(if_snapshot_store != nullptr);
        auto *backward_if_condition = backward_if == nullptr ? nullptr : backward_if->condition();
        expect(backward_if_condition != nullptr && backward_if_condition->isa<LoadInst>());
        auto *backward_if_load = backward_if_condition != nullptr && backward_if_condition->isa<LoadInst>() ?
                                     static_cast<LoadInst *>(backward_if_condition) :
                                     nullptr;
        expect(backward_if_load != nullptr && if_snapshot_store != nullptr &&
               backward_if_load->variable() == if_snapshot_store->variable());
        expect(backward_if_load != nullptr && backward_if_load->variable() != flag);
        auto *switch_snapshot_store = find_store_before(forward_switch, nullptr, selector);
        expect(switch_snapshot_store != nullptr);
        auto *backward_switch_value = backward_switch == nullptr ? nullptr : backward_switch->value();
        expect(backward_switch_value != nullptr && backward_switch_value->isa<LoadInst>());
        auto *backward_switch_load = backward_switch_value != nullptr && backward_switch_value->isa<LoadInst>() ?
                                         static_cast<LoadInst *>(backward_switch_value) :
                                         nullptr;
        expect(backward_switch_load != nullptr && switch_snapshot_store != nullptr &&
               backward_switch_load->variable() == switch_snapshot_store->variable());
        expect(backward_switch_load != nullptr && backward_switch_load->variable() != tag);
    };

    "autodiff_preserves_native_switch_in_backward"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *tag = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        b.store(y, x);
        auto *sw = b.switch_(tag);
        auto *sw_merge = sw->create_merge_block();
        auto *default_block = sw->create_default_block();
        auto *case_block = sw->create_case_block(7);
        b.set_insertion_point(default_block);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(case_block);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {c0, c0});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) == 2u);
    };

    "autodiff_handles_native_pow_int"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        int32_t three = 3;
        auto *exp = m.create_constant(Type::of<int>(), &three);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *pow = b.call(Type::of<float>(), ArithmeticOp::POW_INT, {x, exp});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {pow, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t pow_int_count = 0u;
        bool has_exponent_cast = false;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::POW_INT) {
                pow_int_count++;
            } else if (inst->isa<CastInst>()) {
                auto *cast = static_cast<CastInst *>(inst);
                if (cast->op() == CastOp::STATIC_CAST &&
                    cast->type() == Type::of<float>() &&
                    cast->operand(0)->type() == Type::of<int>()) {
                    has_exponent_cast = true;
                }
            }
        });
        expect(pow_int_count == 2u);
        expect(has_exponent_cast);
    };

    "autodiff_handles_native_smoothstep"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *edge0 = k->create_argument(Type::of<float>(), false);
        auto *edge1 = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {edge0});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {edge1});
        auto *smooth = b.call(Type::of<float>(), ArithmeticOp::SMOOTHSTEP, {edge0, edge1, x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {smooth, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t smoothstep_count = 0u;
        size_t saturate_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arith = static_cast<ArithmeticInst *>(inst);
                if (arith->op() == ArithmeticOp::SMOOTHSTEP) {
                    smoothstep_count++;
                } else if (arith->op() == ArithmeticOp::SATURATE) {
                    saturate_count++;
                }
            }
        });
        expect(smoothstep_count == 1u);
        expect(saturate_count >= 1u);
    };

    "autodiff_accumulate_gradient_marks_reverse_root"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {x});
        auto *loss = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {s, s});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT, {loss, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_reverse_bounded_dynamic_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *n = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, n});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {s, x});
        b.store(y, sum);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 2u);
        size_t cos_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::COS) {
                cos_count++;
            }
        });
        expect(cos_count >= 1u);
    };

    "autodiff_reverse_bounded_dynamic_loop_with_nested_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *n = k->create_argument(Type::of<int>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        int32_t zero = 0;
        int32_t one = 1;
        int32_t two = 2;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *two_c = m.create_constant(Type::of<int>(), &two);
        b.store(y, x);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, n});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *iv_body = b.load(Type::of<int>(), i);
        auto *parity = b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {iv_body, two_c});
        auto *branch_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {parity, zero_c});
        auto *if_inst = b.if_(branch_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *yt0 = b.load(Type::of<float>(), y);
        auto *yt1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {yt0});
        auto *yt2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yt1, x});
        b.store(y, yt2);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::COS, {yf0});
        auto *yf2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {yf1, x});
        b.store(y, yf2);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *gx = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {x});
        b.store(grad_out, gx);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        bool all_terminated = true;
        k->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            all_terminated &= block->is_terminated();
        });
        expect(all_terminated);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 2u);
    };

    "autodiff_inlines_callable_before_reverse_pass"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<float>());
        auto *callee_arg = callee->create_argument(Type::of<float>(), false);
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *sin_x = b.call(Type::of<float>(), ArithmeticOp::SIN, {callee_arg});
        auto *mul = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {sin_x, callee_arg});
        b.return_(mul);
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *call = b.call(Type::of<float>(), callee, {x});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {call, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(count_reachable_insts(k, DerivedInstructionTag::CALL) == 1u);
        auto inline_info = inline_all_pass_run_on_module(&m);
        expect(inline_info.inlined_call_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CALL) == 0u);
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_before_reverse_pass"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *four_c = m.create_constant(Type::of<int>(), &four);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, four_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        b.store(y, s);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_explicit_step_xor_condition"_test = [] {
        auto run_case = [](int32_t start_v, int32_t bound_v, int32_t step_v) {
            Module m;
            BasicBlock *body;
            auto *k = make_kernel_with_body(m, body);
            auto *x = k->create_argument(Type::of<float>(), false);
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *scope = b.autodiff_scope();
            auto *merge = scope->create_merge_block();
            auto *entry = scope->create_entry_block();
            b.set_insertion_point(entry);
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
            auto *y = b.alloca_local(Type::of<float>());
            auto *i = b.alloca_local(Type::of<int>());
            auto *step = b.alloca_local(Type::of<int>());
            int32_t zero_v = 0;
            auto *start = m.create_constant(Type::of<int>(), &start_v);
            auto *bound = m.create_constant(Type::of<int>(), &bound_v);
            auto *step_c = m.create_constant(Type::of<int>(), &step_v);
            auto *zero = m.create_constant(Type::of<int>(), &zero_v);
            b.store(y, x);
            b.store(i, start);
            b.store(step, step_c);
            auto *loop = b.loop();
            auto *prepare = loop->create_prepare_block();
            auto *loop_body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *loop_merge = loop->create_merge_block();
            b.set_insertion_point(prepare);
            auto *iv = b.load(Type::of<int>(), i);
            auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound});
            auto *cond_step = b.load(Type::of<int>(), step);
            auto *neg_step = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {cond_step, zero});
            auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_BIT_XOR, {cmp, neg_step});
            b.cond_br(cond, loop_body, loop_merge);
            b.set_insertion_point(loop_body);
            auto *yv = b.load(Type::of<float>(), y);
            auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yv, x});
            auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {sum});
            b.store(y, s);
            b.br(update);
            b.set_insertion_point(update);
            auto *iv_next_base = b.load(Type::of<int>(), i);
            auto *step_update = b.load(Type::of<int>(), step);
            auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, step_update});
            b.store(i, iv_next);
            b.br(prepare);
            b.set_insertion_point(loop_merge);
            auto *out = b.load(Type::of<float>(), y);
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
            b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();
            auto info = autodiff_pass_run_on_function(k);
            expect(info.transformed_scope_count == 1u);
            expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
            expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
            expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        };
        run_case(0, 6, 2);
        run_case(3, 0, -1);
    };

    "autodiff_unrolls_fixed_trip_loop_with_update_state_store"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        float scale_v = 0.5f;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *scale = m.create_constant(Type::of<float>(), &scale_v);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv = b.load(Type::of<float>(), y);
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {yv});
        b.store(y, s);
        b.br(update);
        b.set_insertion_point(update);
        auto *yu = b.load(Type::of<float>(), y);
        auto *y_next = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yu, scale});
        b.store(y, y_next);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_nested_cfg_before_dce"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        float half = 0.5f;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *half_c = m.create_constant(Type::of<float>(), &half);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, three_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(1);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {d0});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {c0, x});
        b.store(y, c1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *if_y = b.load(Type::of<float>(), y);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {if_y, half_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *t0 = b.load(Type::of<float>(), y);
        auto *t1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {t0, x});
        b.store(y, t1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *f0 = b.load(Type::of<float>(), y);
        auto *f1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {f0, x});
        b.store(y, f1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        auto scalarizer_info = scalarizer_pass_run_on_function(k);
        static_cast<void>(scalarizer_info);
        auto sroa_info = sroa_pass_run_on_function(k);
        static_cast<void>(sroa_info);
        auto dce_info = dce_pass_run_on_function(k);
        static_cast<void>(dce_info);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) >= 6u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 6u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_vector_state_before_dce"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        auto *float2_t = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *p = b.alloca_local(float2_t);
        auto *v = b.alloca_local(float2_t);
        auto *i = b.alloca_local(Type::of<int>());
        uint32_t ix = 0u;
        uint32_t iy = 1u;
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        float c025 = 0.25f;
        float c05 = 0.5f;
        auto *ix_c = m.create_constant(Type::of<uint32_t>(), &ix);
        auto *iy_c = m.create_constant(Type::of<uint32_t>(), &iy);
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *four_c = m.create_constant(Type::of<int>(), &four);
        auto *c025_c = m.create_constant(Type::of<float>(), &c025);
        auto *c05_c = m.create_constant(Type::of<float>(), &c05);
        auto *x2 = b.call(float2_t, ArithmeticOp::AGGREGATE, {x, x});
        auto *v0 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {x2, c025_c});
        b.store(p, x2);
        b.store(v, v0);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, four_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_case = sw->create_case_block(2);
        b.set_insertion_point(sw_default);
        auto *pd0 = b.load(float2_t, p);
        auto *vd0 = b.load(float2_t, v);
        auto *pd1 = b.call(float2_t, ArithmeticOp::BINARY_ADD, {pd0, vd0});
        b.store(p, pd1);
        b.br(sw_merge);
        b.set_insertion_point(sw_case);
        auto *pc0 = b.load(float2_t, p);
        auto *vc0 = b.load(float2_t, v);
        auto *vc1 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {vc0, pc0});
        b.store(v, vc1);
        b.br(sw_merge);
        b.set_insertion_point(sw_merge);
        auto *pl = b.load(float2_t, p);
        auto *px = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {pl, ix_c});
        auto *py = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {pl, iy_c});
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {px, py});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        auto *vt0 = b.load(float2_t, v);
        auto *vt1 = b.call(float2_t, ArithmeticOp::BINARY_MUL, {vt0, c05_c});
        b.store(v, vt1);
        b.br(if_merge);
        b.set_insertion_point(if_false);
        auto *pf0 = b.load(float2_t, p);
        auto *pf1 = b.call(float2_t, ArithmeticOp::BINARY_ADD, {pf0, x2});
        b.store(p, pf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out_p = b.load(float2_t, p);
        auto *out_v = b.load(float2_t, v);
        auto *out_px = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_p, ix_c});
        auto *out_py = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_p, iy_c});
        auto *out_vx = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {out_v, ix_c});
        auto *sum0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {out_px, out_py});
        auto *out = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {sum0, out_vx});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        auto scalarizer_info = scalarizer_pass_run_on_function(k);
        static_cast<void>(scalarizer_info);
        auto sroa_info = sroa_pass_run_on_function(k);
        static_cast<void>(sroa_info);
        auto dce_info = dce_pass_run_on_function(k);
        static_cast<void>(dce_info);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::SWITCH) >= 8u);
        expect(count_reachable_insts(k, DerivedInstructionTag::IF) >= 8u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_continue_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t two = 2;
        int32_t five = 5;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *two_c = m.create_constant(Type::of<int>(), &two);
        auto *five_c = m.create_constant(Type::of<int>(), &five);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, five_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *if_iv = b.load(Type::of<int>(), i);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {if_iv, two_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.continue_(update);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {yf0});
        b.store(y, yf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        auto *ym0 = b.load(Type::of<float>(), y);
        auto *ym1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {ym0, x});
        b.store(y, ym1);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_break_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t three = 3;
        int32_t six = 6;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *three_c = m.create_constant(Type::of<int>(), &three);
        auto *six_c = m.create_constant(Type::of<int>(), &six);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, six_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *yv0 = b.load(Type::of<float>(), y);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {yv0, x});
        auto *s = b.call(Type::of<float>(), ArithmeticOp::SIN, {sum});
        b.store(y, s);
        auto *if_iv = b.load(Type::of<int>(), i);
        auto *if_cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {if_iv, three_c});
        auto *if_inst = b.if_(if_cond);
        auto *if_merge = if_inst->create_merge_block();
        auto *if_true = if_inst->create_true_block();
        auto *if_false = if_inst->create_false_block();
        b.set_insertion_point(if_true);
        b.break_(loop_merge);
        b.set_insertion_point(if_false);
        auto *yf0 = b.load(Type::of<float>(), y);
        auto *yf1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {yf0, x});
        b.store(y, yf1);
        b.br(if_merge);
        b.set_insertion_point(if_merge);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
    };

    "autodiff_unrolls_fixed_trip_loop_with_switch_early_exit_cfg"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *x = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {x});
        auto *y = b.alloca_local(Type::of<float>());
        auto *i = b.alloca_local(Type::of<int>());
        b.store(y, x);
        int32_t zero = 0;
        int32_t one = 1;
        int32_t four = 4;
        int32_t six = 6;
        auto *zero_c = m.create_constant(Type::of<int>(), &zero);
        auto *one_c = m.create_constant(Type::of<int>(), &one);
        auto *six_c = m.create_constant(Type::of<int>(), &six);
        b.store(i, zero_c);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *iv = b.load(Type::of<int>(), i);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, six_c});
        b.cond_br(cond, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_iv = b.load(Type::of<int>(), i);
        auto *sw = b.switch_(switch_iv);
        auto *sw_merge = sw->create_merge_block();
        auto *sw_default = sw->create_default_block();
        auto *sw_continue = sw->create_case_block(1);
        auto *sw_break = sw->create_case_block(4);
        b.set_insertion_point(sw_default);
        auto *d0 = b.load(Type::of<float>(), y);
        auto *d1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {d0, x});
        b.store(y, d1);
        b.br(sw_merge);
        b.set_insertion_point(sw_continue);
        auto *c0 = b.load(Type::of<float>(), y);
        auto *c1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {c0, x});
        b.store(y, c1);
        b.continue_(update);
        b.set_insertion_point(sw_break);
        auto *b0 = b.load(Type::of<float>(), y);
        auto *b1 = b.call(Type::of<float>(), ArithmeticOp::SIN, {b0});
        b.store(y, b1);
        b.break_(loop_merge);
        b.set_insertion_point(sw_merge);
        auto *m0 = b.load(Type::of<float>(), y);
        auto *m1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {m0, x});
        b.store(y, m1);
        b.br(update);
        b.set_insertion_point(update);
        auto *iv_next_base = b.load(Type::of<int>(), i);
        auto *iv_next = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv_next_base, one_c});
        b.store(i, iv_next);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *out = b.load(Type::of<float>(), y);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {out, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = autodiff_pass_run_on_function(k);
        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_SCOPE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::AUTODIFF_INTRINSIC) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::BREAK) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::CONTINUE) == 0u);
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
    reg_autodiff();
    reg_regression();
    return 0;
}
