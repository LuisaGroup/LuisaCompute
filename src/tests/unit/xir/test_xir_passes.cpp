// Test for XIR scalar, memory, CFG, and interprocedural transformation passes.
// This test covers successful rewrites, conservative no-op cases, malformed-input
// rejection, and verifier-preserving behavior across the shared pass pipeline.

#include "ut/ut.hpp"
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/destructure_cfg.h>
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
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/verifier.h>
#include <luisa/core/stl/unordered_map.h>

#include <array>
#include <cfenv>
#include <cmath>
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

static bool block_local_defs_precede_uses(BasicBlock *block) noexcept {
    luisa::unordered_set<const Instruction *> seen;
    for (auto *inst : block->instructions()) {
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            auto *operand = inst->operand(i);
            if (operand->isa<Instruction>()) {
                auto *operand_inst = static_cast<const Instruction *>(operand);
                if (operand_inst->parent_block() == block &&
                    seen.find(operand_inst) == seen.end()) {
                    return false;
                }
            }
        }
        seen.emplace(inst);
    }
    return true;
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

    "algsimpl_float_sub_positive_zero_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_v = 0.0f;
        auto *zero = m.create_constant(Type::of<float>(), &zero_v);
        auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == x);
    };

    "algsimpl_float_sub_negative_zero_not_simplified"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float negative_zero_v = -0.0f;
        auto *negative_zero = m.create_constant(Type::of<float>(), &negative_zero_v);
        auto *sub = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {x, negative_zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
    };

    "algsimpl_float_vector_sub_signed_zero_distinguished"_test = [] {
        Module m;
        auto type = Type::of<float2>();
        auto *f = m.create_callable(type);
        auto *x = f->create_value_argument(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float negative_zero_data[2] = {-0.0f, 0.0f};
        auto *negative_zero = m.create_constant(type, negative_zero_data);
        auto *sub = b.call(type, ArithmeticOp::BINARY_SUB, {x, negative_zero});
        auto *ret = b.return_(sub);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == sub);
    };

    "algsimpl_float_vector_unary_minus_zero_not_simplified"_test = [] {
        Module m;
        auto type = Type::of<float2>();
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_data[2] = {0.0f, -0.0f};
        auto *zero = m.create_constant(type, zero_data);
        auto *neg = b.call(type, ArithmeticOp::UNARY_MINUS, {zero});
        auto *ret = b.return_(neg);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == neg);
    };

    "algsimpl_float_matrix_unary_minus_zero_not_simplified"_test = [] {
        Module m;
        auto type = Type::of<float2x2>();
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float zero_data[4] = {0.0f, -0.0f, 0.0f, 0.0f};
        auto *zero = m.create_constant(type, zero_data);
        auto *neg = b.call(type, ArithmeticOp::UNARY_MINUS, {zero});
        auto *ret = b.return_(neg);

        auto info = algebraic_simplify_pass_run_on_function(f);

        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == neg);
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
        uint8_t index_v = 1u;
        auto *x = m.create_constant(Type::of<int>(), &x_v);
        auto *y = m.create_constant(Type::of<int>(), &y_v);
        auto *z = m.create_constant(Type::of<int>(), &z_v);
        auto *index = m.create_constant(Type::of<uint8_t>(), &index_v);
        auto *aggregate = b.call(type, ArithmeticOp::AGGREGATE, {x, y});
        b.call(type, ArithmeticOp::INSERT, {aggregate, z, index});
        b.return_void();
        auto info = algebraic_simplify_pass_run_on_function(k);
        expect(info.simplified_inst_count == 1u);
    };

    "algsimpl_extract_accepts_all_integer_constant_widths"_test = [] {
        auto run = [](const Type *index_type, const void *index_data) {
            Module m;
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *vector_type = Type::vector(Type::of<int>(), 2u);
            int32_t x_value = 11;
            int32_t y_value = 17;
            auto *x = m.create_constant(Type::of<int>(), &x_value);
            auto *y = m.create_constant(Type::of<int>(), &y_value);
            auto *index = m.create_constant(index_type, index_data);
            auto *aggregate = b.call(vector_type, ArithmeticOp::AGGREGATE, {x, y});
            auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {aggregate, index});
            auto *ret = b.return_(extract);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 1u);
            expect(ret->return_value() == y);
        };
        int8_t i8 = 1;
        uint8_t u8 = 1u;
        int16_t i16 = 1;
        uint16_t u16 = 1u;
        int32_t i32 = 1;
        uint32_t u32 = 1u;
        int64_t i64 = 1;
        uint64_t u64 = 1u;
        run(Type::of<int8_t>(), &i8);
        run(Type::of<uint8_t>(), &u8);
        run(Type::of<int16_t>(), &i16);
        run(Type::of<uint16_t>(), &u16);
        run(Type::of<int32_t>(), &i32);
        run(Type::of<uint32_t>(), &u32);
        run(Type::of<int64_t>(), &i64);
        run(Type::of<uint64_t>(), &u64);
    };

    "algsimpl_aggregate_swizzle_accepts_mixed_integer_widths"_test = [] {
        Module m;
        auto *type = Type::vector(Type::of<float>(), 3u);
        auto *f = m.create_callable(type);
        auto *value = f->create_value_argument(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int8_t index0_value = 0;
        uint16_t index1_value = 1u;
        int64_t index2_value = 2;
        auto *index0 = m.create_constant(Type::of<int8_t>(), &index0_value);
        auto *index1 = m.create_constant(Type::of<uint16_t>(), &index1_value);
        auto *index2 = m.create_constant(Type::of<int64_t>(), &index2_value);
        auto *x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index0});
        auto *y = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index1});
        auto *z = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {value, index2});
        auto *ret = b.return_(b.call(type, ArithmeticOp::AGGREGATE, {x, y, z}));
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value() == value);
    };

    "algsimpl_insert_chain_accepts_mixed_integer_widths"_test = [] {
        Module m;
        auto *type = Type::vector(Type::of<int>(), 3u);
        auto *f = m.create_callable(type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t x_value = 3;
        int32_t y_value = 5;
        int32_t z_value = 7;
        uint8_t index0_value = 0u;
        int16_t index1_value = 1;
        uint64_t index2_value = 2u;
        auto *x = m.create_constant(Type::of<int>(), &x_value);
        auto *y = m.create_constant(Type::of<int>(), &y_value);
        auto *z = m.create_constant(Type::of<int>(), &z_value);
        auto *index0 = m.create_constant(Type::of<uint8_t>(), &index0_value);
        auto *index1 = m.create_constant(Type::of<int16_t>(), &index1_value);
        auto *index2 = m.create_constant(Type::of<uint64_t>(), &index2_value);
        auto *insert0 = b.call(type, ArithmeticOp::INSERT, {m.create_undefined(type), x, index0});
        auto *insert1 = b.call(type, ArithmeticOp::INSERT, {insert0, y, index1});
        auto *insert2 = b.call(type, ArithmeticOp::INSERT, {insert1, z, index2});
        auto *ret = b.return_(insert2);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *aggregate = static_cast<ArithmeticInst *>(ret->return_value());
        expect(aggregate->op() == ArithmeticOp::AGGREGATE);
        expect(aggregate->operand(0u) == x);
        expect(aggregate->operand(1u) == y);
        expect(aggregate->operand(2u) == z);
    };

    "algsimpl_invalid_constant_indices_are_conservative"_test = [] {
        auto run = [](const Type *index_type, const void *index_data) {
            Module m;
            auto *f = m.create_callable(Type::of<int>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *vector_type = Type::vector(Type::of<int>(), 2u);
            auto *zero = m.create_constant_zero(Type::of<int>());
            auto *one = m.create_constant_one(Type::of<int>());
            auto *index = m.create_constant(index_type, index_data);
            auto *aggregate = b.call(vector_type, ArithmeticOp::AGGREGATE, {zero, one});
            auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {aggregate, index});
            auto *ret = b.return_(extract);
            auto info = algebraic_simplify_pass_run_on_function(f);
            expect(info.simplified_inst_count == 0u);
            expect(ret->return_value() == extract);
        };
        int8_t negative = -1;
        float noninteger = 0.0f;
        uint64_t out_of_bounds = std::numeric_limits<uint64_t>::max();
        run(Type::of<int8_t>(), &negative);
        run(Type::of<float>(), &noninteger);
        run(Type::of<uint64_t>(), &out_of_bounds);
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

    "algsimpl_nested_extract_from_aggregate_preserves_path"_test = [] {
        Module m;
        auto *inner_type = Type::array(Type::of<int>(), 2u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *f = m.create_callable(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        auto *inner = b.call(inner_type, ArithmeticOp::AGGREGATE, {one, two});
        auto *outer = b.call(outer_type, ArithmeticOp::AGGREGATE, {inner, inner});
        auto *index0 = m.create_constant_zero(Type::of<uint>());
        auto *index1 = m.create_constant_one(Type::of<uint>());
        auto *extract = b.call(Type::of<int>(), ArithmeticOp::EXTRACT, {outer, index0, index1});
        auto *ret = b.return_(extract);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == extract);
    };

    "algsimpl_nested_insert_into_aggregate_preserves_path"_test = [] {
        Module m;
        auto *inner_type = Type::array(Type::of<int>(), 2u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *f = m.create_callable(outer_type);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *inner = b.call(inner_type, ArithmeticOp::AGGREGATE, {zero, zero});
        auto *outer = b.call(outer_type, ArithmeticOp::AGGREGATE, {inner, inner});
        auto *index0 = m.create_constant_zero(Type::of<uint>());
        auto *index1 = m.create_constant_one(Type::of<uint>());
        auto *insert = b.call(outer_type, ArithmeticOp::INSERT, {outer, zero, index0, index1});
        auto *ret = b.return_(insert);
        auto info = algebraic_simplify_pass_run_on_function(f);
        expect(info.simplified_inst_count == 0u);
        expect(ret->return_value() == insert);
        expect(insert->operand(0u) == outer);
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

    "constfold_lerp_keeps_backend_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 0.5f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "constfold_float_special_values_remain_target_independent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float4>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float nan_v = std::numeric_limits<float>::quiet_NaN();
        float positive_zero_v = 0.0f;
        float negative_zero_v = -0.0f;
        float one_v = 1.0f;
        auto *nan = m.create_constant(Type::of<float>(), &nan_v);
        auto *positive_zero = m.create_constant(Type::of<float>(), &positive_zero_v);
        auto *negative_zero = m.create_constant(Type::of<float>(), &negative_zero_v);
        auto *one = m.create_constant(Type::of<float>(), &one_v);
        auto *min_zero = b.call(Type::of<float>(), ArithmeticOp::MIN,
                                {positive_zero, negative_zero});
        auto *step_nan = b.call(Type::of<float>(), ArithmeticOp::STEP, {one, nan});
        auto *saturate_zero = b.call(Type::of<float>(), ArithmeticOp::SATURATE,
                                     {negative_zero});
        auto *clamp_zero = b.call(Type::of<float>(), ArithmeticOp::CLAMP,
                                  {negative_zero, positive_zero, one});
        [[maybe_unused]] auto clamp_zero_lock = clamp_zero->lock();
        auto *smooth = b.call(Type::of<float>(), ArithmeticOp::SMOOTHSTEP,
                              {positive_zero, one, one});
        auto *result = b.call(Type::of<float4>(), ArithmeticOp::AGGREGATE,
                              {min_zero, step_nan, saturate_zero, smooth});
        auto *ret = b.return_(result);

        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(ret->return_value() == result);
        expect(result->operand(0u) == min_zero);
        expect(result->operand(1u) == step_nan);
        expect(result->operand(2u) == saturate_zero);
        expect(result->operand(3u) == smooth);
        expect(clamp_zero->is_linked());
    };

    "constfold_pow_int_preserves_large_exponent_parity"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = -1.0f;
        int32_t exponent_value = 16777217;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<int>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == -1.0f);
    };

    "constfold_pow_int_decodes_signed_narrow_exponent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = 2.0f;
        int8_t exponent_value = -1;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<int8_t>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.5f);
    };

    "constfold_pow_int_decodes_unsigned_64_bit_exponent"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float base_value = -1.0f;
        uint64_t exponent_value = uint64_t{1} << 32u;
        auto *base = m.create_constant(Type::of<float>(), &base_value);
        auto *exponent = m.create_constant(Type::of<uint64_t>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 1.0f);
    };

    "constfold_pow_int_decodes_vector_exponents_per_lane"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float2>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float2 base_value{2.0f, 2.0f};
        byte2 exponent_value{-1, 3};
        auto *base = m.create_constant(Type::of<float2>(), &base_value);
        auto *exponent = m.create_constant(Type::of<byte2>(), &exponent_value);
        auto *ret = b.return_(b.call(Type::of<float2>(), ArithmeticOp::POW_INT, {base, exponent}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        auto result = static_cast<Constant *>(ret->return_value())->as<float2>();
        expect(result.x == 0.5f);
        expect(result.y == 8.0f);
    };

    "constfold_round_does_not_cross_half_boundary"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto value = std::nextafter(0.5f, 0.0f);
        auto *constant = m.create_constant(Type::of<float>(), &value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::ROUND, {constant}));
        auto info = const_fold_pass_run_on_function(f);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 0.0f);
    };

    "constfold_rint_is_host_rounding_mode_independent"_test = [] {
        auto previous_rounding = std::fegetround();
        auto changed_rounding = std::fesetround(FE_UPWARD) == 0;
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float value = 1.25f;
        auto *constant = m.create_constant(Type::of<float>(), &value);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::RINT, {constant}));
        auto info = const_fold_pass_run_on_function(f);
        if (previous_rounding != -1) { static_cast<void>(std::fesetround(previous_rounding)); }
        expect(changed_rounding);
        expect(info.folded_inst_count == 1u);
        expect(ret->return_value()->isa<Constant>());
        expect(static_cast<Constant *>(ret->return_value())->as<float>() == 1.0f);
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
        expect(info.succeeded());
    };

    "loop_unroll_structured_non_analyzable_loop_rejected"_test = [] {
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
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
        expect(loop->prepare_block() == prepare);
        expect(loop->body_block() == loop_body);
        expect(loop->update_block() == update);
        expect(loop->merge_block() == merge);
    };

    "loop_unroll_structured_counted_loop_rejected"_test = [] {
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
        expect(info.unrolled_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
        expect(prepare->terminator()->isa<ConditionalBranchInst>());
        expect(update->terminator()->isa<BranchInst>());
    };

    "loop_unroll_structured_rejection_precedes_options"_test = [] {
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
        options.unroll_pure_only = true;
        auto info = loop_unroll_pass_run_on_function(k, options);
        expect(info.unrolled_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
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
        expect(info.unrolled_loop_count == 0u);
        expect(info.structured_cfg_error_count == 2u);
    };

    "loop_unroll_preserves_structured_break_continue"_test = [] {
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
        b.cond_br(m.create_undefined(Type::of<bool>()), loop_body, merge);
        b.set_insertion_point(loop_body);
        auto *if_inst = b.if_(m.create_undefined(Type::of<bool>()));
        auto *break_block = if_inst->create_true_block();
        auto *continue_block = if_inst->create_false_block();
        if_inst->set_merge_block(update);
        b.set_insertion_point(break_block);
        auto *break_inst = b.break_(merge);
        b.set_insertion_point(continue_block);
        auto *continue_inst = b.continue_(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
        expect(loop_body->terminator() == if_inst);
        expect(break_block->terminator() == break_inst);
        expect(continue_block->terminator() == continue_inst);
        expect(if_inst->merge_block() == update);
    };

    "loop_unroll_accepts_unstructured_cycle_without_mutation"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *header = k->create_basic_block();
        auto *exit = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.br(header);
        b.set_insertion_point(header);
        auto *cond = m.create_undefined(Type::of<bool>());
        b.cond_br(cond, header, exit);
        b.set_insertion_point(exit);
        b.return_void();
        auto *entry_term = entry->terminator();
        auto *header_term = header->terminator();
        auto info = loop_unroll_pass_run_on_function(k);
        expect(info.succeeded());
        expect(info.unrolled_loop_count == 0u);
        expect(entry->terminator() == entry_term);
        expect(header->terminator() == header_term);
    };

    "loop_unroll_rejects_unreachable_structured_region"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.return_void();

        auto *dead = k->create_basic_block();
        b.set_insertion_point(dead);
        auto *if_inst = b.if_(m.create_undefined(Type::of<bool>()));
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        b.set_insertion_point(true_block);
        b.return_void();
        b.set_insertion_point(false_block);
        b.return_void();

        auto info = loop_unroll_pass_run_on_function(k);
        expect(!info.succeeded());
        expect(info.structured_cfg_error_count == 1u);
        expect(entry->terminator()->isa<ReturnInst>());
        expect(dead->terminator() == if_inst);
        expect(if_inst->true_block() == true_block);
        expect(if_inst->false_block() == false_block);
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

    "dce_preserves_unused_volatile_resource_reads"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *buffer = k->create_resource_argument(Type::buffer(Type::of<int>()));
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *index = m.create_constant_zero(Type::of<uint>());
        b.call(Type::of<int>(), ResourceReadOp::BUFFER_READ, {buffer, index});
        b.call(Type::of<int>(), ResourceReadOp::BUFFER_VOLATILE_READ, {buffer, index});
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        size_t ordinary_count = 0u;
        size_t volatile_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (!inst->isa<ResourceReadInst>()) { return; }
            auto op = static_cast<ResourceReadInst *>(inst)->op();
            ordinary_count += op == ResourceReadOp::BUFFER_READ ? 1u : 0u;
            volatile_count += op == ResourceReadOp::BUFFER_VOLATILE_READ ? 1u : 0u;
        });
        expect(info.removed_inst_count == 1u);
        expect(ordinary_count == 0u);
        expect(volatile_count == 1u);
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
        expect(info.removed_block_count == 0u);
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
        expect(result_loop->merge_block() == loop_merge);
        expect(loop_merge->parent_function() == k);
        expect(loop_merge->terminator()->isa<UnreachableInst>());
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
        expect(loop->merge_block() == merge);
        expect(merge->parent_function() == k);
        expect(merge->terminator()->isa<UnreachableInst>());
    };

    "dce_does_not_truncate_int64_switch_condition"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *entry = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        int64_t selector_value = (int64_t{1} << 32u) + 1;
        auto *selector = m.create_constant(Type::of<int64_t>(), &selector_value);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1);
        auto *default_block = switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = dce_pass_run_on_function(k);
        expect(info.removed_block_count == 0u);
        expect(entry->terminator() == switch_inst);
        expect(switch_inst->value() == selector);
        expect(switch_inst->value()->type() == Type::of<int64_t>());
        expect(static_cast<Constant *>(switch_inst->value())->as<int64_t>() == selector_value);
        expect(switch_inst->case_block(0u) == case_block);
        expect(switch_inst->default_block() == default_block);
        expect(case_block->terminator()->isa<BranchInst>());
        expect(default_block->terminator()->isa<BranchInst>());
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

    "dce_preserves_unreachable_if_merge_shell"_test = [] {
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
        expect(if_inst->merge_block() == merge);
        expect(merge->parent_function() == k);
        expect(merge->terminator()->isa<UnreachableInst>());
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

    "gvn_strict_float_reversed_add_not_merged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = k->create_value_argument(Type::of<float>());
        auto *bv = k->create_value_argument(Type::of<float>());
        auto *add0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {bv, a});
        auto *pair = b.call(Type::of<float2>(), ArithmeticOp::AGGREGATE, {add0, add1});
        b.return_(pair);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 0u);
        expect(pair->operand(0) == add0);
        expect(pair->operand(1) == add1);
    };

    "gvn_integer_reversed_add_merged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *a = k->create_value_argument(Type::of<int>());
        auto *bv = k->create_value_argument(Type::of<int>());
        auto *add0 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {bv, a});
        auto add1_locked = add1->lock();
        auto *pair = b.call(Type::of<int2>(), ArithmeticOp::AGGREGATE, {add0, add1});
        b.return_(pair);
        auto info = gvn_pass_run_on_function(k);
        expect(info.replaced_inst_count == 1u);
        expect(pair->operand(0) == add0);
        expect(pair->operand(1) == add0);
        expect(add1_locked->use_list().empty());
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

    "sccp_bodyless_definition_is_ignored"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto info = sccp_pass_run_on_function(f);
        expect(info.folded_inst_count == 0u);
        expect(info.removed_branch_count == 0u);
    };

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

    "simplify_libcalls_lerp_t_zero_keeps_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 0.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "simplify_libcalls_lerp_t_one_keeps_strict_fp_semantics"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = std::numeric_limits<float>::max();
        float y_v = -std::numeric_limits<float>::max();
        float t_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *y = m.create_constant(Type::of<float>(), &y_v);
        auto *t = m.create_constant(Type::of<float>(), &t_v);
        auto *lerp = b.call(Type::of<float>(), ArithmeticOp::LERP, {x, y, t});
        auto *ret = b.return_(lerp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == lerp);
    };

    "simplify_libcalls_clamp_01_to_saturate"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float x_v = 0.5f, lo_v = 0.0f, hi_v = 1.0f;
        auto *x = m.create_constant(Type::of<float>(), &x_v);
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        auto *ret = b.return_(b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi}));
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::SATURATE);
        expect(static_cast<ArithmeticInst *>(ret->return_value())->operand(0) == x);
    };

    "simplify_libcalls_clamp_negative_zero_not_saturated"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float lo_v = -0.0f, hi_v = 1.0f;
        auto *lo = m.create_constant(Type::of<float>(), &lo_v);
        auto *hi = m.create_constant(Type::of<float>(), &hi_v);
        auto *clamp = b.call(Type::of<float>(), ArithmeticOp::CLAMP, {x, lo, hi});
        auto *ret = b.return_(clamp);
        auto info = simplify_libcalls_pass_run_on_function(f);
        expect(info.simplified_count == 0u);
        expect(ret->return_value() == clamp);
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
        luisa::vector<Value *> arguments;
        luisa::vector<ReturnInst *> returns;
        for (int i = 0; i < 2; ++i) {
            auto *f = m.create_callable(Type::of<uint>());
            auto *x = f->create_value_argument(Type::of<uint>());
            auto *body = f->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *abs = b.call(Type::of<uint>(), ArithmeticOp::ABS, {x});
            arguments.emplace_back(x);
            returns.emplace_back(b.return_(abs));
        }
        auto info = simplify_libcalls_pass_run_on_module(&m);
        expect(info.simplified_count == 2u);
        expect(returns[0]->return_value() == arguments[0]);
        expect(returns[1]->return_value() == arguments[1]);
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

    "reassociate_equal_rank_operands_preserve_ir_order"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int>());
        auto *a = f->create_value_argument(Type::of<int>());
        auto *b_arg = f->create_value_argument(Type::of<int>());
        auto *c = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ab = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, b_arg});
        auto *abc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {ab, c});
        auto *ret = b.return_(abc);
        auto info = reassociate_pass_run_on_function(f);
        expect(info.reassociated_inst_count >= 1u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *result = static_cast<ArithmeticInst *>(ret->return_value());
        expect(result->operand(1u) == c);
        expect(result->operand(0u)->isa<ArithmeticInst>());
        auto *lhs = static_cast<ArithmeticInst *>(result->operand(0u));
        expect(lhs->operand(0u) == a);
        expect(lhs->operand(1u) == b_arg);
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

    "reassociate_strict_float_chain_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *a = k->create_value_argument(Type::of<float>());
        auto *bv = k->create_value_argument(Type::of<float>());
        auto *c = k->create_value_argument(Type::of<float>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *ab = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {a, bv});
        auto *abc = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {ab, c});
        auto *ret = b.return_(abc);
        auto info = reassociate_pass_run_on_function(k);
        expect(info.reassociated_inst_count == 0u);
        expect(ret->return_value() == abc);
        expect(abc->operand(0) == ab);
        expect(abc->operand(1) == c);
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

    "cvp_float_zero_does_not_lose_signed_zero"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<float>());
        auto *x = f->create_value_argument(Type::of<float>());
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *var = b.call(Type::of<float>(), ArithmeticOp::UNARY_MINUS, {x});
        auto *zero = m.create_constant_zero(Type::of<float>());
        auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {var, zero});
        auto *if_inst = b.if_(eq);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        auto *div = b.call(Type::of<float>(), ArithmeticOp::BINARY_DIV,
                           {m.create_constant_one(Type::of<float>()), var});
        b.return_(div);
        b.set_insertion_point(false_block);
        b.return_(zero);
        b.set_insertion_point(merge);
        b.unreachable_();
        auto info = cvp_pass_run_on_function(f);
        expect(info.replaced_inst_count == 0u);
        expect(div->operand(1) == var);
        expect(body->terminator() == if_inst);
        expect(if_inst->merge_block() == merge);
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

    "dead_arg_elim_ray_query_callback_abi_preserved"_test = [] {
        Module m;
        auto *query_type = Type::of<RayQueryAll>();
        auto make_callback = [&] {
            auto *callback = m.create_callable(nullptr);
            callback->create_reference_argument(query_type);
            callback->create_value_argument(Type::of<int>());
            auto *callback_body = callback->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(callback_body);
            b.return_void();
            return callback;
        };
        auto *surface_callback = make_callback();
        auto *procedural_callback = make_callback();

        auto *pipeline_function = m.create_callable(nullptr);
        auto *query = pipeline_function->create_reference_argument(query_type);
        auto *capture = pipeline_function->create_value_argument(Type::of<int>());
        auto *body = pipeline_function->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        std::array<Value *, 1u> captures{capture};
        b.ray_query_pipeline(
            query, surface_callback, procedural_callback,
            luisa::span<Value *const>{captures});
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = dead_arg_elim_pass_run_on_module(&m);
        expect(info.removed_arg_count == 0u);
        expect(surface_callback->arguments().count_size() == 2u);
        expect(procedural_callback->arguments().count_size() == 2u);
        expect(xir_verify_module(&m).succeeded());
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
        auto *div = b.call(Type::of<int>(), ArithmeticOp::BINARY_DIV, {a, bv});
        auto *mod = b.call(Type::of<int>(), ArithmeticOp::BINARY_MOD, {a, bv});
        auto mod_locked = mod->lock();
        auto *ret = b.return_(mod);
        auto info = div_rem_pairs_pass_run_on_function(k);
        expect(info.merged_pair_count == 1u);
        expect(mod_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *sub = static_cast<ArithmeticInst *>(ret->return_value());
        expect(sub->op() == ArithmeticOp::BINARY_SUB);
        expect(sub->operand(0) == a);
        expect(sub->operand(1)->isa<ArithmeticInst>());
        auto *mul = static_cast<ArithmeticInst *>(sub->operand(1));
        expect(mul->op() == ArithmeticOp::BINARY_MUL);
        expect(mul->operand(0) == div);
        expect(mul->operand(1) == bv);
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

    "local_load_elim_entry_backedge_does_not_forward_future_load"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *exit = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *first_load = b.load(Type::of<int>(), alloca);
        auto first_load_lock = first_load->lock();
        auto *increment = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD,
                                 {first_load, m.create_constant_one(Type::of<int>())});
        auto *store = b.store(alloca, increment);
        auto *future_load = b.load(Type::of<int>(), alloca);
        b.cond_br(m.create_undefined(Type::of<bool>()), body, exit);
        b.set_insertion_point(exit);
        b.return_(future_load);

        auto info = local_load_elimination_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(first_load->is_linked());
        expect(future_load->is_linked());
        expect(increment->operand(0u) == first_load_lock.get());
        expect(first_load->next() == increment);
        expect(increment->next() == store);
        expect(store->next() == future_load);
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

    "local_store_forward_nested_partial_store_blocks_uniform_forward"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner = Type::array(Type::of<float>(), 2u);
        auto *outer = Type::array(inner, 2u);
        auto *alloca = b.alloca_local(outer);
        float init_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto *init = m.create_constant(outer, init_data);
        b.store(alloca, init);
        uint32_t zero_value = 0u;
        uint32_t one_value = 1u;
        auto *zero = m.create_constant(Type::of<uint>(), &zero_value);
        auto *one = m.create_constant(Type::of<uint>(), &one_value);
        auto *row = b.gep(inner, alloca, {zero});
        auto *element = b.gep(Type::of<float>(), row, {one});
        b.store(element, m.create_constant_zero(Type::of<float>()));
        auto *load = b.load(outer, alloca);
        auto *ret = b.return_(load);
        auto info = local_store_forward_pass_run_on_function(k);
        expect(info.removed_load_count == 0u);
        expect(ret->return_value() == load);
        expect(load->variable() == alloca);
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
        auto *store1 = b.store(alloca, val1);
        auto store1_locked = store1->lock();
        auto *store2 = b.store(alloca, val2);
        auto *ld = b.load(Type::of<int>(), alloca);
        b.return_(ld);
        auto info = dead_store_elimination_pass_run_on_function(k);
        expect(info.eliminated_store_count == 1u);
        expect(store1_locked->use_list().empty());
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 1u);
        expect(store2->variable() == alloca);
        expect(store2->value() == val2);
        expect(ld->variable() == alloca);
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

    "dse_two_block_straight_line_cycle_terminates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *loop_block = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *store0 = b.store(alloca, m.create_constant_zero(Type::of<int>()));
        [[maybe_unused]] auto store0_lock = store0->lock();
        b.br(loop_block);
        b.set_insertion_point(loop_block);
        auto *store1 = b.store(alloca, m.create_constant_one(Type::of<int>()));
        b.br(body);

        auto info = dead_store_elimination_pass_run_on_function(k);

        expect(info.eliminated_store_count == 1u);
        expect(!store0->is_linked());
        expect(store1->is_linked());
        auto rerun = dead_store_elimination_pass_run_on_function(k);
        expect(rerun.eliminated_store_count == 0u);
        expect(store1->is_linked());
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

    "loop_rotation_rejects_structured_loop_without_mutation"_test = [] {
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
        expect(info.rotated_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop);
        expect(loop->prepare_block() == prepare);
        expect(loop->merge_block() == merge);
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
        expect(info.succeeded());
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
        expect(info.rotated_loop_count == 0u);
        expect(info.structured_cfg_error_count == 2u);
    };
}

// ---- scalar_evolution ----

void reg_scalar_evolution() {

    "scev_argument_stride_and_rerun_are_current"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *stride_arg = k->create_value_argument(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t bound_value = 8;
        auto *bound = m.create_constant(Type::of<int>(), &bound_value);
        b.set_insertion_point(prepare);
        auto *phi = b.phi(Type::of<int>(), {{zero, body}});
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        b.cond_br(cond, loop_body, merge);
        b.set_insertion_point(loop_body);
        auto *constant_sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {zero, one});
        b.br(update);
        b.set_insertion_point(update);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, stride_arg});
        phi->add_incoming(inc, update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_(phi);

        auto first = scev_pass_run_on_function(k);
        expect(first.analyzed_loop_count == 1u);
        auto *phi_scev = scev_get_for_value(phi);
        expect(phi_scev != nullptr);
        expect(phi_scev->kind() == SCEV::Kind::ADD_REC);
        auto *add_rec = static_cast<const SCEVAddRec *>(phi_scev);
        expect(add_rec->stride()->kind() == SCEV::Kind::UNKNOWN);
        expect(static_cast<const SCEVUnknown *>(add_rec->stride())->value() == stride_arg);
        auto *sum_scev = scev_get_for_value(constant_sum);
        expect(sum_scev != nullptr);
        expect(sum_scev->kind() == SCEV::Kind::ADD);
        expect(static_cast<const SCEVAddExpr *>(sum_scev)->operands().size() == 2u);

        inc->set_operand(1u, one);
        auto second = scev_pass_run_on_function(k);
        expect(second.analyzed_loop_count == 1u);
        auto *updated = scev_get_for_value(phi);
        expect(updated != nullptr);
        expect(updated->kind() == SCEV::Kind::ADD_REC);
        auto *updated_rec = static_cast<const SCEVAddRec *>(updated);
        expect(updated_rec->stride()->kind() == SCEV::Kind::CONSTANT);
        expect(static_cast<const SCEVConstant *>(updated_rec->stride())->constant() == one);
    };
}

// ---- scalarizer ----

void reg_scalarizer() {

    "scalarizer_float3_add_scalarized"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        auto *f = m.create_callable(vec_t);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        float a_data[3] = {1.0f, 2.0f, 3.0f};
        float b_data[3] = {4.0f, 5.0f, 6.0f};
        auto *a = m.create_constant(vec_t, a_data);
        auto *bv = m.create_constant(vec_t, b_data);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
        auto add_locked = add->lock();
        auto *ret = b.return_(add);
        auto info = scalarizer_pass_run_on_function(f);
        expect(info.scalarized_inst_count == 1u);
        expect(add_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::AGGREGATE);
        size_t scalar_add_count = 0u;
        size_t vector_add_count = 0u;
        f->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_ADD) {
                if (inst->type() == Type::of<float>()) { ++scalar_add_count; }
                if (inst->type() == vec_t) { ++vector_add_count; }
            }
        });
        expect(scalar_add_count == 3u);
        expect(vector_add_count == 0u);
        expect(block_local_defs_precede_uses(body));
    };

    "scalarizer_chained_vector_ops_preserve_ssa_order"_test = [] {
        Module m;
        auto vec_t = Type::of<float3>();
        auto *f = m.create_callable(vec_t);
        auto *x = f->create_value_argument(vec_t);
        auto *y = f->create_value_argument(vec_t);
        auto *body = f->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {x, y});
        auto *mul = b.call(vec_t, ArithmeticOp::BINARY_MUL, {add, y});
        auto *ret = b.return_(mul);

        auto info = scalarizer_pass_run_on_function(f);

        expect(info.scalarized_inst_count == 2u);
        expect(ret->return_value()->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(ret->return_value())->op() == ArithmeticOp::AGGREGATE);
        expect(block_local_defs_precede_uses(body));

        size_t scalar_add_count = 0u;
        size_t scalar_mul_count = 0u;
        size_t vector_component_op_count = 0u;
        for (auto *inst : body->instructions()) {
            if (!inst->isa<ArithmeticInst>()) continue;
            auto *arith = static_cast<ArithmeticInst *>(inst);
            if (arith->op() == ArithmeticOp::BINARY_ADD) {
                if (arith->type() == Type::of<float>()) { ++scalar_add_count; }
                if (arith->type() == vec_t) { ++vector_component_op_count; }
            }
            if (arith->op() == ArithmeticOp::BINARY_MUL) {
                if (arith->type() == Type::of<float>()) { ++scalar_mul_count; }
                if (arith->type() == vec_t) { ++vector_component_op_count; }
            }
        }
        expect(scalar_add_count == 3u);
        expect(scalar_mul_count == 3u);
        expect(vector_component_op_count == 0u);
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
            auto *a = m.create_constant(vec_t, a_data);
            auto *bv = m.create_constant(vec_t, b_data);
            auto *add = b.call(vec_t, ArithmeticOp::BINARY_ADD, {a, bv});
            b.return_(add);
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
        expect(info.replaced_phi_count == 1u);
        expect(info.succeeded());
        expect(body->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(body->terminator())->target_block() == merge);
        expect(phi->incoming_count() == 1u);
        expect(phi->incoming(0).block == body);
        expect(phi->incoming(0).value->isa<ArithmeticInst>());
        expect(static_cast<ArithmeticInst *>(phi->incoming(0).value)->op() == ArithmeticOp::SELECT);
        expect(count_reachable_blocks(k) == 2u);
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

    "if_conversion_rejects_structured_if_unchanged"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_undefined(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = if_conversion_pass_run_on_function(k);
        expect(info.converted_diamond_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == if_inst);
        expect(if_inst->true_block() == true_block);
        expect(if_inst->false_block() == false_block);
        expect(if_inst->merge_block() == merge);
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
        auto *true_block = k->create_basic_block();
        auto *false_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        b.cond_br(m.create_undefined(Type::of<bool>()), true_block, false_block);
        auto *one = m.create_constant_one(Type::of<int>());
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>(), {{one, true_block}, {two, false_block}});
        auto phi_locked = phi->lock();
        auto *final_add = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {phi, one});
        b.return_(final_add);
        auto info = reg2mem_pass_run_on_function(k);
        expect(info.lowered_phi_count == 1u);
        expect(phi_locked->use_list().empty());
        expect(count_reachable_insts(k, DerivedInstructionTag::PHI) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 3u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOAD) == 1u);
        expect(final_add->operand(0)->isa<LoadInst>());
        expect(final_add->operand(1) == one);
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

    "sroa_aggressive_dynamic_top_level_index_rejected"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *index = k->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *array_type = Type::array(Type::of<float>(), 4u);
        auto *alloca = b.alloca_local(array_type);
        auto *gep = b.gep(Type::of<float>(), alloca, {index});
        auto *load = b.load(Type::of<float>(), gep);
        auto *ret = b.return_(load);
        auto info = sroa_pass_run_on_function(k, {.aggressive = true});
        expect(info.decomposed_alloca_count == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 1u);
        expect(ret->return_value() == load);
        expect(load->variable() == gep);
        expect(gep->base() == alloca);
    };

    "sroa_constant_outer_dynamic_inner_index_is_safe"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *inner_index = k->create_value_argument(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner_type = Type::array(Type::of<float>(), 4u);
        auto *outer_type = Type::array(inner_type, 2u);
        auto *alloca = b.alloca_local(outer_type);
        uint32_t zero_value = 0u;
        auto *zero = m.create_constant(Type::of<uint>(), &zero_value);
        auto *gep = b.gep(Type::of<float>(), alloca, {zero, inner_index});
        auto *load = b.load(Type::of<float>(), gep);
        b.return_(load);
        auto info = sroa_pass_run_on_function(k);
        expect(info.decomposed_alloca_count == 1u);
        expect(info.inserted_alloca_count == 2u);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 2u);
        expect(load->variable()->isa<GEPInst>());
        auto *new_gep = static_cast<GEPInst *>(load->variable());
        expect(new_gep->base()->isa<AllocaInst>());
        expect(new_gep->index_count() == 1u);
        expect(new_gep->index(0) == inner_index);
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
        auto call_locked = call->lock();
        auto *ret = b.return_(call);

        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 1u);
        expect(info.removed_callable_count == 1u);
        expect(call_locked->use_list().empty());
        expect(ret->return_value() == val);
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 0u);
        expect(count_reachable_insts(caller, DerivedInstructionTag::BRANCH) == 0u);
    };

    "inline_recursive_callable_is_skipped"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *self_call = b.call(Type::of<int>(), callee, {});
        b.return_(self_call);
        BasicBlock *caller_body;
        auto *caller = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *call = b.call(Type::of<int>(), callee, {});
        b.return_(call);
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.removed_callable_count == 0u);
        expect(info.skipped_recursive_callable_count == 1u);
        expect(call->is_linked());
        expect(self_call->is_linked());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
    };

    "inline_single_block_callee_preserves_structured_caller"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        auto *one = m.create_constant_one(Type::of<int>());
        b.return_(one);
        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<int>(), callee, {});
        auto call_locked = call->lock();
        auto *if_inst = b.if_(m.create_undefined(Type::of<bool>()));
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        auto *true_ret = b.return_(call);
        b.set_insertion_point(false_block);
        b.return_(m.create_constant_zero(Type::of<int>()));
        b.set_insertion_point(merge);
        b.unreachable_();
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 1u);
        expect(info.skipped_structured_call_count == 0u);
        expect(call_locked->use_list().empty());
        expect(body->terminator() == if_inst);
        expect(if_inst->true_block() == true_block);
        expect(if_inst->false_block() == false_block);
        expect(if_inst->merge_block() == merge);
        expect(true_ret->return_value() == one);
        expect(count_reachable_insts(caller, DerivedInstructionTag::BRANCH) == 0u);
    };

    "inline_multiblock_callee_rejected_in_structured_caller"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<bool>());
        auto *entry = callee->create_body_block();
        auto *left = callee->create_basic_block();
        auto *right = callee->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(m.create_undefined(Type::of<bool>()), left, right);
        b.set_insertion_point(left);
        b.return_(m.create_constant_one(Type::of<bool>()));
        b.set_insertion_point(right);
        b.return_(m.create_constant_zero(Type::of<bool>()));
        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<bool>(), callee, {});
        auto *if_inst = b.if_(call);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());
        auto info = inline_pass_run_on_module(&m);
        expect(info.inlined_call_count == 0u);
        expect(info.skipped_structured_call_count == 1u);
        expect(call->is_linked());
        expect(body->terminator() == if_inst);
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "inline_multiblock_callee_succeeds_after_destructure"_test = [] {
        Module m;
        auto *callee = m.create_callable(Type::of<bool>());
        auto *entry = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *callee_if = b.if_(m.create_undefined(Type::of<bool>()));
        auto *callee_true = callee_if->create_true_block();
        auto *callee_false = callee_if->create_false_block();
        auto *callee_merge = callee_if->create_merge_block();
        b.set_insertion_point(callee_true);
        b.br(callee_merge);
        b.set_insertion_point(callee_false);
        b.br(callee_merge);
        b.set_insertion_point(callee_merge);
        b.return_(m.create_constant_one(Type::of<bool>()));

        BasicBlock *body;
        auto *caller = make_kernel_with_body(m, body);
        b.set_insertion_point(body);
        auto *call = b.call(Type::of<bool>(), callee, {});
        auto call_locked = call->lock();
        auto *if_inst = b.if_(call);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(true_block);
        b.br(merge);
        b.set_insertion_point(false_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 1u);
        auto destructure = destructure_cfg_pass_run_on_module(&m);
        expect(destructure.succeeded());
        expect(destructure.destructured_if_count == 2u);
        expect(xir_verify_module(&m).succeeded());

        auto after = inline_pass_run_on_module(&m);
        expect(after.inlined_call_count == 1u);
        expect(after.skipped_structured_call_count == 0u);
        expect(after.removed_callable_count == 1u);
        expect(call_locked->use_list().empty());
        expect(count_reachable_insts(caller, DerivedInstructionTag::CALL) == 0u);
        expect(xir_verify_module(&m).succeeded());
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
        auto gep_locked = gep->lock();
        auto val_locked = val->lock();
        auto *ret = b.return_(val);
        auto info = transpose_gep_pass_run_on_function(k);
        expect(info.transposed_load_count == 1u);
        expect(gep_locked->use_list().empty());
        expect(val_locked->use_list().empty());
        expect(ret->return_value()->isa<ArithmeticInst>());
        auto *extract = static_cast<ArithmeticInst *>(ret->return_value());
        expect(extract->op() == ArithmeticOp::EXTRACT);
        expect(extract->operand(0)->isa<LoadInst>());
        expect(static_cast<LoadInst *>(extract->operand(0))->variable() == alloca);
        expect(extract->operand(1) == idx0);
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
        auto ld_locked = ld->lock();
        auto *ret = b.return_(ld);
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 1u);
        expect(info.removed_load_count == 1u);
        expect(info.removed_store_count == 1u);
        expect(ld_locked->use_list().empty());
        expect(ret->return_value() == val);
        expect(count_reachable_insts(k, DerivedInstructionTag::ALLOCA) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::STORE) == 0u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOAD) == 0u);
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

    "mem2reg_retains_alloca_with_unreachable_load_store_users"_test = [] {
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
        auto *dead_store = b.store(alloca, dead_load);
        b.unreachable_();
        [[maybe_unused]] auto alloca_lock = alloca->lock();
        [[maybe_unused]] auto ld_lock = ld->lock();
        [[maybe_unused]] auto dead_load_lock = dead_load->lock();
        [[maybe_unused]] auto dead_store_lock = dead_store->lock();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
        expect(info.removed_load_count == 0u);
        expect(info.removed_store_count == 0u);
        expect(alloca->is_linked());
        expect(ld->is_linked());
        expect(dead_load->is_linked());
        expect(dead_store->is_linked());
        expect(dead_store->value() == dead_load);
    };

    "mem2reg_retains_alloca_with_unreachable_load_used_by_owned_instruction"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *dead = k->definition()->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca = b.alloca_local(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        b.store(alloca, one);
        auto *live_load = b.load(Type::of<int>(), alloca);
        b.return_(live_load);

        b.set_insertion_point(dead);
        auto *dead_load = b.load(Type::of<int>(), alloca);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {dead_load, one});
        b.return_(sum);

        [[maybe_unused]] auto alloca_lock = alloca->lock();
        [[maybe_unused]] auto live_load_lock = live_load->lock();
        [[maybe_unused]] auto dead_load_lock = dead_load->lock();
        [[maybe_unused]] auto sum_lock = sum->lock();
        auto info = mem2reg_pass_run_on_function(k);
        expect(info.promoted_alloca_count == 0u);
        expect(info.removed_load_count == 0u);
        expect(info.removed_store_count == 0u);
        expect(alloca->is_linked());
        expect(live_load->is_linked());
        expect(dead_load->is_linked());
        expect(sum->is_linked());
        expect(sum->operand(0u) == dead_load);
        expect(!dead_load->use_list().empty());
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

    "promote_ref_arg_writable_alias_blocks_snapshot"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *read_ref = c->create_reference_argument(Type::of<int>());
        auto *write_ref = c->create_reference_argument(Type::of<int>());
        auto *callee_body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        int32_t two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);
        b.store(write_ref, two);
        auto *loaded_after_store = b.load(Type::of<int>(), read_ref);
        b.return_(loaded_after_store);

        BasicBlock *caller_body;
        auto *k = make_kernel_with_body(m, caller_body);
        b.set_insertion_point(caller_body);
        auto *local = b.alloca_local(Type::of<int>());
        b.store(local, m.create_constant_one(Type::of<int>()));
        auto *call = b.call(Type::of<int>(), c, {local, local});
        b.return_(call);

        auto info = promote_ref_arg_pass_run_on_module(&m);
        expect(info.promoted_ref_arg_count == 0u);
        expect(c->arguments().count_size() == 2u);
        expect(c->arguments().front()->is_reference());
        expect(c->arguments().back()->is_reference());
        expect(call->argument(0u) == local);
        expect(call->argument(1u) == local);
        expect(loaded_after_store->variable() == read_ref);
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
        expect(info.unsupported_outline_count == 0u);
        expect(info.succeeded());
    };

    "outline_instruction_reports_unsupported_without_mutation"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outline = b.outline();
        auto *entry = outline->create_target_block();
        auto *merge = outline->create_merge_block();
        b.set_insertion_point(entry);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = outline_pass_run_on_module(&m);
        expect(info.outlined_func_count == 0u);
        expect(info.unsupported_outline_count == 1u);
        expect(!info.succeeded());
        expect(body->terminator() == outline);
        expect(outline->target_block() == entry);
        expect(outline->merge_block() == merge);
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
        auto *vd = b.cast_(double_vector_type, xir::CastOp::STATIC_CAST, v);
        auto *sm = b.call(matrix_type, ArithmeticOp::AGGREGATE, {b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s}), b.call(vector_type, ArithmeticOp::AGGREGATE, {s, s})});
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
                if (cast->op() == xir::CastOp::STATIC_CAST &&
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

    "autodiff_reverse_projects_vector_scalar_binary_gradients"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto vector_type = Type::of<float3>();
        auto *vector = k->create_argument(vector_type, false);
        auto *scalar = k->create_argument(Type::of<float>(), false);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *grad_out = b.alloca_local(Type::of<float>());
        auto *scope = b.autodiff_scope();
        auto *merge = scope->create_merge_block();
        auto *entry = scope->create_entry_block();
        b.set_insertion_point(entry);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {vector});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {scalar});
        auto *product = b.call(vector_type, ArithmeticOp::BINARY_MUL, {scalar, vector});
        auto *quotient = b.call(vector_type, ArithmeticOp::BINARY_DIV, {vector, scalar});
        auto *sum = b.call(vector_type, ArithmeticOp::BINARY_ADD, {product, quotient});
        auto *biased = b.call(vector_type, ArithmeticOp::BINARY_ADD, {sum, scalar});
        auto *output = b.call(Type::of<float>(), ArithmeticOp::REDUCE_SUM, {biased});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        auto *grad = b.call(Type::of<float>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {scalar});
        b.store(grad_out, grad);
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
                expect(inst->type() == Type::of<float>());
                expect(inst->operand_count() == 1u);
                expect(inst->operand(0u)->type() == vector_type);
            }
        });
        expect(reduce_sum_count == 4u);
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
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == Type::of<double>() &&
                    cast->value()->type() == Type::of<float>()) {
                    has_forward_cast = true;
                }
                if (cast->op() == xir::CastOp::STATIC_CAST &&
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
        auto *xd = b.cast_(double_vector_type, xir::CastOp::STATIC_CAST, x);
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
                if (cast->op() == xir::CastOp::STATIC_CAST &&
                    cast->type() == double_vector_type &&
                    cast->value()->type() == vector_type) {
                    forward_vector_cast_count++;
                }
                if (cast->op() == xir::CastOp::STATIC_CAST &&
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
                if (cast->op() == xir::CastOp::STATIC_CAST &&
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

    "autodiff_fixed_trip_analysis_honors_narrow_integer_wrapping"_test = [] {
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
        auto *value = b.alloca_local(Type::of<float>());
        auto *index = b.alloca_local(Type::of<int8_t>());
        int8_t start_value = 126;
        int8_t bound_value = 0;
        int8_t step_value = 1;
        auto *start = m.create_constant(Type::of<int8_t>(), &start_value);
        auto *bound = m.create_constant(Type::of<int8_t>(), &bound_value);
        auto *step = m.create_constant(Type::of<int8_t>(), &step_value);
        b.store(value, x);
        b.store(index, start);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *current_index = b.load(Type::of<int8_t>(), index);
        auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {current_index, bound});
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *current_value = b.load(Type::of<float>(), value);
        auto *next_value = b.call(Type::of<float>(), ArithmeticOp::SIN,
                                  {b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {current_value, x})});
        b.store(value, next_value);
        b.br(update);
        b.set_insertion_point(update);
        auto *old_index = b.load(Type::of<int8_t>(), index);
        auto *next_index = b.call(Type::of<int8_t>(), ArithmeticOp::BINARY_ADD, {old_index, step});
        b.store(index, next_index);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        auto *output = b.load(Type::of<float>(), value);
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER,
               {output, m.create_constant_one(Type::of<float>())});
        b.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto info = autodiff_pass_run_on_function(k);

        expect(info.transformed_scope_count == 1u);
        expect(count_reachable_insts(k, DerivedInstructionTag::LOOP) == 0u);
        size_t sin_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>() &&
                static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::SIN) {
                sin_count++;
            }
        });
        expect(sin_count == 2u);
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
    reg_scalar_evolution();
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
