//
// Created by Mike Smith on 2022/5/23.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#include <ast/type.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function_builder.h>

namespace luisa::compute::llvm {

class LLVMCodegen : public StmtVisitor {

public:
    static constexpr auto accel_handle_size = sizeof(const void *);
    static constexpr auto buffer_handle_size = sizeof(const void *);
    static constexpr auto texture_handle_size = sizeof(const void *);
    static constexpr auto bindless_array_handle_size = sizeof(const void *);

private:
    struct FunctionContext {
        Function function;
        ::llvm::Function *ir;
        ::llvm::Value *ret;
        ::llvm::BasicBlock *exit_block;
        luisa::unique_ptr<::llvm::IRBuilder<>> builder;
        luisa::unordered_map<uint, ::llvm::Value *> variables;
        luisa::vector<::llvm::BasicBlock *> break_targets;
        luisa::vector<::llvm::BasicBlock *> continue_targets;
        luisa::vector<::llvm::SwitchInst *> switch_stack;
        FunctionContext(Function f, ::llvm::Function *ir, ::llvm::Value *ret,
                        ::llvm::BasicBlock *exit_block,
                        luisa::unique_ptr<::llvm::IRBuilder<>> builder,
                        luisa::unordered_map<uint, ::llvm::Value *> variables) noexcept
            : function{f}, ir{ir}, ret{ret}, exit_block{exit_block},
              builder{std::move(builder)}, variables{std::move(variables)} {}
    };

public:
    static constexpr auto buffer_argument_size = 8u;
    static constexpr auto texture_argument_size = 8u;
    static constexpr auto accel_argument_size = 8u;
    static constexpr auto bindless_array_argument_size = 8u;

private:
    struct LLVMStruct {
        ::llvm::StructType *type;
        luisa::vector<uint> member_indices;
    };

private:
    ::llvm::LLVMContext &_context;
    ::llvm::Module *_module{nullptr};
    luisa::unordered_map<uint64_t, LLVMStruct> _struct_types;
    luisa::unordered_map<uint64_t, ::llvm::Value *> _constants;
    luisa::vector<luisa::unique_ptr<FunctionContext>> _function_stack;

private:
    void _emit_function() noexcept;
    [[nodiscard]] luisa::string _variable_name(Variable v) const noexcept;
    [[nodiscard]] luisa::string _function_name(Function f) const noexcept;
    [[nodiscard]] ::llvm::Function *_create_function(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_kernel_program(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_kernel_context(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_callable_context(Function f) noexcept;
    [[nodiscard]] ::llvm::Type *_create_type(const Type *t) noexcept;
    [[nodiscard]] ::llvm::Value *_create_expr(const Expression *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_unary_expr(const UnaryExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_binary_expr(const BinaryExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_member_expr(const MemberExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_access_expr(const AccessExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_literal_expr(const LiteralExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_ref_expr(const RefExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_constant_expr(const ConstantExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_call_expr(const CallExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_cast_expr(const CastExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_stack_variable(::llvm::Value *v, luisa::string_view name = "") noexcept;
    [[nodiscard]] FunctionContext *_current_context() noexcept;
    void _create_assignment(const Type *dst_type, const Type *src_type, ::llvm::Value *p_dst, ::llvm::Value *p_src) noexcept;

    // built-in make_vector functions
    [[nodiscard]] ::llvm::Value *_make_int2(::llvm::Value *px, ::llvm::Value *py) noexcept;
    [[nodiscard]] ::llvm::Value *_make_int3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept;
    [[nodiscard]] ::llvm::Value *_make_int4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept;
    [[nodiscard]] ::llvm::Value *_make_bool2(::llvm::Value *px, ::llvm::Value *py) noexcept;
    [[nodiscard]] ::llvm::Value *_make_bool3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept;
    [[nodiscard]] ::llvm::Value *_make_bool4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float2(::llvm::Value *px, ::llvm::Value *py) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float2x2(::llvm::Value *p0, ::llvm::Value *p1) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float3x3(::llvm::Value *p0, ::llvm::Value *p1, ::llvm::Value *p2) noexcept;
    [[nodiscard]] ::llvm::Value *_make_float4x4(::llvm::Value *p0, ::llvm::Value *p1, ::llvm::Value *p2, ::llvm::Value *p3) noexcept;

    // literals
    [[nodiscard]] ::llvm::Value *_literal(int x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(uint x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(bool x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(int2 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(uint2 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(bool2 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float2 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(int3 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(uint3 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(bool3 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float3 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(int4 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(uint4 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(bool4 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float4 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float2x2 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float3x3 x) noexcept;
    [[nodiscard]] ::llvm::Value *_literal(float4x4 x) noexcept;

    // constant data
    [[nodiscard]] ::llvm::Value *_create_constant(ConstantData c) noexcept;

    // built-in short-cut logical operators
    [[nodiscard]] ::llvm::Value *_short_circuit_and(const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_short_circuit_or(const Expression *lhs, const Expression *rhs) noexcept;

    // built-in operators
    [[nodiscard]] ::llvm::Value *_builtin_unary_plus(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_unary_minus(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_unary_not(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_unary_bit_not(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_and(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_or(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_xor(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_add(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sub(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mul(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_div(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mod(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_lt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_le(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_gt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_ge(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_eq(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_ne(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_shl(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_shr(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_add_matrix_scalar(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_add_scalar_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_add_matrix_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sub_matrix_scalar(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sub_scalar_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sub_matrix_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mul_matrix_scalar(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mul_scalar_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mul_matrix_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_mul_matrix_vector(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_div_matrix_scalar(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_div_scalar_matrix(
        const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept;

    // built-in cast operators
    [[nodiscard]] ::llvm::Value *_builtin_static_cast(const Type *t_dst, const Type *t_src, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_bitwise_cast(const Type *t_dst, const Type *t_src, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_bool(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_float(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_int(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_uint(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_vector(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_vector_to_vector(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_vector_to_bool_vector(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_vector_to_float_vector(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_vector_to_int_vector(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_vector_to_uint_vector(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_matrix(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_matrix_to_matrix(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept;

    // built-in functions
    [[nodiscard]] ::llvm::Value *_create_builtin_call_expr(const Type *ret_type, CallOp op, luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_all(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_any(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_select(const Type *t_pred, const Type *t_value, ::llvm::Value *pred,
                                                 ::llvm::Value *v_true, ::llvm::Value *v_false) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_clamp(const Type *t, ::llvm::Value *v, ::llvm::Value *lo, ::llvm::Value *hi) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_lerp(const Type *t, ::llvm::Value *a, ::llvm::Value *b, ::llvm::Value *x) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_step(const Type *t, ::llvm::Value *edge, ::llvm::Value *x) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_abs(const Type *t, ::llvm::Value *x) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_min(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_max(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_fma(const Type *t, ::llvm::Value *a, ::llvm::Value *b, ::llvm::Value *c) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_clz(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_ctz(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_popcount(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_reverse(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_isinf(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_isnan(const Type *t, ::llvm::Value *p) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_buffer_read(const Type *t_value, ::llvm::Value *buffer, ::llvm::Value *p_index) noexcept;
    void _builtin_buffer_write(const Type *t_value, ::llvm::Value *buffer, ::llvm::Value *p_index, ::llvm::Value *p_value) noexcept;
    void _builtin_assume(::llvm::Value *p) noexcept;
    void _builtin_unreachable() noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sqrt(const Type *t, ::llvm::Value *x) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_rsqrt(const Type *t, ::llvm::Value *x) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_dot(const Type *t, ::llvm::Value *a, ::llvm::Value *b) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_length_squared(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_normalize(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_floor(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_fract(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_ceil(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_trunc(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_round(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_sin(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_cos(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_tan(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_exp(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_exp2(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_exp10(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_log(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_log2(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_log10(const Type *t, ::llvm::Value *v) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_pow(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_copysign(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_faceforward(
        const Type *t, ::llvm::Value *n, ::llvm::Value *i, ::llvm::Value *nref) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_cross(const Type *t, ::llvm::Value *a, ::llvm::Value *b) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_vector2_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_vector3_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_vector4_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_matrix2_overloaded(luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_matrix3_overloaded(luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_make_matrix4_overloaded(luisa::span<const Expression *const> args) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_exchange(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_desired) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_compare_exchange(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_expected, ::llvm::Value *p_desired) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_add(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_sub(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_and(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_or(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_xor(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_min(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_atomic_fetch_max(
        const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_inverse(const Type *t, ::llvm::Value *pm) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_determinant(const Type *t, ::llvm::Value *pm) noexcept;
    [[nodiscard]] ::llvm::Value *_builtin_transpose(const Type *t, ::llvm::Value *pm) noexcept;

public:
    explicit LLVMCodegen(::llvm::LLVMContext &ctx) noexcept;
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const CommentStmt *stmt) override;
    void visit(const MetaStmt *stmt) override;
    [[nodiscard]] std::unique_ptr<::llvm::Module> emit(Function f) noexcept;
};

}// namespace luisa::compute::llvm
