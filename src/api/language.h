#pragma once

#include <core/platform.h>
#include <api/common.h>

_LUISA_API_DECL_TYPE(LCKernel);
_LUISA_API_DECL_TYPE(LCFunction);
_LUISA_API_DECL_TYPE(LCCallable);
_LUISA_API_DECL_TYPE(LCType);
_LUISA_API_DECL_TYPE(LCExpression);
_LUISA_API_DECL_TYPE(LCConstantData);
_LUISA_API_DECL_TYPE(LCStmt);


/**
 * @brief Enum of unary operations.
 * 
 * Note: We deliberately support *NO* pre and postfix inc/dec operators to avoid possible abuse
 */
typedef enum LCUnaryOp  {
    PLUS,
    MINUS,  // +x, -x
    NOT,    // !x
    BIT_NOT,// ~x
}LCUnaryOp;

/**
 * @brief Enum of binary operations
 * 
 */
typedef enum LCBinaryOp {

    // arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    AND,
    OR,

    // relational
    LESS,
    GREATER,
    LESS_EQUAL,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL
}LCBinaryOp;

/**
 * @brief Enum of call operations.
 * 
 */
typedef enum LCCallOp {

    CUSTOM,

    ALL,
    ANY,

    SELECT,
    CLAMP,
    LERP,
    STEP,

    ABS,
    MIN,
    MAX,

    CLZ,
    CTZ,
    POPCOUNT,
    REVERSE,

    ISINF,
    ISNAN,

    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,

    COS,
    COSH,
    SIN,
    SINH,
    TAN,
    TANH,

    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POW,

    SQRT,
    RSQRT,

    CEIL,
    FLOOR,
    FRACT,
    TRUNC,
    ROUND,

    FMA,
    COPYSIGN,

    CROSS,
    DOT,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACEFORWARD,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,

    SYNCHRONIZE_BLOCK,

    ATOMIC_EXCHANGE,        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    ATOMIC_COMPARE_EXCHANGE,/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    ATOMIC_FETCH_ADD,       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    ATOMIC_FETCH_SUB,       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    ATOMIC_FETCH_AND,       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    ATOMIC_FETCH_OR,        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    ATOMIC_FETCH_XOR,       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    ATOMIC_FETCH_MIN,       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    ATOMIC_FETCH_MAX,       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    BUFFER_READ,  /// [(buffer, index) -> value]: reads the index-th element in buffer
    BUFFER_WRITE, /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    TEXTURE_READ, /// [(texture, coord) -> value]
    TEXTURE_WRITE,/// [(texture, coord, value) -> void]

    BINDLESS_TEXTURE2D_SAMPLE,      //(bindless_array, index: uint, uv: float2): float4
    BINDLESS_TEXTURE2D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float2, level: float): float4
    BINDLESS_TEXTURE2D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    BINDLESS_TEXTURE3D_SAMPLE,      //(bindless_array, index: uint, uv: float3): float4
    BINDLESS_TEXTURE3D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float3, level: float): float4
    BINDLESS_TEXTURE3D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    BINDLESS_TEXTURE2D_READ,        //(bindless_array, index: uint, coord: uint2): float4
    BINDLESS_TEXTURE3D_READ,        //(bindless_array, index: uint, coord: uint3): float4
    BINDLESS_TEXTURE2D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint2, level: uint): float4
    BINDLESS_TEXTURE3D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint3, level: uint): float4
    BINDLESS_TEXTURE2D_SIZE,        //(bindless_array, index: uint): uint2
    BINDLESS_TEXTURE3D_SIZE,        //(bindless_array, index: uint): uint3
    BINDLESS_TEXTURE2D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint2
    BINDLESS_TEXTURE3D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint3

    BINDLESS_BUFFER_READ,//(bindless_array, index: uint): expr->type()

    MAKE_BOOL2,
    MAKE_BOOL3,
    MAKE_BOOL4,
    MAKE_INT2,
    MAKE_INT3,
    MAKE_INT4,
    MAKE_UINT2,
    MAKE_UINT3,
    MAKE_UINT4,
    MAKE_FLOAT2,
    MAKE_FLOAT3,
    MAKE_FLOAT4,

    MAKE_FLOAT2X2,
    MAKE_FLOAT3X3,
    MAKE_FLOAT4X4,

    // optimization hints
    ASSUME,
    UNREACHABLE,

    INSTANCE_TO_WORLD_MATRIX,
    SET_INSTANCE_TRANSFORM,
    SET_INSTANCE_VISIBILITY,
    TRACE_CLOSEST,
    TRACE_ANY
}LCCallOp;

typedef enum LCCastOp {
    STATIC,
    BITWISE
}LCCastOp;

LUISA_EXPORT_API LCKernel luisa_compute_ast_begin_kernel() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_end_kernel(LCKernel) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCCallable luisa_compute_ast_begin_callable() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_end_callable(LCCallable) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_destroy_function(LCFunction) LUISA_NOEXCEPT;


LUISA_EXPORT_API LCConstantData luisa_compute_ast_create_constant_data(LCType t, void * data, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_destroy_constant_data(LCConstantData data) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_set_block_size(uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCExpression luisa_compute_ast_thread_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_block_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_dispatch_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_dispatch_size() LUISA_NOEXCEPT;



LUISA_EXPORT_API LCExpression luisa_compute_ast_local_variable(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_shared_variable(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_constant_variable(LCType t, const void *data) LUISA_NOEXCEPT;


LUISA_EXPORT_API LCExpression luisa_compute_ast_buffer_binding(LCType elem_t, uint64_t buffer, size_t offset_bytes, size_t size_bytes) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_texture_binding(LCType t, uint64_t texture, uint32_t level) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_bindless_array_binding(uint64_t array) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_accel_binding(uint64_t accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCExpression luisa_compute_ast_value_argument(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_reference_argument(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_buffer_argument(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_texture_argument(LCType t) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_bindless_array_argument() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_accel_argument() LUISA_NOEXCEPT;

LUISA_EXPORT_API LCExpression luisa_compute_ast_literal_expr(LCType t, const void *value, const char *meta_value) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_unary_expr(LCType t, LCUnaryOp op, LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_binary_expr(LCType t, LCBinaryOp op, LCExpression lhs, LCExpression rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_member_expr(LCType t, LCExpression self, size_t member_id) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_swizzle_expr(LCType t, LCExpression self, size_t swizzle_size, uint64_t swizzle_code) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_access_expr(LCType t, LCExpression range, LCExpression index) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_cast_expr(LCType t, LCCastOp op, LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_call_expr(LCType t, LCCallOp call_op, LCCallable custom_callable, const LCExpression * args, size_t arg_count) LUISA_NOEXCEPT;



LUISA_EXPORT_API void luisa_compute_ast_break_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_continue_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_return_stmt(const LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt(const void *cond) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_loop_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_stmt(const LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_case_stmt(const LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_default_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_for_stmt(const LCExpression var, const LCExpression cond, const LCExpression update) LUISA_NOEXCEPT;
// LUISA_EXPORT_API LCStmt luisa_compute_ast_meta_stmt(const char *meta_expr) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt_true_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt_false_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_loop_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_case_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_default_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_for_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
// LUISA_EXPORT_API LCStmt luisa_compute_ast_meta_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_assign_stmt(const LCExpression lhs, const LCExpression rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_comment(const char *comment) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_push_scope(LCStmt scope) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_pop_scope(LCStmt scope) LUISA_NOEXCEPT;

// LUISA_EXPORT_API void luisa_compute_ast_push_meta(LCStmtmeta) LUISA_NOEXCEPT;
// LUISA_EXPORT_API void luisa_compute_ast_pop_meta(void *meta) LUISA_NOEXCEPT;