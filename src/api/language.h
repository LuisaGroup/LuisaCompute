#pragma once


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
    LC_OP_PLUS,
    LC_OP_MINUS,  // +x, -x
    LC_OP_NOT,    // !x
    LC_OP_BIT_NOT,// ~x
}LCUnaryOp;

/**
 * @brief Enum of binary operations
 * 
 */
typedef enum LCBinaryOp {

    // arithmetic
    LC_OP_ADD,
    LC_OP_SUB,
    LC_OP_MUL,
    LC_OP_DIV,
    LC_OP_MOD,
    LC_OP_BIT_AND,
    LC_OP_BIT_OR,
    LC_OP_BIT_XOR,
    LC_OP_SHL,
    LC_OP_SHR,
    LC_OP_AND,
    LC_OP_OR,

    // relational
    LC_OP_LESS,
    LC_OP_GREATER,
    LC_OP_LESS_EQUAL,
    LC_OP_GREATER_EQUAL,
    LC_OP_EQUAL,
    LC_OP_NOT_EQUAL
}LCBinaryOp;

/**
 * @brief Enum of call operations.
 * 
 */
typedef enum LCCallOp {

    LC_OP_CUSTOM,

    LC_OP_ALL,
    LC_ANY,

    LC_OP_SELECT,
    LC_OP_CLAMP,
    LC_OP_LERP,
    LC_OP_STEP,

    LC_OP_ABS,
    LC_OP_MIN,
    LC_OP_MAX,

    LC_OP_CLZ,
    LC_OP_CTZ,
    LC_OP_POPCOUNT,
    LC_OP_REVERSE,

    LC_OP_ISINF,
    LC_OP_ISNAN,

    LC_OP_ACOS,
    LC_OP_ACOSH,
    LC_OP_ASIN,
    LC_OP_ASINH,
    LC_OP_ATAN,
    LC_OP_ATAN2,
    LC_OP_ATANH,

    LC_OP_COS,
    LC_OP_COSH,
    LC_OP_SIN,
    LC_OP_SINH,
    LC_OP_TAN,
    LC_OP_TANH,

    LC_OP_EXP,
    LC_OP_EXP2,
    LC_OP_EXP10,
    LC_OP_LOG,
    LC_OP_LOG2,
    LC_OP_LOG10,
    LC_OP_POW,

    LC_OP_SQRT,
    LC_OP_RSQRT,

    LC_OP_CEIL,
    LC_OP_FLOOR,
    LC_OP_FRACT,
    LC_OP_TRUNC,
    LC_OP_ROUND,

    LC_OP_FMA,
    LC_OP_COPYSIGN,

    LC_OP_CROSS,
    LC_OP_DOT,
    LC_OP_LENGTH,
    LC_OP_LENGTH_SQUARED,
    LC_OP_NORMALIZE,
    LC_OP_FACEFORWARD,

    LC_OP_DETERMINANT,
    LC_OP_TRANSPOSE,
    LC_OP_INVERSE,

    LC_OP_SYNCHRONIZE_BLOCK,

    LC_OP_ATOMIC_EXCHANGE,        /// [(atomic_ref, desired) -> old]: stores desired, returns old.
    LC_OP_ATOMIC_COMPARE_EXCHANGE,/// [(atomic_ref, expected, desired) -> old]: stores (old == expected ? desired : old), returns old.
    LC_OP_ATOMIC_FETCH_ADD,       /// [(atomic_ref, val) -> old]: stores (old + val), returns old.
    LC_OP_ATOMIC_FETCH_SUB,       /// [(atomic_ref, val) -> old]: stores (old - val), returns old.
    LC_OP_ATOMIC_FETCH_AND,       /// [(atomic_ref, val) -> old]: stores (old & val), returns old.
    LC_OP_ATOMIC_FETCH_OR,        /// [(atomic_ref, val) -> old]: stores (old | val), returns old.
    LC_OP_ATOMIC_FETCH_XOR,       /// [(atomic_ref, val) -> old]: stores (old ^ val), returns old.
    LC_OP_ATOMIC_FETCH_MIN,       /// [(atomic_ref, val) -> old]: stores min(old, val), returns old.
    LC_OP_ATOMIC_FETCH_MAX,       /// [(atomic_ref, val) -> old]: stores max(old, val), returns old.

    LC_OP_BUFFER_READ,  /// [(buffer, index) -> value]: reads the index-th element in buffer
    LC_OP_BUFFER_WRITE, /// [(buffer, index, value) -> void]: writes value into the index-th element of buffer
    LC_OP_TEXTURE_READ, /// [(texture, coord) -> value]
    LC_OP_TEXTURE_WRITE,/// [(texture, coord, value) -> void]

    LC_OP_BINDLESS_TEXTURE2D_SAMPLE,      //(bindless_array, index: uint, uv: float2): float4
    LC_OP_BINDLESS_TEXTURE2D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float2, level: float): float4
    LC_OP_BINDLESS_TEXTURE2D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE,      //(bindless_array, index: uint, uv: float3): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE_LEVEL,//(bindless_array, index: uint, uv: float3, level: float): float4
    LC_OP_BINDLESS_TEXTURE3D_SAMPLE_GRAD, //(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3): float4
    LC_OP_BINDLESS_TEXTURE2D_READ,        //(bindless_array, index: uint, coord: uint2): float4
    LC_OP_BINDLESS_TEXTURE3D_READ,        //(bindless_array, index: uint, coord: uint3): float4
    LC_OP_BINDLESS_TEXTURE2D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint2, level: uint): float4
    LC_OP_BINDLESS_TEXTURE3D_READ_LEVEL,  //(bindless_array, index: uint, coord: uint3, level: uint): float4
    LC_OP_BINDLESS_TEXTURE2D_SIZE,        //(bindless_array, index: uint): uint2
    LC_OP_BINDLESS_TEXTURE3D_SIZE,        //(bindless_array, index: uint): uint3
    LC_OP_BINDLESS_TEXTURE2D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint2
    LC_OP_BINDLESS_TEXTURE3D_SIZE_LEVEL,  //(bindless_array, index: uint, level: uint): uint3

    LC_OP_BINDLESS_BUFFER_READ,//(bindless_array, index: uint): expr->type()

    LC_OP_MAKE_BOOL2,
    LC_OP_MAKE_BOOL3,
    LC_OP_MAKE_BOOL4,
    LC_OP_MAKE_INT2,
    LC_OP_MAKE_INT3,
    LC_OP_MAKE_INT4,
    LC_OP_MAKE_UINT2,
    LC_OP_MAKE_UINT3,
    LC_OP_MAKE_UINT4,
    LC_OP_MAKE_FLOAT2,
    LC_OP_MAKE_FLOAT3,
    LC_OP_MAKE_FLOAT4,

    LC_OP_MAKE_FLOAT2X2,
    LC_OP_MAKE_FLOAT3X3,
    LC_OP_MAKE_FLOAT4X4,

    // optimization hints
    LC_OP_ASSUME,
    LC_OP_UNREACHABLE,

    LC_OP_INSTANCE_TO_WORLD_MATRIX,
    LC_OP_SET_INSTANCE_TRANSFORM,
    LC_OP_SET_INSTANCE_VISIBILITY,
    LC_OP_TRACE_CLOSEST,
    LC_OP_TRACE_ANY
}LCCallOp;

typedef enum LCCastOp {
    LC_OP_STATIC,
    LC_OP_BITWISE
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