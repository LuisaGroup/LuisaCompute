#pragma once


#include <api/common.h>



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


LUISA_EXPORT_API LCExpression luisa_compute_ast_buffer_binding(LCType elem_t, LCBuffer buffer, size_t offset_bytes, size_t size_bytes) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_texture_binding(LCType t, LCTexture texture, uint32_t level) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_bindless_array_binding(LCBindlessArray array) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_accel_binding(LCAccel accel) LUISA_NOEXCEPT;

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
LUISA_EXPORT_API LCExpression luisa_compute_ast_call_expr(LCType t, LCCallOp call_op, LCCallable custom_callable, LCExpression * args, size_t arg_count) LUISA_NOEXCEPT;



LUISA_EXPORT_API void luisa_compute_ast_break_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_continue_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_return_stmt(LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt(LCExpression cond) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_loop_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_stmt(LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_case_stmt(LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_default_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_for_stmt(LCExpression var, LCExpression cond, LCExpression update) LUISA_NOEXCEPT;
// LUISA_EXPORT_API LCStmt luisa_compute_ast_meta_stmt(const char *meta_expr) LUISA_NOEXCEPT;

LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt_true_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_if_stmt_false_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_loop_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_case_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_switch_default_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCStmt luisa_compute_ast_for_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;
// LUISA_EXPORT_API LCStmt luisa_compute_ast_meta_stmt_scope(LCStmt stmt) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_assign_stmt(LCExpression lhs, LCExpression rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_comment(const char *comment) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_push_scope(LCStmt scope) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_pop_scope(LCStmt scope) LUISA_NOEXCEPT;

// LUISA_EXPORT_API void luisa_compute_ast_push_meta(LCStmtmeta) LUISA_NOEXCEPT;
// LUISA_EXPORT_API void luisa_compute_ast_pop_meta(void *meta) LUISA_NOEXCEPT;