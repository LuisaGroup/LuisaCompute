//
// Created by Mike Smith on 2021/10/27.
//

#pragma once

#include <core/platform.h>

LUISA_EXPORT_API void *luisa_compute_ast_begin_kernel() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_end_kernel(void *kernel) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_begin_callable() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_end_callable(void *callable) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_destroy_function(void *function) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_ast_create_constant_data(const void *t, const void *data, size_t n) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_destroy_constant_data(void *data) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_set_block_size(uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT;

LUISA_EXPORT_API const void *luisa_compute_ast_thread_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_block_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_dispatch_id() LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_dispatch_size() LUISA_NOEXCEPT;

LUISA_EXPORT_API const void *luisa_compute_ast_local_variable(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_shared_variable(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_constant_variable(const void *t, const void *data) LUISA_NOEXCEPT;

LUISA_EXPORT_API const void *luisa_compute_ast_buffer_binding(const void *elem_t, uint64_t buffer, size_t offset_bytes) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_texture_binding(const void *t, uint64_t texture, uint32_t level) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_bindless_array_binding(uint64_t array) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_accel_binding(uint64_t accel) LUISA_NOEXCEPT;

LUISA_EXPORT_API const void *luisa_compute_ast_value_argument(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_reference_argument(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_buffer_argument(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_texture_argument(const void *t) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_bindless_array_argument() LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_accel_argument() LUISA_NOEXCEPT;

LUISA_EXPORT_API const void *luisa_compute_ast_literal_expr(const void *t, const void *value, const char *meta_value) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_unary_expr(const void *t, uint32_t op, const void *expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_binary_expr(const void *t, uint32_t op, const void *lhs, const void *rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_member_expr(const void *t, const void *self, size_t member_id) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_swizzle_expr(const void *t, const void *self, size_t swizzle_size, uint64_t swizzle_code) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_access_expr(const void *t, const void *range, const void *index) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_cast_expr(const void *t, uint32_t op, const void *expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API const void *luisa_compute_ast_call_expr(const void *t, uint32_t call_op, const void *custom_callable, const void *args, size_t arg_count) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_break_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_continue_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_return_stmt(const void *expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_if_stmt(const void *cond) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_loop_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_switch_stmt(const void *expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_case_stmt(const void *expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_default_stmt() LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_for_stmt(const void *var, const void *cond, const void *update) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_meta_stmt(const char *meta_expr) LUISA_NOEXCEPT;

LUISA_EXPORT_API void *luisa_compute_ast_if_stmt_true_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_if_stmt_false_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_loop_stmt_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_switch_stmt_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_switch_case_stmt_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_switch_default_stmt_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_for_stmt_scope(void *stmt) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_ast_meta_stmt_scope(void *stmt) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_assign_stmt(uint32_t op, const void *lhs, const void *rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_comment(const char *comment) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_push_scope(void *scope) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_pop_scope(void *scope) LUISA_NOEXCEPT;

LUISA_EXPORT_API void luisa_compute_ast_push_meta(void *meta) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_ast_pop_meta(void *meta) LUISA_NOEXCEPT;