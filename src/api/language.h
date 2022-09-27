#pragma once

#include <core/platform.h>

#define _LUISA_API_DECL_TYPE(TypeName) typedef struct _##TypeName{}_##TypeName; typedef _##TypeName * TypeName
_LUISA_API_DECL_TYPE(LCKernel);
_LUISA_API_DECL_TYPE(LCFunction);
_LUISA_API_DECL_TYPE(LCCallable);
_LUISA_API_DECL_TYPE(LCType);
_LUISA_API_DECL_TYPE(LCExpression);
_LUISA_API_DECL_TYPE(LCConstantData);
// typedef struct LCArg{
//     LCType *type;
//     const char *name;
// };
// typedef struct _LCClosure {
//     void (*function)(void *data);
//     void * data;
// }_LCClosure;


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

LUISA_EXPORT_API LCType luisa_compute_type_from_description(const char*) LUISA_NOEXCEPT;


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
LUISA_EXPORT_API LCExpression luisa_compute_ast_unary_expr(LCType t, uint32_t op, LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_binary_expr(LCType t, uint32_t op, LCExpression lhs, LCExpression rhs) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_member_expr(LCType t, LCExpression self, size_t member_id) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_swizzle_expr(LCType t, LCExpression self, size_t swizzle_size, uint64_t swizzle_code) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_access_expr(LCType t, LCExpression range, LCExpression index) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_cast_expr(LCType t, uint32_t op, LCExpression expr) LUISA_NOEXCEPT;
LUISA_EXPORT_API LCExpression luisa_compute_ast_call_expr(LCType t, uint32_t call_op, LCCallable custom_callable, const LCExpression * args, size_t arg_count) LUISA_NOEXCEPT;
