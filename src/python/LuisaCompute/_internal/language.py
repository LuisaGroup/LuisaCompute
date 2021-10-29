from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t
from .config import dll


dll.luisa_compute_ast_begin_kernel.restype = c_void_p
dll.luisa_compute_ast_begin_kernel.argtypes = []


def ast_begin_kernel():
    return dll.luisa_compute_ast_begin_kernel()


dll.luisa_compute_ast_end_kernel.restype = None
dll.luisa_compute_ast_end_kernel.argtypes = [c_void_p]


def ast_end_kernel(kernel):
    dll.luisa_compute_ast_end_kernel(kernel)


dll.luisa_compute_ast_begin_callable.restype = c_void_p
dll.luisa_compute_ast_begin_callable.argtypes = []


def ast_begin_callable():
    return dll.luisa_compute_ast_begin_callable()


dll.luisa_compute_ast_end_callable.restype = None
dll.luisa_compute_ast_end_callable.argtypes = [c_void_p]


def ast_end_callable(callable):
    dll.luisa_compute_ast_end_callable(callable)


dll.luisa_compute_ast_destroy_function.restype = None
dll.luisa_compute_ast_destroy_function.argtypes = [c_void_p]


def ast_destroy_function(function):
    dll.luisa_compute_ast_destroy_function(function)


dll.luisa_compute_ast_create_constant_data.restype = c_void_p
dll.luisa_compute_ast_create_constant_data.argtypes = [c_void_p, c_void_p, c_size_t]


def ast_create_constant_data(t, data, n):
    return dll.luisa_compute_ast_create_constant_data(t, data, n)


dll.luisa_compute_ast_destroy_constant_data.restype = None
dll.luisa_compute_ast_destroy_constant_data.argtypes = [c_void_p]


def ast_destroy_constant_data(data):
    dll.luisa_compute_ast_destroy_constant_data(data)


dll.luisa_compute_ast_set_block_size.restype = None
dll.luisa_compute_ast_set_block_size.argtypes = [c_uint32, c_uint32, c_uint32]


def ast_set_block_size(sx, sy, sz):
    dll.luisa_compute_ast_set_block_size(sx, sy, sz)


dll.luisa_compute_ast_thread_id.restype = c_void_p
dll.luisa_compute_ast_thread_id.argtypes = []


def ast_thread_id():
    return dll.luisa_compute_ast_thread_id()


dll.luisa_compute_ast_block_id.restype = c_void_p
dll.luisa_compute_ast_block_id.argtypes = []


def ast_block_id():
    return dll.luisa_compute_ast_block_id()


dll.luisa_compute_ast_dispatch_id.restype = c_void_p
dll.luisa_compute_ast_dispatch_id.argtypes = []


def ast_dispatch_id():
    return dll.luisa_compute_ast_dispatch_id()


dll.luisa_compute_ast_dispatch_size.restype = c_void_p
dll.luisa_compute_ast_dispatch_size.argtypes = []


def ast_dispatch_size():
    return dll.luisa_compute_ast_dispatch_size()


dll.luisa_compute_ast_local_variable.restype = c_void_p
dll.luisa_compute_ast_local_variable.argtypes = [c_void_p]


def ast_local_variable(t):
    return dll.luisa_compute_ast_local_variable(t)


dll.luisa_compute_ast_shared_variable.restype = c_void_p
dll.luisa_compute_ast_shared_variable.argtypes = [c_void_p]


def ast_shared_variable(t):
    return dll.luisa_compute_ast_shared_variable(t)


dll.luisa_compute_ast_constant_variable.restype = c_void_p
dll.luisa_compute_ast_constant_variable.argtypes = [c_void_p, c_void_p]


def ast_constant_variable(t, data):
    return dll.luisa_compute_ast_constant_variable(t, data)


dll.luisa_compute_ast_buffer_binding.restype = c_void_p
dll.luisa_compute_ast_buffer_binding.argtypes = [c_void_p, c_uint64, c_size_t]


def ast_buffer_binding(elem_t, buffer, offset_bytes):
    return dll.luisa_compute_ast_buffer_binding(elem_t, buffer, offset_bytes)


dll.luisa_compute_ast_texture_binding.restype = c_void_p
dll.luisa_compute_ast_texture_binding.argtypes = [c_void_p, c_uint64]


def ast_texture_binding(t, texture):
    return dll.luisa_compute_ast_texture_binding(t, texture)


dll.luisa_compute_ast_heap_binding.restype = c_void_p
dll.luisa_compute_ast_heap_binding.argtypes = [c_uint64]


def ast_heap_binding(heap):
    return dll.luisa_compute_ast_heap_binding(heap)


dll.luisa_compute_ast_accel_binding.restype = c_void_p
dll.luisa_compute_ast_accel_binding.argtypes = [c_uint64]


def ast_accel_binding(accel):
    return dll.luisa_compute_ast_accel_binding(accel)


dll.luisa_compute_ast_value_argument.restype = c_void_p
dll.luisa_compute_ast_value_argument.argtypes = [c_void_p]


def ast_value_argument(t):
    return dll.luisa_compute_ast_value_argument(t)


dll.luisa_compute_ast_reference_argument.restype = c_void_p
dll.luisa_compute_ast_reference_argument.argtypes = [c_void_p]


def ast_reference_argument(t):
    return dll.luisa_compute_ast_reference_argument(t)


dll.luisa_compute_ast_buffer_argument.restype = c_void_p
dll.luisa_compute_ast_buffer_argument.argtypes = [c_void_p]


def ast_buffer_argument(t):
    return dll.luisa_compute_ast_buffer_argument(t)


dll.luisa_compute_ast_texture_argument.restype = c_void_p
dll.luisa_compute_ast_texture_argument.argtypes = [c_void_p]


def ast_texture_argument(t):
    return dll.luisa_compute_ast_texture_argument(t)


dll.luisa_compute_ast_heap_argument.restype = c_void_p
dll.luisa_compute_ast_heap_argument.argtypes = []


def ast_heap_argument():
    return dll.luisa_compute_ast_heap_argument()


dll.luisa_compute_ast_accel_argument.restype = c_void_p
dll.luisa_compute_ast_accel_argument.argtypes = []


def ast_accel_argument():
    return dll.luisa_compute_ast_accel_argument()


dll.luisa_compute_ast_literal_expr.restype = c_void_p
dll.luisa_compute_ast_literal_expr.argtypes = [c_void_p, c_void_p]


def ast_literal_expr(t, value):
    return dll.luisa_compute_ast_literal_expr(t, value)


dll.luisa_compute_ast_unary_expr.restype = c_void_p
dll.luisa_compute_ast_unary_expr.argtypes = [c_void_p, c_uint32, c_void_p]


def ast_unary_expr(t, op, expr):
    return dll.luisa_compute_ast_unary_expr(t, op, expr)


dll.luisa_compute_ast_binary_expr.restype = c_void_p
dll.luisa_compute_ast_binary_expr.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p]


def ast_binary_expr(t, op, lhs, rhs):
    return dll.luisa_compute_ast_binary_expr(t, op, lhs, rhs)


dll.luisa_compute_ast_member_expr.restype = c_void_p
dll.luisa_compute_ast_member_expr.argtypes = [c_void_p, c_void_p, c_size_t]


def ast_member_expr(t, self, member_id):
    return dll.luisa_compute_ast_member_expr(t, self, member_id)


dll.luisa_compute_ast_swizzle_expr.restype = c_void_p
dll.luisa_compute_ast_swizzle_expr.argtypes = [c_void_p, c_void_p, c_size_t, c_uint64]


def ast_swizzle_expr(t, self, swizzle_size, swizzle_code):
    return dll.luisa_compute_ast_swizzle_expr(t, self, swizzle_size, swizzle_code)


dll.luisa_compute_ast_access_expr.restype = c_void_p
dll.luisa_compute_ast_access_expr.argtypes = [c_void_p, c_void_p, c_void_p]


def ast_access_expr(t, range, index):
    return dll.luisa_compute_ast_access_expr(t, range, index)


dll.luisa_compute_ast_cast_expr.restype = c_void_p
dll.luisa_compute_ast_cast_expr.argtypes = [c_void_p, c_uint32, c_void_p]


def ast_cast_expr(t, op, expr):
    return dll.luisa_compute_ast_cast_expr(t, op, expr)


dll.luisa_compute_ast_call_expr.restype = c_void_p
dll.luisa_compute_ast_call_expr.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p, c_size_t]


def ast_call_expr(t, call_op, custom_callable, args, arg_count):
    return dll.luisa_compute_ast_call_expr(t, call_op, custom_callable, args, arg_count)


dll.luisa_compute_ast_break_stmt.restype = None
dll.luisa_compute_ast_break_stmt.argtypes = []


def ast_break_stmt():
    dll.luisa_compute_ast_break_stmt()


dll.luisa_compute_ast_continue_stmt.restype = None
dll.luisa_compute_ast_continue_stmt.argtypes = []


def ast_continue_stmt():
    dll.luisa_compute_ast_continue_stmt()


dll.luisa_compute_ast_return_stmt.restype = None
dll.luisa_compute_ast_return_stmt.argtypes = [c_void_p]


def ast_return_stmt(expr):
    dll.luisa_compute_ast_return_stmt(expr)


dll.luisa_compute_ast_if_stmt.restype = c_void_p
dll.luisa_compute_ast_if_stmt.argtypes = [c_void_p]


def ast_if_stmt(cond):
    return dll.luisa_compute_ast_if_stmt(cond)


dll.luisa_compute_ast_loop_stmt.restype = c_void_p
dll.luisa_compute_ast_loop_stmt.argtypes = []


def ast_loop_stmt():
    return dll.luisa_compute_ast_loop_stmt()


dll.luisa_compute_ast_switch_stmt.restype = c_void_p
dll.luisa_compute_ast_switch_stmt.argtypes = [c_void_p]


def ast_switch_stmt(expr):
    return dll.luisa_compute_ast_switch_stmt(expr)


dll.luisa_compute_ast_case_stmt.restype = c_void_p
dll.luisa_compute_ast_case_stmt.argtypes = [c_void_p]


def ast_case_stmt(expr):
    return dll.luisa_compute_ast_case_stmt(expr)


dll.luisa_compute_ast_default_stmt.restype = c_void_p
dll.luisa_compute_ast_default_stmt.argtypes = []


def ast_default_stmt():
    return dll.luisa_compute_ast_default_stmt()


dll.luisa_compute_ast_for_stmt.restype = c_void_p
dll.luisa_compute_ast_for_stmt.argtypes = [c_void_p, c_void_p, c_void_p]


def ast_for_stmt(var, cond, update):
    return dll.luisa_compute_ast_for_stmt(var, cond, update)


dll.luisa_compute_ast_assign_stmt.restype = None
dll.luisa_compute_ast_assign_stmt.argtypes = [c_uint32, c_void_p, c_void_p]


def ast_assign_stmt(op, lhs, rhs):
    dll.luisa_compute_ast_assign_stmt(op, lhs, rhs)


dll.luisa_compute_ast_comment.restype = None
dll.luisa_compute_ast_comment.argtypes = [c_char_p]


def ast_comment(comment):
    dll.luisa_compute_ast_comment(comment.encode())


dll.luisa_compute_ast_push_scope.restype = None
dll.luisa_compute_ast_push_scope.argtypes = [c_void_p]


def ast_push_scope(scope):
    dll.luisa_compute_ast_push_scope(scope)


dll.luisa_compute_ast_pop_scope.restype = None
dll.luisa_compute_ast_pop_scope.argtypes = [c_void_p]


def ast_pop_scope(scope):
    dll.luisa_compute_ast_pop_scope(scope)
