from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t
from .config import dll


dll.luisa_compute_type_from_description.restype = c_void_p
dll.luisa_compute_type_from_description.argtypes = [c_char_p]


def type_from_description(desc):
    return dll.luisa_compute_type_from_description(desc.encode())


dll.luisa_compute_type_description.restype = c_char_p
dll.luisa_compute_type_description.argtypes = [c_void_p]


def type_description(t):
    return bytes(dll.luisa_compute_type_description(t)).decode()


dll.luisa_compute_type_size.restype = c_size_t
dll.luisa_compute_type_size.argtypes = [c_void_p]


def type_size(t):
    return dll.luisa_compute_type_size(t)


dll.luisa_compute_type_alignment.restype = c_size_t
dll.luisa_compute_type_alignment.argtypes = [c_void_p]


def type_alignment(t):
    return dll.luisa_compute_type_alignment(t)


dll.luisa_compute_type_dimension.restype = c_size_t
dll.luisa_compute_type_dimension.argtypes = [c_void_p]


def type_dimension(t):
    return dll.luisa_compute_type_dimension(t)


dll.luisa_compute_type_member_count.restype = c_size_t
dll.luisa_compute_type_member_count.argtypes = [c_void_p]


def type_member_count(t):
    return dll.luisa_compute_type_member_count(t)


dll.luisa_compute_type_member_types.restype = c_void_p
dll.luisa_compute_type_member_types.argtypes = [c_void_p]


def type_member_types(t):
    return dll.luisa_compute_type_member_types(t)


dll.luisa_compute_type_element_type.restype = c_void_p
dll.luisa_compute_type_element_type.argtypes = [c_void_p]


def type_element_type(t):
    return dll.luisa_compute_type_element_type(t)


dll.luisa_compute_type_is_array.restype = c_int
dll.luisa_compute_type_is_array.argtypes = [c_void_p]


def type_is_array(t):
    return dll.luisa_compute_type_is_array(t)


dll.luisa_compute_type_is_scalar.restype = c_int
dll.luisa_compute_type_is_scalar.argtypes = [c_void_p]


def type_is_scalar(t):
    return dll.luisa_compute_type_is_scalar(t)


dll.luisa_compute_type_is_vector.restype = c_int
dll.luisa_compute_type_is_vector.argtypes = [c_void_p]


def type_is_vector(t):
    return dll.luisa_compute_type_is_vector(t)


dll.luisa_compute_type_is_matrix.restype = c_int
dll.luisa_compute_type_is_matrix.argtypes = [c_void_p]


def type_is_matrix(t):
    return dll.luisa_compute_type_is_matrix(t)


dll.luisa_compute_type_is_structure.restype = c_int
dll.luisa_compute_type_is_structure.argtypes = [c_void_p]


def type_is_structure(t):
    return dll.luisa_compute_type_is_structure(t)


dll.luisa_compute_type_is_buffer.restype = c_int
dll.luisa_compute_type_is_buffer.argtypes = [c_void_p]


def type_is_buffer(t):
    return dll.luisa_compute_type_is_buffer(t)


dll.luisa_compute_type_is_texture.restype = c_int
dll.luisa_compute_type_is_texture.argtypes = [c_void_p]


def type_is_texture(t):
    return dll.luisa_compute_type_is_texture(t)


dll.luisa_compute_type_is_heap.restype = c_int
dll.luisa_compute_type_is_heap.argtypes = [c_void_p]


def type_is_heap(t):
    return dll.luisa_compute_type_is_heap(t)


dll.luisa_compute_type_is_accel.restype = c_int
dll.luisa_compute_type_is_accel.argtypes = [c_void_p]


def type_is_accel(t):
    return dll.luisa_compute_type_is_accel(t)
