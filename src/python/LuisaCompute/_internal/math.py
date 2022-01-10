from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t, c_float
from .config import dll


dll.luisa_compute_int2_create.restype = c_void_p
dll.luisa_compute_int2_create.argtypes = [c_int, c_int]


def int2_create(v0, v1):
    return dll.luisa_compute_int2_create(v0, v1)


dll.luisa_compute_int2_destroy.restype = None
dll.luisa_compute_int2_destroy.argtypes = [c_void_p]


def int2_destroy(v):
    dll.luisa_compute_int2_destroy(v)


dll.luisa_compute_int3_create.restype = c_void_p
dll.luisa_compute_int3_create.argtypes = [c_int, c_int, c_int]


def int3_create(v0, v1, v2):
    return dll.luisa_compute_int3_create(v0, v1, v2)


dll.luisa_compute_int3_destroy.restype = None
dll.luisa_compute_int3_destroy.argtypes = [c_void_p]


def int3_destroy(v):
    dll.luisa_compute_int3_destroy(v)


dll.luisa_compute_int4_create.restype = c_void_p
dll.luisa_compute_int4_create.argtypes = [c_int, c_int, c_int, c_int]


def int4_create(v0, v1, v2, v3):
    return dll.luisa_compute_int4_create(v0, v1, v2, v3)


dll.luisa_compute_int4_destroy.restype = None
dll.luisa_compute_int4_destroy.argtypes = [c_void_p]


def int4_destroy(v):
    dll.luisa_compute_int4_destroy(v)


dll.luisa_compute_uint2_create.restype = c_void_p
dll.luisa_compute_uint2_create.argtypes = [c_uint32, c_uint32]


def uint2_create(v0, v1):
    return dll.luisa_compute_uint2_create(v0, v1)


dll.luisa_compute_uint2_destroy.restype = None
dll.luisa_compute_uint2_destroy.argtypes = [c_void_p]


def uint2_destroy(v):
    dll.luisa_compute_uint2_destroy(v)


dll.luisa_compute_uint3_create.restype = c_void_p
dll.luisa_compute_uint3_create.argtypes = [c_uint32, c_uint32, c_uint32]


def uint3_create(v0, v1, v2):
    return dll.luisa_compute_uint3_create(v0, v1, v2)


dll.luisa_compute_uint3_destroy.restype = None
dll.luisa_compute_uint3_destroy.argtypes = [c_void_p]


def uint3_destroy(v):
    dll.luisa_compute_uint3_destroy(v)


dll.luisa_compute_uint4_create.restype = c_void_p
dll.luisa_compute_uint4_create.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]


def uint4_create(v0, v1, v2, v3):
    return dll.luisa_compute_uint4_create(v0, v1, v2, v3)


dll.luisa_compute_uint4_destroy.restype = None
dll.luisa_compute_uint4_destroy.argtypes = [c_void_p]


def uint4_destroy(v):
    dll.luisa_compute_uint4_destroy(v)


dll.luisa_compute_float2_create.restype = c_void_p
dll.luisa_compute_float2_create.argtypes = [c_float, c_float]


def float2_create(v0, v1):
    return dll.luisa_compute_float2_create(v0, v1)


dll.luisa_compute_float2_destroy.restype = None
dll.luisa_compute_float2_destroy.argtypes = [c_void_p]


def float2_destroy(v):
    dll.luisa_compute_float2_destroy(v)


dll.luisa_compute_float3_create.restype = c_void_p
dll.luisa_compute_float3_create.argtypes = [c_float, c_float, c_float]


def float3_create(v0, v1, v2):
    return dll.luisa_compute_float3_create(v0, v1, v2)


dll.luisa_compute_float3_destroy.restype = None
dll.luisa_compute_float3_destroy.argtypes = [c_void_p]


def float3_destroy(v):
    dll.luisa_compute_float3_destroy(v)


dll.luisa_compute_float4_create.restype = c_void_p
dll.luisa_compute_float4_create.argtypes = [c_float, c_float, c_float, c_float]


def float4_create(v0, v1, v2, v3):
    return dll.luisa_compute_float4_create(v0, v1, v2, v3)


dll.luisa_compute_float4_destroy.restype = None
dll.luisa_compute_float4_destroy.argtypes = [c_void_p]


def float4_destroy(v):
    dll.luisa_compute_float4_destroy(v)


dll.luisa_compute_bool2_create.restype = c_void_p
dll.luisa_compute_bool2_create.argtypes = [c_int, c_int]


def bool2_create(v0, v1):
    return dll.luisa_compute_bool2_create(v0, v1)


dll.luisa_compute_bool2_destroy.restype = None
dll.luisa_compute_bool2_destroy.argtypes = [c_void_p]


def bool2_destroy(v):
    dll.luisa_compute_bool2_destroy(v)


dll.luisa_compute_bool3_create.restype = c_void_p
dll.luisa_compute_bool3_create.argtypes = [c_int, c_int, c_int]


def bool3_create(v0, v1, v2):
    return dll.luisa_compute_bool3_create(v0, v1, v2)


dll.luisa_compute_bool3_destroy.restype = None
dll.luisa_compute_bool3_destroy.argtypes = [c_void_p]


def bool3_destroy(v):
    dll.luisa_compute_bool3_destroy(v)


dll.luisa_compute_bool4_create.restype = c_void_p
dll.luisa_compute_bool4_create.argtypes = [c_int, c_int, c_int, c_int]


def bool4_create(v0, v1, v2, v3):
    return dll.luisa_compute_bool4_create(v0, v1, v2, v3)


dll.luisa_compute_bool4_destroy.restype = None
dll.luisa_compute_bool4_destroy.argtypes = [c_void_p]


def bool4_destroy(v):
    dll.luisa_compute_bool4_destroy(v)


dll.luisa_compute_float2x2_create.restype = c_void_p
dll.luisa_compute_float2x2_create.argtypes = [c_float, c_float, c_float, c_float]


def float2x2_create(m00, m01, m10, m11):
    return dll.luisa_compute_float2x2_create(m00, m01, m10, m11)


dll.luisa_compute_float2x2_destroy.restype = None
dll.luisa_compute_float2x2_destroy.argtypes = [c_void_p]


def float2x2_destroy(m):
    dll.luisa_compute_float2x2_destroy(m)


dll.luisa_compute_float3x3_create.restype = c_void_p
dll.luisa_compute_float3x3_create.argtypes = [c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]


def float3x3_create(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    return dll.luisa_compute_float3x3_create(m00, m01, m02, m10, m11, m12, m20, m21, m22)


dll.luisa_compute_float3x3_destroy.restype = None
dll.luisa_compute_float3x3_destroy.argtypes = [c_void_p]


def float3x3_destroy(m):
    dll.luisa_compute_float3x3_destroy(m)


dll.luisa_compute_float4x4_create.restype = c_void_p
dll.luisa_compute_float4x4_create.argtypes = [c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]


def float4x4_create(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33):
    return dll.luisa_compute_float4x4_create(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33)


dll.luisa_compute_float4x4_destroy.restype = None
dll.luisa_compute_float4x4_destroy.argtypes = [c_void_p]


def float4x4_destroy(m):
    dll.luisa_compute_float4x4_destroy(m)
