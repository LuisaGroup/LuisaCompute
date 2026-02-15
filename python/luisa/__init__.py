"""
LuisaCompute Python DSL v2

A multistage programming system for GPU/CPU compute shaders with complete
type hinting support and automatic constant folding.
"""

# lang
from .lang.ir import (
    IROp, Value, ConstantValue, ArgumentValue, InstructionValue,
    IRInstruction, IRBasicBlock, IRFunction, IRModule,
)
from .lang.builder import IRBuilder
from .lang.multistage import StagedFunction, kernel, callable, static_range, unrolled, StaticIf, StaticWhile
from .lang.types import (
    # Base types
    Type, Scalar, Vector, Matrix, Array, Struct,
    Buffer, Texture2D, Texture3D, BindlessArray, Accel, RayQuery, Callable, Void, Ref,

    # Scalar types
    bool_, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    float16, float32, float64,

    # Vector types
    int2, int3, int4, uint2, uint3, uint4,
    float2, float3, float4, bool2, bool3, bool4,
    half2, half3, half4,
    short2, short3, short4, ushort2, ushort3, ushort4,
    long2, long3, long4, ulong2, ulong3, ulong4,

    # Matrix types
    float2x2, float3x3, float4x4,

    # Utilities
    get_element_type, get_length,
    is_scalar_type, is_vector_type, is_integer_type, is_float_type,
    promote_types,
)
# Struct decorator
from .lang.types import struct

# lang - utilities
from .lang.util import UnrolledRange

# Code generation
from .codegen import (
    serialize_function,
    serialize_module,
    pprint,
    pprint_to_file,
)

# Builtins (language operations)
from .lang.builtins import (
    # Math
    sqrt, abs, sin, cos, tan, asin, acos, atan, atan2,
    exp, exp2, log, log2, log10,
    floor, ceil, round, trunc, fract, saturate,
    normalize, length, length_squared,
    min, max, clamp, lerp, step, smoothstep, pow,
    dot, cross, distance, reflect, refract, faceforward,
    transpose, inverse, determinant,
    # Special registers
    dispatch_id, thread_id, block_id, dispatch_size,
    kernel_id, object_id,
    # Synchronization
    sync_block,
    # Type casting
    cast, bitcast,
    # Print
    device_print,
    # Assertions
    assume, device_assert, unreachable,
    # Profiling
    clock,
    # Buffer
    buffer_read, buffer_write, buffer_size, buffer_device_address,
    # Texture2D
    texture2d_read, texture2d_write, texture2d_sample, texture2d_sample_level, texture2d_size,
    # Texture3D
    texture3d_read, texture3d_write, texture3d_sample, texture3d_size,
    # Device address
    device_address_load, device_address_store,
    # Atomic
    atomic_exchange, atomic_compare_exchange,
    atomic_add, atomic_sub,
    atomic_and, atomic_or, atomic_xor,
    atomic_min, atomic_max,
    # Warp
    warp_is_first_active_lane, warp_first_active_lane, warp_active_count_bits,
    warp_sum, warp_product, warp_min, warp_max, warp_all, warp_any, warp_all_equal,
    warp_prefix_sum, warp_prefix_product, warp_prefix_count_bits,
    warp_read_lane, warp_read_first_lane,
    warp_bit_and, warp_bit_or, warp_bit_xor, warp_bit_mask,
    # Ray tracing
    Ray, TriangleHit, ProceduralHit, CommittedHit,
    trace_closest, trace_any, ray_query_all, ray_query_any,
    ray_query_world_space_ray, ray_query_proceed,
    ray_query_committed_hit, ray_query_candidate_triangle_hit, ray_query_candidate_procedural_hit,
    ray_query_commit_triangle, ray_query_commit_procedural, ray_query_terminate,
    accel_instance_transform, accel_instance_user_id, accel_instance_visibility_mask,
    make_ray,
)

# Version
__version__ = "2.0.0-alpha"

__all__ = [
    # lang
    "IROp", "Value", "ConstantValue", "ArgumentValue", "InstructionValue",
    "IRInstruction", "IRBasicBlock", "IRFunction", "IRModule",
    "IRBuilder",
    "StagedFunction", "kernel", "callable", "static_range", "unrolled", "StaticIf", "StaticWhile",
    "Type", "Scalar", "Vector", "Matrix", "Array", "Struct",
    "Buffer", "Texture2D", "Texture3D", "BindlessArray", "Accel", "RayQuery", "Callable", "Void", "Ref",
    "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
    "int2", "int3", "int4", "uint2", "uint3", "uint4",
    "float2", "float3", "float4", "bool2", "bool3", "bool4",
    "half2", "half3", "half4",
    "short2", "short3", "short4", "ushort2", "ushort3", "ushort4",
    "long2", "long3", "long4", "ulong2", "ulong3", "ulong4",
    "float2x2", "float3x3", "float4x4",
    "get_element_type", "get_length",
    "is_scalar_type", "is_vector_type", "is_integer_type", "is_float_type",
    "promote_types",
    "struct",

    # util
    "unrolled", "UnrolledRange",

    # codegen
    "serialize_function", "serialize_module",
    "pprint", "pprint_to_file",

    # builtins
    "sqrt", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "exp2", "log", "log2", "log10",
    "floor", "ceil", "round", "trunc", "fract", "saturate",
    "normalize", "length", "length_squared",
    "min", "max", "clamp", "lerp", "step", "smoothstep", "pow",
    "dot", "cross", "distance", "reflect", "refract", "faceforward",
    "transpose", "inverse", "determinant",
    "dispatch_id", "thread_id", "block_id", "dispatch_size",
    "kernel_id", "object_id",
    "sync_block",
    "cast", "bitcast",
    "device_print",
    "assume", "device_assert", "unreachable",
    "clock",
    "buffer_read", "buffer_write", "buffer_size", "buffer_device_address",
    "texture2d_read", "texture2d_write", "texture2d_sample", "texture2d_sample_level", "texture2d_size",
    "texture3d_read", "texture3d_write", "texture3d_sample", "texture3d_size",
    "device_address_load", "device_address_store",
    "atomic_exchange", "atomic_compare_exchange",
    "atomic_add", "atomic_sub",
    "atomic_and", "atomic_or", "atomic_xor",
    "atomic_min", "atomic_max",
    "warp_is_first_active_lane", "warp_first_active_lane", "warp_active_count_bits",
    "warp_sum", "warp_product", "warp_min", "warp_max", "warp_all", "warp_any", "warp_all_equal",
    "warp_prefix_sum", "warp_prefix_product", "warp_prefix_count_bits",
    "warp_read_lane", "warp_read_first_lane",
    "warp_bit_and", "warp_bit_or", "warp_bit_xor", "warp_bit_mask",
    "Ray", "TriangleHit", "ProceduralHit", "CommittedHit",
    "trace_closest", "trace_any", "ray_query_all", "ray_query_any",
    "ray_query_world_space_ray", "ray_query_proceed",
    "ray_query_committed_hit", "ray_query_candidate_triangle_hit", "ray_query_candidate_procedural_hit",
    "ray_query_commit_triangle", "ray_query_commit_procedural", "ray_query_terminate",
    "accel_instance_transform", "accel_instance_user_id", "accel_instance_visibility_mask",
    "make_ray",
]
