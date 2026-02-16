"""
LuisaCompute Python DSL v2.

This package provides a modern, high-performance DSL for LuisaCompute,
allowing GPU programming directly in Python with a focus on ease of use,
performance, and advanced meta-programming features.
"""

from __future__ import annotations

# Export builtins - MOVED TO END TO AVOID CIRCULAR IMPORTS
from .lang import builtins
from .lang.builtins import *
# Export core DSL components
from .lang.jit import (StagedFunction, UnrolledRange, callable, kernel,
                       static_range, unrolled)
from .lang.ops import StaticIf, StaticWhile
# Export router utilities
from .lang.router import (RoutedFunction, extract_constant_value,
                          extract_vector_components, is_constant_value,
                          is_vector_constant, router)
# Export Const, static, and Shared
from .lang.types import (  # Base types; Type aliases; Internal aliases; Utilities
    Accel, Array, BindlessArray, Bool, Bool2, Bool3, Bool4, Buffer, Byte,
    Byte2, Byte3, Byte4, Callable, Const, Double, Double2, Double2x2, Double3, Double3x3,
    Double4, Double4x4, Float, Float2, Float2x2, Float3, Float3x3, Float4,
    Float4x4, Half, Half2, Half2x2, Half3, Half3x3, Half4, Half4x4, Int, Int2,
    Int3, Int4, Long, Long2, Long3, Long4, Matrix, RayQuery, Ref, Scalar,
    Shared, Short, Short2, Short3, Short4, Struct, Texture2D, Texture3D, Type,
    UByte, UByte2, UByte3, UByte4, UInt, UInt2, UInt3, UInt4, ULong, ULong2,
    ULong3, ULong4, UShort, UShort2, UShort3, UShort4, Vector, bool2, bool3,
    bool4, bool_t, byte2, byte3, byte4, byte_t, double2, double2x2, double3,
    double3x3, double4, double4x4, double_t, float2, float2x2, float3,
    float3x3, float4, float4x4, float_t, get_alignment, get_broadcast_type,
    get_element_type, get_length, half2, half2x2, half3, half3x3, half4,
    half4x4, half_t, int2, int3, int4, int_t, is_arithmetic_type, is_bool_type,
    is_const_value, is_data_type, is_float_type, is_integer_type,
    is_matrix_type, is_resource_type, is_scalar_type, is_vector_type, long2,
    long3, long4, long_t, promote_types, short2, short3, short4, short_t,
    static, struct, ubyte2, ubyte3, ubyte4, ubyte_t, uint2, uint3, uint4,
    uint_t, ulong2, ulong3, ulong4, ulong_t, ushort2, ushort3, ushort4,
    ushort_t)
# Export codegen utilities
from .printer import pprint
from .serialize import serialize_function, serialize_module
from .transform.builder import Builder, set_current_builder
from .transform.ir import (ArgumentValue, ConstantValue, Function,
                           InstructionValue, Module, Value)
from .transform.op import Op
# Version info
from .version import __version__

__all__ = [
    # JIT
    "kernel", "callable", "StagedFunction", "static_range", "unrolled", "UnrolledRange",
    "StaticIf", "StaticWhile",

    # Builder
    "Builder", "set_current_builder", "Op", "Module", "Function",
    "Value", "ConstantValue", "ArgumentValue", "InstructionValue",

    # Types
    "Type", "Scalar", "Vector", "Matrix", "Array", "Struct", "Ref",
    "Buffer", "Texture2D", "Texture3D", "BindlessArray", "Accel", "RayQuery", "Callable",
    "Bool", "Byte", "UByte", "Short", "UShort", "Int", "UInt", "Long", "ULong", "Half", "Float", "Double",
    "Bool2", "Bool3", "Bool4", "Byte2", "Byte3", "Byte4", "UByte2", "UByte3", "UByte4",
    "Short2", "Short3", "Short4", "UShort2", "UShort3", "UShort4",
    "Int2", "Int3", "Int4", "UInt2", "UInt3", "UInt4",
    "Long2", "Long3", "Long4", "ULong2", "ULong3", "ULong4",
    "Half2", "Half3", "Half4", "Float2", "Float3", "Float4", "Double2", "Double3", "Double4",
    "Float2x2", "Float3x3", "Float4x4", "Double2x2", "Double3x3", "Double4x4", "Half2x2", "Half3x3", "Half4x4",
    "struct", "get_alignment", "is_data_type", "get_element_type", "get_length",
    "is_scalar_type", "is_vector_type", "is_matrix_type", "is_arithmetic_type",
    "is_integer_type", "is_float_type", "is_bool_type", "is_resource_type",
    "promote_types", "get_broadcast_type",

    # Internal aliases
    "bool_t", "byte_t", "ubyte_t", "short_t", "ushort_t", "int_t", "uint_t", "long_t", "ulong_t", "half_t", "float_t", "double_t",
    "bool2", "bool3", "bool4", "byte2", "byte3", "byte4", "ubyte2", "ubyte3", "ubyte4",
    "short2", "short3", "short4", "ushort2", "ushort3", "ushort4",
    "int2", "int3", "int4", "uint2", "uint3", "uint4",
    "long2", "long3", "long4", "ulong2", "ulong3", "ulong4",
    "half2", "half3", "half4", "float2", "float3", "float4", "double2", "double3", "double4",
    "float2x2", "float3x3", "float4x4", "double2x2", "double3x3", "double4x4", "half2x2", "half3x3", "half4x4",

    "pprint", "serialize_function", "serialize_module",

    # Router utilities
    "router", "RoutedFunction", "is_constant_value", "extract_constant_value",
    "is_vector_constant", "extract_vector_components",

    # Compile-time constants
    "Const", "static", "is_const_value",

    # Shared memory
    "Shared",

    # Builtins
]

# Add all builtins to __all__
__all__ += builtins.__all__
