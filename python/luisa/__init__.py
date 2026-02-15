"""
LuisaCompute Python DSL v2.

This package provides a modern, high-performance DSL for LuisaCompute,
allowing GPU programming directly in Python with a focus on ease of use,
performance, and advanced meta-programming features.
"""

from __future__ import annotations

# Export core DSL components
from .lang.staged import StagedFunction, kernel, callable, static_range, unrolled, UnrolledRange
from .lang.ops import StaticIf, StaticWhile
from .lang.builder import Builder, set_current_builder
from .lang.ir import (
    Op, Module, Function,
    Value, ConstantValue, ArgumentValue, InstructionValue
)
from .lang.type import (
    # Base types
    Type, Scalar, Vector, Matrix, Array, Struct, Ref,
    Buffer, Texture2D, Texture3D, BindlessArray, Accel, RayQuery,
    # Type aliases
    Bool, Byte, UByte, Short, UShort, Int, UInt, Long, ULong, Half, Float, Double,
    Bool2, Bool3, Bool4, Byte2, Byte3, Byte4, UByte2, UByte3, UByte4,
    Short2, Short3, Short4, UShort2, UShort3, UShort4,
    Int2, Int3, Int4, UInt2, UInt3, UInt4,
    Long2, Long3, Long4, ULong2, ULong3, ULong4,
    Half2, Half3, Half4, Float2, Float3, Float4, Double2, Double3, Double4,
    Float2x2, Float3x3, Float4x4, Double2x2, Double3x3, Double4x4, Half2x2, Half3x3, Half4x4,
    # Internal aliases
    bool_t, byte_t, ubyte_t, short_t, ushort_t, int_t, uint_t, long_t, ulong_t, half_t, float_t, double_t,
    bool2, bool3, bool4, byte2, byte3, byte4, ubyte2, ubyte3, ubyte4,
    short2, short3, short4, ushort2, ushort3, ushort4,
    int2, int3, int4, uint2, uint3, uint4,
    long2, long3, long4, ulong2, ulong3, ulong4,
    half2, half3, half4, float2, float3, float4, double2, double3, double4,
    float2x2, float3x3, float4x4, double2x2, double3x3, double4x4, half2x2, half3x3, half4x4,
    # Utilities
    get_alignment, is_data_type, struct, get_element_type, get_length,
    is_scalar_type, is_vector_type, is_matrix_type, is_arithmetic_type,
    is_integer_type, is_float_type, is_bool_type, is_resource_type,
    promote_types, get_broadcast_type,
)

# Export builtins
from .lang.builtin import *

# Export router utilities
from .lang.router import (
    router, RoutedFunction, 
    is_constant_value, extract_constant_value,
    VectorValue
)

# Export codegen utilities
from .codegen.pretty_printer import pprint
from .codegen.json_serializer import serialize_function, serialize_module

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
    "Buffer", "Texture2D", "Texture3D", "BindlessArray", "Accel", "RayQuery",
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
    "router", "RoutedFunction", "is_constant_value", "extract_constant_value", "VectorValue",
    
    # Builtins
]
from .lang import builtin
__all__ += builtin.__all__
