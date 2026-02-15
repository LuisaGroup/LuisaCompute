"""
Language Definition for the LuisaCompute Python DSL v2.

This package contains everything the user interacts with when writing DSL code.
"""

from __future__ import annotations

# Types
from .types import (
    # Base types
    Type, Scalar, Vector, Matrix, Array, Struct, Ref,
    Buffer, Texture2D, Texture3D, BindlessArray, Accel, RayQuery, Callable,
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
    # Const and Shared
    Const, static, Shared, is_const_value, extract_const_value,
)

# JIT compilation
from .jit import (
    StagedFunction, kernel, callable, static_range, unrolled, UnrolledRange,
    Specialization, SpecializedFunctionProxy, StagedFunctionDecorator
)

# Control flow
from .ops import StaticIf, StaticWhile
from .control_flow import IfStmt, WhileStmt, ForRangeStmt, UnrolledForStmt, SwitchStmt

# Router
from .router import (
    router, RoutedFunction, 
    is_constant_value, extract_constant_value,
    is_vector_constant, extract_vector_components
)

# Runtime operators
from .ops import (
    binop, unaryop, compare, boolop,
    if_, switch, for_, while_, loop_scope, while_scope,
    call, subscript, subscript_assign, attribute,
    return_, local_assign, local_var_assign, set_location,
    load, maybe_load, store,
    is_ir_value, to_ir_value, try_to_ir_value,
    LuisaRange
)

# Builtins
from . import builtins

__all__ = [
    # Types
    'Type', 'Scalar', 'Vector', 'Matrix', 'Array', 'Struct', 'Ref',
    'Buffer', 'Texture2D', 'Texture3D', 'BindlessArray', 'Accel', 'RayQuery', 'Callable',
    'Bool', 'Byte', 'UByte', 'Short', 'UShort', 'Int', 'UInt', 'Long', 'ULong', 'Half', 'Float', 'Double',
    'Bool2', 'Bool3', 'Bool4', 'Byte2', 'Byte3', 'Byte4', 'UByte2', 'UByte3', 'UByte4',
    'Short2', 'Short3', 'Short4', 'UShort2', 'UShort3', 'UShort4',
    'Int2', 'Int3', 'Int4', 'UInt2', 'UInt3', 'UInt4',
    'Long2', 'Long3', 'Long4', 'ULong2', 'ULong3', 'ULong4',
    'Half2', 'Half3', 'Half4', 'Float2', 'Float3', 'Float4', 'Double2', 'Double3', 'Double4',
    'Float2x2', 'Float3x3', 'Float4x4', 'Double2x2', 'Double3x3', 'Double4x4', 'Half2x2', 'Half3x3', 'Half4x4',
    'bool_t', 'byte_t', 'ubyte_t', 'short_t', 'ushort_t', 'int_t', 'uint_t', 'long_t', 'ulong_t', 'half_t', 'float_t', 'double_t',
    'bool2', 'bool3', 'bool4', 'byte2', 'byte3', 'byte4', 'ubyte2', 'ubyte3', 'ubyte4',
    'short2', 'short3', 'short4', 'ushort2', 'ushort3', 'ushort4',
    'int2', 'int3', 'int4', 'uint2', 'uint3', 'uint4',
    'long2', 'long3', 'long4', 'ulong2', 'ulong3', 'ulong4',
    'half2', 'half3', 'half4', 'float2', 'float3', 'float4', 'double2', 'double3', 'double4',
    'float2x2', 'float3x3', 'float4x4', 'double2x2', 'double3x3', 'double4x4', 'half2x2', 'half3x3', 'half4x4',
    'get_alignment', 'is_data_type', 'struct', 'get_element_type', 'get_length',
    'is_scalar_type', 'is_vector_type', 'is_matrix_type', 'is_arithmetic_type',
    'is_integer_type', 'is_float_type', 'is_bool_type', 'is_resource_type',
    'promote_types', 'get_broadcast_type',
    
    # Const and Shared
    'Const', 'static', 'Shared', 'is_const_value', 'extract_const_value',
    
    # JIT
    'StagedFunction', 'kernel', 'callable', 'static_range', 'unrolled', 'UnrolledRange',
    'Specialization', 'SpecializedFunctionProxy', 'StagedFunctionDecorator',
    
    # Control flow
    'StaticIf', 'StaticWhile', 'IfStmt', 'WhileStmt', 'ForRangeStmt', 'UnrolledForStmt', 'SwitchStmt',
    
    # Router
    'router', 'RoutedFunction', 'is_constant_value', 'extract_constant_value',
    'is_vector_constant', 'extract_vector_components',
    
    # Runtime operators
    'binop', 'unaryop', 'compare', 'boolop',
    'if_', 'switch', 'for_', 'while_', 'loop_scope', 'while_scope',
    'call', 'subscript', 'subscript_assign', 'attribute',
    'return_', 'local_assign', 'local_var_assign', 'set_location',
    'load', 'maybe_load', 'store',
    'is_ir_value', 'to_ir_value', 'try_to_ir_value', 'LuisaRange',
    
    # Builtins
    'builtins',
]

# Re-export all builtins
__all__ += builtins.__all__
