"""
Core language components for the LuisaCompute Python DSL v2.
"""

from __future__ import annotations

# Re-export core decorators and types
from .jit import kernel, callable, static_range, unrolled, UnrolledRange
from .ops import StaticIf, StaticWhile
from .types import (
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
    # Utilities
    get_alignment, is_data_type, struct, get_element_type, get_length,
    is_scalar_type, is_vector_type, is_matrix_type, is_arithmetic_type,
    is_integer_type, is_float_type, is_bool_type, is_resource_type,
    promote_types, get_broadcast_type,
    Const, static, Shared, is_const_value
)

# Re-export router
from .router import router, RoutedFunction

# Re-export inspect utilities
from .inspect import (
    get_ir_ast,
    analyze_control_flow, count_instructions, find_operations
)

# Export builtins - AT THE END TO AVOID CIRCULAR IMPORTS
from . import builtins

__all__ = [
    "kernel", "callable", "static_range", "unrolled", "UnrolledRange",
    "StaticIf", "StaticWhile",
    "Type", "Scalar", "Vector", "Matrix", "Array", "Struct", "Ref",
    "Buffer", "Texture2D", "Texture3D", "BindlessArray", "Accel", "RayQuery",
    "Bool", "Byte", "UByte", "Short", "UShort", "Int", "UInt", "Long", "ULong", "Half", "Float", "Double",
    "struct", "get_alignment", "is_data_type", "get_element_type", "get_length",
    "is_scalar_type", "is_vector_type", "is_matrix_type", "is_arithmetic_type",
    "is_integer_type", "is_float_type", "is_bool_type", "is_resource_type",
    "promote_types", "get_broadcast_type",
    "Const", "static", "Shared", "is_const_value",
    "router", "RoutedFunction",
    "get_ir_ast",
    "analyze_control_flow", "count_instructions", "find_operations",
    "builtins"
]
