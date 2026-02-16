"""
Core language components for the LuisaCompute Python DSL v2.
"""

from __future__ import annotations

# Export builtins - AT THE END TO AVOID CIRCULAR IMPORTS
from . import builtins
# Re-export inspect utilities
from .inspect import (analyze_control_flow, count_instructions,
                      find_operations, get_ir_ast)
# Re-export core decorators and types
from .jit import UnrolledRange, callable, kernel, static_range, unrolled
from .ops import StaticIf, StaticWhile
# Re-export router
from .router import RoutedFunction, router
from .types import (Accel, Array,  # Base types; Type aliases; Utilities
                    BindlessArray, Bool, Bool2, Bool3, Bool4, Buffer, Byte,
                    Byte2, Byte3, Byte4, Const, Double, Double2, Double2x2,
                    Double3, Double3x3, Double4, Double4x4, Float, Float2,
                    Float2x2, Float3, Float3x3, Float4, Float4x4, Half, Half2,
                    Half2x2, Half3, Half3x3, Half4, Half4x4, Int, Int2, Int3,
                    Int4, Long, Long2, Long3, Long4, Matrix, RayQuery, Ref,
                    Scalar, Shared, Short, Short2, Short3, Short4, Struct,
                    Texture2D, Texture3D, Type, UByte, UByte2, UByte3, UByte4,
                    UInt, UInt2, UInt3, UInt4, ULong, ULong2, ULong3, ULong4,
                    UShort, UShort2, UShort3, UShort4, Vector, get_alignment,
                    get_broadcast_type, get_element_type, get_length,
                    is_arithmetic_type, is_bool_type, is_const_value,
                    is_data_type, is_float_type, is_integer_type,
                    is_matrix_type, is_resource_type, is_scalar_type,
                    is_vector_type, promote_types, static, struct)

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
