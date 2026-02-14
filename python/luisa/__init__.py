"""
LuisaCompute Python DSL v2

A multistage programming system for GPU/CPU compute shaders with complete
type hinting support and automatic constant folding.
"""

# Type system
from .dsl_types import (
    # Base types
    Type, Scalar, Vector, Matrix, Array, Struct,
    Buffer, Texture2D, Texture3D, BindlessArray, Accel, RayQuery, Callable, Void,
    
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

# IR
from .ir import (
    IROp, Value, ConstantValue, ArgumentValue, InstructionValue,
    IRInstruction, IRBasicBlock, IRFunction, IRModule,
)

# Builder
from .builder import IRBuilder

# Staged functions
from .staged import StagedFunction, kernel, callable

# Builtins
from .builtins import (
    sqrt, abs, sin, cos, tan, asin, acos, atan, atan2,
    exp, exp2, log, log2, log10,
    floor, ceil, round, trunc, fract, saturate,
    normalize, length, length_squared,
    min, max, clamp, lerp, step, smoothstep, pow,
    dot, cross, distance, reflect, refract, faceforward,
    transpose, inverse, determinant,
)

# Version
__version__ = "2.0.0-alpha"

__all__ = [
    # Types
    "Type", "Scalar", "Vector", "Matrix", "Array", "Struct",
    "Buffer", "Texture2D", "Texture3D", "BindlessArray", "Accel", "RayQuery", "Callable", "Void",
    
    # Scalar types
    "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
    
    # Vector types
    "int2", "int3", "int4", "uint2", "uint3", "uint4",
    "float2", "float3", "float4", "bool2", "bool3", "bool4",
    "half2", "half3", "half4",
    "short2", "short3", "short4", "ushort2", "ushort3", "ushort4",
    "long2", "long3", "long4", "ulong2", "ulong3", "ulong4",
    
    # Matrix types
    "float2x2", "float3x3", "float4x4",
    
    # Utilities
    "get_element_type", "get_length",
    "is_scalar_type", "is_vector_type", "is_integer_type", "is_float_type",
    "promote_types",
    
    # IR
    "IROp", "Value", "ConstantValue", "ArgumentValue", "InstructionValue",
    "IRInstruction", "IRBasicBlock", "IRFunction", "IRModule",
    
    # Builder
    "IRBuilder",
    
    # Staged
    "StagedFunction", "kernel", "callable",
    
    # Builtins - Math
    "sqrt", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "exp2", "log", "log2", "log10",
    "floor", "ceil", "round", "trunc", "fract", "saturate",
    "normalize", "length", "length_squared",
    "min", "max", "clamp", "lerp", "step", "smoothstep", "pow",
    "dot", "cross", "distance", "reflect", "refract", "faceforward",
    "transpose", "inverse", "determinant",
]
