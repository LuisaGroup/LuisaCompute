"""
Type system for the LuisaCompute Python DSL v2.

This module defines all types used in the DSL, including scalar types,
vector types, matrix types, arrays, structs, and resource types.
"""

from __future__ import annotations
from typing import Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, auto

if TYPE_CHECKING:
    pass


class ScalarType(Enum):
    """Scalar data types."""
    BOOL = auto()
    INT8 = auto()
    UINT8 = auto()
    INT16 = auto()
    UINT16 = auto()
    INT32 = auto()
    UINT32 = auto()
    INT64 = auto()
    UINT64 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()


# Base type class
@dataclass(frozen=True)
class Type:
    """Base class for all types in the DSL."""
    
    def __repr__(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class Scalar(Type):
    """Scalar type."""
    dtype: ScalarType
    
    def __repr__(self) -> str:
        return f"{self.dtype.name.lower()}"
    
    # Predefined scalar type constructors
    @classmethod
    def bool(cls) -> Scalar:
        return cls(ScalarType.BOOL)
    
    @classmethod
    def int8(cls) -> Scalar:
        return cls(ScalarType.INT8)
    
    @classmethod
    def uint8(cls) -> Scalar:
        return cls(ScalarType.UINT8)
    
    @classmethod
    def int16(cls) -> Scalar:
        return cls(ScalarType.INT16)
    
    @classmethod
    def uint16(cls) -> Scalar:
        return cls(ScalarType.UINT16)
    
    @classmethod
    def int32(cls) -> Scalar:
        return cls(ScalarType.INT32)
    
    @classmethod
    def uint32(cls) -> Scalar:
        return cls(ScalarType.UINT32)
    
    @classmethod
    def int64(cls) -> Scalar:
        return cls(ScalarType.INT64)
    
    @classmethod
    def uint64(cls) -> Scalar:
        return cls(ScalarType.UINT64)
    
    @classmethod
    def float16(cls) -> Scalar:
        return cls(ScalarType.FLOAT16)
    
    @classmethod
    def float32(cls) -> Scalar:
        return cls(ScalarType.FLOAT32)
    
    @classmethod
    def float64(cls) -> Scalar:
        return cls(ScalarType.FLOAT64)


@dataclass(frozen=True)
class Vector(Type):
    """Vector type (e.g., float3, int4)."""
    element: Scalar
    size: int  # 2, 3, or 4
    
    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Vector size must be 2, 3, or 4, got {self.size}")
    
    def __repr__(self) -> str:
        return f"{self.element}[{self.size}]"


@dataclass(frozen=True)
class Matrix(Type):
    """Matrix type (e.g., float3x3)."""
    element: Scalar  # typically float32
    size: int  # 2, 3, or 4
    
    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Matrix size must be 2, 3, or 4, got {self.size}")
    
    def __repr__(self) -> str:
        return f"{self.element}[{self.size}]x[{self.size}]"


@dataclass(frozen=True)
class Array(Type):
    """Fixed-size array type."""
    element: Type
    size: int
    
    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(f"Array size must be positive, got {self.size}")
    
    def __repr__(self) -> str:
        return f"array<{self.element}, {self.size}>"


@dataclass(frozen=True)
class Struct(Type):
    """Struct type."""
    name: str
    fields: tuple[tuple[str, Type], ...]
    alignment: int = 4
    
    def __repr__(self) -> str:
        field_strs = [f"{name}: {typ}" for name, typ in self.fields]
        return f"struct {self.name} {{{', '.join(field_strs)}}}"
    
    def get_field_type(self, field_name: str) -> Optional[Type]:
        """Get the type of a field by name."""
        for name, typ in self.fields:
            if name == field_name:
                return typ
        return None
    
    def get_field_index(self, field_name: str) -> int:
        """Get the index of a field by name."""
        for i, (name, _) in enumerate(self.fields):
            if name == field_name:
                return i
        raise KeyError(f"Field '{field_name}' not found in struct {self.name}")


@dataclass(frozen=True)
class Buffer(Type):
    """Buffer type (GPU memory)."""
    element: Type
    
    def __repr__(self) -> str:
        return f"buffer<{self.element}>"


@dataclass(frozen=True)
class Texture2D(Type):
    """2D texture type."""
    element: Scalar
    
    def __repr__(self) -> str:
        return f"texture2d<{self.element}>"


@dataclass(frozen=True)
class Texture3D(Type):
    """3D texture type."""
    element: Scalar
    
    def __repr__(self) -> str:
        return f"texture3d<{self.element}>"


@dataclass(frozen=True)
class BindlessArray(Type):
    """Bindless array type."""
    
    def __repr__(self) -> str:
        return "bindless_array"


@dataclass(frozen=True)
class Accel(Type):
    """Acceleration structure type for ray tracing."""
    
    def __repr__(self) -> str:
        return "accel"


@dataclass(frozen=True)
class RayQuery(Type):
    """Ray query type."""
    query_any: bool  # True for RayQueryAny, False for RayQueryAll
    
    def __repr__(self) -> str:
        return "ray_query_any" if self.query_any else "ray_query_all"


@dataclass(frozen=True)
class Callable(Type):
    """Callable function type."""
    arg_types: tuple[Type, ...]
    ret_type: Optional[Type]
    
    def __repr__(self) -> str:
        arg_str = ', '.join(str(t) for t in self.arg_types)
        ret_str = str(self.ret_type) if self.ret_type else "void"
        return f"({arg_str}) -> {ret_str}"


@dataclass(frozen=True)
class Void(Type):
    """Void type."""
    
    def __repr__(self) -> str:
        return "void"


# Type alias for any type
AnyType = Union[
    Scalar, Vector, Matrix, Array, Struct,
    Buffer, Texture2D, Texture3D, BindlessArray,
    Accel, RayQuery, Callable, Void
]


# ============================================================================
# Predefined type aliases for convenience
# ============================================================================

# Scalar types
bool_ = Scalar.bool()
int8 = Scalar(ScalarType.INT8)
uint8 = Scalar(ScalarType.UINT8)
int16 = Scalar(ScalarType.INT16)
uint16 = Scalar(ScalarType.UINT16)
int32 = Scalar(ScalarType.INT32)
uint32 = Scalar(ScalarType.UINT32)
int64 = Scalar(ScalarType.INT64)
uint64 = Scalar(ScalarType.UINT64)
float16 = Scalar(ScalarType.FLOAT16)
float32 = Scalar(ScalarType.FLOAT32)
float64 = Scalar(ScalarType.FLOAT64)

# Short aliases
int_ = int32
uint = uint32
float_ = float32

# Common vector types
int2 = Vector(int32, 2)
int3 = Vector(int32, 3)
int4 = Vector(int32, 4)
uint2 = Vector(uint32, 2)
uint3 = Vector(uint32, 3)
uint4 = Vector(uint32, 4)
float2 = Vector(float32, 2)
float3 = Vector(float32, 3)
float4 = Vector(float32, 4)
bool2 = Vector(bool_, 2)
bool3 = Vector(bool_, 3)
bool4 = Vector(bool_, 4)

# Half-precision vector types
half2 = Vector(float16, 2)
half3 = Vector(float16, 3)
half4 = Vector(float16, 4)

# 16-bit integer vector types
short2 = Vector(int16, 2)
short3 = Vector(int16, 3)
short4 = Vector(int16, 4)
ushort2 = Vector(uint16, 2)
ushort3 = Vector(uint16, 3)
ushort4 = Vector(uint16, 4)

# 64-bit integer vector types
long2 = Vector(int64, 2)
long3 = Vector(int64, 3)
long4 = Vector(int64, 4)
ulong2 = Vector(uint64, 2)
ulong3 = Vector(uint64, 3)
ulong4 = Vector(uint64, 4)

# Matrix types
float2x2 = Matrix(float32, 2)
float3x3 = Matrix(float32, 3)
float4x4 = Matrix(float32, 4)


# ============================================================================
# Type utility functions
# ============================================================================

def get_element_type(t: Type) -> Type:
    """Get the element type of a vector, matrix, or array."""
    if isinstance(t, Vector):
        return t.element
    elif isinstance(t, Matrix):
        return Vector(t.element, t.size)
    elif isinstance(t, Array):
        return t.element
    else:
        raise TypeError(f"Type {t} has no element type")


def get_length(t: Type) -> int:
    """Get the length of a vector, matrix dimension, or array size."""
    if isinstance(t, Vector):
        return t.size
    elif isinstance(t, Matrix):
        return t.size
    elif isinstance(t, Array):
        return t.size
    elif isinstance(t, Scalar):
        return 1
    else:
        raise TypeError(f"Type {t} has no length")


def is_scalar_type(t: Type) -> bool:
    """Check if type is a scalar."""
    return isinstance(t, Scalar)


def is_vector_type(t: Type) -> bool:
    """Check if type is a vector."""
    return isinstance(t, Vector)


def is_matrix_type(t: Type) -> bool:
    """Check if type is a matrix."""
    return isinstance(t, Matrix)


def is_arithmetic_type(t: Type) -> bool:
    """Check if type is arithmetic (scalar or vector of int/uint/float)."""
    if isinstance(t, Scalar):
        return t.dtype not in (ScalarType.BOOL,)
    elif isinstance(t, Vector):
        return t.element.dtype not in (ScalarType.BOOL,)
    return False


def is_integer_type(t: Type) -> bool:
    """Check if type is an integer (scalar or vector)."""
    if isinstance(t, Scalar):
        return t.dtype in (
            ScalarType.INT8, ScalarType.UINT8,
            ScalarType.INT16, ScalarType.UINT16,
            ScalarType.INT32, ScalarType.UINT32,
            ScalarType.INT64, ScalarType.UINT64,
        )
    elif isinstance(t, Vector):
        return is_integer_type(t.element)
    return False


def is_float_type(t: Type) -> bool:
    """Check if type is a float (scalar or vector)."""
    if isinstance(t, Scalar):
        return t.dtype in (ScalarType.FLOAT16, ScalarType.FLOAT32, ScalarType.FLOAT64)
    elif isinstance(t, Vector):
        return is_float_type(t.element)
    return False


def is_bool_type(t: Type) -> bool:
    """Check if type is a bool (scalar or vector)."""
    if isinstance(t, Scalar):
        return t.dtype == ScalarType.BOOL
    elif isinstance(t, Vector):
        return is_bool_type(t.element)
    return False


def is_resource_type(t: Type) -> bool:
    """Check if type is a resource type (buffer, texture, etc.)."""
    return isinstance(t, (Buffer, Texture2D, Texture3D, BindlessArray, Accel))


def promote_types(t1: Type, t2: Type) -> Type:
    """Promote two types to a common type for operations."""
    # Same type
    if t1 == t2:
        return t1
    
    # Scalar vs Vector broadcasting
    if isinstance(t1, Scalar) and isinstance(t2, Vector):
        if t1 == t2.element:
            return t2
    if isinstance(t1, Vector) and isinstance(t2, Scalar):
        if t1.element == t2:
            return t1
    
    # Both vectors - must have same size
    if isinstance(t1, Vector) and isinstance(t2, Vector):
        if t1.size != t2.size:
            raise TypeError(f"Cannot promote vectors of different sizes: {t1} and {t2}")
        # Promote element types
        promoted = promote_types(t1.element, t2.element)
        return Vector(promoted, t1.size)
    
    # Both scalars - use type precedence
    if isinstance(t1, Scalar) and isinstance(t2, Scalar):
        precedence = [
            ScalarType.BOOL,
            ScalarType.INT8, ScalarType.UINT8,
            ScalarType.INT16, ScalarType.UINT16,
            ScalarType.INT32, ScalarType.UINT32,
            ScalarType.INT64, ScalarType.UINT64,
            ScalarType.FLOAT16,
            ScalarType.FLOAT32,
            ScalarType.FLOAT64,
        ]
        idx1 = precedence.index(t1.dtype)
        idx2 = precedence.index(t2.dtype)
        return t1 if idx1 > idx2 else t2
    
    raise TypeError(f"Cannot promote types {t1} and {t2}")


def python_type_to_dsl(py_type: type) -> Optional[Type]:
    """Convert a Python type to a DSL type."""
    mapping = {
        bool: bool_,
        int: int32,
        float: float32,
    }
    return mapping.get(py_type)


def get_broadcast_type(t1: Type, t2: Type) -> Optional[Type]:
    """
    Get the broadcast type for two types.
    Returns None if broadcasting is not possible.
    """
    # Same type
    if t1 == t2:
        return t1
    
    # Scalar can broadcast to vector
    if isinstance(t1, Scalar) and isinstance(t2, Vector):
        if t1 == t2.element:
            return t2
    if isinstance(t1, Vector) and isinstance(t2, Scalar):
        if t1.element == t2:
            return t1
    
    return None
