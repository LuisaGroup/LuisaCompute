"""
Type system for the LuisaCompute Python DSL v2.

This module defines all types used in the DSL, including scalar types,
vector types, matrix types, arrays, structs, and resource types.
"""

from __future__ import annotations
from typing import Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
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

    def __call__(self, arg: Any) -> Any:
        """Support casting syntax like Float(x)."""
        from .ir import Value
        from .builder import get_current_builder

        builder = get_current_builder()

        if not isinstance(arg, Value):
            from .multistage import to_ir_value
            arg = to_ir_value(builder, arg)

        return builder.cast(arg, self)


class Ref:
    """Reference type marker (e.g., for mutable function arguments)."""

    def __class_getitem__(cls, item):
        return cls


# Helper for class-level properties
class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


@dataclass(frozen=True)
class Scalar(Type):
    """Scalar type."""
    dtype: ScalarType

    def __repr__(self) -> str:
        mapping = {
            ScalarType.BOOL: "i1",
            ScalarType.INT8: "i8",
            ScalarType.UINT8: "u8",
            ScalarType.INT16: "i16",
            ScalarType.UINT16: "u16",
            ScalarType.INT32: "i32",
            ScalarType.UINT32: "u32",
            ScalarType.INT64: "i64",
            ScalarType.UINT64: "u64",
            ScalarType.FLOAT16: "f16",
            ScalarType.FLOAT32: "f32",
            ScalarType.FLOAT64: "f64",
        }
        return mapping.get(self.dtype, self.dtype.name.lower())

    # Predefined scalar type constructors (as class properties)
    @classproperty
    def bool(cls) -> Scalar:
        return cls(ScalarType.BOOL)

    @classproperty
    def byte(cls) -> Scalar:
        return cls(ScalarType.INT8)

    @classproperty
    def ubyte(cls) -> Scalar:
        return cls(ScalarType.UINT8)

    @classproperty
    def short(cls) -> Scalar:
        return cls(ScalarType.INT16)

    @classproperty
    def ushort(cls) -> Scalar:
        return cls(ScalarType.UINT16)

    @classproperty
    def int(cls) -> Scalar:
        return cls(ScalarType.INT32)

    @classproperty
    def uint(cls) -> Scalar:
        return cls(ScalarType.UINT32)

    @classproperty
    def long(cls) -> Scalar:
        return cls(ScalarType.INT64)

    @classproperty
    def ulong(cls) -> Scalar:
        return cls(ScalarType.UINT64)

    @classproperty
    def half(cls) -> Scalar:
        return cls(ScalarType.FLOAT16)

    @classproperty
    def float(cls) -> Scalar:
        return cls(ScalarType.FLOAT32)

    @classproperty
    def double(cls) -> Scalar:
        return cls(ScalarType.FLOAT64)


@dataclass(frozen=True)
class Vector(Type):
    """Vector type (e.g., Float3, Int4)."""
    element: Scalar
    size: int  # 2, 3, or 4

    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Vector size must be 2, 3, or 4, got {self.size}")

    def __repr__(self) -> str:
        return f"<{self.size} x {self.element}>"

    def __class_getitem__(cls, item):
        """Support Vector[type, dim] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Vector requires [type, size]")
        return cls(element=item[0], size=item[1])


@dataclass(frozen=True)
class Matrix(Type):
    """Matrix type (e.g., Float3x3)."""
    element: Scalar  # typically Float32
    size: int  # 2, 3, or 4

    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Matrix size must be 2, 3, or 4, got {self.size}")

    def __repr__(self) -> str:
        return f"[ {self.size} x <{self.size} x {self.element}> ]"

    def __class_getitem__(cls, item):
        """Support Matrix[type, dim] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Matrix requires [type, size]")
        return cls(element=item[0], size=item[1])


@dataclass(frozen=True)
class Array(Type):
    """Fixed-size array type."""
    element: Type
    size: int

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(f"Array size must be positive, got {self.size}")

    def __repr__(self) -> str:
        return f"[{self.size} x {self.element}]"

    def __class_getitem__(cls, item):
        """Support Array[type, size] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Array requires [type, size]")
        return cls(element=item[0], size=item[1])


@dataclass(frozen=True)
class Struct(Type):
    """Struct type."""
    name: str
    fields: tuple[tuple[str, Type], ...]
    alignment: int = 4

    def __repr__(self) -> str:
        field_types = [str(typ) for name, typ in self.fields]
        return f"{{ {', '.join(field_types)} }}"

    def __class_getitem__(cls, items):
        """
        Support Struct[T1, T2, ..., align] syntax for anonymous structs.
        If the last item is an integer, it's used as alignment.
        """
        if not isinstance(items, tuple):
            items = (items,)
        
        fields = []
        alignment = None
        
        # Check if last item is alignment
        if len(items) > 0 and isinstance(items[-1], int):
            alignment = items[-1]
            types = items[:-1]
        else:
            types = items
            
        for i, t in enumerate(types):
            if not is_data_type(t):
                raise TypeError(f"Struct member must be a data type, got {t}")
            fields.append((f"_{i}", t))
            
        if alignment is None:
            alignment = max((get_alignment(t) for _, t in fields), default=4)
            
        return cls(name="anonymous_struct", fields=tuple(fields), alignment=alignment)

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

    def __class_getitem__(cls, item):
        """Support Buffer[Float] syntax."""
        return cls(element=item)


@dataclass(frozen=True)
class Texture2D(Type):
    """2D texture type."""
    element: Scalar

    def __repr__(self) -> str:
        return f"texture2d<{self.element}>"

    def __class_getitem__(cls, item):
        """Support Texture2D[Float] syntax."""
        return cls(element=item)


@dataclass(frozen=True)
class Texture3D(Type):
    """3D texture type."""
    element: Scalar

    def __repr__(self) -> str:
        return f"texture3d<{self.element}>"

    def __class_getitem__(cls, item):
        """Support Texture3D[Float] syntax."""
        return cls(element=item)


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
        return f"{ret_str} ({arg_str})"

    def __class_getitem__(cls, item):
        """Support Callable[[arg_types], ret_type] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Callable requires [[arg_types], ret_type]")
        arg_types, ret_type = item
        if isinstance(arg_types, list):
            arg_types = tuple(arg_types)
        else:
            arg_types = (arg_types,)
        return cls(arg_types=arg_types, ret_type=ret_type)


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
# Predefined type aliases for convenience (aligned with var.h)
# ============================================================================

# Scalar types
Bool = Scalar.bool
Byte = Scalar.byte
UByte = Scalar.ubyte
Short = Scalar.short
UShort = Scalar.ushort
Int = Scalar.int
UInt = Scalar.uint
Long = Scalar.long
ULong = Scalar.ulong
Half = Scalar.half
Float = Scalar.float
Double = Scalar.double

# Vector types
Bool2, Bool3, Bool4 = Vector[Bool, 2], Vector[Bool, 3], Vector[Bool, 4]
Byte2, Byte3, Byte4 = Vector[Byte, 2], Vector[Byte, 3], Vector[Byte, 4]
UByte2, UByte3, UByte4 = Vector[UByte, 2], Vector[UByte, 3], Vector[UByte, 4]
Short2, Short3, Short4 = Vector[Short, 2], Vector[Short, 3], Vector[Short, 4]
UShort2, UShort3, UShort4 = Vector[UShort, 2], Vector[UShort, 3], Vector[UShort, 4]
Int2, Int3, Int4 = Vector[Int, 2], Vector[Int, 3], Vector[Int, 4]
UInt2, UInt3, UInt4 = Vector[UInt, 2], Vector[UInt, 3], Vector[UInt, 4]
Long2, Long3, Long4 = Vector[Long, 2], Vector[Long, 3], Vector[Long, 4]
ULong2, ULong3, ULong4 = Vector[ULong, 2], Vector[ULong, 3], Vector[ULong, 4]
Half2, Half3, Half4 = Vector[Half, 2], Vector[Half, 3], Vector[Half, 4]
Float2, Float3, Float4 = Vector[Float, 2], Vector[Float, 3], Vector[Float, 4]
Double2, Double3, Double4 = Vector[Double, 2], Vector[Double, 3], Vector[Double, 4]

# Matrix types
Float2x2, Float3x3, Float4x4 = Matrix[Float, 2], Matrix[Float, 3], Matrix[Float, 4]
Double2x2, Double3x3, Double4x4 = Matrix[Double, 2], Matrix[Double, 3], Matrix[Double, 4]
Half2x2, Half3x3, Half4x4 = Matrix[Half, 2], Matrix[Half, 3], Matrix[Half, 4]

# Lowercase aliases for internal use (with _t postfix)
bool_t = Bool
byte_t, ubyte_t = Byte, UByte
short_t, ushort_t = Short, UShort
int_t, uint_t = Int, UInt
long_t, ulong_t = Long, ULong
half_t, float_t, double_t = Half, Float, Double


# ============================================================================
# Type utility functions
# ============================================================================

def get_element_type(t: Type) -> Type:
    """Get the element type of a vector, matrix, or array."""
    if isinstance(t, Vector):
        return t.element
    elif isinstance(t, Matrix):
        return Vector[t.element, t.size]
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


def get_alignment(t: Type) -> int:
    """Get the alignment of a type in bytes."""
    if isinstance(t, Scalar):
        mapping = {
            ScalarType.BOOL: 1,
            ScalarType.INT8: 1, ScalarType.UINT8: 1,
            ScalarType.INT16: 2, ScalarType.UINT16: 2,
            ScalarType.INT32: 4, ScalarType.UINT32: 4,
            ScalarType.INT64: 8, ScalarType.UINT64: 8,
            ScalarType.FLOAT16: 2,
            ScalarType.FLOAT32: 4,
            ScalarType.FLOAT64: 8,
        }
        return mapping.get(t.dtype, 4)
    if isinstance(t, Vector):
        # 2 elements -> 2 * align, 3 or 4 elements -> 4 * align
        base = get_alignment(t.element)
        return (2 if t.size == 2 else 4) * base
    if isinstance(t, Matrix):
        # Alignment of a matrix is the alignment of its columns
        return get_alignment(Vector[t.element, t.size])
    if isinstance(t, Array):
        return get_alignment(t.element)
    if isinstance(t, Struct):
        return t.alignment
    return 4


def is_scalar_type(t: Type) -> bool:
    """Check if type is a scalar."""
    return isinstance(t, Scalar)


def is_vector_type(t: Type) -> bool:
    """Check if type is a vector."""
    return isinstance(t, Vector)


def is_matrix_type(t: Type) -> bool:
    """Check if type is a matrix."""
    return isinstance(t, Matrix)


def is_data_type(t: Type) -> bool:
    """Check if type is a data type (scalar, vector, matrix, array, struct)."""
    return isinstance(t, (Scalar, Vector, Matrix, Array, Struct))


def is_arithmetic_type(t: Type) -> bool:
    """Check if type is arithmetic (scalar or vector of int/UInt/float)."""
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
        return Vector[promoted, t1.size]

    # Both scalars - use type precedence
    if isinstance(t1, Scalar) and isinstance(t2, Scalar):
        precedence = [
            ScalarType.BOOL,
            ScalarType.INT8, ScalarType.UINT8,
            ScalarType.INT16, ScalarType.UINT16,
            ScalarType.INT32, ScalarType.UINT32,
            ScalarType.INT64, ScalarType.UINT64,
            ScalarType.FLOAT16, ScalarType.FLOAT32,
            ScalarType.FLOAT64,
        ]
        idx1 = precedence.index(t1.dtype)
        idx2 = precedence.index(t2.dtype)
        return t1 if idx1 > idx2 else t2

    raise TypeError(f"Cannot promote types {t1} and {t2}")


def python_type_to_dsl(py_type: type) -> Optional[Type]:
    """Convert a Python type to a DSL type."""
    mapping = {
        bool: Bool,
        int: Int,
        float: Float,
    }
    return mapping.get(py_type)


def value_to_type(value: Any) -> Optional[Type]:
    """Infer DSL type from a Python value."""
    if value is None:
        return Void()
    if isinstance(value, bool):
        return Bool
    if isinstance(value, int):
        return Int
    if isinstance(value, float):
        return Float
    return None


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


# ============================================================================
# Struct Decorator
# ============================================================================

# Registry of defined struct types
_struct_registry: dict[str, type] = {}


def _struct_impl(cls: type, align: Optional[int] = None) -> type:
    """Implementation of the struct decorator."""
    # Get annotations
    annotations = getattr(cls, '__annotations__', {})
    if not annotations:
        raise TypeError(f"Struct {cls.__name__} must have annotated fields")

    # Build field list
    fields = []
    for name, ann_type in annotations.items():
        dsl_type = None
        if isinstance(ann_type, type):
            # Convert Python type to DSL type
            dsl_type = python_type_to_dsl(ann_type)
            if dsl_type is None:
                raise TypeError(f"Field '{name}' has unsupported type: {ann_type}")
        elif isinstance(ann_type, Type):
            dsl_type = ann_type
        else:
            raise TypeError(f"Field '{name}' has unsupported type annotation: {ann_type}")
            
        if not is_data_type(dsl_type):
            raise TypeError(f"Struct member '{name}' must be a data type, got {dsl_type}")
            
        fields.append((name, dsl_type))

    # Compute alignment if not specified
    if align is None:
        align = max((get_alignment(typ) for _, typ in fields), default=4)

    # Create Struct type
    struct_type = Struct(
        name=cls.__name__,
        fields=tuple(fields),
        alignment=align
    )

    # Store in registry
    _struct_registry[cls.__name__] = cls

    # Attach type info to class
    cls._dsl_type = struct_type  # pylint: disable=protected-access
    cls._dsl_fields = {name: typ for name, typ in fields}  # pylint: disable=protected-access

    # Add methods
    @classmethod
    def get_dsl_type(cls) -> Struct:
        """Get the DSL type for this struct."""
        return cls._dsl_type  # pylint: disable=protected-access

    cls.get_dsl_type = get_dsl_type

    return cls


def struct(arg=None, *, align: Optional[int] = None):
    """
    Decorator to define a struct type for the DSL.
    
    Usage:
        @struct
        class Particle:
            position: Float3
            velocity: Float3
            mass: Float
            
        @struct(align=16)
        class AlignedStruct:
            x: Int
    """
    if arg is not None and not isinstance(arg, int) and callable(arg):
        # Used as @struct
        return _struct_impl(arg)
    
    # Used as @struct(align=n)
    def decorator(cls):
        return _struct_impl(cls, align=align)
    return decorator


def get_struct_type(name: str) -> Optional[type]:
    """Get a registered struct type by name."""
    return _struct_registry.get(name)
