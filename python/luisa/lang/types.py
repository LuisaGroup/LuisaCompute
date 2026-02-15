"""
Type system for the LuisaCompute Python DSL v2.

This module defines all types used in the DSL, including scalar types,
vector types, matrix types, arrays, structs, and resource types.
"""

from __future__ import annotations
import inspect
from typing import Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum, auto

if TYPE_CHECKING:
    from ..transform.op import Op


# ============================================================================
# Scalar Type Enum
# ============================================================================

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


# ============================================================================
# Base Type Class
# ============================================================================

@dataclass(frozen=True)
class Type:
    """Base class for all types in the DSL."""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self, arg: Any) -> Any:
        """Support casting syntax like Float(x)."""
        from ..transform.builder import get_current_builder
        from .ops import to_ir_value

        builder = get_current_builder()

        if not hasattr(arg, 'type') or arg.type is None:
            arg = to_ir_value(arg)

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


# ============================================================================
# Scalar Type
# ============================================================================

@dataclass(frozen=True)
class Scalar(Type):
    """Scalar type."""
    dtype: ScalarType

    def __str__(self) -> str:
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


# ============================================================================
# Vector Type
# ============================================================================

@dataclass(frozen=True)
class Vector(Type):
    """Vector type (e.g., Float3, Int4)."""
    element: Scalar
    size: int  # 2, 3, or 4

    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Vector size must be 2, 3, or 4, got {self.size}")

    def __str__(self) -> str:
        return f"<{self.size} x {self.element}>"

    def __class_getitem__(cls, item):
        """Support Vector[type, dim] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Vector requires [type, size]")
        return cls(element=item[0], size=item[1])
    
    def __call__(self, *components):
        """
        Construct a vector value from components.
        
        Supports:
            Float3(1.0, 2.0, 3.0)  # 3 components
            Float3([1.0, 2.0, 3.0])# From list/tuple
            Float3(1.0)            # All components set to 1.0 (broadcast)
            Float3(x, y, z)        # From DSL values
        
        Returns:
            - A tuple of constants for constant folding (if all args are constants)
            - An IR Value for DSL construction (if any arg is a DSL value)
        """
        from ..transform.ir import Value, ConstantValue
        from ..transform.builder import get_current_builder
        
        # Handle single argument construction from list/tuple
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            components = tuple(components[0])
            
        # Check component count
        if len(components) == 1:
            # Broadcast scalar to all components
            val = components[0]
            if isinstance(val, Value):
                # Emit vector broadcast in IR
                from ..transform.op import Op
                builder = get_current_builder()
                return builder._emit(Op.CALL_BUILTIN, self, [f"make_vector_{self.size}", val])
            # Single constant - replicate to all components
            components = (val,) * self.size
        elif len(components) != self.size:
            raise ValueError(f"Vector<{self.element}, {self.size}> requires {self.size} components, got {len(components)}")
        
        # Check if all components are constants
        if all(not isinstance(c, Value) for c in components):
            # Convert all to the element type
            converted = []
            for c in components:
                if isinstance(c, ConstantValue):
                    converted.append(c.value)
                else:
                    converted.append(c)
            
            # Return as a tuple for constant folding
            return tuple(converted)
        
        # Some components are DSL values - emit IR
        from ..transform.op import Op
        builder = get_current_builder()
        return builder._emit(Op.CALL_BUILTIN, self, [f"make_vector_{self.size}", *components])


# ============================================================================
# Matrix Type
# ============================================================================

@dataclass(frozen=True)
class Matrix(Type):
    """Matrix type (e.g., Float3x3)."""
    element: Scalar  # typically Float32
    size: int  # 2, 3, or 4

    def __post_init__(self):
        if self.size not in (2, 3, 4):
            raise ValueError(f"Matrix size must be 2, 3, or 4, got {self.size}")

    def __str__(self) -> str:
        return f"[{self.size} x <{self.size} x {self.element}>]"

    def __class_getitem__(cls, item):
        """Support Matrix[type, dim] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Matrix requires [type, size]")
        return cls(element=item[0], size=item[1])

    def __call__(self, *components):
        """
        Construct a matrix value from components.
        
        Supports:
            Float2x2(1.0, 2.0, 3.0, 4.0)  # 4 components (row-major)
            Float2x2([1,2,3,4])           # From list/tuple
            Float2x2(1.0)                 # Diagonal elements set to 1.0 (identity * scalar)
            Float2x2(x, y, z, w)          # From DSL values
        """
        from ..transform.ir import Value, ConstantValue
        from ..transform.builder import get_current_builder
        from ..transform.op import Op
        
        # Handle single argument construction from list/tuple
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            components = tuple(components[0])
            
        num_elements = self.size * self.size
        
        if len(components) == 1:
            # Identity * scalar (diagonal)
            val = components[0]
            if isinstance(val, Value):
                return get_current_builder()._emit(Op.CALL_BUILTIN, self, [f"make_matrix_{self.size}", val])
            
            # Constant diagonal
            diag_val = val if not isinstance(val, ConstantValue) else val.value
            result = []
            for r in range(self.size):
                for c in range(self.size):
                    result.append(diag_val if r == c else 0.0)
            return tuple(result)
            
        if len(components) != num_elements:
            raise ValueError(f"Matrix<{self.element}, {self.size}> requires {num_elements} components, got {len(components)}")
            
        # Check if all components are constants
        if all(not isinstance(c, Value) for c in components):
            converted = []
            for c in components:
                converted.append(c.value if isinstance(c, ConstantValue) else c)
            return tuple(converted)
            
        # Emit IR
        return get_current_builder()._emit(Op.CALL_BUILTIN, self, [f"make_matrix_{self.size}", *components])


# ============================================================================
# Array Type
# ============================================================================

@dataclass(frozen=True)
class Array(Type):
    """Fixed-size array type."""
    element: Type
    size: int

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(f"Array size must be positive, got {self.size}")

    def __str__(self) -> str:
        return f"[{self.size} x {self.element}]"

    def __class_getitem__(cls, item):
        """Support Array[type, size] syntax."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Array requires [type, size]")
        return cls(element=item[0], size=item[1])

    def __call__(self, *elements):
        """Construct an array from elements."""
        from ..transform.ir import Value, ConstantValue
        from ..transform.builder import get_current_builder
        from ..transform.op import Op
        
        if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
            elements = tuple(elements[0])
            
        if len(elements) != self.size:
            raise ValueError(f"Array<{self.element}, {self.size}> requires {self.size} elements, got {len(elements)}")
            
        if all(not isinstance(e, Value) for e in elements):
            return tuple(e.value if isinstance(e, ConstantValue) else e for e in elements)
            
        return get_current_builder()._emit(Op.CALL_BUILTIN, self, [f"make_array_{self.size}", *elements])


# ============================================================================
# Struct Type
# ============================================================================

@dataclass(frozen=True)
class Struct(Type):
    """Struct type."""
    name: str
    fields: tuple[tuple[str, Type], ...]
    alignment: int = 4

    def __str__(self) -> str:
        field_types = [str(typ) for name, typ in self.fields]
        return f"{{ {', '.join(field_types)} }}"

    def __call__(self, *args, **kwargs):
        """
        Construct a struct from arguments.
        
        Supports:
            Point(x=1.0, y=2.0)  # Named arguments
            Point(1.0, 2.0)      # Positional arguments
            Point([1.0, 2.0])    # From list/tuple
        """
        from ..transform.ir import Value, ConstantValue
        from ..transform.builder import get_current_builder
        from ..transform.op import Op
        
        # Determine elements based on positional or named arguments
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and not kwargs:
            elements = list(args[0])
        elif args:
            if kwargs:
                raise ValueError("Cannot mix positional and named arguments in struct construction")
            elements = list(args)
        else:
            # Map kwargs to fields
            field_dict = {name: i for i, (name, _) in enumerate(self.fields)}
            elements = [None] * len(self.fields)
            for name, val in kwargs.items():
                if name not in field_dict:
                    raise KeyError(f"Struct {self.name} has no field {name}")
                elements[field_dict[name]] = val
            
            if any(e is None for e in elements):
                missing = [self.fields[i][0] for i, e in enumerate(elements) if e is None]
                raise ValueError(f"Missing values for fields: {', '.join(missing)}")

        if len(elements) != len(self.fields):
            raise ValueError(f"Struct {self.name} requires {len(self.fields)} fields, got {len(elements)}")
            
        if all(not isinstance(e, Value) for e in elements):
            return tuple(e.value if isinstance(e, ConstantValue) else e for e in elements)
            
        return get_current_builder()._emit(Op.CALL_BUILTIN, self, [f"make_struct_{self.name}", *elements])

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

    # Add methods
    @classmethod
    def get_dsl_type(cls) -> Struct:
        """Get the DSL type for this struct."""
        return cls._dsl_type  # pylint: disable=protected-access


# ============================================================================
# Resource Types
# ============================================================================

@dataclass(frozen=True)
class Buffer(Type):
    """Buffer type (GPU memory)."""
    element: Type

    def __str__(self) -> str:
        return f"buffer<{self.element}>"

    def __class_getitem__(cls, item):
        """Support Buffer[Float] syntax."""
        return cls(element=item)


@dataclass(frozen=True)
class Texture2D(Type):
    """2D texture type."""
    element: Scalar

    def __str__(self) -> str:
        return f"texture2d<{self.element}>"

    def __class_getitem__(cls, item):
        """Support Texture2D[Float] syntax."""
        return cls(element=item)


@dataclass(frozen=True)
class Texture3D(Type):
    """3D texture type."""
    element: Scalar

    def __str__(self) -> str:
        return f"texture3d<{self.element}>"

    def __class_getitem__(cls, item):
        """Support Texture3D[Float] syntax."""
        return cls(element=item)


@dataclass(frozen=True)
class BindlessArray(Type):
    """Bindless array type."""

    def __str__(self) -> str:
        return "bindless_array"


@dataclass(frozen=True)
class Accel(Type):
    """Acceleration structure type for ray tracing."""

    def __str__(self) -> str:
        return "accel"


@dataclass(frozen=True)
class RayQuery(Type):
    """Ray query type."""
    query_any: bool  # True for RayQueryAny, False for RayQueryAll

    def __str__(self) -> str:
        return "ray_query_any" if self.query_any else "ray_query_all"


@dataclass(frozen=True)
class Callable(Type):
    """Callable function type."""
    arg_types: tuple[Type, ...]
    ret_type: Optional[Type]

    def __str__(self) -> str:
        arg_str = ', '.join(str(t) for t in self.arg_types)
        ret_str = str(self.ret_type) if self.ret_type is not None else "void"
        return f"({arg_str}) -> {ret_str}"

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


# ============================================================================
# Type Aliases
# ============================================================================

AnyType = Optional[Union[
    Scalar, Vector, Matrix, Array, Struct,
    Buffer, Texture2D, Texture3D, BindlessArray,
    Accel, RayQuery, Callable
]]

# Void is now None
Void = None

# ============================================================================
# Predefined Type Aliases
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

# Lowercase aliases for internal use (with _t postfix for scalars to avoid conflicts)
bool_t = Bool
byte_t, ubyte_t = Byte, UByte
short_t, ushort_t = Short, UShort
int_t, uint_t = Int, UInt
long_t, ulong_t = Long, ULong
half_t, float_t, double_t = Half, Float, Double

# Vector and matrix lowercase aliases (no _t suffix needed as they don't conflict)
bool2, bool3, bool4 = Bool2, Bool3, Bool4
byte2, byte3, byte4 = Byte2, Byte3, Byte4
ubyte2, ubyte3, ubyte4 = UByte2, UByte3, UByte4
short2, short3, short4 = Short2, Short3, Short4
ushort2, ushort3, ushort4 = UShort2, UShort3, UShort4
int2, int3, int4 = Int2, Int3, Int4
uint2, uint3, uint4 = UInt2, UInt3, UInt4
long2, long3, long4 = Long2, Long3, Long4
ulong2, ulong3, ulong4 = ULong2, ULong3, ULong4
half2, half3, half4 = Half2, Half3, Half4
float2, float3, float4 = Float2, Float3, Float4
double2, double3, double4 = Double2, Double3, Double4

float2x2, float3x3, float4x4 = Float2x2, Float3x3, Float4x4
double2x2, double3x3, double4x4 = Double2x2, Double3x3, Double4x4
half2x2, half3x3, half4x4 = Half2x2, Half3x3, Half4x4


# ============================================================================
# Type Conversion Logic
# ============================================================================

def value_to_type(value: Any) -> Optional[Type]:
    """Infer DSL type from a Python value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return Bool
    if isinstance(value, int):
        return Int
    if isinstance(value, float):
        return Float
    return None


def annotation_to_type(ann: Any) -> tuple[Optional[Type], bool]:
    """Convert a Python type annotation to a DSL type and a reference flag."""
    if ann is None or ann is inspect.Parameter.empty:
        return None, False

    # Handle direct type references
    if isinstance(ann, Type):
        return ann, False

    # Handle Python built-in types
    py_type = python_type_to_dsl(ann)
    if py_type is not None:
        return py_type, False

    # Handle generic types like Buffer[Float]
    origin = getattr(ann, '__origin__', None)
    args = getattr(ann, '__args__', None)

    if origin is not None and args is not None:
        # Handle Ref[T]
        if origin.__name__ == 'Ref' or getattr(origin, '__name__', None) == 'Ref':
            elem_type, _ = annotation_to_type(args[0])
            return elem_type, True

        # Handle Buffer[T]
        if origin.__name__ == 'Buffer' or getattr(origin, '__name__', None) == 'buffer':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None:
                return Buffer(element=elem_type), False

        # Handle Texture2D[T]
        if origin.__name__ == 'Texture2D' or getattr(origin, '__name__', None) == 'Texture2D':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None and isinstance(elem_type, Scalar):
                return Texture2D(element=elem_type), False

        # Handle Texture3D[T]
        if origin.__name__ == 'Texture3D' or getattr(origin, '__name__', None) == 'Texture3D':
            elem_type, _ = annotation_to_type(args[0])
            if elem_type is not None and isinstance(elem_type, Scalar):
                return Texture3D(element=elem_type), False

    return None, False


def python_type_to_dsl(py_type: type) -> Optional[Type]:
    """Convert a Python type to a DSL type."""
    mapping = {
        bool: Bool,
        int: Int,
        float: Float,
    }
    return mapping.get(py_type)


# ============================================================================
# Type Utility Functions
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


def get_length(t: Any) -> int:
    """Get the length of a vector, matrix dimension, array size, or struct field count."""
    if isinstance(t, Vector):
        return t.size
    elif isinstance(t, Matrix):
        return t.size * t.size
    elif isinstance(t, Array):
        return t.size
    elif isinstance(t, Scalar):
        return 1
    elif isinstance(t, Struct):
        return len(t.fields)
    elif hasattr(t, '_dsl_type') and isinstance(t._dsl_type, Struct):
        return len(t._dsl_type.fields)
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

    def __init__(self, *args, **kwargs):
        field_names = [f[0] for f in self._dsl_type.fields]
        if args:
            if kwargs:
                raise ValueError("Cannot mix positional and named arguments")
            if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == len(field_names):
                args = tuple(args[0])
            if len(args) != len(field_names):
                raise ValueError(f"Expected {len(field_names)} arguments, got {len(args)}")
            for name, val in zip(field_names, args):
                setattr(self, name, val)
        else:
            for name in field_names:
                if name not in kwargs:
                    raise ValueError(f"Missing argument: {name}")
                setattr(self, name, kwargs[name])
            if len(kwargs) > len(field_names):
                extra = set(kwargs.keys()) - set(field_names)
                raise ValueError(f"Extra arguments: {', '.join(extra)}")

    def to_tuple(self):
        field_names = [f[0] for f in self._dsl_type.fields]
        return tuple(getattr(self, name) for name in field_names)

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.to_tuple() == other.to_tuple()

    def __repr__(self):
        field_names = [f[0] for f in self._dsl_type.fields]
        fields_str = ", ".join(f"{name}={getattr(self, name)!r}" for name in field_names)
        return f"{self.__class__.__name__}({fields_str})"

    cls.__init__ = __init__
    cls.to_tuple = to_tuple
    cls.__eq__ = __eq__
    cls.__repr__ = __repr__

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


# ============================================================================
# Const and Shared Types
# ============================================================================

class Const:
    """
    Mark a value as a compile-time constant.
    
    Usage:
        # With explicit type
        a = Const[Float](1.0)
        b = Const[Int](42)
        
        # Without explicit type (inferred from value)
        c = Const(sin(1.0))
        
        # Multiple values (creates a tuple)
        d = Const[Float](1.0, 2.0, 3.0)
    
    Variables marked with Const are kept as Python values during DSL execution
    and are not converted to DSL variables (no alloca/store).
    """
    
    def __class_getitem__(cls, item):
        """Support Const[Type] syntax - returns a typed Const constructor."""
        return _TypedConst(item)
    
    def __new__(cls, *values):
        """
        Create a const value with type inferred from the value.
        
        If multiple values are provided, they are wrapped in a tuple.
        """
        if len(values) == 1:
            return _ConstValue(values[0])
        else:
            return _ConstValue(values)


class _TypedConst:
    """A const constructor with an explicit type."""
    
    def __init__(self, typ: Type):
        self.type = typ
    
    def __call__(self, *values, **kwargs):
        """
        Create a const value with the specified type.
        
        This validates that the initial values match the target type.
        """
        # Support broadcasting or exact match
        if len(values) == 1 and not kwargs:
            val = values[0]
            # If it's a single value, it's either a scalar broadcast or a tuple/list
            if isinstance(val, (list, tuple)):
                # Exact element count check for aggregate types
                required = get_length(self.type)
                if len(val) != required:
                    raise ValueError(f"Const[{self.type}] requires {required} elements, got {len(val)}")
                
                # Construct the type from the sequence
                result = self.type(val)
                return _ConstValue(result, explicit_type=self.type)
            
            # Scalar broadcast to aggregate is handled by T.__call__
            # Or it's a simple scalar Const[Float](1.0)
            result = self.type(val)
            return _ConstValue(result, explicit_type=self.type)
        else:
            # Multiple arguments or named arguments - must match required length
            required = get_length(self.type)
            
            # For structs, we might have kwargs
            if kwargs:
                result = self.type(*values, **kwargs)
            else:
                if len(values) != required:
                    raise ValueError(f"Const[{self.type}] requires {required} elements, got {len(values)}")
                result = self.type(*values)
                
            return _ConstValue(result, explicit_type=self.type)


class _ConstValue:
    """
    Wrapper for a compile-time constant value.
    
    This is used internally to mark values that should not be converted
to DSL variables.
    """
    
    def __init__(self, value: Any, explicit_type: Optional[Type] = None):
        self._raw_value = value
        self._explicit_type = explicit_type
        
        # Unwrap nested ConstValue
        if isinstance(value, _ConstValue):
            self._raw_value = value._raw_value
            if explicit_type is None:
                self._explicit_type = value._explicit_type
        
        # Unwrap ConstantValue
        elif hasattr(value, 'type') and hasattr(value, 'value'):
            # It's a ConstantValue-like object
            self._raw_value = value.value
            if explicit_type is None:
                self._explicit_type = value.type
    
    @property
    def value(self) -> Any:
        """Get the raw Python value."""
        return self._raw_value
    
    @property
    def dsl_type(self) -> Optional[Type]:
        """Get the explicit DSL type if specified."""
        return self._explicit_type
    
    def __repr__(self) -> str:
        if self._explicit_type:
            return f"Const[{self._explicit_type}]({self._raw_value!r})"
        return f"Const({self._raw_value!r})"
    
    # Delegate arithmetic to the raw value
    def __add__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value + other)
    
    def __radd__(self, other):
        return _ConstValue(other + self._raw_value)
    
    def __sub__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value - other)
    
    def __rsub__(self, other):
        return _ConstValue(other - self._raw_value)
    
    def __mul__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value * other)
    
    def __rmul__(self, other):
        return _ConstValue(other * self._raw_value)
    
    def __truediv__(self, other):
        if isinstance(other, _ConstValue):
            other = other._raw_value
        return _ConstValue(self._raw_value / other)
    
    def __rtruediv__(self, other):
        return _ConstValue(other / self._raw_value)
    
    def __neg__(self):
        return _ConstValue(-self._raw_value)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        return _ConstValue(abs(self._raw_value))
    
    # Comparisons
    def __eq__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value == other._raw_value
        return self._raw_value == other
    
    def __ne__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value != other._raw_value
        return self._raw_value != other
    
    def __lt__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value < other._raw_value
        return self._raw_value < other
    
    def __le__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value <= other._raw_value
        return self._raw_value <= other
    
    def __gt__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value > other._raw_value
        return self._raw_value > other
    
    def __ge__(self, other):
        if isinstance(other, _ConstValue):
            return self._raw_value >= other._raw_value
        return self._raw_value >= other
    
    def __bool__(self):
        return bool(self._raw_value)
    
    def __float__(self):
        return float(self._raw_value)
    
    def __int__(self):
        return int(self._raw_value)
    
    def __getitem__(self, index):
        """Allow indexing into const tuples/arrays."""
        return _ConstValue(self._raw_value[index])


def static(*values):
    """
    Mark value(s) as statically evaluated (compile-time constants).
    
    This is a shorthand for Const() that makes it explicit the value is
    evaluated at compile time (statically) rather than on the device.
    
    Usage:
        # Single value
        a = static(sin(1.0))
        
        # Multiple values
        b, c = static(1.0, 2.0)
    
    Variables marked with static() are kept as Python values and are not
    converted to DSL variables (no alloca/store).
    """
    if len(values) == 1:
        return _ConstValue(values[0])
    else:
        return tuple(_ConstValue(v) for v in values)


class Shared:
    """
    Mark a variable as shared memory (GPU threadgroup memory).
    
    Usage:
        # Shared memory array
        shared_buf = Shared[Float, 256]  # 256 floats of shared memory
        
        # Shared memory with initialization
        shared_val = Shared[Float](0.0)
    
    This is a placeholder for future implementation of GPU shared memory.
    Currently, it creates a normal DSL variable.
    """
    
    def __class_getitem__(cls, params):
        """
        Support Shared[Type] or Shared[Type, size] syntax.
        
        Returns a Shared memory constructor.
        """
        if isinstance(params, tuple):
            if len(params) == 2:
                typ, size = params
                return _SharedConstructor(typ, size)
            else:
                raise TypeError("Shared[...] expects 1 or 2 parameters: Shared[Type] or Shared[Type, size]")
        else:
            # Just the type, no size
            return _SharedConstructor(params, None)


class _SharedConstructor:
    """Constructor for shared memory variables."""
    
    def __init__(self, typ: Type, size: Optional[int] = None):
        self.type = typ
        self.size = size
    
    def __call__(self, *initial_values):
        """
        Create a shared memory variable.
        
        For now, this creates a normal DSL variable (placeholder implementation).
        In the future, this will allocate GPU shared memory.
        """
        # TODO: Implement actual shared memory allocation
        # For now, just mark it as a shared variable
        return _SharedValue(self.type, self.size, initial_values)


class _SharedValue:
    """Wrapper for a shared memory variable."""
    
    def __init__(self, typ: Type, size: Optional[int], initial_values: tuple):
        self.type = typ
        self.size = size
        self.initial_values = initial_values
        self._is_shared = True
    
    def __repr__(self) -> str:
        if self.size:
            return f"Shared[{self.type}, {self.size}]"
        return f"Shared[{self.type}]"
    
    def __getitem__(self, index):
        """Allow indexing into shared arrays."""
        # TODO: Implement shared memory access
        pass


def is_const_value(val: Any) -> bool:
    """Check if a value is a compile-time constant (Const or _ConstValue)."""
    return isinstance(val, (_ConstValue, Const))


def extract_const_value(val: Any) -> Any:
    """Extract the raw Python value from a const wrapper."""
    if isinstance(val, _ConstValue):
        return val.value
    return val
