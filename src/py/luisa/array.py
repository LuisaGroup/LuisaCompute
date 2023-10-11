from .dylibs import lcapi
from .types import dtype_of, to_lctype, nameof
from .atomic import int_atomic_functions, float_atomic_functions
from .types import uint, uint, short, ushort
class Array:
    def __init__(self, arr):
        if type(arr) is Array:
            self.arrayType = arr.arrayType
            self.values = arr.values.copy()
        else:
            self.arrayType = deduce_array_type(arr)
            self.values = list(arr)

    def __len__(self):
        return self.arrayType.size

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, value):
        assert dtype_of(value) == self.arrayType.dtype
        self.values[idx] = value

    def copy(self):
        return Array(self)

    def to_bytes(self):
        packed_bytes = b''
        for x in self.values:
            packed_bytes += lcapi.to_bytes(x)
        assert len(packed_bytes) == self.arrayType.luisa_type.size()
        return packed_bytes

    def __repr__(self):
        return '[' + ','.join(repr(x) for x in self.values) + ']'


def array(arr):
    return Array(arr)


class ArrayType:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        assert type(size) is int and size > 0
        self.luisa_type = lcapi.Type.from_(f'array<{to_lctype(dtype).description()},{self.size}>')
        self.size_bytes = self.luisa_type.size()

    def __call__(self, data):
        assert self == deduce_array_type(data)
        return Array(data)

    def __repr__(self):
        return f'ArrayType({self.size},{nameof(self.dtype)})'

    def __eq__(self, other):
        return type(other) is ArrayType and self.dtype == other.dtype and self.size == other.size

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.size) ^ 2958463956743103


def deduce_array_type(arr):
    assert len(arr) > 0
    dtype = dtype_of(arr[0])
    for x in arr:
        if dtype_of(x) != dtype:
            raise TypeError("all elements of array must be of same type")
    return ArrayType(dtype=dtype, size=len(arr))


class SharedArrayType:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        assert type(size) is int and size > 0
        self.luisa_type = lcapi.Type.from_(f'array<{to_lctype(dtype).description()},{self.size}>')
        # disable atomic operations if it's not an int buffer
        if dtype in {int, uint, short, ushort}:
            for f in int_atomic_functions:
                setattr(self, f.__name__, f)
        if dtype == float:
            for f in float_atomic_functions:
                setattr(self, f.__name__, f)

    def __repr__(self):
        return f'SharedArrayType({self.size},{nameof(self.dtype)})'

    def __eq__(self, other):
        return type(other) is SharedArrayType and self.dtype == other.dtype and self.size == other.size

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.size) ^ 1058367271709454336
