import lcapi
from .types import dtype_of, to_lctype

class _Array:
    def __init__(self, arrayType, arg):
        self.arrayType = arrayType
        if type(arg) is _Array:
            assert arrayType == arg.arrayType
            self.values = arr.values.copy()
        else:
            assert len(arg) == arrayType.size
            for x in arg:
                assert dtype_of(x) == self.arrayType.dtype
            self.values = list(arg)

    def to_bytes(self):
        packed_bytes = b''
        for x in self.values:
            packed_bytes += lcapi.to_bytes(x)
        assert len(packed_bytes) == self.arrayType.luisa_type.size()
        return packed_bytes


class ArrayType:
    def __init__(self, dtype, size):
        self.dtype = dtype
        self.size = size
        assert type(size) is int and size>0
        self.luisa_type = lcapi.Type.from_(f'array<{to_lctype(dtype).description()},{self.size}>')

    def __call__(self, data):
        return _Array(self, data)

    def __eq__(self, other):
        return type(other) is ArrayType and self.dtype == other.dtype and self.size == other.size

