import lcapi
from .types import basic_types

class _Array:
    def __init__(self, dtype, arg):
        self.dtype = dtype
        if type(arg) is _Array:
            assert dtype == arg.dtype
            self.packed_bytes = arr.packed_bytes
        else:
            assert len(arg) == dtype.dimension()
            self.packed_bytes = b''
            for x in arg:
                assert basic_types[type(x)] == dtype.element()
                self.packed_bytes += lcapi.to_bytes(x)

    def to_bytes(self):
        assert len(self.packed_bytes) == self.dtype.size()
        return self.packed_bytes


class ArrayType:
    def __init__(self, dtype, size = None):
        if size is None:
            # given 1 argument: from lcapi.Type
            assert type(dtype) is lcapi.Type and dtype.is_array()
            self.dtype = dtype.element()
            self.size = dtype.dimension()
            self.luisa_type = lcapi.Type.from_(f'array<{self.dtype.description()},{self.size}>')
            assert self.luisa_type == dtype
        else:
            # given 2 argument: python type & size
            self.dtype = basic_types[dtype]
            self.size = size
            assert type(size) is int and size>0
            self.luisa_type = lcapi.Type.from_(f'array<{self.dtype.description()},{self.size}>')

    def __call__(self, data):
        return _Array(self.luisa_type, data)

