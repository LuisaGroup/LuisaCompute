import lcapi
from .types import basic_types

class ArrayType:
    def __init__(self, dtype, size):
        self.dtype = basic_types[dtype]
        self.size = size
        assert type(size) is int and size>0
    def luisa_type(self):
        return lcapi.Type.from_(f'array<{self.dtype.description()},{self.size}>')
    def __call__(self, data):
        pass # TODO initialize array on host