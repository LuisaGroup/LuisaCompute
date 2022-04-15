import lcapi
from .types import basic_types
from .arraytype import ArrayType, _Array


class _Struct:
    @staticmethod
    def make_getter(name):
        def f(self):
            return self.values[self.structType.idx_dict[name]]
        return f
    @staticmethod
    def make_setter(name):
        def f(self, value):
            self.values[self.structType.idx_dict[name]] = value
        return f

    @staticmethod
    def cast(dtype, value):
        if dtype.is_basic():
            assert basic_types[type(value)] == dtype
        elif dtype.is_array():
            if type(value) is not _Array:
                assert len(value) == dtype.dimension() and basic_types[type(value[0])] == dtype.element()
                value = _Array(dtype, value)
        elif dtype.is_structure():
            assert type(value) is _Struct and value.structType.luisa_type == dtype
        else:
            assert False
        return value

    def __init__(self, basetype, **kwargs):
        self.structType = basetype
        self.values = []
        assert len(kwargs.items()) == len(self.structType.membertype)
        for name, value in kwargs.items():
            idx = self.structType.idx_dict[name]
            dtype = self.structType.membertype[idx]
            self.values.append(self.cast(dtype, value))
            setattr(_Struct, name, property(self.make_getter(name), self.make_setter(name)))

    def to_bytes(self):
        packed_bytes = b''
        for idx, value in enumerate(self.values):
            dtype = self.structType.membertype[idx]
            curr_align = dtype.alignment()
            while len(packed_bytes) % curr_align != 0:
                packed_bytes += b'\0'
            if dtype.is_basic():
                packed_bytes += lcapi.to_bytes(value)
            elif dtype.is_array():
                packed_bytes += value.to_bytes()
            elif dtype.is_structure():
                packed_bytes += value.to_bytes()
            else:
                assert False
        while len(packed_bytes) % self.structType.alignment != 0:
            packed_bytes += b'\0'
        assert len(packed_bytes) == self.structType.luisa_type.size()
        return packed_bytes



class StructType:
    def __init__(self, **kwargs):
        # initialize from dict (name->type)
        self.membertype = []
        self.idx_dict = {}
        self.alignment = 1
        for idx, (name, dtype) in enumerate(kwargs.items()):
            self.idx_dict[name] = idx
            if dtype in basic_types:
                self.membertype.append(basic_types[dtype])
            elif type(dtype) in (ArrayType, StructType):
                self.membertype.append(dtype.luisa_type)
            else:
                raise Exception("unrecognized struct member data type")
            self.alignment = max(self.alignment, self.membertype[idx].alignment())
        # compute lcapi.Type
        type_string = f'struct<{self.alignment},' +  ','.join([x.description() for x in self.membertype]) + '>'
        self.luisa_type = lcapi.Type.from_(type_string)

    def __call__(self, **kwargs):
        return _Struct(self, **kwargs)


