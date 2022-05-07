import lcapi
from .types import dtype_of, to_lctype
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
        assert dtype_of(value) == dtype
        return value

    def __init__(self, structType, **kwargs):
        self.structType = structType
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
            lctype = to_lctype(dtype)
            curr_align = lctype.alignment()
            while len(packed_bytes) % curr_align != 0:
                packed_bytes += b'\0'
            if lctype.is_basic():
                packed_bytes += lcapi.to_bytes(value)
            elif lctype.is_array():
                packed_bytes += value.to_bytes()
            elif lctype.is_structure():
                packed_bytes += value.to_bytes()
            else:
                assert False
        while len(packed_bytes) % self.structType.alignment != 0:
            packed_bytes += b'\0'
        assert len(packed_bytes) == self.structType.luisa_type.size()
        return packed_bytes



class StructType:
    def __init__(self, alignment = 1, **kwargs):
        # initialize from dict (name->type)
        self.membertype = [] # index -> member dtype
        self.idx_dict = {} # attr -> index
        self.method_dict = {} # attr -> callable (if any)
        self.alignment = alignment
        for idx, (name, dtype) in enumerate(kwargs.items()):
            self.idx_dict[name] = idx
            lctype = to_lctype(dtype) # also checks if it's valid dtype
            self.membertype.append(dtype)
            self.alignment = max(self.alignment, lctype.alignment())
        # compute lcapi.Type
        type_string = f'struct<{self.alignment},' +  ','.join([to_lctype(x).description() for x in self.membertype]) + '>'
        self.luisa_type = lcapi.Type.from_(type_string)

    def __call__(self, **kwargs):
        return _Struct(self, **kwargs)

    def __repr__(self):
        return 'StructType(' + ','.join([f'{x}:{(lambda x: getattr(x,"__name__",None) or repr(x))(self.membertype[self.idx_dict[x]])}' for x in self.idx_dict]) + ')'

    def __eq__(self, other):
        return type(other) is StructType and self.idx_dict == other.idx_dict and self.membertype == other.membertype and self.alignment == other.alignment

    def __hash__(self):
        return hash(self.luisa_type.description()) ^ 7178987438397

    def add_method(self, name, func):
        # check name collision
        if name in self.idx_dict:
            raise NameError("struct method can't have same name as its data members")
        # add method to structtype
        self.method_dict[name] = func


