import lcapi
from .types import dtype_of, to_lctype, nameof

class Struct:
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

    def __init__(self, copy_source = None, alignment = 1, **kwargs):
        if copy_source is not None: # copy from another struct
            assert len(kwargs) == 0 and type(copy_source) is Struct
            self.structType = copy_source.structType
            self.values = copy_source.values.copy()
        else:
            self.structType = deduce_struct_type(kwargs, alignment=alignment)
            self.values = [value for name, value in kwargs.items()]
            for name in kwargs:
                setattr(Struct, name, property(self.make_getter(name), self.make_setter(name)))

    def copy(self):
        return Struct(copy_source = self)
        
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

    def __repr__(self):
        idd = self.structType.idx_dict
        return '{' + ', '.join([name + ':' + repr(self.values[idd[name]]) for name in idd]) + '}'

def struct(alignment = 1, **kwargs):
    assert 'copy_source' not in kwargs
    return Struct(alignment=alignment, **kwargs)

def deduce_struct_type(kwargs, alignment = 1):
    return StructType(alignment, **{name: dtype_of(kwargs[name]) for name in kwargs})


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
        # ensure order
        assert deduce_struct_type(kwargs, alignment=self.alignment) == self
        t = Struct(alignment=self.alignment, **kwargs)
        t.structType = self # override it's type tag to retain method_dict
        return t

    def __repr__(self):
        return f'StructType[{self.alignment}](' + ', '.join([f'{x}:{nameof(self.membertype[self.idx_dict[x]])}' for x in self.idx_dict]) + ')'

    def __eq__(self, other):
        return type(other) is StructType and self.idx_dict == other.idx_dict and self.membertype == other.membertype and self.alignment == other.alignment

    def __hash__(self):
        return hash(self.luisa_type.description()) ^ 7178987438397

    def add_method(self, func, name=None):
        if name is None:
            name = func.__name__
        # check name collision
        if name in self.idx_dict:
            raise NameError("struct method can't have same name as its data members")
        # add method to structtype
        self.method_dict[name] = func


