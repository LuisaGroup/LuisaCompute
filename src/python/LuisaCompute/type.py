import ctypes

from ._internal.type import *
from ._internal.logging import *


class Type:
    bool = bool
    int = int
    uint = "uint"
    float = float
    bool2 = "vector<bool,2>"
    int2 = "vector<int,2>"
    uint2 = "vector<uint,2>"
    float2 = "vector<float,2>"
    bool3 = "vector<bool,3>"
    int3 = "vector<int,3>"
    uint3 = "vector<uint,3>"
    float3 = "vector<float,3>"
    bool4 = "vector<bool,4>"
    int4 = "vector<int,4>"
    uint4 = "vector<uint,4>"
    float4 = "vector<float,4>"
    float2x2 = "matrix<2>"
    float3x3 = "matrix<3>"
    float4x4 = "matrix<4>"

    class Field:
        def __init__(self, t, name):
            self._type = t
            self._name = name

        def __str__(self):
            return f"{self.name}: {self.type}"

        def __repr__(self):
            return f"{self.name}: {self.type}"

        @property
        def type(self):
            return self._type

        @property
        def name(self):
            return self._name

    def __init__(self, desc, fields=None):
        if isinstance(desc, str):
            self._as_parameter_ = type_from_description("".join(desc.lower().split()))
        else:
            self._as_parameter_ = Type.of(desc)._as_parameter_
        self._fields = fields
        if fields:
            assert len(self._fields) == len(self.members)

    @property
    def description(self):
        return type_description(self)

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description

    @property
    def size(self):
        return type_size(self)

    @property
    def alignment(self):
        return type_alignment(self)

    @property
    def dimension(self):
        return type_dimension(self)

    @property
    def fields(self):
        return self._fields

    def field(self, name):
        for f in self.fields:
            if f.name == name:
                return f.type

    def has_field(self, name):
        return any(f.name == name for f in self.fields)

    @property
    def members(self):
        count = type_member_count(self)
        ptr = ctypes.cast(type_member_types(self), ctypes.POINTER(c_void_p))
        return [Type(type_description(ptr[i])) for i in range(count)]

    @property
    def element(self):
        return Type(type_description(type_element_type(self)))

    @property
    def is_array(self):
        return bool(type_is_array(self))

    @property
    def is_scalar(self):
        return bool(type_is_scalar(self))

    @property
    def is_vector(self):
        return bool(type_is_vector(self))

    @property
    def is_matrix(self):
        return bool(type_is_matrix(self))

    @property
    def is_structure(self):
        return bool(type_is_structure(self))

    @property
    def is_buffer(self):
        return bool(type_is_buffer(self))

    @property
    def is_texture(self):
        return bool(type_is_texture(self))

    @property
    def is_heap(self):
        return bool(type_is_heap(self))

    @property
    def is_accel(self):
        return bool(type_is_accel(self))

    @staticmethod
    def of(t):
        if t == int:
            return Type("int")
        elif t == float:
            return Type("float")
        elif t == bool:
            return Type("bool")
        elif isinstance(t, str):
            return Type(t)
        elif isinstance(t, Type):
            return t
        elif isinstance(t, dict):
            return Type.struct(0, **t)
        else:  # should be iterable
            return Type.tuple(*t)

    @staticmethod
    def struct(alignment=None, **fields):
        if not fields:
            raise ValueError("empty structs are not allowed")
        member_types = [Type.of(v) for v in fields.values()]
        member_alignment = max(m.alignment for m in member_types)
        if not alignment:
            alignment = member_alignment
        assert alignment >= member_alignment
        desc = f"struct<{alignment},{','.join(m.description for m in member_types)}>"
        member_names = [f.strip() for f in fields.keys()]
        assert len(set(member_names)) == len(member_names)
        for name in member_names:
            assert name.startswith("_") or name and name[0].isalpha()
            assert all(n.isalnum() or n == "_" for n in name)
        fields = [Type.Field(t, n) for t, n in zip(member_types, member_names)]
        return Type(desc, fields)

    @staticmethod
    def tuple(*args):
        if not args:
            raise ValueError("empty tuples are not allowed")
        members = [Type.of(t) for t in args]
        alignment = max(m.alignment for m in members)
        desc = f"struct<{alignment},{','.join(m.description for m in members)}>"
        return Type.of(desc)

    @staticmethod
    def vector(t, n):
        t = Type.of(t)
        assert t.is_scalar
        assert n == 2 or n == 3 or n == 4
        return Type(f"vector<{t},{int(n)}>")

    @staticmethod
    def matrix(n):
        assert n == 2 or n == 3 or n == 4
        return Type(f"matrix<{int(n)}>")

    @staticmethod
    def array(t, n):
        t = Type.of(t)
        n = int(n)
        assert n > 0
        return Type(f"array<{t},{n}>")
