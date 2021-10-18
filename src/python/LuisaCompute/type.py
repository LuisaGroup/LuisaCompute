import ctypes

from ._internal.type import *
from ._internal.logging import *


class Type:
    def __init__(self, desc):
        self._as_parameter_ = type_from_description(desc.lower())

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
        else:  # should be iterable
            return Type.struct(*t)

    @staticmethod
    def struct(*args, **kwargs):
        if not args:
            raise ValueError("empty structures are not allowed")
        members = [Type.of(t) for t in args]
        ma = max(m.alignment for m in members)
        alignment = kwargs["alignment"] if "alignment" in kwargs else 0
        if alignment != 0 and alignment < ma:
            log_warning(f"Request alignment {alignment} is smaller than needed {ma}.")
        alignment = max(alignment, ma)
        desc = f"struct<{alignment},{','.join(m.description for m in members)}>"
        return Type.of(desc)


Bool = Type("bool")
Int = Type("int")
UInt = Type("uint")
Float = Type("float")
Bool2 = Type("vector<bool,2>")
Int2 = Type("vector<int,2>")
UInt2 = Type("vector<uint,2>")
Float2 = Type("vector<float,2>")
Bool3 = Type("vector<bool,3>")
Int3 = Type("vector<int,3>")
UInt3 = Type("vector<uint,3>")
Float3 = Type("vector<float,3>")
Bool4 = Type("vector<bool,4>")
Int4 = Type("vector<int,4>")
UInt4 = Type("vector<uint,4>")
Float4 = Type("vector<float,4>")
Mat2 = Float2x2 = Type("matrix<2>")
Mat3 = Float3x3 = Type("matrix<3>")
Mat4 = Float4x4 = Type("matrix<4>")
