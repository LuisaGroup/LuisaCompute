from .dylibs import lcapi
from .dylibs.lcapi import int2, float2, bool2, uint2, int3, float3, bool3, uint3, int4, float4, bool4, uint4
from .dylibs.lcapi import float2x2, float3x3, float4x4


class uint:
    pass


class ushort:
    pass


class half:
    pass


class short:
    pass


class long:
    pass


class ulong:
    pass


class ushort2:
    pass


class half2:
    pass


class short2:
    pass


class ushort3:
    pass


class half3:
    pass


class short3:
    pass


class ushort4:
    pass


class half4:
    pass


class short4:
    pass


class long2:
    pass


class ulong2:
    pass


class long3:
    pass


class ulong3:
    pass


class long4:
    pass


class ulong4:
    pass


_bit16_types = {ushort, half, short, short2, half2, ushort2, short3, half3, ushort3, short4, half4, ushort4}
_bit64_types = {long, ulong, long2, ulong2, long3, ulong3, long4, ulong4}


def is_bit16_types(dtype):
    return dtype in _bit16_types

def is_bit64_types(dtype):
    return dtype in _bit64_types


scalar_dtypes = {int, float, bool, uint, ushort, half, short, long, ulong}
vector_dtypes = {int2, float2, bool2, uint2, int3, float3, bool3, uint3, int4, float4, bool4, uint4, short2, half2,
                 ushort2, short3, half3, ushort3, short4, half4, ushort4, long2, ulong2, long3, ulong3, long4, ulong4}
matrix_dtypes = {float2x2, float3x3, float4x4}

scalar_and_vector_dtypes = {*scalar_dtypes, *vector_dtypes}
vector_and_matrix_dtypes = {*vector_dtypes, *matrix_dtypes}
basic_dtypes = {*scalar_dtypes, *vector_dtypes, *matrix_dtypes}
arithmetic_dtypes = {int, uint, float, short, ushort, long, ulong, half,
                     int2, uint2, float2, int3, uint3, float3, int4, uint4, float4, short2, half2,
                     ushort2, short3, half3, ushort3, short4, half4, ushort4,
                     long2, ulong2, long3, ulong3, long4, ulong4}


def nameof(dtype):
    return getattr(dtype, '__name__', None) or repr(dtype)


def vector16(dtype, length):  # (float, 2) -> float2
    if length == 1:
        return dtype
    name = dtype.__name__ + str(length)
    return eval(name)


def vector(dtype, length):  # (float, 2) -> float2
    if length == 1:
        return dtype
    name = dtype.__name__ + str(length)
    if is_bit16_types(dtype):
        return eval(name)
    return getattr(lcapi, name)


def vector32(dtype, length):  # (float, 2) -> float2
    if length == 1:
        return dtype
    name = dtype.__name__ + str(length)
    return getattr(lcapi, name)


def length_of(dtype):  # float2 -> 2
    if hasattr(dtype, 'size'):
        return dtype.size
    if dtype in scalar_dtypes:
        return 1
    assert dtype in vector_dtypes or dtype in matrix_dtypes
    return int(dtype.__name__[-1])


# Note: matrix subscripted is vector, not its element
def element_of(dtype):  # float2 -> float
    if hasattr(dtype, 'dtype'):
        return dtype.dtype
    if dtype in scalar_dtypes:
        return dtype
    if dtype in matrix_dtypes:
        return float
    assert dtype in vector_dtypes
    return {'int': int,
            'float': float,
            'bool': bool,
            'uint': uint,
            'short': short,
            'half': half,
            'ushort': ushort,
            'long': long,
            'ulong': ulong}[
        dtype.__name__[:-1]]


basic_dtype_to_lctype_dict = {
    int: lcapi.Type.from_("int"),
    float: lcapi.Type.from_("float"),
    bool: lcapi.Type.from_("bool"),
    uint: lcapi.Type.from_("uint"),
    short: lcapi.Type.from_("short"),
    long: lcapi.Type.from_("long"),
    ulong: lcapi.Type.from_("ulong"),
    half: lcapi.Type.from_("half"),
    ushort: lcapi.Type.from_("ushort"),
    int2: lcapi.Type.from_("vector<int,2>"),
    uint2: lcapi.Type.from_("vector<uint,2>"),
    bool2: lcapi.Type.from_("vector<bool,2>"),
    float2: lcapi.Type.from_("vector<float,2>"),
    int3: lcapi.Type.from_("vector<int,3>"),
    uint3: lcapi.Type.from_("vector<uint,3>"),
    bool3: lcapi.Type.from_("vector<bool,3>"),
    float3: lcapi.Type.from_("vector<float,3>"),
    int4: lcapi.Type.from_("vector<int,4>"),
    uint4: lcapi.Type.from_("vector<uint,4>"),
    bool4: lcapi.Type.from_("vector<bool,4>"),
    float4: lcapi.Type.from_("vector<float,4>"),
    short2: lcapi.Type.from_("vector<short,2>"),
    ushort2: lcapi.Type.from_("vector<ushort,2>"),
    half2: lcapi.Type.from_("vector<half,2>"),
    short3: lcapi.Type.from_("vector<short,3>"),
    ushort3: lcapi.Type.from_("vector<ushort,3>"),
    half3: lcapi.Type.from_("vector<half,3>"),
    short4: lcapi.Type.from_("vector<short,4>"),
    ushort4: lcapi.Type.from_("vector<ushort,4>"),
    half4: lcapi.Type.from_("vector<half,4>"),
    float2x2: lcapi.Type.from_("matrix<2>"),
    float3x3: lcapi.Type.from_("matrix<3>"),
    float4x4: lcapi.Type.from_("matrix<4>"),
    long2: lcapi.Type.from_("vector<long,2>"),
    ulong2: lcapi.Type.from_("vector<ulong,2>"),
    long3: lcapi.Type.from_("vector<long,3>"),
    ulong3: lcapi.Type.from_("vector<ulong,3>"),
    long4: lcapi.Type.from_("vector<long,4>"),
    ulong4: lcapi.Type.from_("vector<ulong,4>"),
}

basic_lctype_to_dtype_dict = {
    basic_dtype_to_lctype_dict[x]: x for x in basic_dtype_to_lctype_dict
}


# dtype: {int, ..., int3, ..., ArrayType(...), StructType(...), BufferType(...), type}

class CallableType:
    pass


class BuiltinFuncType:
    pass


class BuiltinFuncBuilder:
    def __init__(self, builder):
        self.builder = builder
        self.__name__ = builder.__name__

    def __call__(self, *args, DO_NOT_CALL):
        pass


# class ref:
#     def __init__(self, dtype):
#         self.dtype = dtype


def dtype_of(val):
    if type(val).__name__ == "module":
        return type(val)
    if type(val) is str:
        return str
    if type(val) in basic_dtypes:
        return type(val)
    if type(val).__name__ == "Array":
        return val.arrayType
    if type(val).__name__ == "Struct":
        return val.structType
    if type(val).__name__ == "Buffer":
        return val.bufferType
    if type(val).__name__ == "RayQuery":
        return val.queryType
    if type(val).__name__ == "Image2D":
        return val.texture2DType
    if type(val).__name__ == "Image3D":
        return val.texture3DType
    if type(val).__name__ == "BindlessArray":
        return type(val)
    if type(val).__name__ == "Accel":
        return type(val)
    if type(val).__name__ == "IndirectDispatchBuffer":
        return type(val)
    if type(val).__name__ == "func":
        return CallableType
    if type(val).__name__ == "BuiltinFuncBuilder":
        return type(val)
    if type(val) is list:
        raise Exception("list is unsupported. Convert to Array instead.")
    if type(val).__name__ in {"ArrayType", "StructType", "BufferType", "IndirectBufferType", "RayQueryAllType",
                              "RayQueryAnyType", "SharedArrayType"} or val in basic_dtypes:
        return type
    if type(val).__name__ == "function":
        raise Exception(f"dtype_of ({val}): unrecognized type. Did you forget to decorate with luisa.func?")


def to_lctype(dtype):
    if type(dtype).__name__ in {"ArrayType", "StructType", "BufferType", "Texture2DType", "Texture3DType", "CustomType",
                                "RayQueryAllType", "RayQueryAnyType", "SharedArrayType"}:
        return dtype.luisa_type
    if not hasattr(dtype, "__name__"):
        raise TypeError(f"{dtype} is not a valid data type")
    if dtype.__name__ == "BindlessArray":
        return lcapi.Type.from_("bindless_array")
    if dtype.__name__ == "Accel":
        return lcapi.Type.from_("accel")
    if dtype.__name__ == "IndirectDispatchBuffer":
        return lcapi.Type.custom("LC_IndirectDispatchBuffer")
    if dtype in basic_dtype_to_lctype_dict:
        return basic_dtype_to_lctype_dict[dtype]
    raise TypeError(f"{dtype} is not a valid data type")


def from_lctype(lctype):
    if lctype in basic_lctype_to_dtype_dict:
        return basic_lctype_to_dtype_dict[lctype]
    raise Exception(f"from_lctype({lctype}:{lctype.description()}): unsupported")


_implicit_map = {
    int: 1,
    uint: 1,
    int2: 1,
    uint2: 1,
    int3: 1,
    uint3: 1,
    int4: 1,
    uint4: 1,
    float: 1,
    float2: 1,
    float3: 1,
    float4: 1,
    short: 1,
    ushort: 1,
    short2: 1,
    half2: 1,
    ushort2: 1,
    short3: 1,
    half3: 1,
    ushort3: 1,
    short4: 1,
    half4: 1,
    ushort4: 1,
    long: 1,
    ulong: 1,
    long2: 1,
    ulong2: 1,
    long3: 1,
    ulong3: 1,
    long4: 1,
    ulong4: 1,
}


def implicit_covertable(src, dst):
    return (src == dst) or (
            _implicit_map.get(src) is not None and \
            _implicit_map.get(src) is not None and \
            length_of(src) == length_of(dst))
