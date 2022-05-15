import lcapi
from lcapi import int2, float2, bool2, uint2, int3, float3, bool3, uint3, int4, float4, bool4, uint4
from lcapi import float2x2, float3x3, float4x4

class uint:
    pass

scalar_dtypes = {int, float, bool, uint}
vector_dtypes = {int2, float2, bool2, uint2, int3, float3, bool3, uint3, int4, float4, bool4, uint4}
matrix_dtypes = {float2x2, float3x3, float4x4}
basic_dtypes = {*scalar_dtypes, *vector_dtypes, *matrix_dtypes}
arithmetic_dtypes = {int, float, int2, float2, int3, float3, int4, float4}


def nameof(dtype):
    return getattr(dtype, '__name__', None) or repr(dtype)


def vector(dtype, length): # (float, 2) -> float2
    if length==1:
        return dtype
    return getattr(lcapi, dtype.__name__ + str(length))

def length_of(dtype): # float2 -> 2
    if hasattr(dtype, 'size'):
        return dtype.size
    if dtype in scalar_dtypes:
        return 1
    assert dtype in vector_dtypes or dtype in matrix_dtypes
    return int(dtype.__name__[-1])

# Note: matrix subscripted is vector, not its element
def element_of(dtype): # float2 -> float
    if hasattr(dtype, 'dtype'):
        return dtype.dtype
    if dtype in scalar_dtypes:
        return dtype
    if dtype in matrix_dtypes:
        return float
    assert dtype in vector_dtypes
    return {'int':int, 'float':float, 'bool':bool, 'uint':uint}[dtype.__name__[:-1]]


basic_dtype_to_lctype_dict = {
    int:    lcapi.Type.from_("int"),
    float:  lcapi.Type.from_("float"),
    bool:   lcapi.Type.from_("bool"),
    uint:   lcapi.Type.from_("uint"),
    int2:   lcapi.Type.from_("vector<int,2>"),
    uint2:  lcapi.Type.from_("vector<uint,2>"),
    bool2:  lcapi.Type.from_("vector<bool,2>"),
    float2: lcapi.Type.from_("vector<float,2>"),
    int3:   lcapi.Type.from_("vector<int,3>"),
    uint3:  lcapi.Type.from_("vector<uint,3>"),
    bool3:  lcapi.Type.from_("vector<bool,3>"),
    float3: lcapi.Type.from_("vector<float,3>"),
    int4:   lcapi.Type.from_("vector<int,4>"),
    uint4:  lcapi.Type.from_("vector<uint,4>"),
    bool4:  lcapi.Type.from_("vector<bool,4>"),
    float4: lcapi.Type.from_("vector<float,4>"),
    float2x2:   lcapi.Type.from_("matrix<2>"),
    float3x3:   lcapi.Type.from_("matrix<3>"),
    float4x4:   lcapi.Type.from_("matrix<4>")
}

basic_lctype_to_dtype_dict = {
    basic_dtype_to_lctype_dict[x]:x for x in basic_dtype_to_lctype_dict
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
    if type(val).__name__ == "Texture2D":
        return val.texture2DType
    if type(val).__name__ == "BindlessArray":
        return type(val)
    if type(val).__name__ == "Accel":
        return type(val)
    if type(val).__name__ == "func":
        return CallableType
    if type(val).__name__ == "BuiltinFuncBuilder":
        return type(val)
    if type(val) is list:
        raise Exception("list is unsupported. Convert to Array instead.")
    if type(val).__name__ in {"ArrayType", "StructType", "BufferType"} or val in basic_dtypes:
        return type
    if type(val).__name__ == "function":
        raise Exception(f"dtype_of ({val}): unrecognized type. Did you forget to decorate with luisa.func?")
    raise Exception(f"dtype_of ({val}): unrecognized type")


def to_lctype(dtype):
    if type(dtype).__name__ in {"ArrayType", "StructType", "BufferType", "Texture2DType"}:
        return dtype.luisa_type
    if not hasattr(dtype, "__name__"):
        raise TypeError(f"{dtype} is not a valid data type")
    if dtype.__name__ == "BindlessArray":
        return lcapi.Type.from_("bindless_array")
    if dtype.__name__ == "Accel":
        return lcapi.Type.from_("accel")
    if dtype in basic_dtype_to_lctype_dict:
        return basic_dtype_to_lctype_dict[dtype]
    raise TypeError(f"{dtype} is not a valid data type")

def from_lctype(lctype):
    if lctype in basic_lctype_to_dtype_dict:
        return basic_lctype_to_dtype_dict[lctype]
    raise Exception(f"from_lctype({lctype}:{lctype.description()}): unsupported")

