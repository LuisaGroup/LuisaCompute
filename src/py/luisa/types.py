import lcapi


basic_type_dict = {
    int: lcapi.Type.from_("int"),
    float: lcapi.Type.from_("float"),
    bool: lcapi.Type.from_("bool"),
    lcapi.int2: lcapi.Type.from_("vector<int,2>"),
    lcapi.uint2: lcapi.Type.from_("vector<uint,2>"),
    lcapi.bool2: lcapi.Type.from_("vector<bool,2>"),
    lcapi.float2: lcapi.Type.from_("vector<float,2>"),
    lcapi.int3: lcapi.Type.from_("vector<int,3>"),
    lcapi.uint3: lcapi.Type.from_("vector<uint,3>"),
    lcapi.bool3: lcapi.Type.from_("vector<bool,3>"),
    lcapi.float3: lcapi.Type.from_("vector<float,3>"),
    lcapi.int4: lcapi.Type.from_("vector<int,4>"),
    lcapi.uint4: lcapi.Type.from_("vector<uint,4>"),
    lcapi.bool4: lcapi.Type.from_("vector<bool,4>"),
    lcapi.float4: lcapi.Type.from_("vector<float,4>"),
    lcapi.float2x2: lcapi.Type.from_("matrix<2>"),
    lcapi.float3x3: lcapi.Type.from_("matrix<3>"),
    lcapi.float4x4: lcapi.Type.from_("matrix<4>")
}

basic_lctype_dict = {
    lcapi.Type.from_("int") : int,
    lcapi.Type.from_("float") : float,
    lcapi.Type.from_("bool") : bool,
    lcapi.Type.from_("vector<int,2>") : lcapi.int2,
    lcapi.Type.from_("vector<uint,2>") : lcapi.uint2,
    lcapi.Type.from_("vector<bool,2>") : lcapi.bool2,
    lcapi.Type.from_("vector<float,2>") : lcapi.float2,
    lcapi.Type.from_("vector<int,3>") : lcapi.int3,
    lcapi.Type.from_("vector<uint,3>") : lcapi.uint3,
    lcapi.Type.from_("vector<bool,3>") : lcapi.bool3,
    lcapi.Type.from_("vector<float,3>") : lcapi.float3,
    lcapi.Type.from_("vector<int,4>") : lcapi.int4,
    lcapi.Type.from_("vector<uint,4>") : lcapi.uint4,
    lcapi.Type.from_("vector<bool,4>") : lcapi.bool4,
    lcapi.Type.from_("vector<float,4>") : lcapi.float4,
    lcapi.Type.from_("matrix<2>") : lcapi.float2x2,
    lcapi.Type.from_("matrix<3>") : lcapi.float3x3,
    lcapi.Type.from_("matrix<4>") : lcapi.float4x4
}


# dtype: {int, ..., int3, ..., ArrayType(...), StructType(...), BufferType(...), type}
# type annotation should be in the form of dtype

class CallableType:
    pass

def dtype_of(val):
    if type(val) is str:
        return str
    if type(val) in basic_type_dict:
        return type(val)
    if type(val).__name__ == "_Array":
        return val.arrayType
    if type(val).__name__ == "_Struct":
        return val.structType
    if type(val).__name__ == "Buffer":
        return val.bufferType
    if type(val).__name__ == "kernel":
        assert val.is_device_callable
        return CallableType
    if type(val) is list:
        raise Exception("list is unsupported. Convert to Array instead.")
    if type(val).__name__ in {"ArrayType", "StructType", "BufferType"} or val in basic_type_dict:
        return type
    raise Exception(f"dtype_of ({val}): unrecognized type")


def to_lctype(dtype):
    if type(dtype).__name__ in {"ArrayType", "StructType", "BufferType"}:
        return dtype.luisa_type
    if dtype in basic_type_dict:
        return basic_type_dict[dtype]
    raise Exception(f"to_lctype({dtype}): unrecognized type")

def from_lctype(lctype):
    if lctype in basic_lctype_dict:
        return basic_lctype_dict[lctype]
    raise Exception(f"from_lctype({lctype}:{lctype.description()}): unsupported")


# class types:
#     i32 = lcapi.Type.from_("int")
#     u32 = lcapi.Type.from_("uint")
#     f32 = lcapi.Type.from_("float")
#     bool_ = lcapi.Type.from_("bool")

#     int2 = lcapi.Type.from_("vector<int,2>")
#     uint2 = lcapi.Type.from_("vector<uint,2>")
#     bool2 = lcapi.Type.from_("vector<bool,2>")
#     float2 = lcapi.Type.from_("vector<float,2>")
#     int3 = lcapi.Type.from_("vector<int,3>")
#     uint3 = lcapi.Type.from_("vector<uint,3>")
#     bool3 = lcapi.Type.from_("vector<bool,3>")
#     float3 = lcapi.Type.from_("vector<float,3>")
#     int4 = lcapi.Type.from_("vector<int,4>")
#     uint4 = lcapi.Type.from_("vector<uint,4>")
#     bool4 = lcapi.Type.from_("vector<bool,4>")
#     float4 = lcapi.Type.from_("vector<float,4>")

#     float2x2 = lcapi.Type.from_("matrix<2>")
#     float3x3 = lcapi.Type.from_("matrix<3>")
#     float4x4 = lcapi.Type.from_("matrix<4>")