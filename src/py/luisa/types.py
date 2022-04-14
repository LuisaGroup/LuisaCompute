import lcapi

# class types:
#     i32 = lcapi.Type.from_("int")
#     f32 = lcapi.Type.from_("float")
#     bool_ = lcapi.Type.from_("bool")

#     int2: lcapi.Type.from_("vector<int,2>")
#     uint2: lcapi.Type.from_("vector<uint,2>")
#     bool2: lcapi.Type.from_("vector<bool,2>")
#     float2: lcapi.Type.from_("vector<float,2>")
#     int3: lcapi.Type.from_("vector<int,3>")
#     uint3: lcapi.Type.from_("vector<uint,3>")
#     bool3: lcapi.Type.from_("vector<bool,3>")
#     float3: lcapi.Type.from_("vector<float,3>")
#     int4: lcapi.Type.from_("vector<int,4>")
#     uint4: lcapi.Type.from_("vector<uint,4>")
#     bool4: lcapi.Type.from_("vector<bool,4>")
#     float4: lcapi.Type.from_("vector<float,4>")

#     float2x2: lcapi.Type.from_("matrix<2>")
#     float3x3: lcapi.Type.from_("matrix<3>")
#     float4x4: lcapi.Type.from_("matrix<4>")

scalar_types = {
    int: lcapi.Type.from_("int"),
    float: lcapi.Type.from_("float"),
    bool: lcapi.Type.from_("bool")
}

basic_types = {
    **scalar_types,
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
