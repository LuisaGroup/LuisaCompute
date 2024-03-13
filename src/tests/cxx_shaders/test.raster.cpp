#include "luisa/std.hpp"
#include "luisa/raster/attributes.hpp"
using namespace luisa::shader;
struct Appdata {
    [[POSITION]] float3 positon;
    [[NORMAL]] float3 norm;
    [[INSTANCE_ID]] uint inst_id;
    [[VERTEX_ID]] uint vert_id;
};
struct v2p {
    [[POSITION]] float4 position;
    float2 uv;
    float4 color;
    v2p(float4 position) : position(position) {}
};
[[VERTEX_SHADER]] v2p vert(Appdata data, float time) {
    v2p o{float4(data.positon, 1.f)};
    if (object_id() == 114514) {
        o.position.y += sin(time) * 0.1f;
        o.color = float4(0.3f, 0.6f, 0.7f, 1.0f);
    } else if (object_id() == 1919810) {
        o.color = float4(0.7f, 0.6f, 0.3f, 1.0f);
    } else {
        o.color = float4();
    }
    o.uv = float2(0.5);
    return o;
}

[[PIXEL_SHADER]] float4 frag(v2p i, float time) {
    return i.color;
}