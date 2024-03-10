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
    float3 color;
};
[[VERTEX_SHADER]] v2p vert(Appdata data, float4x4 mvp) {
    v2p p;
    p.position = mvp * float4(data.positon, 1.f);
    p.color = (mvp * float4(data.norm, 0.f)).xyz;
    return p;
}

struct Output {
    float4 v0;
    float4 v1;
};
[[PIXEL_SHADER]] Output frag(v2p i, float3 color) {
    Output o;
    o.v0 = float4(color, 1.f);
    o.v1 = float4(i.color, 1.f);
    return o;
}