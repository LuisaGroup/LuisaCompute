#include "luisa/std.hpp"

using namespace luisa::shader;

auto tea(uint32 v0, uint32 v1) {
    uint32 s0 = 0;
    for (uint32 n = 0; n < 4; ++n) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

auto halton(uint32 i, uint32 b) {
    auto f = 1.0f;
    auto invB = 1.0f / b;
    auto r = 0.0f;
    while (i > 0u) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    };
    return r;
}

auto rand(uint32 f, uint2 p) {
    auto i = tea(p.x, p.y) + f;
    auto rx = halton(i, 2u);
    auto ry = halton(i, 3u);
    return float2(rx, ry);
}

[[kernel_2d(16, 16)]] 
int kernel(Buffer<float4> &buffer, Accel &accel, uint32 frame_index) {
    auto coord = dispatch_id().xy;
    auto size = dispatch_size().xy;
    auto p = (float2(coord) + rand(frame_index, coord)) / (float2(size));
    p = p * 2.0f - 1.0f;
    auto color = float3(0.3f, 0.5f, 0.7f);
    Ray ray{float3(p * float2(1.0f, -1.0f), 1.0f), float3(0.0f, 0.0f, -1.0f), 0.0f, 1e10f};
    auto hit = accel.trace_closest(ray, 255u);
    if (!hit.miss()) {
        float3 red = float3(1.0f, 0.0f, 0.0f);
        float3 green = float3(0.0f, 1.0f, 0.0f);
        float3 blue = float3(0.0f, 0.0f, 1.0f);
        color = hit.interpolate(red, green, blue);
    }
    float3 old = buffer.load(coord.y * size.x + coord.x).xyz;
    float t = 1.0f / (float(frame_index) + 1.0f);
    buffer.store(coord.y * size.x + coord.x, float4(lerp(old, color, t), 1.0f));
    return 0;
}