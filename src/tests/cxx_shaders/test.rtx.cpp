#include "luisa-shader/std.hpp"

using namespace luisa::shader;

namespace luisa::shader {
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
auto lcg(uint32 &state) {
    const uint32 lcg_a = 1664525u;
    const uint32 lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return static_cast<float>(state & 0x00ffffffu) *
           (1.0f / static_cast<float>(0x01000000u));
}

template<primitive T, bool_family B>
    requires(vec_dim_v<T> == vec_dim_v<B> || vec_dim_v<B> == 1)
extern T ite(B bool_v, T true_v, T false_v) {
    return select(false_v, true_v, bool_v);
}

auto make_onb(float3 &normal) {
    // auto ff = abs(normal.x);
    // float3 binormal = normalize(ite(
    //     abs(normal.x) > abs(normal.z),
    //         float3(-normal.y, normal.x, 0.0f),
    //         float3(0.0f, -normal.z, normal.y))
    // );
}

auto rand(uint32 f, uint2 p) {
    auto i = tea(p.x, p.y) + f;
    auto rx = halton(i, 2u);
    auto ry = halton(i, 3u);
    return float2(rx, ry);
}

[[kernel_2d(16, 16)]] int kernel(Buffer<float4> &buffer, Accel &accel, uint32 frame_index) {
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
}// namespace luisa::shader
