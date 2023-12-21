#include "luisa-shader/std.hpp"
#include "luisa-shader/internal/resource/bindless_array.hpp"

using namespace luisa::shader;
namespace luisa::shader {
using uint = uint32;
/// pi
const auto pi = 3.14159265358979323846264338327950288f;
/// pi/2
const auto pi_over_two = 1.57079632679489661923132169163975144f;
/// pi/4
const auto pi_over_four = 0.785398163397448309615660845819875721f;
/// 1/pi
const auto inv_pi = 0.318309886183790671537767526745028724f;
/// 2/pi
const auto two_over_pi = 0.636619772367581343075535053490057448f;
/// sqrt(2)
const auto sqrt_two = 1.41421356237309504880168872420969808f;
/// 1/sqrt(2)
const auto inv_sqrt_two = 0.707106781186547524400844362104849039f;
/// 1-epsilon
const auto one_minus_epsilon = 0x1.fffffep-1f;
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
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
    Onb(
        float3 tangent,
        float3 binormal,
        float3 normal) : tangent(tangent), binormal(binormal), normal(normal) {}
};

auto make_onb(float3 normal) {
    auto binormal = normalize(ite(
        abs(normal.x) > abs(normal.z),
        float3(-normal.y, normal.x, 0.0f),
        float3(0.0f, -normal.z, normal.y)));
    float3 tangent = normalize(cross(binormal, normal));
    return Onb{tangent, binormal, normal};
}
auto radians(float deg) {
    return deg * pi / 180.f;
}
auto degree(float rad) { return rad * inv_pi * 180.f; };
auto generate_ray(float2 p) {
    const float fov = radians(27.8f);
    const float3 origin(-0.01f, 0.995f, 5.0f);
    float3 pixel = origin + float3(p * tan(0.5f * fov), -1.0f);
    float3 direction = normalize(pixel - origin);
    return Ray(origin, direction);
}
auto cosine_sample_hemisphere(float2 u) {
    float r = sqrt(u.x);
    float phi = 2.f * pi * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
}
auto balanced_heuristic(float pdf_a, float pdf_b) {
    return pdf_a / max(pdf_a + pdf_b, 1e-4f);
}

auto rand(uint32 f, uint2 p) {
    auto i = tea(p.x, p.y) + f;
    auto rx = halton(i, 2u);
    auto ry = halton(i, 3u);
    return float2(rx, ry);
}

struct Triangle {
    uint i0;
    uint i1;
    uint i2;
};

[[kernel_2d(16, 16)]] int kernel(
    Image<float> &image,
    Image<uint> &seed_image,
    Accel &accel,
    BindlessArray &heap,
    Buffer<float3> &vertex_buffer,
    uint2 resolution) {
    auto coord = dispatch_id().xy;
    auto size = dispatch_size().xy;
    auto state = seed_image.load(coord).x;
    auto rx = lcg(state);
    auto ry = lcg(state);
    auto pixel = (float2(coord) + float2(rx, ry)) / float2(size) * 2.f - 1.f;
    float3 radiance;
    const uint spp_per_dispatch = 64;
    for (uint i = 0; i < spp_per_dispatch; ++i) {
        auto ray = generate_ray(pixel * float2(1.f, -1.f));
        float pdf_bsdf = 0.;
        const float3 light_position = float3(-0.24f, 1.98f, 0.16f);
        const float3 light_u = float3(-0.24f, 1.98f, -0.22f) - light_position;
        const float3 light_v = float3(0.23f, 1.98f, 0.16f) - light_position;
        const float3 light_emission = float3(17.0f, 12.0f, 4.0f);
        auto light_area = length(cross(light_u, light_v));
        auto light_normal = normalize(cross(light_u, light_v));
        for (uint depth = 0; depth < 10; ++depth) {
            auto hit = accel.trace_closest(ray);
            if (hit.miss()) {
                break;
            }
            auto triangle = heap.buffer_read<Triangle>(hit.inst, hit.prim);
            float3 p0 = vertex_buffer.load(triangle.i0);
            float3 p1 = vertex_buffer.load(triangle.i1);
            float3 p2 = vertex_buffer.load(triangle.i2);
            float3 p = hit.interpolate(p0, p1, p2);
            float3 n = normalize(cross(p1 - p0, p2 - p0));
            float cos_wo = dot(-ray.dir(), n);
            if (cos_wo < 1e-4f) { break; };
        }
    }
    return 0;
}
}// namespace luisa::shader
