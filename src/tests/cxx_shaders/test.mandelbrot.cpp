#include "luisa/std.hpp"

#ifndef MANDELBROT_OUTPUT_AS_IMAGE
#define MANDELBROT_OUTPUT_AS_IMAGE 1
#endif

namespace luisa::shader::mandelbrot {

[[export]] static constexpr bool kMandelbrotOutputAsImage = MANDELBROT_OUTPUT_AS_IMAGE;
template <bool> trait MandelbrotOutput;
template <> trait MandelbrotOutput<true> { using type = Image<float>; };
template <> trait MandelbrotOutput<false> { using type = Buffer<float4>; };
using MandelbrotResource = typename MandelbrotOutput<kMandelbrotOutputAsImage>::type;

float4 mandelbrot(uint2 tid, uint2 tsize) {
    const float x = float(tid.x) / (float)tsize.x;
    const float y = float(tid.y) / (float)tsize.y;
    const float2 uv = float2(x, y);
    float n = 0.0f;
    float2 c = float2(-0.444999992847442626953125f, 0.0f);
    c = c + (uv - float2(0.5f, 0.5f)) * 2.3399999141693115234375f;
    float2 z = float2(0.f, 0.f);
    const int M = 128;
    for (int i = 0; i < M; i++) {
        z = float2((z.x * z.x) - (z.y * z.y), (2.0f * z.x) * z.y) + c;
        if (dot(z, z) > 2.0f) {
            break;
        }
        n += 1.0f;
    }
    // we use a simple cosine palette to determine color:
    // http://iquilezles.org/www/articles/palettes/palettes.htm
    const float t = float(n) / float(M);
    const float3 d = float3(0.3f, 0.3f, 0.5f);
    const float3 e = float3(-0.2f, -0.3f, -0.5f);
    const float3 f = float3(2.1f, 2.0f, 3.0f);
    const float3 g = float3(0.0f, 0.1f, 0.0f);
    return float4(d + (e * cos(((f * t) + g) * 2.f * PI)), 1.0f);
}

[[kernel_2d(16, 16)]] 
void kernel(MandelbrotResource& output) {
    const uint2 tid = dispatch_id().xy;
    const uint2 tsize = dispatch_size().xy; 
    const uint32 row_pitch = tsize.x;
    store_2d(output, row_pitch, tid, mandelbrot(tid, tsize));
}

}// namespace luisa::shader