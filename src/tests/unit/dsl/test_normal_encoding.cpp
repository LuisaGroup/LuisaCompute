// Test for octahedral normal encoding/decoding.
//
// This test implements octahedral normal encoding, a technique for efficiently
// storing unit vectors (normals) using only 32 bits with minimal precision loss.
//
// Algorithm:
// 1. Project 3D normal onto octahedron (|x|+|y|+|z| = 1)
// 2. Unfold octahedron to 2D square [-1,1]^2
// 3. Quantize to 16-bit per component
//
// Advantages over naive spherical coordinates:
// - Uniform distribution of precision
// - Fast encode/decode
// - Only 32 bits per normal
//
// Reference: "A Survey of Efficient Representations for Independent Unit Vectors"
// by Cigolle et al., JCGT 2014

#include "ut/ut.hpp"
#include "test_device.h"
#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

// Octahedral encoding function (host version)
// Encodes a unit vector into a 32-bit unsigned integer
inline uint oct_encode(float3 n) noexcept {
    // Helper to fold the octahedron
    constexpr auto oct_wrap = [](float2 v) noexcept {
        return (1.0f - abs(v.yx())) * select(make_float2(-1.0f), make_float2(1.0f), v >= 0.0f);
    };

    // Project onto octahedron and unfold
    float2 p = n.xy() * (1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));

    // Fold negative z hemisphere
    p = n.z >= 0.0f ? p : oct_wrap(p);

    // Quantize to 16-bit unsigned integers
    uint2 u = make_uint2(clamp(round((p * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));

    // Pack into 32-bit: lower 16 bits = x, upper 16 bits = y
    return u.x | (u.y << 16u);
}

void test_normal_encoding(Device &device) {
    // Random number generator for test normals
    std::default_random_engine random{std::random_device{}()};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

    // Generate test normals
    static constexpr uint n = 1024u * 1024u;
    luisa::vector<float3> normals;
    luisa::vector<uint> encoded_normals;
    normals.reserve(n);
    encoded_normals.reserve(n);

    // Generate random normalized vectors
    for (uint i = 0u; i < n; i++) {
        for (;;) {
            float x = dist(random);
            float y = dist(random);
            float z = dist(random);
            float3 v = make_float3(x, y, z);
            // Reject points outside unit sphere for uniform distribution
            if (float vv = dot(v, v); vv > 1e-4f && vv < 1.0f) {
                v = normalize(v);
                normals.emplace_back(v);
                encoded_normals.emplace_back(oct_encode(v));
                break;
            }
        }
    }

    // Create GPU buffers
    Buffer<float3> decoded_normal_buffer = device.create_buffer<float3>(n);
    Buffer<uint> encoded_normal_buffer = device.create_buffer<uint>(n);
    Stream stream = device.create_stream();

    // Decoding kernel (device version)
    Kernel1D kernel = [&] {
        auto oct_decode = [](Expr<uint> u) noexcept {
            // Unpack 16-bit components
            Float2 p = make_float2(
                cast<float>((u & 0xffffu)) * (1.0f / 65535.0f),
                cast<float>((u >> 16u)) * (1.0f / 65535.0f));

            // Map from [0,1] to [-1,1]
            p = p * 2.0f - 1.0f;

            // Reconstruct z component (octahedron property: |x|+|y|+|z|=1)
            Float3 n = make_float3(p, 1.0f - abs(p.x) - abs(p.y));

            // Unfold if in lower hemisphere
            Float t = saturate(-n.z);
            return normalize(make_float3(n.xy() + select(t, -t, n.xy() >= 0.0f), n.z));
        };

        // Decode each normal
        Float3 encoded = oct_decode(encoded_normal_buffer->read(dispatch_x()));
        decoded_normal_buffer->write(dispatch_x(), encoded);
    };

    // Execute decoding kernel
    Shader shader = device.compile(kernel);
    luisa::vector<float3> decoded_normals(n);

    stream << encoded_normal_buffer.copy_from(luisa::span{encoded_normals})
           << shader().dispatch(n)
           << decoded_normal_buffer.copy_to(luisa::span{decoded_normals})
           << synchronize();

    // Print sample results
    for (uint i = 0u; i < 1024u; i++) {
        float3 u = normalize(normals[i]);
        float3 v = normalize(decoded_normals[i]);
        // Compute angular error in degrees
        float e = degrees(std::sqrt(dot(u - v, u - v)));
        LUISA_INFO(
            "original = ({}, {}, {}), decoded = ({}, {}, {}), error = {} deg.",
            u.x, u.y, u.z, v.x, v.y, v.z, e);
    }

    // Compute error statistics
    float min_error = std::numeric_limits<float>::max();
    float max_error = 0.0f;
    float sum_error = 0.0;
    for (uint i = 0u; i < n; i++) {
        float3 u = normalize(normals[i]);
        float3 v = normalize(decoded_normals[i]);
        float e = degrees(std::sqrt(dot(u - v, u - v)));
        if (e < min_error) { min_error = e; }
        if (e > max_error) { max_error = e; }
        sum_error += e;
    }
    LUISA_INFO(
        "error: min = {} deg, max = {} deg, avg = {} deg.",
        min_error, max_error, sum_error / n);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device(argc, argv);
    auto &device = dc.device;
    test_normal_encoding(device);
    return 0;
}
