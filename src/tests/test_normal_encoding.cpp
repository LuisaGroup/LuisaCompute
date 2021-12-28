//
// Created by Mike Smith on 2021/12/27.
//

#include <random>
#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    auto device = context.create_device("metal");

    std::default_random_engine random{std::random_device{}()};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

    constexpr auto oct_encode = [](float3 n) noexcept {
        constexpr auto oct_wrap = [](float2 v) noexcept {
            return (1.0f - abs(v.yx())) * select(make_float2(-1.0f), make_float2(1.0f), v >= 0.0f);
        };
        auto p = n.xy() * (1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));
        p = n.z >= 0.0f ? p : oct_wrap(p);// in [-1, 1]
        auto u = make_uint2(clamp(round((p * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
        return u.x | (u.y << 16u);
    };

    static constexpr auto n = 1024u * 1024u;
    luisa::vector<float3> normals;
    luisa::vector<uint> encoded_normals;
    normals.reserve(n);
    encoded_normals.reserve(n);
    for (auto i = 0u; i < n; i++) {
        for (;;) {
            auto x = dist(random);
            auto y = dist(random);
            auto z = dist(random);
            auto v = make_float3(x, y, z);
            if (auto vv = dot(v, v); vv > 1e-4f && vv < 1.0f) {
                v = normalize(v);
                normals.emplace_back(v);
                encoded_normals.emplace_back(oct_encode(v));
                break;
            }
        }
    }

    auto decoded_normal_buffer = device.create_buffer<float3>(n);
    auto encoded_normal_buffer = device.create_buffer<uint>(n);
    auto stream = device.create_stream();

    Kernel1D kernel = [&] {
        constexpr auto oct_decode = [](luisa::compute::Expr<uint> u) noexcept {
            using namespace luisa::compute;
            auto p = make_float2(
                cast<float>((u & 0xffffu) * (1.0f / 65535.0f)),
                cast<float>((u >> 16u) * (1.0f / 65535.0f)));
            p = p * 2.0f - 1.0f;// map to [-1, 1]
            auto n = make_float3(p, 1.0f - abs(p.x) - abs(p.y));
            auto t = saturate(-n.z);
            return normalize(make_float3(n.xy() + select(t, -t, n.xy() >= 0.0f), n.z));
        };
        decoded_normal_buffer[dispatch_x()] = oct_decode(encoded_normal_buffer[dispatch_x()]);
    };

    auto shader = device.compile(kernel);
    luisa::vector<float3> decoded_normals(n);

    stream << encoded_normal_buffer.copy_from(encoded_normals.data())
           << shader().dispatch(n)
           << decoded_normal_buffer.copy_to(decoded_normals.data())
           << synchronize();

    for (auto i = 0u; i < 1024u; i++) {
        auto u = normalize(normals[i]);
        auto v = normalize(decoded_normals[i]);
        auto e = degrees(std::sqrt(dot(u - v, u - v)));
        LUISA_INFO(
            "original = ({}, {}, {}), decoded = ({}, {}, {}), error = {} deg.",
            u.x, u.y, u.z, v.x, v.y, v.z, e);
    }

    auto min_error = std::numeric_limits<float>::max();
    auto max_error = 0.0f;
    auto sum_error = 0.0;
    for (auto i = 0u; i < n; i++) {
        auto u = normalize(normals[i]);
        auto v = normalize(decoded_normals[i]);
        auto e = degrees(std::sqrt(dot(u - v, u - v)));
        if (e < min_error) { min_error = e; }
        if (e > max_error) { max_error = e; }
        sum_error += e;
    }
    LUISA_INFO(
        "error: min = {} deg, max = {} deg, avg = {} deg.",
        min_error, max_error, sum_error / n);
}
