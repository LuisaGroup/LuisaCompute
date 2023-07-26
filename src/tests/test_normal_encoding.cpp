#include <random>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    std::default_random_engine random{std::random_device{}()};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

    constexpr auto oct_encode = [](float3 n) noexcept {
        constexpr auto oct_wrap = [](float2 v) noexcept {
            return (1.0f - abs(v.yx())) * select(make_float2(-1.0f), make_float2(1.0f), v >= 0.0f);
        };
        float2 p = n.xy() * (1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));
        p = n.z >= 0.0f ? p : oct_wrap(p);// in [-1, 1]
        uint2 u = make_uint2(clamp(round((p * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
        return u.x | (u.y << 16u);
    };

    static constexpr uint n = 1024u * 1024u;
    luisa::vector<float3> normals;
    luisa::vector<uint> encoded_normals;
    normals.reserve(n);
    encoded_normals.reserve(n);
    for (uint i = 0u; i < n; i++) {
        for (;;) {
            float x = dist(random);
            float y = dist(random);
            float z = dist(random);
            float3 v = make_float3(x, y, z);
            if (float vv = dot(v, v); vv > 1e-4f && vv < 1.0f) {
                v = normalize(v);
                normals.emplace_back(v);
                encoded_normals.emplace_back(oct_encode(v));
                break;
            }
        }
    }

    Buffer<float3> decoded_normal_buffer = device.create_buffer<float3>(n);
    Buffer<uint> encoded_normal_buffer = device.create_buffer<uint>(n);
    Stream stream = device.create_stream();

    Kernel1D kernel = [&] {
        constexpr auto oct_decode = [](luisa::compute::Expr<uint> u) noexcept {
            using namespace luisa::compute;
            Float2 p = make_float2(
                cast<float>((u & 0xffffu) * (1.0f / 65535.0f)),
                cast<float>((u >> 16u) * (1.0f / 65535.0f)));
            p = p * 2.0f - 1.0f;// map to [-1, 1]
            Float3 n = make_float3(p, 1.0f - abs(p.x) - abs(p.y));
            Float t = saturate(-n.z);
            return normalize(make_float3(n.xy() + select(t, -t, n.xy() >= 0.0f), n.z));
        };
        Float3 encoded = oct_decode(encoded_normal_buffer->read(dispatch_x()));
        decoded_normal_buffer->write(dispatch_x(), encoded);
    };

    Shader shader = device.compile(kernel);
    luisa::vector<float3> decoded_normals(n);

    stream << encoded_normal_buffer.copy_from(encoded_normals.data())
           << shader().dispatch(n)
           << decoded_normal_buffer.copy_to(decoded_normals.data())
           << synchronize();

    for (uint i = 0u; i < 1024u; i++) {
        float3 u = normalize(normals[i]);
        float3 v = normalize(decoded_normals[i]);
        float e = degrees(std::sqrt(dot(u - v, u - v)));
        LUISA_INFO(
            "original = ({}, {}, {}), decoded = ({}, {}, {}), error = {} deg.",
            u.x, u.y, u.z, v.x, v.y, v.z, e);
    }

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

