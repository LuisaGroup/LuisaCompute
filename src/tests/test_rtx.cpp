//
// Created by Mike Smith on 2021/6/23.
//

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_info();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                      12.92f * x,
                      x <= 0.00031308f);
    };

    Callable halton = [](UInt i, UInt b) noexcept {
        Float f = def(1.0f);
        Float invB = 1.0f / b;
        Float r = def(0.0f);
        $while (i > 0u) {
            f = f * invB;
            r = r + f * (i % b);
            i = i / b;
        };
        return r;
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Callable rand = [&](UInt f, UInt2 p) noexcept {
        UInt i = tea(p.x, p.y) + f;
        Float rx = halton(i, 2u);
        Float ry = halton(i, 3u);
        return make_float2(rx, ry);
    };

    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float2 p = (make_float2(coord) + rand(frame_index, coord)) /
                       make_float2(dispatch_size().xy()) * 2.0f -
                   1.0f;
        Float3 color = def<float3>(0.3f, 0.5f, 0.7f);
        Var<Ray> ray = make_ray(
            make_float3(p * make_float2(1.0f, -1.0f), 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        Var<TriangleHit> hit = accel.intersect(ray, {});
        $if (!hit->miss()) {
            constexpr float3 red = float3(1.0f, 0.0f, 0.0f);
            constexpr float3 green = float3(0.0f, 1.0f, 0.0f);
            constexpr float3 blue = float3(0.0f, 0.0f, 1.0f);
            color = triangle_interpolate(hit.bary, red, green, blue);
        };
        Float3 old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
        Float t = 1.0f / (frame_index + 1.0f);
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 hdr = hdr_image.read(i).xyz();
        UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };
    Kernel1D set_transform_kernel = [&](AccelVar accel, Float4x4 matrix, UInt offset) noexcept {
        accel.set_instance_transform(dispatch_id().x + offset, matrix);
    };
    Stream stream = device.create_stream();
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data());

    Accel accel = device.create_accel();
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(1.5f));
    accel.emplace_back(mesh, translation(float3(-0.25f, 0.0f, 0.1f)) *
                                 rotation(float3(0.0f, 0.0f, 1.0f), 0.5f));
    stream << mesh.build()
           << accel.build();

    auto colorspace_shader = device.compile(colorspace_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto set_transform_shader = device.compile(set_transform_kernel);

    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    Clock clock;
    clock.tic();
    static constexpr uint spp = 1024u;
    for (uint i = 0u; i < spp; i++) {
        float t = static_cast<float>(i) * (1.0f / spp);
        vertices[2].y = 0.5f - 0.2f * t;
        float4x4 m = translation(float3(-0.25f + t * 0.15f, 0.0f, 0.1f)) *
                     rotation(float3(0.0f, 0.0f, 1.0f), 0.5f + t * 0.5f);

        stream << vertex_buffer.copy_from(vertices.data())
               << set_transform_shader(accel, m, 1u).dispatch(1)
               << mesh.build()
               << accel.build()
               << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
        if (i == 511u) {
            float4x4 mm = translation(make_float3(0.0f, 0.0f, 0.3f)) *
                          rotation(make_float3(0.0f, 0.0f, 1.0f), radians(180.0f));
            accel.emplace_back(mesh, mm, true);
            stream << accel.update_instance_buffer();
        }
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(pixels.data())
           << synchronize();
    double time = clock.toc();
    LUISA_INFO("Time: {} ms", time);
    stbi_write_png("test_rtx.png", width, height, 4, pixels.data(), 0);
}
