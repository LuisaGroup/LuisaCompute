//
// Created by Mike Smith on 2024/9/20.
//

#include <stb/stb_image_write.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    static constexpr uint width = 512u;
    static constexpr uint height = 512u;

    static constexpr auto mesh_keyframe_count = 3u;
    static constexpr auto curve_keyframe_count = 2u;

    // curve
    static constexpr auto control_point_count = 50u;
    static constexpr auto curve_basis = CurveBasis::CATMULL_ROM;
    static constexpr auto control_points_per_segment = segment_control_point_count(curve_basis);
    static constexpr auto segment_count = control_point_count - control_points_per_segment + 1u;

    luisa::vector<float4> control_points;
    control_points.reserve(control_point_count * curve_keyframe_count);
    for (auto k = 0u; k < curve_keyframe_count; k++) {
        for (auto i = 0u; i < control_point_count; i++) {
            auto x = cos(i * pi / 5.f) * (1.f - .01f * i);
            auto y = i * .02f;
            auto z = sin(i * pi / 5.f) * (1.f - .01f * i);
            auto t = static_cast<float>(i) / static_cast<float>(control_point_count - 1u);// [0, 1]
            auto r = .03f + .03f * sin(t * 10.f * pi - .5f * pi);
            control_points.emplace_back(make_float4(x, y + k * .1f, z, r));
        }
    }
    luisa::vector<uint> segments;
    segments.reserve(segment_count);
    for (auto i = 0u; i < segment_count; i++) {
        segments.emplace_back(i);
    }

    auto control_point_buffer = device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << control_point_buffer.copy_from(control_points.data())
           << segment_buffer.copy_from(segments.data());

    AccelOption curve_option;
    curve_option.motion.keyframe_count = curve_keyframe_count;
    auto curve = device.create_curve(curve_basis, control_point_buffer, segment_buffer, curve_option);

    // mesh
    std::array vertices{
        // keyframe 0
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(-0.1f, 0.5f, 0.0f),
        // keyframe 1
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.55f, 0.0f),
        // keyframe 2
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.1f, 0.5f, 0.0f),
    };
    std::array indices{0u, 1u, 2u};

    auto vertex_buffer = device.create_buffer<float3>(3u * mesh_keyframe_count);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data());

    AccelOption mesh_option;
    mesh_option.motion.keyframe_count = mesh_keyframe_count;
    mesh_option.motion.time_start = 0.f;
    mesh_option.motion.time_end = 1.f;
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, mesh_option);

    AccelMotionOption motion_option;
    motion_option.mode = AccelMotionMode::SRT;
    motion_option.keyframe_count = 3u;
    auto motion_instance = device.create_motion_instance(curve, motion_option);
    luisa::vector<MotionInstanceTransformSRT> motion_transforms;
    for (auto i = 0; i < motion_option.keyframe_count; i++) {
        auto angle = i * radians(15.f);
        auto transform = MotionInstanceTransformSRT{
            .pivot = {0.f, 0.f, 0.f},
            .quaternion = {0.f, sin(angle / 2.f), 0.f, cos(angle / 2.f)},
            .scale = {1.f, 1.f, 1.f},
            .shear = {0.f, 0.f, 0.f},
            .translation = {0.f, -.5f, 0.f}};
        motion_transforms.emplace_back(transform);
    }
    motion_instance.set_keyframes(motion_transforms);

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
        Float rz = halton(i, 5u);
        return make_float3(rx, ry, rz);
    };

    Callable generate_ray = [](Float2 p) noexcept {
        constexpr auto origin = make_float3(0.f, 1.5f, 2.5f);
        constexpr auto target = make_float3(0.f, 0.f, 0.f);
        auto up = make_float3(0.f, 1.f, 0.f);
        auto front = normalize(target - origin);
        auto right = normalize(cross(front, up));
        up = cross(right, front);
        auto fov = radians(45.f);
        auto aspect = static_cast<float>(width) /
                      static_cast<float>(height);
        auto image_plane_height = tan(fov / 2.f);
        auto image_plane_width = aspect * image_plane_height;
        up *= image_plane_height;
        right *= image_plane_width;
        auto uv = p / make_float2(make_uint2(width, height)) * 2.f - 1.f;
        auto ray_origin = origin;
        auto ray_direction = normalize(uv.x * right - uv.y * up + front);
        return make_ray(ray_origin, ray_direction);
    };

    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto color = def<float3>(0.3f, 0.5f, 0.7f);
        auto u = rand(frame_index, coord);
        auto ray = generate_ray(make_float2(coord) + u.xy());
        auto time = u.z * 1.f;
        auto hit = accel.intersect_motion(ray, time, {.curve_bases = {curve_basis}});
        $if (!hit->miss()) {
            constexpr auto red = make_float3(1.0f, 0.0f, 0.0f);
            constexpr auto green = make_float3(0.0f, 1.0f, 0.0f);
            constexpr auto blue = make_float3(0.0f, 0.0f, 1.0f);
            color = triangle_interpolate(hit.bary, red, green, blue);
        };
        auto old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
        auto t = 1.0f / (frame_index + 1.0f);
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 hdr = hdr_image.read(i).xyz();
        UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };

    auto accel = device.create_accel();
    accel.emplace_back(mesh, translation(-.3f, 0.f, 0.f) * scaling(2.f));
    // accel.emplace_back(curve);
    accel.emplace_back(motion_instance);
    stream << curve.build()
           << mesh.build()
           << motion_instance.build()
           << accel.build();

    auto colorspace_shader = device.compile(colorspace_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);

    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    Clock clock;
    clock.tic();
    static constexpr uint spp = 1024u;
    for (uint i = 0u; i < spp; i++) {
        stream << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(pixels.data())
           << synchronize();
    double time = clock.toc();
    LUISA_INFO("Time: {} ms", time);
    stbi_write_png("test_motion_blur.png", width, height, 4, pixels.data(), 0);
}
