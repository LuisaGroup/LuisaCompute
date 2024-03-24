#include <luisa/luisa-compute.h>

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

    static constexpr auto control_point_count = 50u;
    static constexpr auto curve_basis = CurveBasis::CATMULL_ROM;
    static constexpr auto control_points_per_segment = segment_control_point_count(curve_basis);
    static constexpr auto segment_count = control_point_count - control_points_per_segment + 1u;

    luisa::vector<float4> control_points;
    control_points.reserve(control_point_count);
    for (auto i = 0u; i < control_point_count; i++) {
        auto x = cos(i * pi / 5.f) * (1.f - .01f * i);
        auto y = i * .02f;
        auto z = sin(i * pi / 5.f) * (1.f - .01f * i);
        auto t = static_cast<float>(i) / static_cast<float>(control_point_count - 1u);// [0, 1]
        auto r = .03f + .03f * sin(t * 10.f * pi - .5f * pi);
        control_points.emplace_back(make_float4(x, y, z, r));
    }
    luisa::vector<uint> segments;
    segments.reserve(segment_count);
    for (auto i = 0u; i < segment_count; i++) {
        segments.emplace_back(i);
    }

    auto control_point_buffer = device.create_buffer<float4>(control_point_count);
    auto segment_buffer = device.create_buffer<uint>(segment_count);

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << control_point_buffer.copy_from(control_points.data())
           << segment_buffer.copy_from(segments.data());

    auto curve = device.create_curve(curve_basis, control_point_buffer, segment_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(curve);

    stream << curve.build()
           << accel.build()
           << synchronize();

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    auto make_sampler_kernel = device.compile<2u>([&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    });

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    static constexpr auto resolution = make_uint2(800u, 600u);

    Callable generate_ray = [](Float2 p) noexcept {
        constexpr auto origin = make_float3(0.f, 1.5f, 2.5f);
        constexpr auto target = make_float3(0.f, 0.f, 0.f);
        auto up = make_float3(0.f, 1.f, 0.f);
        auto front = normalize(target - origin);
        auto right = normalize(cross(front, up));
        up = cross(right, front);
        auto fov = radians(45.f);
        auto aspect = static_cast<float>(resolution.x) /
                      static_cast<float>(resolution.y);
        auto image_plane_height = tan(fov / 2.f);
        auto image_plane_width = aspect * image_plane_height;
        up *= image_plane_height;
        right *= image_plane_width;
        auto uv = p / make_float2(resolution) * 2.f - 1.f;
        auto ray_origin = origin;
        auto ray_direction = normalize(uv.x * right - uv.y * up + front);
        return make_ray(ray_origin, ray_direction);
    };

    auto render = device.compile<2u>(
        [&](AccelVar accel, ImageFloat image, ImageUInt seed_image) noexcept {
            set_block_size(16u, 16u, 1u);
            auto coord = dispatch_id().xy();
            auto state = seed_image.read(coord).x;
            auto ux = lcg(state);
            auto uy = lcg(state);
            seed_image.write(coord, make_uint4(state));
            auto pixel = make_float2(coord) + make_float2(ux, uy);
            auto ray = generate_ray(pixel);
            auto hit = accel.intersect(ray, {.curve_bases = {curve_basis}});
            auto color = def(make_float3());
            $if (hit->is_curve()) {
                auto u = hit->curve_parameter();
                auto i0 = hit->prim;
                auto c = [&] {
                    if constexpr (curve_basis == CurveBasis::PIECEWISE_LINEAR) {
                        auto p0 = control_point_buffer->read(i0 + 0u);
                        auto p1 = control_point_buffer->read(i0 + 1u);
                        return CurveEvaluator::create(curve_basis, p0, p1);
                    } else {
                        auto p0 = control_point_buffer->read(i0 + 0u);
                        auto p1 = control_point_buffer->read(i0 + 1u);
                        auto p2 = control_point_buffer->read(i0 + 2u);
                        auto p3 = control_point_buffer->read(i0 + 3u);
                        return CurveEvaluator::create(curve_basis, p0, p1, p2, p3);
                    }
                }();
                auto ps = ray->origin() + hit->distance() * ray->direction();
                auto eval = c->evaluate(u, ps);
                color = make_float3(hit->curve_parameter(), eval.v(-ray->direction()), .5f);//eval.normal * .5f + .5f;
            };
            auto old = image.read(coord);
            image.write(coord, old + make_float4(color, 1.f));
        });

    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    auto hdr_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    auto clear = device.compile<2>([&](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.f));
    });

    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    auto hdr2ldr = device.compile<2>([&](ImageFloat hdr_image, ImageFloat ldr_image, Bool is_hdr) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = hdr.xyz() / hdr.w;
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    });

    Window window{"Display", resolution, true};
    auto swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });

    stream << clear(hdr_image).dispatch(resolution)
           << make_sampler_kernel(seed_image).dispatch(resolution);
    while (!window.should_close()) {
        stream << render(accel, hdr_image, seed_image).dispatch(resolution)
               << hdr2ldr(hdr_image, ldr_image, false).dispatch(resolution)
               << swap_chain.present(ldr_image);
        window.poll_events();
    }
}
