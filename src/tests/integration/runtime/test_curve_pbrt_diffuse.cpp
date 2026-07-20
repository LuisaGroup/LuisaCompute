// Diffuse shading test for curve rendering.
//
// This test renders curves with simple diffuse (Lambertian) shading
// for comparison with the more complex hair BSDF in test_curve_pbrt.cpp.
// Features:
// - PBRT curve file parsing
// - Simple diffuse BSDF
// - Direct lighting with shadow rays
// - Cosine-weighted hemisphere sampling
// - Interactive camera control

#include "ut/ut.hpp"
#include "test_device.h"
#include "pbrt_curve_parser.h"

#include "reference_image.h"
#include <filesystem>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Orthonormal basis for shading frame
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    // Transform vector from local to world space
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
    // Transform vector from world to local space
    [[nodiscard]] Float3 to_local(Expr<float3> v) const noexcept {
        return make_float3(dot(v, tangent), dot(v, binormal), dot(v, normal));
    }
};

void test_curve_pbrt_diffuse(Device &device) {

    auto argv = boost::ut::detail::cfg::largv;

    log_level_verbose();

    auto curve_path = std::filesystem::path{argv[2]};
    if (!std::filesystem::is_regular_file(curve_path)) {
        boost::ut::expect(false) << "PBRT curve input file does not exist: " << curve_path.string();
        return;
    }
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    // Create device and parse curve file
    auto parsed_curve = luisa::test::parse_pbrt_curve_file(curve_path);
    if (!parsed_curve) {
        boost::ut::expect(false) << parsed_curve.error;
        return;
    }
    auto [control_points, segments, aabb_min, aabb_max] = std::move(parsed_curve.data);
    auto control_point_count = static_cast<uint>(control_points.size());
    auto segment_count = static_cast<uint>(segments.size());
    auto extent = aabb_max - aabb_min;
    auto center = (aabb_max + aabb_min) * .5f;
    auto scaling_factor = std::max({extent.x, extent.y, extent.z});
    LUISA_INFO("Control Points: {}, Segments: {}, AABB: {} -> {}, Extent = {}, Scaling Factor = {}",
               control_point_count, segment_count, aabb_min, aabb_max, extent, scaling_factor);

    // Compute normalization transform
    auto M = scaling(1.f / scaling_factor) * translation(-center);
    auto invM = inverse(M);
    auto N = transpose(inverse(make_float3x3(M)));
    auto invN = inverse(N);

    // Setup curve geometry
    static constexpr auto curve_basis = CurveBasis::CATMULL_ROM;
    auto control_point_buffer = device.create_buffer<float4>(control_point_count);
    auto segment_buffer = device.create_buffer<uint>(segment_count);

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << control_point_buffer.copy_from(luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << synchronize();
    control_points = {};
    segments = {};

    auto curve = device.create_curve(curve_basis, control_point_buffer, segment_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(curve, M);

    stream << curve.build()
           << accel.build()
           << synchronize();

    // Random number generation
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

    // Cosine-weighted hemisphere sampling for diffuse BSDF
    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    static constexpr auto resolution = make_uint2(512u);

    // Camera with rotatable view
    Callable generate_ray = [](Float2 p, Float angle) noexcept {
        auto origin = make_float3(sin(angle) * 2.f, 0.f, cos(angle) * 2.f);
        auto target = make_float3(0.f, 0.f, 0.f);
        auto up = def(make_float3(0.f, 1.f, 0.f));
        auto front = normalize(target - origin);
        auto right = normalize(cross(front, up));
        up = cross(right, front);
        auto fov = radians(35.f);
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

    // Build orthonormal basis from normal vector
    Callable make_onb = [](const Float3 &normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    // Path tracing render kernel with diffuse shading
    auto render = device.compile<2u>(
        [&](AccelVar accel, ImageFloat image, ImageUInt seed_image, Float view_angle) noexcept {
            set_block_size(16u, 16u, 1u);
            auto coord = dispatch_id().xy();
            auto state = seed_image.read(coord).x;
            auto ux = lcg(state);
            auto uy = lcg(state);
            seed_image.write(coord, make_uint4(state));
            auto pixel = make_float2(coord) + make_float2(ux, uy);
            auto ray = generate_ray(pixel, view_angle);
            auto color = def(make_float3());
            auto beta = def(make_float3(1.f));

            // Path tracing loop
            $for (depth, 10u) {
                auto hit = accel.intersect(ray, {.curve_bases = {curve_basis}});
                $if (!hit->is_curve()) { $break; };
                auto light_color = make_float3(100.f);
                auto u = hit->curve_parameter();
                auto i0 = hit->prim;

                // Read curve control points
                auto p0 = control_point_buffer->read(i0 + 0u);
                auto p1 = control_point_buffer->read(i0 + 1u);
                auto p2 = control_point_buffer->read(i0 + 2u);
                auto p3 = control_point_buffer->read(i0 + 3u);
                auto c = CurveEvaluator::create(curve_basis, p0, p1, p2, p3);

                // Compute intersection point and normal
                auto ps_local = ray->origin() + hit->distance() * ray->direction();
                auto ps = make_float3(invM * make_float4(ps_local, 1.f));
                auto eval = c->evaluate(u, ps_local);
                auto p = make_float3(M * make_float4(eval.position, 1.f));

                // Transform normal to world space
                auto n = normalize(N * eval.normal);
                auto onb = make_onb(n);
                auto wo = -ray->direction();
                auto wo_local = onb->to_local(wo);
                auto albedo = .8f;

                // Direct lighting with shadow rays
                {
                    auto light_dir = make_float3(-0.376047f, 0.758426f, 0.532333f);
                    auto wi_local = normalize(onb->to_local(light_dir));
                    // Lambertian BRDF: albedo / π
                    auto direct = light_color * max(wi_local.z, 0.f) * albedo * inv_pi;
                    auto shadow_ray = make_ray(p + n * 1e-4f, light_dir);
                    auto occluded = accel->intersect_any(shadow_ray, {.curve_bases = {curve_basis}});
                    color += beta * ite(dsl::isnan(reduce_sum(direct)), 0.f, direct) *
                             ite(occluded, 0.f, 1.f);
                }

                // Indirect lighting: cosine-weighted sampling
                {
                    auto wi_local = cosine_sample_hemisphere(make_float2(lcg(state), lcg(state)));
                    beta = beta * albedo;// Multiply by albedo
                    $if (all(beta <= 1e-3f) | dsl::isnan(reduce_sum(beta))) { $break; };
                    auto wi = onb->to_world(wi_local);
                    ray = make_ray(p + n * 1e-4f, wi);
                }
            };
            seed_image.write(coord, make_uint4(state));
            auto old = image.read(coord);
            image.write(coord, old + make_float4(color, 1.f));
        });

    // Setup display
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

    if (!opts.offline) {
        Window window{"Display", resolution};
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

        // Interactive render loop
        Clock clock;
        auto viewing_angle = pi;
        auto dirty = true;
        auto last_time = 0.;
        stream << make_sampler_kernel(seed_image).dispatch(resolution);
        Framerate framerate;
        while (!window.should_close()) {
            if (dirty) {
                stream << clear(hdr_image).dispatch(resolution);
                dirty = false;
            }
            stream << render(accel, hdr_image, seed_image, viewing_angle).dispatch(resolution)
                   << hdr2ldr(hdr_image, ldr_image, false).dispatch(resolution)
                   << swap_chain.present(ldr_image);
            window.poll_events();
            static constexpr auto speed = 1e-3f;
            auto curr_time = clock.toc();
            auto delta_time = curr_time - last_time;
            last_time = curr_time;
            if (window.is_key_down(KEY_LEFT)) {
                viewing_angle = static_cast<float>(viewing_angle - speed * delta_time);
                dirty = true;
            } else if (window.is_key_down(KEY_RIGHT)) {
                viewing_angle = static_cast<float>(viewing_angle + speed * delta_time);
                dirty = true;
            } else if (window.is_key_down(KEY_ESCAPE) ||
                       window.is_key_down(KEY_Q)) {
                window.set_should_close(true);
            }
            framerate.record();
            LUISA_INFO("FPS: {}", framerate.report());
        }

        // Save final image
        luisa::vector<std::byte> pixels(ldr_image.view().size_bytes());
        stream << hdr2ldr(hdr_image, ldr_image, false).dispatch(resolution)
               << ldr_image.copy_to(luisa::span{pixels})
               << synchronize();
        stbi_write_png("test_curve_pbrt.png", resolution.x, resolution.y, 4, pixels.data(), 0);
        return;
    } else {
        auto viewing_angle = pi;
        luisa::vector<std::byte> pixels(ldr_image.view().size_bytes());
        stream << make_sampler_kernel(seed_image).dispatch(resolution)
               << clear(hdr_image).dispatch(resolution);
        for (auto i = 0u; i < 256u; i++) {
            stream << render(accel, hdr_image, seed_image, viewing_angle).dispatch(resolution);
        }
        stream << hdr2ldr(hdr_image, ldr_image, false).dispatch(resolution)
               << ldr_image.copy_to(luisa::span{pixels})
               << synchronize();
        if (opts.compare_path) {
            auto result = luisa::test::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison [test_curve_pbrt_diffuse]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) {
                boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
                return;
            }
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        LUISA_INFO("Usage: {} <backend> <pbrt-curve-file>.", argc > 0 ? argv[0] : "test_curve_pbrt_diffuse");
        return 2;
    }
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_curve_pbrt_diffuse(device);
}
