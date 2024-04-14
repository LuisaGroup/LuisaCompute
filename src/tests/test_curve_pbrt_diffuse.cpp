#include <fstream>
#include <stb/stb_image_write.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
    [[nodiscard]] Float3 to_local(Expr<float3> v) const noexcept {
        return make_float3(dot(v, tangent), dot(v, binormal), dot(v, normal));
    }
};

[[nodiscard]] auto parse_pbrt_curve_file(const std::filesystem::path &path) noexcept {
    std::ifstream file{path};
    if (!file.is_open()) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to open curve file: {}",
            path.string());
    }
    luisa::vector<float4> control_points;
    luisa::vector<uint> segments;
    static constexpr auto inf = std::numeric_limits<float>::infinity();
    auto aabb_min = make_float3(inf);
    auto aabb_max = make_float3(-inf);

    auto eof = [&] { return file.peek() == EOF; };
    auto peek = [&] {
        LUISA_ASSERT(!eof(), "Unexpected EOF.");
        return static_cast<char>(file.peek());
    };
    auto pop = [&] {
        LUISA_ASSERT(!eof(), "Unexpected EOF.");
        return static_cast<char>(file.get());
    };
    auto match = [&](char c) noexcept {
        auto x = pop();
        LUISA_ASSERT(x == c,
                     "Unexpected character: {} (expected {})",
                     x, c);
    };
    auto skip_whitespaces = [&] {
        while (!eof() && std::isspace(peek())) { pop(); }
    };
    auto read_string = [&] {
        skip_whitespaces();
        match('"');
        static std::string s;
        s.clear();
        while (peek() != '"') { s.push_back(pop()); }
        match('"');
        return s;
    };
    auto read_token = [&] {
        skip_whitespaces();
        static std::string s;
        s.clear();
        while (!eof() && !std::isspace(peek())) { s.push_back(pop()); }
        return s;
    };
    auto read_float = [&]() noexcept {
        skip_whitespaces();
        static std::string s;
        s.clear();
        auto is_digit = [](char c) noexcept {
            return std::isdigit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E';
        };
        while (!eof() && is_digit(peek())) { s.push_back(pop()); }
        auto p = static_cast<size_t>(0u);
        auto x = std::stof(s, &p);
        LUISA_ASSERT(p == s.size(), "Failed to parse float: {}", s);
        return x;
    };

    auto parse_curve = [&]() noexcept -> bool {
        skip_whitespaces();
        if (eof()) { return false; }
        auto token = read_token();
        LUISA_ASSERT(token == "Shape", "Unexpected token: {}", token);
        LUISA_ASSERT(read_string() == "curve", "Unexpected shape: {}", token);
        static luisa::vector<float3> vertices;
        vertices.clear();
        auto radius_max = 0.;
        auto radius_min = 0.;
        skip_whitespaces();
        while (!eof() && peek() == '"') {
            auto prop = read_string();
            skip_whitespaces();
            match('[');
            if (prop == "point3 P") {
                skip_whitespaces();
                while (peek() != ']') {
                    auto x = read_float();
                    auto y = read_float();
                    auto z = read_float();
                    auto p = make_float3(x, y, z);
                    aabb_min = min(aabb_min, p);
                    aabb_max = max(aabb_max, p);
                    vertices.emplace_back(make_float3(x, y, z));
                    skip_whitespaces();
                }
            } else if (prop == "float width" || prop == "float width0") {
                radius_max = read_float();
            } else if (prop == "float width1") {
                radius_min = read_float();
            } else {
                while (peek() != ']') { pop(); }
            }
            skip_whitespaces();
            match(']');
            skip_whitespaces();
        }
        LUISA_ASSERT(!vertices.empty(), "Empty curve.");
        LUISA_ASSERT(radius_max > 0.f, "Invalid curve radius: {}", radius_max);
        auto offset = static_cast<uint>(control_points.size());
        auto n = static_cast<double>(vertices.size() - 1u);
        for (auto i = 0u; i < vertices.size(); i++) {
            auto v = vertices[i];
            auto r = std::lerp(radius_max, radius_min, i / n);
            control_points.emplace_back(make_float4(v, static_cast<float>(r)));
        }
        for (auto i = 0u; i < vertices.size() - 3u; i++) {
            segments.emplace_back(offset + i);
        }
        return true;
    };
    while (parse_curve()) {}
    return std::make_tuple(std::move(control_points), std::move(segments), aabb_min, aabb_max);
}

int main(int argc, char *argv[]) {

    log_level_verbose();
    Context context{argv[0]};

    if (argc < 3) {
        LUISA_INFO("Usage: {} <backend> <pbrt-curve-file>. "
                   "<backend>: cuda, dx, cpu, metal",
                   argv[0]);
        exit(1);
    }

    auto device = context.create_device(argv[1]);
    auto [control_points, segments, aabb_min, aabb_max] = parse_pbrt_curve_file(argv[2]);
    auto control_point_count = static_cast<uint>(control_points.size());
    auto segment_count = static_cast<uint>(segments.size());
    auto extent = aabb_max - aabb_min;
    auto center = (aabb_max + aabb_min) * .5f;
    auto scaling_factor = std::max({extent.x, extent.y, extent.z});
    LUISA_INFO("Control Points: {}, Segments: {}, AABB: {} -> {}, Extent = {}, Scaling Factor = {}",
               control_point_count, segment_count, aabb_min, aabb_max, extent, scaling_factor);

    auto M = scaling(1.f / scaling_factor) * translation(-center);
    auto invM = inverse(M);
    auto N = transpose(inverse(make_float3x3(M)));
    auto invN = inverse(N);

    static constexpr auto curve_basis = CurveBasis::CATMULL_ROM;
    auto control_point_buffer = device.create_buffer<float4>(control_point_count);
    auto segment_buffer = device.create_buffer<uint>(segment_count);

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << control_point_buffer.copy_from(control_points.data())
           << segment_buffer.copy_from(segments.data())
           << synchronize();
    control_points = {};
    segments = {};

    auto curve = device.create_curve(curve_basis, control_point_buffer, segment_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(curve, M);

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

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    static constexpr auto resolution = make_uint2(512u);

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

    Callable make_onb = [](const Float3 &normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    auto render = device.compile<2u>(
        [&](AccelVar accel, ImageFloat image, ImageUInt seed_image, Float view_angle) noexcept {
            set_block_size(16u, 16u, 1u);
            auto coord = dispatch_id().xy();
            auto state = seed_image.read(coord).x;
            auto ux = lcg(state);
            auto uy = lcg(state);
            auto pixel = make_float2(coord) + make_float2(ux, uy);
            auto ray = generate_ray(pixel, view_angle);
            auto color = def(make_float3());
            auto beta = def(make_float3(1.f));
            $for (depth, 10u) {
                auto hit = accel.intersect(ray, {.curve_bases = {curve_basis}});
                $if (!hit->is_curve()) { $break; };
                auto light_color = make_float3(100.f);
                auto u = hit->curve_parameter();
                auto i0 = hit->prim;
                auto p0 = control_point_buffer->read(i0 + 0u);
                auto p1 = control_point_buffer->read(i0 + 1u);
                auto p2 = control_point_buffer->read(i0 + 2u);
                auto p3 = control_point_buffer->read(i0 + 3u);
                auto c = CurveEvaluator::create(curve_basis, p0, p1, p2, p3);
                auto ps_local = ray->origin() + hit->distance() * ray->direction();
                auto ps = make_float3(invM * make_float4(ps_local, 1.f));
                auto eval = c->evaluate(u, ps_local);
                auto p = make_float3(M * make_float4(eval.position, 1.f));
                auto n = normalize(N * eval.normal);
                auto onb = make_onb(n);
                auto wo = -ray->direction();
                auto wo_local = onb->to_local(wo);
                auto albedo = .8f;
                // Eval light
                {
                    auto light_dir = make_float3(-0.376047f, 0.758426f, 0.532333f);
                    auto wi_local = normalize(onb->to_local(light_dir));
                    auto direct = light_color * max(wi_local.z, 0.f) * albedo * inv_pi;
                    auto shadow_ray = make_ray(p + n * 1e-4f, light_dir);
                    auto occluded = accel->intersect_any(shadow_ray, {.curve_bases = {curve_basis}});
                    color += beta * ite(dsl::isnan(reduce_sum(direct)), 0.f, direct) *
                             ite(occluded, 0.f, 1.f);
                }
                // Sample BSDF. For simplicity, we uniformly sample the sphere.
                {
                    auto wi_local = cosine_sample_hemisphere(make_float2(lcg(state), lcg(state)));
                    beta = beta * albedo;
                    $if (all(beta <= 1e-3f) | dsl::isnan(reduce_sum(beta))) { $break; };
                    auto wi = onb->to_world(wi_local);
                    ray = make_ray(p + n * 1e-4f, wi);
                }
            };
            seed_image.write(coord, make_uint4(state));
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

    luisa::vector<std::byte> pixels(ldr_image.view().size_bytes());
    stream << hdr2ldr(hdr_image, ldr_image, false).dispatch(resolution)
           << ldr_image.copy_to(pixels.data())
           << synchronize();
    stbi_write_png("test_curve_pbrt.png", resolution.x, resolution.y, 4, pixels.data(), 0);
}
