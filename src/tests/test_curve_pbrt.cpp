#include <fstream>
#include <stb/stb_image_write.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
constexpr static float PI = 3.14159265358979323846f;
auto sqr(auto x) noexcept {
    return x * x;
}
auto powi(Float x, uint32_t i) noexcept {
    Float y = x;
    Float z = 1.0f;
    while (i > 0) {
        if (i & 1) { z *= y; }
        y *= y;
        i >>= 1;
    }
    return z;
}
auto abs_cos_theta(Float3 w) noexcept {
    return abs(w.z);
}
Float I0(Float x) noexcept {
    Float val = 0;
    Float x2i = 1;
    int64_t ifact = 1;
    int i4 = 1;
    // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
    for (int i = 0; i < 10; ++i) {
        if (i > 1)
            ifact *= i;
        val += x2i / (static_cast<float>(i4) * sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}
inline Float Logistic(Float x, Float s) noexcept {
    x = abs(x);
    return exp(-x / s) / (s * sqr(1.f + exp(-x / s)));
}

inline Float LogisticCDF(Float x, Float s) noexcept {
    return 1.f / (1.f + exp(-x / s));
}

inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) noexcept {
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

inline Float FrDielectric(Float cosTheta_i, Float eta) noexcept {
    cosTheta_i = clamp(cosTheta_i, -1.0f, 1.0f);
    // Potentially flip interface orientation for Fresnel equations
    $if (cosTheta_i < 0.0f) {
        eta = 1.f / eta;
        cosTheta_i = -cosTheta_i;
    };

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1.0f - sqr(cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / sqr(eta);
    Float ret;
    $if (sin2Theta_t >= 1.f) {
        ret = 1.f;
    }
    $else {
        Float cosTheta_t = sqrt(1.f - sin2Theta_t);

        Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
        Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
        ret = (sqr(r_parl) + sqr(r_perp)) * .5f;
    };
    return ret;
}

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

// From PBRT-v4
class HairBsdf {
    static constexpr int pMax = 3;
    Float h;
    Float eta;
    Float3 sigma_a;
    Float beta_m;
    Float beta_n;
    Float alpha;
    Float s;
    std::array<Float, pMax> sin2kAlpha{}, cos2kAlpha{};
    std::array<Float, pMax + 1> v{};
public:
    HairBsdf(Float h, Float eta, Float3 sigma_a, Float beta_m, Float beta_n, Float alpha)
        : h{h}, eta{eta}, sigma_a{sigma_a}, beta_m{beta_m}, beta_n{beta_n}, alpha{alpha} {
        v[0] = sqr(0.726f * beta_m + 0.812f * sqr(beta_m) + 3.7f * powi(beta_m, 20));
        v[1] = .25f * v[0];
        v[2] = 4.f * v[0];
        for (int p = 3; p <= pMax; ++p)
            v[p] = v[2];

        static const Float SqrtPiOver8 = 0.626657069f;
        s = SqrtPiOver8 * (0.265f * beta_n + 1.194f * sqr(beta_n) + 5.372f * powi(beta_n, 22));

        sin2kAlpha[0] = sin(radians(alpha));
        cos2kAlpha[0] = sqrt(1.0f - sqr(sin2kAlpha[0]));
        for (int i = 1; i < pMax; ++i) {
            sin2kAlpha[i] = 2.f * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = sqr(cos2kAlpha[i - 1]) - sqr(sin2kAlpha[i - 1]);
        }
    }
    static Float Mp(Float cosTheta_i, Float cosTheta_o, Float sinTheta_i,
                    Float sinTheta_o, Float v) {
        Float a = cosTheta_i * cosTheta_o / v, b = sinTheta_i * sinTheta_o / v;
        Float mp = ite(v <= .1f,
                       (exp(log10(a) - b - 1.f / v + 0.6931f + log(1.f / (2.f * v)))),
                       (exp(-b) * I0(a)) / (sinh(1.f / v) * 2.f * v));
        return mp;
    }

    static std::array<Float3, pMax + 1> Ap(Float cosTheta_o,
                                           Float eta, Float h,
                                           Float3 T) {
        std::array<Float3, pMax + 1> ap{};
        // Compute $p=0$ attenuation at initial cylinder intersection
        Float cosGamma_o = sqrt(1.f - sqr(h));
        Float cosTheta = cosTheta_o * cosGamma_o;
        Float f = FrDielectric(cosTheta, eta);
        ap[0] = make_float3(f);

        // Compute $p=1$ attenuation term
        ap[1] = sqr(1.0f - f) * T;

        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p)
            ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of scattering
        $if (all(1.0f - T * f != 0.0f)) {
            ap[pMax] = ap[pMax - 1] * f * T / (1.0f - T * f);
        };

        return ap;
    }

    Float3 f(Float3 wo, Float3 wi) const {
        // Compute hair coordinate system terms related to _wo_
        Float sinTheta_o = wo.x;
        Float cosTheta_o = sqrt(1.f - sqr(sinTheta_o));
        Float phi_o = atan2(wo.z, wo.y);
        Float gamma_o = asin(h);

        // Compute hair coordinate system terms related to _wi_
        Float sinTheta_i = wi.x;
        Float cosTheta_i = sqrt(1.f - sqr(sinTheta_i));
        Float phi_i = atan2(wi.z, wi.y);

        // Compute $\cos\,\thetat$ for refracted ray
        Float sinTheta_t = sinTheta_o / eta;
        Float cosTheta_t = sqrt(1.f - sqr(sinTheta_t));

        // Compute $\gammat$ for refracted ray
        Float etap = sqrt(sqr(eta) - sqr(sinTheta_o)) / cosTheta_o;
        Float sinGamma_t = h / etap;
        Float cosGamma_t = sqrt(1.f - sqr(sinGamma_t));
        Float gamma_t = asin(sinGamma_t);

        // Compute the transmittance _T_ of a single path through the cylinder
        Float3 T = exp(-sigma_a * (2.f * cosGamma_t / cosTheta_t));

        // Evaluate hair BSDF
        Float phi = phi_i - phi_o;
        std::array<Float3, pMax + 1> ap = Ap(cosTheta_o, eta, h, T);
        Float3 fsum = make_float3(0.f);

        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin\,\thetao$ and $\cos\,\thetao$ terms accounting for scales
            Float sinThetap_o, cosThetap_o;
            if (p == 0) {
                sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
                cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (p == 1) {
                sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
                cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
            } else if (p == 2) {
                sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
                cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
            } else {
                sinThetap_o = sinTheta_o;
                cosThetap_o = cosTheta_o;
            }

            // Handle out-of-range $\cos\,\thetao$ from scale adjustment
            cosThetap_o = abs(cosThetap_o);

            fsum += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) * ap[p] *
                    Np(phi, p, s, gamma_o, gamma_t);
        }
        // Compute contribution of remaining terms after _pMax_
        fsum +=
            Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) * ap[pMax] / (2 * PI);

        $if (abs_cos_theta(wi) > 0.0f) {
            fsum /= abs_cos_theta(wi);
        };

        return fsum;
    }
    static inline Float Phi(int p, Float gamma_o, Float gamma_t) noexcept {
        return static_cast<float>(2 * p) * gamma_t - 2.f * gamma_o + static_cast<float>(p) * PI;
    }

    static inline Float Np(Float phi, int p, Float s, Float gamma_o,
                           Float gamma_t) noexcept {
        Float dphi = phi - Phi(p, gamma_o, gamma_t);
        // Remap _dphi_ to $[-\pi,\pi]$
        $while (dphi > PI) {
            dphi -= 2 * PI;
        };
        $while (dphi < -PI) {
            dphi += 2 * PI;
        };
        return TrimmedLogistic(dphi, s, -PI, PI);
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
    auto invN = transpose(make_float3x3(M));
    auto N = inverse(invN);

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
                auto t_local = c->tangent(u);
                auto n = normalize(N * eval.normal);
                auto t = normalize(N * t_local);
                Callable make_onb = [](Float3 normal, Float3 tangent) noexcept {
                    auto binormal = normalize(cross(normal, tangent));
                    return def<Onb>(tangent, binormal, normal);
                };
                auto onb = make_onb(n, t);
                auto wo = -ray->direction();
                auto wo_local = onb->to_local(wo);
                Float h = eval.h(wo_local);
                Float eta = 1.55f;
                Float beta_m = 0.3f;
                Float beta_n = 0.3f;
                Float alpha = 2.0f;
                Float3 sigma_a = ([&] {
                    auto eumelaninSigma_a = make_float3(0.419f, 0.697f, 1.37f);
                    auto pheomelaninSigma_a = make_float3(0.187f, 0.4f, 1.05f);
                    auto ce = 0.5f;
                    auto cp = 0.2f;
                    return ce * eumelaninSigma_a + cp * pheomelaninSigma_a;
                })();
                auto bsdf = HairBsdf(h, eta, sigma_a, beta_m, beta_n, alpha);
                auto p_curve = c->position(u);
                // Eval light
                {
                    auto light_dir = make_float3(-0.376047f, 0.758426f, 0.532333f);
                    auto wi_local = normalize(onb->to_local(light_dir));
                    auto f = bsdf.f(wo_local, wi_local);
                    auto direct = light_color * abs(wi_local.z) * f;
                    auto r = p_curve.w / abs(wi_local.z) + 1e-4f;
                    auto o = make_float3(M * make_float4(p_curve.xyz() + wi_local * r, 1.f));
                    auto shadow_ray = make_ray(o, light_dir);
                    auto occluded = accel->intersect_any(shadow_ray, {.curve_bases = {curve_basis}});
                    color += beta * ite(dsl::isnan(reduce_sum(direct)), 0.f, direct) *
                             ite(occluded, 0.f, 1.f);
                }
                // Sample BSDF. For simplicity, we uniformly sample the sphere.
                {
                    auto wi_local = [&]() noexcept {
                        auto u = make_float2(lcg(state), lcg(state));
                        Float z = 1.f - 2.f * u.x;
                        Float r = sqrt(1.f - sqr(z));
                        Float phi = 2.f * pi * u.y;
                        return make_float3(r * cos(phi), r * sin(phi), z);
                    }();
                    auto f = bsdf.f(wo_local, wi_local);
                    beta = beta * f * abs(wi_local.z) * 4.f * pi;
                    $if (all(beta <= 0.f) | dsl::isnan(reduce_sum(beta))) { $break; };
                    auto wi = onb->to_world(wi_local);
                    auto r = p_curve.w / abs(wi_local.z) + 1e-4f;
                    auto o = make_float3(M * make_float4(p_curve.xyz() + wi_local * r, 1.f));
                    ray = make_ray(o, wi);
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
    auto viewing_angle = PI;
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
