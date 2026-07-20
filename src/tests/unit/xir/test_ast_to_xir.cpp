// Test for AST (Abstract Syntax Tree) to XIR (Extended Intermediate Representation) conversion.
// This test demonstrates a path tracing renderer implemented using DSL,
// showcasing complex GPU compute patterns including ray marching, SDF (Signed Distance Field)
// rendering, and Monte Carlo sampling. The test converts the render kernel to XIR format
// and outputs both text and JSON representations.

#include <luisa/luisa-compute.h>
#include <luisa/xir/verifier.h>
#include <yyjson.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    // Rendering constants for the path tracer
    static constexpr int max_ray_depth = 6;                              // Maximum recursion depth for ray tracing
    static constexpr float eps = 1e-4f;                                  // Epsilon for numerical precision
    static constexpr float inf = 1e10f;                                  // Infinity representation
    static constexpr float fov = 0.23f;                                  // Field of view
    static constexpr float dist_limit = 100.0f;                          // Maximum ray distance
    static constexpr float3 camera_pos = make_float3(0.0f, 0.32f, 3.7f); // Camera position
    static constexpr float3 light_pos = make_float3(-1.5f, 0.6f, 0.3f);  // Light source position
    static constexpr float3 light_normal = make_float3(1.0f, 0.0f, 0.0f);// Light direction
    static constexpr float light_radius = 2.0f;                          // Light source radius

    Clock clock;

    // Callable to compute intersection with area light source
    Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
        Float cos_w = dot(-d, light_normal);
        Float dist = dot(d, light_pos - pos);
        Float D = dist / cos_w;
        Float dist_to_center = distance_squared(light_pos, pos + D * d);
        Bool valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < light_radius * light_radius;
        return ite(valid, D, inf);
    };

    // Tiny Encryption Algorithm (TEA) for pseudo-random number generation
    Callable tea = [](UInt v0, UInt v1) noexcept {
        Var s0 = 0u;
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    // Linear Congruential Generator (LCG) for random number generation
    Callable rand = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
    };

    // Generate random outgoing ray direction based on surface normal
    Callable out_dir = [&rand](Float3 n, UInt &seed) noexcept {
        Float3 u = ite(
            abs(n.y) < 1.0f - eps,
            normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f))),
            make_float3(1.f, 0.f, 0.f));
        Float3 v = cross(n, u);
        Float phi = 2.0f * pi * rand(seed);
        Float ay = sqrt(rand(seed));
        Float ax = sqrt(1.0f - ay * ay);
        return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
    };

    // Create nested SDF pattern for procedural texturing
    Callable make_nested = [](Float f) noexcept {
        static constexpr float freq = 40.0f;
        f *= freq;
        f = ite(f < 0.f, ite(cast<int>(f) % 2 == 0, 1.f - fract(f), fract(f)), f);
        return (f - 0.2f) * (1.0f / freq);
    };

    // Signed Distance Function (SDF) defining the 3D scene geometry
    Callable sdf = [&make_nested](Float3 o) noexcept {
        Float wall = min(o.y + 0.1f, o.z + 0.4f);
        Float sphere = distance(o, make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
        Float3 q = abs(o - make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
        Float box = length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
        Float3 O = o - make_float3(-0.8f, 0.3f, 0.0f);
        Float2 d = make_float2(length(make_float2(O.x, O.z)) - 0.3f, abs(O.y) - 0.3f);
        Float cylinder = min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
        Float geometry = make_nested(min(min(sphere, box), cylinder));
        Float g = max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
        return min(wall, g);
    };

    // Ray marching algorithm to find intersection with SDF scene
    Callable ray_march = [&sdf](Float3 p, Float3 d) noexcept {
        Float dist = def(0.0f);
        $for (j, 100) {
            Float s = sdf(p + dist * d);
            $if (s <= 1e-6f | dist >= inf) { $break; };
            dist += s;
        };
        return min(dist, inf);
    };

    // Compute surface normal using finite differences on SDF
    Callable sdf_normal = [&sdf](Float3 p) noexcept {
        static constexpr float d = 1e-3f;
        Float3 n = def(make_float3());
        Float sdf_center = sdf(p);
        for (uint i = 0; i < 3; i++) {
            Float3 inc = p;
            inc[i] += d;
            n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
        }
        return normalize(n);
    };

    // Find next hit point along ray, returning distance, normal, and color
    Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal, Float3 &c, Float3 pos, Float3 d) noexcept {
        closest = inf;
        normal = make_float3();
        c = make_float3();
        Float ray_march_dist = ray_march(pos, d);
        $if (ray_march_dist < min(dist_limit, closest)) {
            closest = ray_march_dist;
            Float3 hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            Int t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
            c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
        };
    };

    // Main render kernel implementing path tracing
    Kernel2D render_kernel = [&](ImageUInt seed_image, ImageFloat accum_image, UInt frame_index) noexcept {
        set_block_size(16u, 8u, 1u);

        Float2 resolution = make_float2(dispatch_size().xy());
        UInt2 coord = dispatch_id().xy();

        // Initialize seed and accumulator on first frame
        $if (frame_index == 0u) {
            seed_image.write(coord, make_uint4(tea(coord.x, coord.y)));
            accum_image.write(coord, make_float4(make_float3(0.0f), 1.0f));
        };

        // Compute camera ray
        Float aspect_ratio = resolution.x / resolution.y;
        Float3 pos = def(camera_pos);
        UInt seed = seed_image.read(coord).x;
        Float ux = rand(seed);
        Float uy = rand(seed);
        Float2 uv = make_float2(cast<float>(dispatch_id().x) + ux, cast<float>(dispatch_size().y - 1u - dispatch_id().y) + uy);
        Float3 d = make_float3(
            2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
        d = normalize(d);

        // Path tracing loop
        Float3 throughput = def(make_float3(1.0f, 1.0f, 1.0f));
        Float hit_light = def(0.0f);
        $for (depth, max_ray_depth) {
            Float closest = def(0.0f);
            Float3 normal = def(make_float3());
            Float3 c = def(make_float3());
            next_hit(closest, normal, c, pos, d);
            Float dist_to_light = intersect_light(pos, d);
            $if (dist_to_light < closest) {
                hit_light = 1.0f;
                $break;
            };
            $if (length_squared(normal) == 0.0f) { $break; };
            Float3 hit_pos = pos + closest * d;
            d = out_dir(normal, seed);
            pos = hit_pos + 1e-4f * d;
            throughput *= c;
        };

        // Accumulate color with temporal blending
        Float3 accum_color = lerp(accum_image.read(coord).xyz(), throughput.xyz() * hit_light, 1.0f / (frame_index + 1.0f));
        accum_image.write(coord, make_float4(accum_color, 1.0f));
        seed_image.write(coord, make_uint4(seed));
    };

    // Convert AST to XIR (Extended IR)
    auto module = xir::ast_to_xir_translate(render_kernel.function()->function(), {});
    if (module == nullptr) {
        LUISA_WARNING("AST-to-XIR translation returned a null module.");
        return 1;
    }
    auto verification = xir::xir_verify_module(module.get());
    if (!verification.succeeded()) {
        LUISA_WARNING("AST-to-XIR produced invalid IR: {}",
                      verification.errors.front().message);
        return 1;
    }
    auto text = xir::xir_to_text_translate(module.get(), true);
    if (text.empty() || text.find("module;") == luisa::string::npos) {
        LUISA_WARNING("XIR text translation produced invalid output.");
        return 1;
    }

    // Convert XIR to JSON for inspection
    auto json = xir::xir_to_json_translate(module.get());
    auto *json_doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
    auto *json_root = json_doc == nullptr ? nullptr : yyjson_doc_get_root(json_doc);
    auto json_valid = yyjson_is_obj(json_root) &&
                      yyjson_get_bool(yyjson_obj_get(json_root, "ok")) &&
                      yyjson_get_uint(yyjson_obj_get(json_root, "function_count")) >= 1u &&
                      yyjson_is_str(yyjson_obj_get(json_root, "text"));
    if (json_doc != nullptr) { yyjson_doc_free(json_doc); }
    if (!json_valid) {
        LUISA_WARNING("XIR JSON translation produced an invalid module snapshot.");
        return 1;
    }
    return 0;
}
