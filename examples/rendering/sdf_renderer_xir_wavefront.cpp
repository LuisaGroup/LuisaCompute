#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <string_view>
#include <algorithm>

#include <stb/stb_image_write.h>
#include "common/reference_compare.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct alignas(16) SdfFrame {
    uint2 coord;
    uint seed;
    uint depth;
    float3 pos;
    float3 d;
    float3 throughput;
    float hit_light;
    uint frame_index;
};

LUISA_STRUCT(SdfFrame, coord, seed, depth, pos, d, throughput, hit_light, frame_index){};

namespace {

constexpr uint kMaxRayDepth = 6u;
constexpr float kEps = 1e-4f;
constexpr float kInf = 1e10f;
constexpr float kFov = 0.23f;
constexpr float kDistLimit = 100.0f;
constexpr float3 kCameraPos = make_float3(0.0f, 0.32f, 3.7f);
constexpr float3 kLightPos = make_float3(-1.5f, 0.6f, 0.3f);
constexpr float3 kLightNormal = make_float3(1.0f, 0.0f, 0.0f);
constexpr float kLightRadius = 2.0f;

constexpr uint kWidth = 1280u;
constexpr uint kHeight = 720u;
constexpr uint kPathCount = kWidth * kHeight;

constexpr uint kCounterSetup = 0u;
constexpr uint kCounterBounce0 = 1u;
constexpr uint kCounterBounce1 = 2u;
constexpr uint kCounterFinalize = 3u;
constexpr uint kCounterTotal = 4u;

}

int main(int argc, char *argv[]) {
    Clock clock;

    Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
        auto cos_w = dot(-d, kLightNormal);
        auto dist = dot(d, kLightPos - pos);
        auto D = dist / cos_w;
        auto dist_to_center = distance_squared(kLightPos, pos + D * d);
        auto valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < kLightRadius * kLightRadius;
        return ite(valid, D, kInf);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        Var s0 = 0u;
        for (auto n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Callable rand = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
    };

    Callable out_dir = [&rand](Float3 n, UInt &seed) noexcept {
        auto u = ite(
            abs(n.y) < 1.0f - kEps,
            normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f))),
            make_float3(1.f, 0.f, 0.f));
        auto v = cross(n, u);
        auto phi = 2.0f * std::numbers::pi_v<float> * rand(seed);
        auto ay = sqrt(rand(seed));
        auto ax = sqrt(1.0f - ay * ay);
        return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
    };

    Callable make_nested = [](Float f) noexcept {
        static constexpr auto freq = 40.0f;
        f *= freq;
        f = ite(f < 0.f, ite(cast<int>(f) % 2 == 0, 1.f - fract(f), fract(f)), f);
        return (f - 0.2f) * (1.0f / freq);
    };

    Callable sdf = [&make_nested](Float3 o) noexcept {
        auto wall = min(o.y + 0.1f, o.z + 0.4f);
        auto sphere = distance(o, make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
        auto q = abs(o - make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
        auto box = length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
        auto O = o - make_float3(-0.8f, 0.3f, 0.0f);
        auto d = make_float2(length(make_float2(O.x, O.z)) - 0.3f, abs(O.y) - 0.3f);
        auto cylinder = min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
        auto geometry = make_nested(min(min(sphere, box), cylinder));
        auto g = max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
        return min(wall, g);
    };

    Callable ray_march = [&sdf](Float3 p, Float3 d) noexcept {
        auto dist = def(0.0f);
        $for (j, 100) {
            auto s = sdf(p + dist * d);
            $if (s <= 1e-6f | dist >= kInf) { $break; };
            dist += s;
        };
        return min(dist, kInf);
    };

    Callable sdf_normal = [&sdf](Float3 p) noexcept {
        static constexpr auto d = 1e-3f;
        auto n = def(make_float3());
        auto sdf_center = sdf(p);
        for (auto i = 0; i < 3; i++) {
            auto inc = p;
            inc[i] += d;
            n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
        }
        return normalize(n);
    };

    Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal, Float3 &c, Float3 pos, Float3 d) noexcept {
        closest = kInf;
        normal = make_float3();
        c = make_float3();
        auto ray_march_dist = ray_march(pos, d);
        $if (ray_march_dist < min(kDistLimit, closest)) {
            closest = ray_march_dist;
            auto hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            auto t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
            c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
        };
    };

    Kernel1D reset_counters_kernel = [](BufferVar<uint> counters) noexcept {
        counters->write(dispatch_id().x, 0u);
    };

    Kernel1D init_seed_kernel = [&](ImageUInt seed_image, UInt frame_index) noexcept {
        auto coord = make_uint2(dispatch_id().x % kWidth, dispatch_id().x / kWidth);
        $if (frame_index == 0u) {
            seed_image.write(coord, make_uint4(tea(coord.x, coord.y)));
        };
    };

    // C0_setup: thread per path; init frame, enqueue to bounce_q[0]
    Kernel1D setup_kernel = [&](BufferVar<SdfFrame> frames,
                                BufferVar<uint> bounce_q0,
                                BufferVar<uint> counters,
                                ImageUInt seed_image,
                                UInt frame_index) noexcept {
        auto path_id = dispatch_id().x;
        $if (path_id >= kPathCount) { $return(); };
        auto coord = make_uint2(path_id % kWidth, path_id / kWidth);
        auto seed = seed_image.read(coord).x;
        auto resolution = make_float2(static_cast<float>(kWidth), static_cast<float>(kHeight));
        auto aspect_ratio = resolution.x / resolution.y;
        auto ux = rand(seed);
        auto uy = rand(seed);
        auto uv = make_float2(cast<float>(coord.x) + ux,
                              static_cast<float>(kHeight - 1u) - cast<float>(coord.y) + uy);
        auto d = make_float3(
            2.0f * kFov * uv / resolution.y - kFov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
        d = normalize(d);
        Var<SdfFrame> frame;
        frame.coord = coord;
        frame.seed = seed;
        frame.depth = 0u;
        frame.pos = kCameraPos;
        frame.d = d;
        frame.throughput = make_float3(1.0f, 1.0f, 1.0f);
        frame.hit_light = 0.0f;
        frame.frame_index = frame_index;
        frames->write(path_id, frame);
        auto slot = counters->atomic(kCounterBounce0).fetch_add(1u);
        bounce_q0->write(slot, path_id);
    };

    // bounce kernel template: read from in_q, do one bounce iteration, write to either out_bounce_q or finalize_q
    auto build_bounce_kernel = [&]() {
        return [&](BufferVar<SdfFrame> frames,
                   BufferVar<uint> in_q,
                   BufferVar<uint> out_bounce_q,
                   BufferVar<uint> finalize_q,
                   BufferVar<uint> counters,
                   UInt out_bounce_counter_idx,
                   UInt task_count) noexcept {
            auto task = dispatch_id().x;
            $if (task >= task_count) { $return(); };
            auto path_id = in_q->read(task);
            auto frame = frames->read(path_id);
            auto pos = def(frame.pos);
            auto d = def(frame.d);
            auto throughput = def(frame.throughput);
            auto seed = def(frame.seed);
            auto hit_light = def(frame.hit_light);
            auto depth = def(frame.depth);
            auto exited = def(false);
            auto closest = def(0.0f);
            auto normal = def(make_float3());
            auto c = def(make_float3());
            next_hit(closest, normal, c, pos, d);
            auto dist_to_light = intersect_light(pos, d);
            $if (dist_to_light < closest) {
                hit_light = 1.0f;
                exited = true;
            }
            $else {
                $if (length_squared(normal) == 0.0f) {
                    exited = true;
                }
                $else {
                    auto hit_pos = pos + closest * d;
                    d = out_dir(normal, seed);
                    pos = hit_pos + 1e-4f * d;
                    throughput *= c;
                };
            };
            depth = depth + 1u;
            $if (depth >= kMaxRayDepth) { exited = true; };
            frame.pos = pos;
            frame.d = d;
            frame.throughput = throughput;
            frame.seed = seed;
            frame.hit_light = hit_light;
            frame.depth = depth;
            frames->write(path_id, frame);
            $if (exited) {
                auto slot = counters->atomic(kCounterFinalize).fetch_add(1u);
                finalize_q->write(slot, path_id);
            }
            $else {
                auto slot = counters->atomic(out_bounce_counter_idx).fetch_add(1u);
                out_bounce_q->write(slot, path_id);
            };
        };
    };

    Kernel1D bounce_kernel = build_bounce_kernel();

    // C_finalize: read from finalize_q, write accum + seed
    Kernel1D finalize_kernel = [&](BufferVar<SdfFrame> frames,
                                   BufferVar<uint> in_q,
                                   ImageUInt seed_image,
                                   ImageFloat accum_image,
                                   UInt task_count) noexcept {
        auto task = dispatch_id().x;
        $if (task >= task_count) { $return(); };
        auto path_id = in_q->read(task);
        auto frame = frames->read(path_id);
        auto coord = frame.coord;
        auto frame_index = frame.frame_index;
        auto contribution = frame.throughput * frame.hit_light;
        $if (frame_index == 0u) {
            accum_image.write(coord, make_float4(contribution, 1.0f));
        }
        $else {
            auto prev = accum_image.read(coord).xyz();
            auto accum = lerp(prev, contribution, 1.0f / (cast<float>(frame_index) + 1.0f));
            accum_image.write(coord, make_float4(accum, 1.0f));
        };
        seed_image.write(coord, make_uint4(frame.seed));
    };

    auto linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };
    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = linear_to_srgb(hdr.xyz() / hdr.w * scale);
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    LUISA_INFO("Recorded AST in {} ms.", clock.toc());

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [-c <reference.png>].", argv[0]);
        exit(1);
    }
    bool force_offline = false;
    uint user_spp = 0u;
    std::optional<std::filesystem::path> compare_path;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
        } else if (std::string_view{argv[i]} == "--spp" && i + 1 < argc) {
            user_spp = static_cast<uint>(std::atoi(argv[++i]));
        }
    }

    Context context{argv[0]};
    Device device = context.create_device(argv[1]);

    auto reset_counters = device.compile(reset_counters_kernel);
    auto init_seed = device.compile(init_seed_kernel);
    auto setup = device.compile(setup_kernel);
    auto bounce = device.compile(bounce_kernel);
    auto finalize = device.compile(finalize_kernel);
    auto hdr2ldr = device.compile(hdr2ldr_kernel);

    auto frames = device.create_buffer<SdfFrame>(kPathCount);
    auto bounce_q0 = device.create_buffer<uint>(kPathCount);
    auto bounce_q1 = device.create_buffer<uint>(kPathCount);
    auto finalize_q = device.create_buffer<uint>(kPathCount);
    auto counters = device.create_buffer<uint>(kCounterTotal);
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, kWidth, kHeight);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, kWidth, kHeight);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, kWidth, kHeight);

    Stream stream = device.create_stream(StreamTag::COMPUTE);
    static constexpr uint block_x = 256u;
    auto round_up = [](uint x, uint block) { return ((x + block - 1u) / block) * block; };
    static constexpr auto total_spp_default = 1024u;
    auto total_spp = user_spp == 0u ? total_spp_default : user_spp;
    LUISA_INFO("Wavefront SDF renderer: {} spp, {} paths, {} max depth", total_spp, kPathCount, kMaxRayDepth);

    auto t0 = clock.toc();

    luisa::vector<uint> host_counters(kCounterTotal);
    for (uint frame_index = 0u; frame_index < total_spp; frame_index++) {
        stream << reset_counters(counters).dispatch(kCounterTotal);
        if (frame_index == 0u) {
            stream << init_seed(seed_image, frame_index).dispatch(round_up(kPathCount, block_x));
        }
        stream << setup(frames, bounce_q0, counters, seed_image, frame_index).dispatch(round_up(kPathCount, block_x));
        uint active = 0u;
        while (true) {
            stream << counters.copy_to(host_counters.data()) << synchronize();
            uint count = active == 0u ? host_counters[kCounterBounce0] : host_counters[kCounterBounce1];
            if (count == 0u) { break; }
            uint out_idx = active == 0u ? kCounterBounce1 : kCounterBounce0;
            uint reset_idx = active == 0u ? kCounterBounce0 : kCounterBounce1;
            auto in_q = active == 0u ? bounce_q0.view() : bounce_q1.view();
            auto out_q = active == 0u ? bounce_q1.view() : bounce_q0.view();
            stream << bounce(frames, in_q, out_q, finalize_q, counters, out_idx, count).dispatch(round_up(count, block_x));
            stream << reset_counters(counters.view(reset_idx, 1u)).dispatch(1u);
            active = 1u - active;
        }
        uint finalize_count = host_counters[kCounterFinalize];
        if (finalize_count > 0u) {
            stream << finalize(frames, finalize_q, seed_image, accum_image, finalize_count)
                          .dispatch(round_up(finalize_count, block_x));
        }
    }

    stream << synchronize();
    auto avg_fps = total_spp / (clock.toc() - t0) * 1000;
    LUISA_INFO("Wavefront SDF: {} samples/s", avg_fps);

    luisa::vector<uint8_t> host_image(kWidth * kHeight * 4u);
    stream << hdr2ldr(accum_image, ldr_image, 2.0f).dispatch(kWidth, kHeight)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("sdf-renderer-wavefront.png", kWidth, kHeight, 4, host_image.data(), 0);
    if (force_offline && compare_path) {
        auto result = luisa::ref::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(host_image.data()),
            kWidth, kHeight, 4,
            *compare_path);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) { return 1; }
    }
    return 0;
}
