//
// Created by Mike Smith on 2021/6/25.
//

#include <atomic>
#include <numbers>
#include <numeric>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// Credit: https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py
int main(int argc, char *argv[]) {

    log_level_verbose();

    static constexpr auto max_ray_depth = 6;
    static constexpr auto eps = 1e-4f;
    static constexpr auto inf = 1e10f;
    static constexpr auto fov = 0.23f;
    static constexpr auto dist_limit = 100.0f;
    static constexpr auto camera_pos = make_float3(0.0f, 0.32f, 3.7f);
    static constexpr auto light_pos = make_float3(-1.5f, 0.6f, 0.3f);
    static constexpr auto light_normal = make_float3(1.0f, 0.0f, 0.0f);
    static constexpr auto light_radius = 2.0f;

    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;

    Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
        auto cos_w = dot(-d, light_normal);
        auto dist = dot(d, light_pos - pos);
        auto dist_to_light = def(inf);
        $if(cos_w > 0.0f & dist > 0.0f) {
            auto D = dist / cos_w;
            auto dist_to_center = distance_squared(light_pos, pos + D * d);
            $if(dist_to_center < light_radius * light_radius) {
                dist_to_light = D;
            };
        };
        return dist_to_light;
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
        auto u = def<float3>(1.0f, 0.0f, 0.0f);
        $if(abs(n.y) < 1.0f - eps) {
            u = normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f)));
        };
        auto v = cross(n, u);
        auto phi = 2.0f * std::numbers::pi_v<float> * rand(seed);
        auto ay = sqrt(rand(seed));
        auto ax = sqrt(1.0f - ay * ay);
        return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
    };

    Callable make_nested = [](Float f) noexcept {
        static constexpr auto freq = 40.0f;
        f *= freq;
        $if(f < 0.0f) {
            auto ff = floor(f);
            f = select(f - ff, ff + 1.0f - f, f.cast<int>() % 2 == 0);
        };
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
        $for(j, 100) {
            auto s = sdf(p + dist * d);
            $if(s <= 1e-6f | dist >= inf) { $break; };
            dist += s;
        };
        return min(dist, inf);
    };

    Callable sdf_normal = [&sdf](Float3 p) noexcept {
        static constexpr auto d = 1e-3f;
        auto n = def<float3>();
        auto sdf_center = sdf(p);
        for (auto i = 0; i < 3; i++) {
            auto inc = p;
            inc[i] += d;
            n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
        }
        return normalize(n);
    };

    Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal, Float3 &c, Float3 pos, Float3 d) noexcept {
        closest = inf;
        normal = make_float3();
        c = make_float3();
        auto ray_march_dist = ray_march(pos, d);
        $if(ray_march_dist < min(dist_limit, closest)) {
            closest = ray_march_dist;
            auto hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            auto t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
            c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
        };
    };

    Kernel2D render_kernel = [&](BufferUInt seed_image, BufferFloat4 accum_image, UInt frame_index, UInt2 offset) noexcept {
        set_block_size(8u, 8u, 1u);

        auto resolution = make_float2(width, height);
        auto global_id = dispatch_x() + dispatch_y() * width;
        auto coord = dispatch_id().xy() + offset;

        $if(frame_index == 0u) {
            seed_image.write(global_id, tea(coord.x, coord.y));
            accum_image.write(global_id, make_float4(make_float3(0.0f), 1.0f));
        };

        auto aspect_ratio = resolution.x / resolution.y;
        auto pos = def(camera_pos);
        auto seed = seed_image.read(global_id);
        auto ux = rand(seed);
        auto uy = rand(seed);
        auto uv = make_float2(cast<float>(coord.x) + ux, cast<float>(height - 1u - coord.y) + uy);
        auto d = make_float3(
            2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
        d = normalize(d);
        auto throughput = def<float3>(1.0f, 1.0f, 1.0f);
        auto hit_light = def(0.0f);
        $for(depth, max_ray_depth) {
            auto closest = def(0.0f);
            auto normal = def<float3>();
            auto c = def<float3>();
            next_hit(closest, normal, c, pos, d);
            auto dist_to_light = intersect_light(pos, d);
            $if(dist_to_light < closest) {
                hit_light = 1.0f;
                $break;
            };
            $if(length_squared(normal) == 0.0f) { $break; };
            auto hit_pos = pos + closest * d;
            d = out_dir(normal, seed);
            pos = hit_pos + 1e-4f * d;
            throughput *= c;
        };
        auto accum_color = accum_image.read(global_id).xyz() + throughput.zyx() * hit_light;
        accum_image.write(global_id, make_float4(accum_color, 1.0f));
        seed_image.write(global_id, seed);
    };

    Context context{argv[0]};
    auto device_upper = context.create_device("cuda");
    auto device_lower = context.create_device("cuda");

    static constexpr auto height_upper = 400u;
    static constexpr auto height_lower = height - height_upper;

    auto seed_image_upper = device_upper.create_buffer<uint>(width * height_upper);
    auto accum_image_upper = device_upper.create_buffer<float4>(width * height_upper);
    auto render_upper = device_upper.compile(render_kernel);
    auto stream_upper = device_upper.create_stream();
    auto swap_event_upper = device_upper.create_event();
    auto copy_event_upper = device_upper.create_event();

    auto seed_image_lower = device_lower.create_buffer<uint>(width * height_lower);
    auto accum_image_lower = device_lower.create_buffer<float4>(width * height_lower);
    auto render_lower = device_lower.compile(render_kernel);
    auto stream_lower = device_lower.create_stream();
    auto swap_event_lower = device_lower.create_event();
    auto copy_event_lower = device_lower.create_event();

    cv::Mat cv_image{height, width, CV_32FC4, cv::Scalar::all(1.0)};
    cv::Mat cv_back_image{height, width, CV_32FC4, cv::Scalar::all(1.0)};

    Clock clock;
    auto last_t = clock.toc();

    static constexpr auto interval = 10u;
    static constexpr auto total_spp = 500000u;
    std::vector<double> fps;
    fps.reserve(total_spp);
    for (auto spp = 0u; spp < total_spp; spp += interval) {

        // swap buffers
        copy_event_upper.synchronize();
        copy_event_lower.synchronize();
        std::swap(cv_image, cv_back_image);
        stream_upper << swap_event_upper.signal();
        stream_lower << swap_event_lower.signal();

        // render
        auto command_buffer_upper = stream_upper.command_buffer();
        auto command_buffer_lower = stream_lower.command_buffer();
        for (auto frame = spp; frame < spp + interval && frame < total_spp; frame++) {
            command_buffer_upper << render_upper(seed_image_upper, accum_image_upper, frame, make_uint2(0u, 0u)).dispatch(width, height_upper);
            command_buffer_lower << render_lower(seed_image_lower, accum_image_lower, frame, make_uint2(0u, height_upper)).dispatch(width, height_lower);
        }
        command_buffer_upper << swap_event_upper.wait()
                             << accum_image_upper.copy_to(cv_back_image.data)
                             << copy_event_upper.signal();
        command_buffer_lower << swap_event_lower.wait()
                             << accum_image_lower.copy_to(cv_back_image.data + sizeof(float4) * width * height_upper)
                             << copy_event_lower.signal();

        // display
        cv_image *= 1.0 / std::max(spp, 1u);
        auto mean = std::max(cv::mean(cv::mean(cv_image))[0], 1e-3);
        cv::sqrt(cv_image * (0.24 / mean), cv_image);
        cv::imshow("Display", cv_image);
        if (auto key = cv::waitKey(1); key == 'q' || key == 27) { break; }
        auto t = clock.toc();
        LUISA_INFO(
            "{:.2f} samples/s [{}/{}]",
            fps.emplace_back(interval * 1000.0 / (t - last_t)),
            spp + interval, total_spp);
        last_t = t;
    }
    LUISA_INFO("Average: {} samples/s.", std::reduce(fps.cbegin(), fps.cend(), 0.0) / fps.size());
}
