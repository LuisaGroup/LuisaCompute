//
// Created by Mike Smith on 2021/9/17.
//

#include <random>
#include <numbers>
#include <queue>
#include <thread>

#include <asio.hpp>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/sugar.h>

#include <network/render_tile.h>
#include <network/render_config.h>
#include <network/render_worker.h>

using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] auto create_shader(Device &device) {

    static constexpr auto max_ray_depth = 6;
    static constexpr auto eps = 1e-4f;
    static constexpr auto inf = 1e10f;
    static constexpr auto fov = 0.23f;
    static constexpr auto dist_limit = 100.0f;
    static constexpr auto camera_pos = make_float3(0.0f, 0.32f, 3.7f);
    static constexpr auto light_pos = make_float3(-1.5f, 0.6f, 0.3f);
    static constexpr auto light_normal = make_float3(1.0f, 0.0f, 0.0f);
    static constexpr auto light_radius = 2.0f;

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
        $for(j) : $range(100) {
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

    Kernel2D render_kernel = [&](BufferUInt seed_image, BufferFloat4 output_image, UInt2 tile_offset, UInt2 frame_size, UInt sub_frame_index) noexcept {
        set_block_size(16u, 8u, 1u);

        auto coord = dispatch_id().xy() + tile_offset;
        $if(all(coord < frame_size)) {
            auto global_id = dispatch_x() + dispatch_y() * dispatch_size_x();
            auto resolution = make_float2(frame_size);
            auto aspect_ratio = resolution.x / resolution.y;
            auto pos = def(camera_pos);
            auto seed = seed_image[global_id];
            auto ux = rand(seed);
            auto uy = rand(seed);
            auto uv = make_float2(make_uint2(coord.x, frame_size.y - 1u - coord.y)) + make_float2(ux, uy);
            auto d = make_float3(
                2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
            d = normalize(d);
            auto throughput = def<float3>(1.0f, 1.0f, 1.0f);
            auto hit_light = def(0.0f);
            $for(depth) : $range(max_ray_depth) {
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
            auto color = throughput.xyz() * hit_light;
            auto old = output_image[global_id].xyz();
            output_image[global_id] = make_float4(lerp(old, color, 1.0f / cast<float>(sub_frame_index + 1u)), 1.0f);
            seed_image[global_id] = seed;
        };
    };

    return device.compile(render_kernel);
}

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#else
    auto device = context.create_device("ispc");
#endif
    auto stream = device.create_stream();
    auto shader = create_shader(device);
    Buffer<uint> seed_buffer;
    Buffer<float4> render_buffer;
    std::vector<float4> tile_buffer;
    auto frame_size = make_uint2();
    auto tile_size = make_uint2();
    auto render_id = std::numeric_limits<uint32_t>::max();
    auto tile_spp = 0u;
    auto worker = RenderWorker::create("127.0.0.1", 12345u);
    std::atomic_bool should_stop = false;

    std::queue<RenderTile> tiles;
    std::mutex mutex;
    std::future<void> future;

    auto render = [&] {
        while (!should_stop.load()) {
            if (auto item = [&]() noexcept -> std::optional<RenderTile> {
                    std::scoped_lock lock{mutex};
                    if (tiles.empty()) { return std::nullopt; }
                    auto tile = tiles.front();
                    tiles.pop();
                    return tile;
                }();
                item && item->render_id() == render_id) {
                auto tile = *item;
                Clock clock;
                auto command_buffer = stream.command_buffer();
                for (auto i = 0u; i < tile_spp; i++) {
                    command_buffer << shader(seed_buffer, render_buffer, tile.offset(), frame_size, i).dispatch(tile_size);
                }
                command_buffer << render_buffer.copy_to(tile_buffer.data())
                               << commit();
                stream << synchronize();
                worker->finish(tile, tile_buffer, tile_size);
            } else {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(1ms);
            }
        }
    };

    Clock clock;
    worker->set_config_handler([&](const RenderConfig &config) noexcept {
              LUISA_INFO(
                  "RenderConfig: scene = {}, render_id = {}, resolution = {}x{}, "
                  "spp = {}, tile_size = {}x{}, tile_spp = {}, max_tiles_in_flight = {}.",
                  config.scene(), config.render_id(), config.resolution().x, config.resolution().y,
                  config.spp(), config.tile_size().x, config.tile_size().y, config.tile_spp(), config.tiles_in_flight());
              should_stop = true;
              if (future.valid()) { future.wait(); }
              frame_size = config.resolution();
              tile_size = config.tile_size();
              tile_spp = config.tile_spp();
              render_id = config.render_id();
              auto tile_pixel_count = tile_size.x * tile_size.y;
              seed_buffer = device.create_buffer<uint>(tile_pixel_count);
              render_buffer = device.create_buffer<float4>(tile_pixel_count);
              tile_buffer.resize(tile_pixel_count);
              tiles = {};
              // fill seed buffer
              std::mt19937 random{std::random_device{}()};
              std::vector<uint> seeds(tile_pixel_count);
              for (auto &s : seeds) { s = random(); }
              stream << seed_buffer.copy_from(seeds.data());
              should_stop = false;
              future = std::async(std::launch::async, render);
          })
        .set_render_handler([&](const RenderTile &tile) noexcept {
            std::scoped_lock lock{mutex};
            tiles.emplace(tile);
        })
        .run();
}
