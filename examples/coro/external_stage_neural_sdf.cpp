#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

#include "coro/external_stage_common.h"
#include "coro/neural_bunny_sdf.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace luisa::compute::coro::example;

namespace {

constexpr auto distance_schema = "luisa.coro.example.neural-sdf.distance";
constexpr auto normal_schema = "luisa.coro.example.neural-sdf.normal";

struct Options {
    uint width{640u};
    uint height{360u};
    bool write_image{true};
};

[[nodiscard]] Options parse_options(int argc, char *argv[]) noexcept {
    Options options;
    for (auto i = 2; i < argc; ++i) {
        if (std::string_view{argv[i]} == "--test") {
            options.width = 64u;
            options.height = 64u;
            options.write_image = false;
        }
    }
    return options;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <backend> [--test]", argv[0]);
        return 1;
    }
    auto options = parse_options(argc, argv);
    auto frame_count = options.width * options.height;

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto hdr = device.create_image<float>(
        PixelStorage::FLOAT4, options.width, options.height);
    auto ldr = device.create_image<float>(
        PixelStorage::BYTE4, options.width, options.height);
    auto hit_mask = device.create_buffer<uint>(frame_count);

    Callable random = [](UInt &state) noexcept {
        state = state * 747796405u + 2891336453u;
        auto word = ((state >> ((state >> 28u) + 4u)) ^ state) *
                    277803737u;
        word = (word >> 22u) ^ word;
        return cast<float>(word & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Coroutine<void(Image<float>, Buffer<uint>)> coroutine =
        [&random](ImageFloat output, BufferUInt hits) noexcept {
            constexpr auto max_bounces = 2u;
            constexpr auto max_steps = 96u;
            constexpr auto max_distance = 8.0f;
            auto coord = dispatch_id().xy();
            auto resolution = make_float2(dispatch_size().xy());
            auto uv = (make_float2(coord) + .5f - resolution * .5f) /
                      resolution.y;
            Var origin = make_float3(-3.0f, 0.0f, 0.05f);
            Var direction = normalize(make_float3(1.5f, uv.x, uv.y));
            Var radiance = make_float3(0.0f);
            Var throughput = make_float3(1.0f);
            Var any_hit = false;
            Var seed = coord.x * 1973u + coord.y * 9277u + 89173u;

            $for (bounce, max_bounces) {
                Var travel = 0.0f;
                Var hit = false;
                Var material = 0.0f;
                $for (step, max_steps) {
                    Var query = origin + direction * travel;
                    Var sample = make_float2(max_distance, -1.0f);
                    $suspend(
                        "neural_distance",
                        coro_stage(distance_schema)
                            .read("point", query)
                            .write("sample", sample));
                    auto epsilon = max(1e-4f, 1e-3f * travel);
                    $if (sample.x < epsilon) {
                        hit = true;
                        material = sample.y;
                        $break;
                    };
                    travel += max(sample.x * .8f, 5e-4f);
                    $if (travel > max_distance) { $break; };
                };

                $if (!hit) {
                    auto sky = make_float3(.08f, .12f, .20f) +
                               make_float3(.30f, .36f, .42f) *
                                   max(direction.z, 0.0f);
                    radiance += throughput * sky;
                    $break;
                };

                any_hit = true;
                Var hit_point = origin + direction * travel;
                Var normal = make_float3(0.0f, 0.0f, 1.0f);
                $suspend(
                    "neural_normal",
                    coro_stage(normal_schema)
                        .read("point", hit_point)
                        .write("normal", normal));
                $if (dot(normal, direction) > 0.0f) { normal = -normal; };

                auto light = normalize(make_float3(-.4f, -.2f, 1.0f));
                auto albedo = ite(
                    material > .5f,
                    make_float3(.72f, .84f, .98f),
                    make_float3(.78f, .68f, .68f));
                radiance += throughput * albedo *
                            (0.04f + 0.55f * max(dot(normal, light), 0.0f));

                auto phi = 6.28318530718f * random(seed);
                auto z = sqrt(random(seed));
                auto r = sqrt(max(1.0f - z * z, 0.0f));
                auto tangent = normalize(ite(
                    abs(normal.z) < .999f,
                    cross(normal, make_float3(0.0f, 0.0f, 1.0f)),
                    make_float3(1.0f, 0.0f, 0.0f)));
                auto bitangent = cross(normal, tangent);
                direction = normalize(
                    tangent * (r * cos(phi)) +
                    bitangent * (r * sin(phi)) + normal * z);
                origin = hit_point + normal * 2e-3f;
                throughput *= albedo * .45f;
            };

            output.write(coord, make_float4(radiance, 1.0f));
            hits.write(coord.x + coord.y * dispatch_size_x(),
                       cast<uint>(any_hit));
        };

    auto distance_views = find_external_stages(
        coroutine.graph(), distance_schema);
    auto normal_views = find_external_stages(
        coroutine.graph(), normal_schema);
    auto bunny = make_neural_bunny_sdf();
    auto scene = make_neural_bunny_scene(bunny);
    auto scene_normal = make_neural_bunny_normal(scene);

    auto layout = CoroFrameStorageLayout::make_aos(
        coroutine.frame(), frame_count);
    auto frames = device.create_byte_buffer(layout.size_bytes);
    auto routes = device.create_buffer<uint>(frame_count);
    auto scheduled_routes = device.create_buffer<uint>(frame_count);
    auto io_plan = coro_frame_make_io_plan(
        coroutine.graph(), coroutine.frame().frame_field_count(),
        CoroFrameIOPlanConfig{.externalize_target_token = true});

    auto outgoing_routes = [&](size_t source) noexcept {
        return collect_external_stage_routes(coroutine.graph(), source);
    };

    auto entry_outputs = io_plan.transition_output_fields[0u];
    auto entry_routes = outgoing_routes(0u);
    Kernel1D generate = [&coroutine, layout, entry_outputs, entry_routes,
                         frame_count, options](
                            ByteBufferVar frame_storage, ImageFloat output,
                            BufferUInt hits, BufferUInt route_buffer) noexcept {
        auto index = dispatch_x();
        $if (index >= frame_count) { $return(); };
        auto x = index % options.width;
        auto y = index / options.width;
        auto frame = coroutine.instantiate(
            make_uint3(x, y, 0u),
            make_uint3(options.width, options.height, 1u));
        frame.target_token = 0u;
        coroutine.entry()(frame, output, hits);
        Var next = 0u;
        Var next_route = 0u;
        for (auto route : entry_routes) {
            $if (frame.target_token == route.token) {
                next = route.target;
                next_route = route.boundary;
            };
        }
        route_buffer.write(index, next_route);
        for (auto target = 0u; target < entry_outputs.size(); ++target) {
            $if (next == static_cast<uint>(target)) {
                coro_frame_store(
                    frame_storage, index, frame, layout, false,
                    luisa::span{entry_outputs[target]}, false, false);
            };
        }
    };

    auto generate_shader = device.compile(generate);
    Kernel1D snapshot_routes = [](BufferUInt source,
                                  BufferUInt destination) noexcept {
        auto index = dispatch_x();
        destination.write(index, source.read(index));
    };
    auto snapshot_routes_shader = device.compile(snapshot_routes);
    auto make_stage_shader =
        [&](const ExternalStageView &view, auto &&evaluate) {
            auto reconstruct_slots = merge_stage_slots(
                view.stage->reconstruct_slot_span());
            auto writeback_slots = merge_stage_slots(
                view.stage->required_writeback_slot_span());
            auto point = &view.binding("point");
            auto result_name = view.extension->schema() == distance_schema ?
                                   luisa::string_view{"sample"} :
                                   luisa::string_view{"normal"};
            auto result = &view.binding(result_name);
            auto route = static_cast<uint>(view.boundary->index + 1u);
            Kernel1D kernel = [&coroutine, layout, reconstruct_slots,
                               writeback_slots, point, result, route,
                               frame_count,
                               evaluate = std::forward<decltype(evaluate)>(
                                   evaluate)](
                                  ByteBufferVar frame_storage,
                                  BufferUInt scheduled_route_buffer) noexcept {
                auto index = dispatch_x();
                $if (index >= frame_count) { $return(); };
                $if (scheduled_route_buffer.read(index) != route) {
                    $return();
                };
                auto frame = CoroFrame::create(&coroutine.frame());
                coro_frame_load_into(
                    frame, frame_storage, index, layout, false,
                    luisa::span{reconstruct_slots}, false, false);
                auto p = point->read<float3>(frame);
                evaluate(*result, frame, p);
                coro_frame_store(
                    frame_storage, index, frame, layout, false,
                    luisa::span{writeback_slots}, false, false);
            };
            return device.compile(kernel);
        };

    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>>> distance_stages;
    distance_stages.reserve(distance_views.size());
    for (auto &&view : distance_views) {
        distance_stages.emplace_back(make_stage_shader(
            view,
            [&scene](const CoroSlotAccess &result, CoroFrame &frame,
                     Float3 p) noexcept {
                result.write<float2>(frame, scene(p));
            }));
    }
    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>>> normal_stages;
    normal_stages.reserve(normal_views.size());
    for (auto &&view : normal_views) {
        normal_stages.emplace_back(make_stage_shader(
            view,
            [&scene_normal](const CoroSlotAccess &result, CoroFrame &frame,
                            Float3 p) noexcept {
                result.write<float3>(frame, scene_normal(p));
            }));
    }

    auto make_resume_shader = [&](const ExternalStageView &view) {
        auto node = view.boundary->to_index;
        auto route = static_cast<uint>(view.boundary->index + 1u);
        auto input_slots = io_plan.input_fields[node];
        auto output_slots = io_plan.transition_output_fields[node];
        auto next_routes = outgoing_routes(node);
        Kernel1D kernel = [&coroutine, layout, node, route, input_slots,
                           output_slots, next_routes, frame_count](
                              ByteBufferVar frame_storage, ImageFloat output,
                              BufferUInt hits, BufferUInt route_buffer,
                              BufferUInt scheduled_route_buffer) noexcept {
            auto index = dispatch_x();
            $if (index >= frame_count) { $return(); };
            $if (scheduled_route_buffer.read(index) != route) { $return(); };
            auto frame = CoroFrame::create(&coroutine.frame());
            coro_frame_load_into(
                frame, frame_storage, index, layout, false,
                luisa::span{input_slots}, false, false);
            frame.target_token = CoroFrame::TERMINAL_TOKEN;
            coroutine[node](frame, output, hits);
            Var next = 0u;
            Var next_route = 0u;
            for (auto candidate : next_routes) {
                $if (frame.target_token == candidate.token) {
                    next = candidate.target;
                    next_route = candidate.boundary;
                };
            }
            route_buffer.write(index, next_route);
            for (auto target = 0u; target < output_slots.size(); ++target) {
                $if (next == static_cast<uint>(target)) {
                    coro_frame_store(
                        frame_storage, index, frame, layout, false,
                        luisa::span{output_slots[target]}, false, false);
                };
            }
        };
        return device.compile(kernel);
    };

    using ResumeShader =
        Shader1D<ByteBuffer, Image<float>, Buffer<uint>, Buffer<uint>,
                 Buffer<uint>>;
    luisa::vector<ResumeShader> distance_resumes;
    distance_resumes.reserve(distance_views.size());
    for (auto &&view : distance_views) {
        distance_resumes.emplace_back(make_resume_shader(view));
    }
    luisa::vector<ResumeShader> normal_resumes;
    normal_resumes.reserve(normal_views.size());
    for (auto &&view : normal_views) {
        normal_resumes.emplace_back(make_resume_shader(view));
    }
    Kernel2D hdr_to_ldr = [](ImageFloat source, ImageFloat destination) {
        auto coord = dispatch_id().xy();
        auto color = source.read(coord).xyz();
        color = pow(max(color, 0.0f), make_float3(1.0f / 2.2f));
        destination.write(coord, make_float4(saturate(color), 1.0f));
    };
    auto hdr_to_ldr_shader = device.compile(hdr_to_ldr);

    Clock clock;
    stream << generate_shader(frames, hdr, hit_mask, routes)
                  .dispatch(frame_count);
    // One loop iteration advances every active path through at most one
    // distance query and one normal query. Two bounces each contain at most
    // 96 distance steps, so 192 iterations are a complete static bound.
    for (auto iteration = 0u; iteration < 192u; ++iteration) {
        stream << snapshot_routes_shader(routes, scheduled_routes)
                      .dispatch(frame_count);
        for (auto &&shader : distance_stages) {
            stream << shader(frames, scheduled_routes).dispatch(frame_count);
        }
        for (auto &&shader : distance_resumes) {
            stream << shader(frames, hdr, hit_mask, routes,
                             scheduled_routes)
                          .dispatch(frame_count);
        }
        stream << snapshot_routes_shader(routes, scheduled_routes)
                      .dispatch(frame_count);
        for (auto &&shader : normal_stages) {
            stream << shader(frames, scheduled_routes).dispatch(frame_count);
        }
        for (auto &&shader : normal_resumes) {
            stream << shader(frames, hdr, hit_mask, routes,
                             scheduled_routes)
                          .dispatch(frame_count);
        }
    }

    luisa::vector<float> host_hdr(
        static_cast<size_t>(frame_count) * 4u);
    luisa::vector<uint> host_hits(frame_count);
    stream << hdr.copy_to(luisa::span{host_hdr})
           << hit_mask.copy_to(luisa::span{host_hits}) << synchronize();
    auto elapsed_ms = clock.toc();

    auto hit_count = static_cast<size_t>(std::count(
        host_hits.begin(), host_hits.end(), 1u));
    auto min_value = std::numeric_limits<float>::max();
    auto max_value = std::numeric_limits<float>::lowest();
    for (auto i = 0u; i < frame_count; ++i) {
        for (auto channel = 0u; channel < 3u; ++channel) {
            auto value = host_hdr[i * 4u + channel];
            LUISA_ASSERT(std::isfinite(value),
                         "Neural-SDF render produced a non-finite value at "
                         "pixel {}, channel {}.",
                         i, channel);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }
    LUISA_ASSERT(hit_count > frame_count / 20u &&
                     hit_count < frame_count * 19u / 20u,
                 "Neural-SDF path tracing hit coverage is invalid: {}/{}.",
                 hit_count, frame_count);
    LUISA_ASSERT(max_value - min_value > .1f,
                 "Neural-SDF path tracing output is unexpectedly uniform.");

    if (options.write_image) {
        luisa::vector<uint8_t> host_ldr(
            static_cast<size_t>(frame_count) * 4u);
        stream << hdr_to_ldr_shader(hdr, ldr)
                      .dispatch(options.width, options.height)
               << ldr.copy_to(luisa::span{host_ldr}) << synchronize();
        LUISA_ASSERT(
            stbi_write_png("coro_neural_sdf.png", options.width,
                           options.height, 4, host_ldr.data(), 0) != 0,
            "Failed to write 'coro_neural_sdf.png'.");
    }

    LUISA_INFO(
        "Neural-SDF external-stage path tracing passed on '{}': {}x{}, "
        "hits={}/{}, elapsed={:.3f} ms{}.",
        argv[1], options.width, options.height, hit_count, frame_count,
        elapsed_ms,
        options.write_image ? ", wrote coro_neural_sdf.png" : "");
    return 0;
}
