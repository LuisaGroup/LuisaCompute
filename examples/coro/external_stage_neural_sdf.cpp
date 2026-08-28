#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

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

void validate_neural_sdf_values(
    Device &device, Stream &stream,
    const Callable<float(float3)> &bunny) noexcept {
    // Independently evaluated from the original GLSL using column-major GLSL
    // mat4 construction and the exact nonlinear operator placement. This
    // catches transcription mistakes that can still produce a smooth but
    // completely different zero level set.
    luisa::vector<float3> points{
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(.1f, .2f, .3f),
        make_float3(-.4f, .1f, .2f),
        make_float3(.3f, -.2f, .4f),
        make_float3(-.7f, .2f, -.1f),
        make_float3(.9f, 0.0f, 0.0f),
        make_float3(0.0f, .8f, 0.0f),
        make_float3(0.0f, 0.0f, .8f)};
    luisa::vector<float> expected{
        -.161558315f, .075506943f, .066108586f, .118490206f,
        .171548502f, .522333055f, .226816883f, .118556469f};
    auto point_buffer = device.create_buffer<float3>(points.size());
    auto value_buffer = device.create_buffer<float>(points.size());
    Kernel1D evaluate = [&bunny](BufferVar<float3> input,
                                 BufferFloat output) noexcept {
        auto index = dispatch_x();
        output.write(index, bunny(input.read(index)));
    };
    auto evaluate_shader = device.compile(evaluate);
    luisa::vector<float> actual(points.size());
    stream << point_buffer.copy_from(luisa::span{points})
           << evaluate_shader(point_buffer, value_buffer)
                  .dispatch(static_cast<uint>(points.size()))
           << value_buffer.copy_to(luisa::span{actual}) << synchronize();
    for (auto i = 0u; i < actual.size(); ++i) {
        LUISA_ASSERT(
            std::abs(actual[i] - expected[i]) <= 2e-4f,
            "Neural bunny GLSL parity failed at probe {}: got {}, expected "
            "{}.",
            i, actual[i], expected[i]);
    }
}

class NeuralSdfHandler final
    : public WavefrontCoroSchedulerExtensionHandler<
          Image<float>, Buffer<uint>> {

private:
    struct StageKernel {
        size_t queue_index{0u};
        Shader1D<ByteBuffer, Buffer<uint>, uint, uint> shader;
    };

    Callable<float2(float3)> _scene;
    Callable<float3(float3)> _normal;
    luisa::vector<StageKernel> _kernels;

private:
    template<typename Evaluate>
    void _prepare_stage(
        const WavefrontCoroExtensionPrepareContext &context,
        const WavefrontCoroExtensionStage &stage,
        luisa::string_view result_name,
        Evaluate evaluate) noexcept {
        auto reconstruct_slots = stage.dataflow->reconstruct_slots;
        auto writeback_slots = stage.dataflow->required_def.slots;
        auto *point = &stage.binding("point");
        auto *result = &stage.binding(result_name);
        auto *desc = &context.frame_desc;
        Kernel1D kernel = [desc, point, result,
                           layout = context.frame_layout,
                           soa = context.global_memory_soa,
                           reconstruct_slots, writeback_slots,
                           evaluate = std::move(evaluate)](
                              ByteBufferVar frame_storage,
                              BufferUInt frame_indices,
                              UInt frame_capacity, UInt count) noexcept {
            auto x = dispatch_x();
            $if (x >= count) { $return(); };
            auto frame_index = frame_indices.read(x);
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(
                frame, frame_storage, frame_index, frame_capacity,
                layout, soa, luisa::span{reconstruct_slots},
                false, false);
            evaluate(*result, frame, point->read<float3>(frame));
            coro_frame_store(
                frame_storage, frame_index, frame_capacity, frame,
                layout, soa, luisa::span{writeback_slots},
                false, false);
        };
        auto label = luisa::format(
            "wavefront_extension_neural_sdf_{}", stage.queue_index);
        auto shader = coro::detail::coro_scheduler_label_shader(
            context.device.compile(
                kernel,
                coro::detail::coro_scheduler_shader_option(
                    context.shader_option, label)),
            label);
        _kernels.emplace_back(StageKernel{
            .queue_index = stage.queue_index,
            .shader = std::move(shader)});
    }

public:
    NeuralSdfHandler(
        Callable<float2(float3)> scene,
        Callable<float3(float3)> normal) noexcept
        : _scene{std::move(scene)}, _normal{std::move(normal)} {}

    [[nodiscard]] luisa::string_view name() const noexcept override {
        return "neural-sdf";
    }

    [[nodiscard]] bool can_handle(
        const WavefrontCoroExtensionStage &stage) const noexcept override {
        auto schema = stage.extension->schema();
        return stage.extension->version() == 1u &&
               (schema == distance_schema || schema == normal_schema);
    }

    void prepare(
        const WavefrontCoroExtensionPrepareContext &context,
        const WavefrontCoroExtensionStage &stage) noexcept override {
        if (stage.extension->schema() == distance_schema) {
            auto *scene = &_scene;
            _prepare_stage(
                context, stage, "sample",
                [scene](const CoroSlotAccess &result,
                        CoroFrame &frame, Float3 point) noexcept {
                    result.write<float2>(frame, (*scene)(point));
                });
        } else {
            auto *normal = &_normal;
            _prepare_stage(
                context, stage, "normal",
                [normal](const CoroSlotAccess &result,
                         CoroFrame &frame, Float3 point) noexcept {
                    result.write<float3>(frame, (*normal)(point));
                });
        }
    }

    void dispatch(
        const WavefrontCoroExtensionDispatchContext &context,
        ImageView<float>, BufferView<uint>) noexcept override {
        auto kernel = std::find_if(
            _kernels.begin(), _kernels.end(),
            [&](auto &&candidate) noexcept {
                return candidate.queue_index ==
                       context.stage.queue_index;
            });
        LUISA_ASSERT(kernel != _kernels.end(),
                     "Neural-SDF handler has no prepared stage {}.",
                     context.stage.queue_index);
        context.stream << kernel->shader(
                              context.frame_buffer,
                              context.frame_indices,
                              context.frame_capacity,
                              context.frame_count)
                              .dispatch(context.frame_count);
    }
};

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
            auto centered =
                (make_float2(coord) + .5f - resolution * .5f) /
                resolution.y;
            // Shadertoy's fragment origin is bottom-left, while the host-side
            // PNG receives row zero as the top scanline.
            auto uv = make_float2(centered.x, -centered.y);
            // Match the reference shader's default camera. Even with no
            // mouse input it pitches both the origin and ray direction by
            // 0.5 radians; looking straight down +X only shows an ambiguous
            // side projection of the learned field.
            constexpr auto camera_cos = 0.877582562f;
            constexpr auto camera_sin = 0.479425539f;
            Var origin = make_float3(
                -3.0f * camera_cos, 0.0f, 3.0f * camera_sin);
            auto camera_ray = normalize(make_float3(1.5f, uv.x, uv.y));
            Var direction = make_float3(
                camera_cos * camera_ray.x + camera_sin * camera_ray.z,
                camera_ray.y,
                camera_cos * camera_ray.z - camera_sin * camera_ray.x);
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

    auto bunny = make_neural_bunny_sdf();
    validate_neural_sdf_values(device, stream, bunny);
    auto scene = make_neural_bunny_scene(bunny);
    auto scene_normal = make_neural_bunny_normal(scene);
    WavefrontCoroScheduler<Image<float>, Buffer<uint>> scheduler{
        device, coroutine,
        WavefrontCoroSchedulerConfig{
            .thread_count = frame_count,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = false,
            .execution_block_size = 256u,
            .largest_continuation_first = true,
            .incremental_continuation_counts = true}};
    auto neural_sdf = luisa::make_shared<NeuralSdfHandler>(
        std::move(scene), std::move(scene_normal));
    scheduler.register_extension_handler(neural_sdf);

    Kernel2D hdr_to_ldr = [](ImageFloat source, ImageFloat destination) {
        auto coord = dispatch_id().xy();
        auto color = source.read(coord).xyz();
        color = pow(max(color, 0.0f), make_float3(1.0f / 2.2f));
        destination.write(coord, make_float4(saturate(color), 1.0f));
    };
    auto hdr_to_ldr_shader = device.compile(hdr_to_ldr);

    Clock clock;
    scheduler(hdr, hit_mask)
        .dispatch(options.width, options.height)(stream);

    luisa::vector<float> host_hdr(
        static_cast<size_t>(frame_count) * 4u);
    luisa::vector<uint> host_hits(frame_count);
    stream << hdr.copy_to(luisa::span{host_hdr})
           << hit_mask.copy_to(luisa::span{host_hits}) << synchronize();
    auto elapsed_ms = clock.toc();

    auto hit_count = static_cast<size_t>(std::count(
        host_hits.begin(), host_hits.end(), 1u));
    uint64_t hit_y_sum = 0u;
    for (auto i = 0u; i < frame_count; ++i) {
        if (host_hits[i] != 0u) {
            hit_y_sum += i / options.width;
        }
    }
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
    LUISA_ASSERT(
        hit_y_sum * 2u >
            static_cast<uint64_t>(hit_count) * (options.height - 1u),
        "Neural-SDF camera/image Y convention is inverted: hit-row sum {} "
        "for {} hits over {} rows.",
        hit_y_sum, hit_count, options.height);
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
