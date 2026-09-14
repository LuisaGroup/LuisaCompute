// Rendering integration for shared suspending callables.
// Covers nested texture filtering, divergent reflection paths, volume integration,
// layered materials, all schedulers, and frame growth with repeated call sites.
#include "ut/ut.hpp"
#include "coro_test_utils.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/coro/coro_frame_storage.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <filesystem>
#include <cstdlib>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
constexpr uint32_t width = 37u, height = 29u, count = width * height;
using Render = Coroutine<void(Buffer<float4>, Buffer<float4>)>;

void test_render(Device &device, uint32_t scene) {
    auto stream = device.create_stream();
    auto texture = device.create_buffer<float4>(256u);
    auto output = device.create_buffer<float4>(count);
    auto reference = device.create_buffer<float4>(count);
    luisa::vector<float4> texels(256u);
    for (uint32_t i = 0; i < 256u; ++i) {
        texels[i] = make_float4(float(i % 16u) / 15.f, float(i / 16u) / 15.f,
                                float((i * 13u) % 31u) / 30.f, 1.f);
    }
    stream << texture.copy_from(luisa::span{texels});
    // Build identical arithmetic with suspension enabled/disabled. The direct
    // kernel is an independent execution path through normal callable lowering.
    auto make_sample = [scene](bool suspend) {
        return Callable<float3(Buffer<float4>, float2)>{[=](BufferFloat4 tex, Float2 uv) {
            if (scene != 2u) { luisa::compute::detail::FunctionBuilder::current()->mark_noinline(); }
            UInt2 xy = make_uint2(fract(abs(uv)) * 16.f);
            UInt index = xy.y * 16u + xy.x;
            Float3 first = tex.read(index).xyz();
            if (suspend) {
                $if ((index & 3u) != 0u) { $suspend("texture_filter"); };
            }
            return first * 0.625f + tex.read((index + 17u) % 256u).xyz() * 0.375f;
        }};
    };
    auto sample = make_sample(true);
    auto direct_sample = make_sample(false);
    auto make_material = [scene](auto &lookup) {
        return Callable<float3(Buffer<float4>, float3)>{[&](BufferFloat4 tex, Float3 p) {
            if (scene != 2u) { luisa::compute::detail::FunctionBuilder::current()->mark_noinline(); }
            auto a = lookup(tex, p.xy() * 0.3f + 0.51f);
            auto b = lookup(tex, p.yz() * 0.7f + 0.23f);
            return a * 0.7f + b * 0.3f;
        }};
    };
    // Two, four, and eight shared levels, with early returns and a shared leaf
    // called both through the chain and directly from intermediate levels.
    // The deepest case leaves ordinary callable inlining enabled: suspension
    // discovery must preserve shared source tokens without a noinline hint.
    using Material = Callable<float3(Buffer<float4>, float3)>;
    auto depth = std::array{2u, 4u, 8u}[scene];
    luisa::vector<Material> materials, direct_materials;
    materials.emplace_back(make_material(sample));
    direct_materials.emplace_back(make_material(direct_sample));
    auto append_layer = [scene](auto &chain, auto &lookup, uint32_t level) {
        Material next = [&](BufferFloat4 tex, Float3 p) {
            if (scene != 2u) { luisa::compute::detail::FunctionBuilder::current()->mark_noinline(); }
            Float3 saved = p * 0.03f + 0.2f;
            auto first = chain.back()(tex, p);
            $if (p.x > 0.f) { $return(first * 0.8f + saved * 0.2f); };
            auto second = lookup(tex, p.xz() + float(level) * 0.07f);
            return first * 0.7f + second * 0.2f + saved * 0.1f;
        };
        chain.emplace_back(std::move(next));
    };
    for (uint32_t level = 2u; level < depth; ++level) {
        append_layer(materials, sample, level);
        append_layer(direct_materials, direct_sample, level);
    }
    auto &material = materials.back();
    auto &direct_material = direct_materials.back();
    auto render = [scene](auto &shade, BufferFloat4 &out, BufferFloat4 &tex) {
        UInt i = dispatch_x();
        Float2 uv = (make_float2(make_uint2(i % width, i / width)) + 0.5f) / make_float2(float(width), float(height)) * 2.f - 1.f;
        Float3 origin = make_float3(uv * 0.85f, -3.f);
        Float3 direction = normalize(make_float3(uv * 0.2f, 1.f));
        Float3 radiance = make_float3(0.f);
        Float3 throughput = make_float3(1.f);
        if (scene == 0u) {
            // Analytic sphere reflections and a textured environment. Rays have
            // different path lengths; caller ray state survives nested samples.
            $for (bounce, 4u) {
                auto b = dot(origin, direction);
                auto discriminant = b * b - dot(origin, origin) + 1.f;
                $if (discriminant <= 0.f) {
                    radiance += throughput * shade(tex, direction * 2.f);
                    $break;
                };
                auto t = -b - sqrt(discriminant);
                $if (t <= 0.001f) { $break; };
                auto p = origin + t * direction;
                auto normal = normalize(p);
                auto albedo = shade(tex, p);
                radiance += throughput * albedo * (0.15f + 0.65f * max(dot(normal, normalize(make_float3(-1.f, 2.f, -3.f))), 0.f));
                throughput *= albedo * 0.4f;
                direction = direction - 2.f * dot(direction, normal) * normal;
                origin = p + normal * 0.002f;
            };
        } else if (scene == 1u) {
            // Variable-length front-to-back volume integration with nested
            // filtering and early opacity termination.
            Float transmittance = 1.f;
            $for (step, 5u + i % 9u) {
                auto p = origin + direction * (1.5f + step.cast<float>() * 0.17f);
                auto color = shade(tex, p);
                auto density = clamp(0.2f + 0.6f * color.z, 0.f, 0.9f);
                radiance += color * (transmittance * density);
                transmittance *= 1.f - density;
                $if (transmittance < 0.02f) { $break; };
            };
        } else {
            // Multiple material callsites with a dynamically indexed local
            // aggregate, branch-specific layers, and loop-carried color.
            Var<std::array<float3, 3u>> layers;
            layers[0u] = shade(tex, origin);
            layers[1u] = shade(tex, direction);
            layers[2u] = make_float3(0.1f);
            $if (uv.x * uv.y > 0.f) {
                layers[2u] = shade(tex, origin + direction);
            };
            $for (layer, 3u) {
                radiance = radiance * 0.35f + layers[(layer + i) % 3u] * 0.65f;
            };
        }
        out.write(i, make_float4(radiance, 1.f));
    };
    Render task = [&](BufferFloat4 out, BufferFloat4 tex) { render(material, out, tex); };
    Kernel1D direct = [&](BufferFloat4 out, BufferFloat4 tex) { render(direct_material, out, tex); };
    ShaderOption shader_option;
    shader_option.enable_cache = false;
    auto shader = device.compile(direct, shader_option);
    luisa::vector<float4> expected(count), actual(count);
    stream << shader(reference, texture).dispatch(count)
           << reference.copy_to(luisa::span{expected}) << synchronize();
    auto poison = [&] {
        std::fill(actual.begin(), actual.end(), make_float4(std::numeric_limits<float>::quiet_NaN()));
        stream << output.copy_from(luisa::span{actual});
    };
    auto bytes = task.frame_desc().frame_type()->size();
    auto frame_budget = std::array{192u, 256u, 448u}[scene];
    LUISA_INFO("Rendering scene {}: frame={} bytes, fields={}, nodes={}, callsites={}, analysis_states={}",
               scene, bytes, task.frame_desc().frame_field_count(), task.graph().node_count(),
               task.graph().call_graph().edges.size(), task.graph().call_graph().analysis_state_count);
    expect(task.graph().call_graph().functions.size() == depth + 1u);
    auto root_sites = std::array{2u, 1u, 3u}[scene];
    expect(task.graph().call_graph().edges.size() == root_sites + 2u * (depth - 1u));
    size_t source_token_count = 0u;
    for (auto &&function : task.graph().call_graph().functions) {
        for (auto token : function.resume_tokens) {
            ++source_token_count;
            expect(token == task.trigger_token(1u));
        }
    }
    expect(source_token_count == 1u) << "call depth and inlining eligibility cannot introduce suspend tokens";
    expect(task.graph().node_count() == 2u) << "all material sites share the texture resume";
    // These small scenes carry rays/colors/indices, never the texture contents.
    // Budgets leave room for control-state changes while catching excess spills
    // and keeping the deepest scene within the persistent scheduler's memory.
    expect(bytes <= frame_budget) << "unexpected frame inflation";
    auto check = [&](const char *scheduler) {
        stream << output.copy_to(luisa::span{actual}) << synchronize();
        float error = 0.f, minimum = 1e30f, maximum = -1e30f;
        bool finite = true;
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t c = 0; c < 4u; ++c) {
                finite &= std::isfinite(actual[i][c]) && std::isfinite(expected[i][c]);
                error = std::max(error, std::abs(actual[i][c] - expected[i][c]));
            }
            minimum = std::min(minimum, expected[i].x);
            maximum = std::max(maximum, expected[i].x);
        }
        LUISA_INFO("Rendering scene {} / {}: max_error={}", scene, scheduler, error);
        if (auto *directory = std::getenv("LUISA_CORO_RENDER_OUTPUT")) {
            std::filesystem::create_directories(directory);
            luisa::vector<uint8_t> pixels(count * 3u);
            for (uint32_t i = 0; i < count; ++i) {
                for (uint32_t c = 0; c < 3u; ++c) {
                    pixels[i * 3u + c] = static_cast<uint8_t>(
                        std::clamp(std::isfinite(actual[i][c]) ? actual[i][c] : 0.f, 0.f, 1.f) * 255.f + 0.5f);
                }
            }
            auto file = std::filesystem::path{directory} / luisa::format("scene_{}_{}.png", scene, scheduler).c_str();
            expect(stbi_write_png(file.string().c_str(), width, height, 3, pixels.data(), width * 3u) != 0);
        }
        expect(finite && error < 2e-4f) << scheduler;
        expect(maximum - minimum > 0.05f) << "reference must be nondegenerate";
    };
    {
        StateMachineCoroSchedulerConfig config;
        config.shader_option = shader_option;
        StateMachineCoroScheduler scheduler{device, task, config};
        poison();
        stream << scheduler(output, texture).dispatch(count);
        check("state_machine");
    }
    for (bool soa : {false, true}) {
        WavefrontCoroSchedulerConfig config;
        config.thread_count = 67u;
        config.shader_option = shader_option;
        config.global_memory_soa = soa;
        config.frame_buffer_compaction = true;
        WavefrontCoroScheduler scheduler{device, task, config};
        poison();
        stream << scheduler(output, texture).dispatch(count);
        check(soa ? "wavefront_soa" : "wavefront_aos");
    }
    {
        PersistentThreadsCoroSchedulerConfig config;
        config.thread_count = 256u;
        // Keep the 420-byte deep-chain frame below a 32 KiB shared-memory
        // budget, including the scheduler's queue and bookkeeping.
        config.block_size = 64u;
        config.shader_option = shader_option;
        PersistentThreadsCoroScheduler scheduler{device, task, config};
        poison();
        stream << scheduler(output, texture).dispatch(count);
        check("persistent");
    }
}

void test_frame_scaling(Device &device) {
    Callable<float(float)> leaf = [](Float x) {
        luisa::compute::detail::FunctionBuilder::current()->mark_noinline();
        Var<std::array<float, 32u>> scratch;
        $for (j, 32u) { scratch[j] = x + j.cast<float>(); };
        $suspend("shared_scratch");
        Float sum = 0.f;
        $for (j, 32u) { sum += scratch[j]; };
        return sum;
    };
    size_t first_size = 0u;
    auto output = device.create_buffer<float>(count);
    auto stream = device.create_stream();
    for (uint32_t sites : {1u, 4u, 16u}) {
        Coroutine<void(Buffer<float>)> task = [&](BufferFloat out) {
            Float sum = 0.f;
            for (uint32_t site = 0; site < sites; ++site) {
                sum += leaf(dispatch_x().cast<float>() + float(site));
            }
            out.write(dispatch_x(), sum);
        };
        auto bytes = task.frame_desc().frame_type()->size();
        if (sites == 1u) { first_size = bytes; }
        LUISA_INFO("Frame scaling: {} callsites, {} bytes, {} fields", sites, bytes, task.frame_desc().frame_field_count());
        expect(task.graph().node_count() == 2u);
        expect(task.graph().call_graph().edges.size() == sites);
        expect(task.graph().call_graph().functions.size() == 2u);
        expect(task.graph().call_graph().functions.front().resume_tokens.empty());
        auto &tokens = task.graph().call_graph().functions.back().resume_tokens;
        expect(tokens.size() == 1u);
        if (tokens.size() == 1u) { expect(tokens.front() == task.trigger_token(1u)); }
        // One 128-byte activation plus bounded scalar control/accumulator state.
        // Sixteen sites must not allocate sixteen copies of scratch.
        expect(bytes <= first_size + 16u) << "frame grows with callsite count";
        WavefrontCoroSchedulerConfig config;
        config.thread_count = 67u;
        config.frame_buffer_compaction = true;
        config.shader_option.enable_cache = false;
        WavefrontCoroScheduler scheduler{device, task, config};
        luisa::vector<float> actual(count, std::numeric_limits<float>::quiet_NaN());
        stream << output.copy_from(luisa::span{actual});
        stream << scheduler(output).dispatch(count);
        stream << output.copy_to(luisa::span{actual}) << synchronize();
        bool correct = true;
        for (uint32_t i = 0; i < count; ++i) {
            float expected = 32.f * (float(sites * i) + float(sites * (sites - 1u)) * 0.5f) + float(sites) * 496.f;
            correct &= std::isfinite(actual[i]) && std::abs(actual[i] - expected) < 0.01f;
        }
        expect(correct);
    }
}
}// namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    auto dc = luisa::test::coro_test::create_device(options);
    "coro_reflection_rendering"_test = [&] { test_render(dc.device, 0u); };
    "coro_volume_rendering"_test = [&] { test_render(dc.device, 1u); };
    "coro_layered_material_rendering"_test = [&] { test_render(dc.device, 2u); };
    "coro_shared_activation_frame_scaling"_test = [&] { test_frame_scaling(dc.device); };
}
