#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/passes/coroutine.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/ast/function_builder.h>
#include <stb/stb_image_write.h>
#include "common/reference_compare.h"
#include <luisa/gui/window.h>
#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

// clang-format off
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

namespace {

using FuncBuilder = luisa::compute::detail::FunctionBuilder;

}// namespace

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [-c <reference.png>].", argv[0]);
        exit(1);
    }
    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    Device device = context.create_device(argv[1]);

    // Load the Cornell Box scene.
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        std::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    std::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (auto i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{p[i + 0u], p[i + 1u], p[i + 2u]});
    }
    LUISA_INFO("Loaded mesh with {} shape(s) and {} vertices.",
               obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    std::vector<Mesh> meshes;
    std::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        LUISA_INFO("Processing shape '{}' at index {} with {} triangle(s).",
                   shape.name, index, triangle_count);
        std::vector<uint> indices;
        indices.reserve(t.size());
        for (auto i : t) { indices.emplace_back(i.vertex_index); }
        auto &&triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        auto &&mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build();
    }
    auto accel = device.create_accel({});
    for (auto &&m : meshes) { accel.emplace_back(m, make_float4x4(1.0f)); }
    stream << heap.update() << accel.build() << synchronize();

    float3 materials_array[] = {
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.14f, 0.45f, 0.091f),
        make_float3(0.63f, 0.065f, 0.05f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.725f, 0.71f, 0.68f),
        make_float3(0.0f),
    };
    auto materials = device.create_buffer<float3>(8);
    stream << materials.copy_from(luisa::span{materials_array, std::size(materials_array)});

    auto linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    auto tea = [](UInt v0, UInt v1) noexcept {
        auto s0 = def(0u);
        for (auto n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        auto p = dispatch_id().xy();
        auto state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    auto lcg = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto make_onb = [](const Float3 &normal) noexcept {
        auto binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    auto generate_ray = [](Float2 p) noexcept {
        static constexpr auto fov = radians(27.8f);
        static constexpr auto origin = make_float3(-0.01f, 0.995f, 5.0f);
        auto pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        auto direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    auto cosine_sample_hemisphere = [](Float2 u) noexcept {
        auto r = sqrt(u.x);
        auto phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    auto balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    auto spp_per_dispatch = device.backend_name() == "metal" ||
                                    device.backend_name() == "cpu" ||
                                    device.backend_name() == "fallback"
                                ? 1u
                                : 64u;

    // The path tracing kernel with $suspend markers.
    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        set_block_size(16u, 16u, 1u);
        auto &&heap_ref = heap;
        auto &&vertex_buffer_ref = vertex_buffer;
        auto &&materials_ref = materials;
        auto coord = dispatch_id().xy();
        auto frame_size = min(resolution.x, resolution.y).cast<float>();
        auto state = seed_image.read(coord).x;
        auto rx = lcg(state);
        auto ry = lcg(state);
        auto pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        auto radiance = def(make_float3(0.0f));
        $for (i, spp_per_dispatch) {
            auto ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            auto beta = def(make_float3(1.0f));
            auto pdf_bsdf = def(0.0f);
            constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
            auto light_area = length(cross(light_u, light_v));
            auto light_normal = normalize(cross(light_u, light_v));
            $for (depth, 10u) {
                $suspend("intersect");
                auto hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                auto triangle = heap_ref->buffer<Triangle>(hit.inst).read(hit.prim);
                auto p0 = vertex_buffer_ref->read(triangle.i0);
                auto p1 = vertex_buffer_ref->read(triangle.i1);
                auto p2 = vertex_buffer_ref->read(triangle.i2);
                auto p = triangle_interpolate(hit.bary, p0, p1, p2);
                auto n = normalize(cross(p1 - p0, p2 - p0));
                auto cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };
                auto albedo = materials_ref->read(hit.inst);

                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        auto pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        auto mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                $suspend("sample_light");
                auto ux_light = lcg(state);
                auto uy_light = lcg(state);
                auto p_light = light_position + ux_light * light_u + uy_light * light_v;
                auto pp = offset_ray_origin(p, n);
                auto pp_light = offset_ray_origin(p_light, light_normal);
                auto d_light = distance(pp, pp_light);
                auto wi_light = normalize(pp_light - pp);
                auto shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                auto occluded = accel.intersect_any(shadow_ray, {});
                auto cos_wi_light = dot(wi_light, n);
                auto cos_light = -dot(light_normal, wi_light);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    auto pdf_light = (d_light * d_light) / (light_area * cos_light);
                    auto pdf_bsdf = cos_wi_light * inv_pi;
                    auto mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    auto bsdf = albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };

                $suspend("sample_bsdf");
                Var<Onb> onb = make_onb(n);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;

                auto l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                auto q = max(l, 0.05f);
                auto r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        radiance /= static_cast<float>(spp_per_dispatch);
        $suspend("write_film");
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    // --- XIR coroutine split pipeline ---
    // 1. AST → XIR
    auto original_function = raytracing_kernel.function()->function();
    auto module = xir::ast_to_xir_translate(original_function, {});

    // 2. Inline all callables, then coro split.
    xir::inline_all_pass_run_on_module(module.get());
    xir::CoroutineSplitInfo split_info;
    for (auto *f : module->function_list()) {
        if (f->derived_function_tag() != xir::DerivedFunctionTag::KERNEL) continue;
        split_info = xir::coroutine_split_run_on_function(f);
        break;
    }
    LUISA_ASSERT(split_info.is_supported && split_info.changed, "coroutine_split failed");
    LUISA_INFO("Coroutine split: {} continuation(s), frame has {} slot(s).",
               split_info.continuations.size(), split_info.frame_slots.size());
    for (size_t i = 0; i < split_info.continuations.size(); i++) {
        auto &c = split_info.continuations[i];
        luisa::string suspends_str;
        for (auto s : c.outgoing_suspends) { suspends_str += std::to_string(s) + " "; }
        LUISA_INFO("  cont[{}] id={} suspends=[{}]", i, c.id, suspends_str);
    }

    // 4. Translate each continuation to AST (already structured CF from split).
    luisa::vector<luisa::shared_ptr<const FuncBuilder>> continuation_asts;
    continuation_asts.reserve(split_info.continuations.size());
    for (auto &cont : split_info.continuations) {
        auto *callable = static_cast<xir::FunctionDefinition *>(cont.callable);
        auto ast = xir::xir_to_ast_translate(*callable, {});
        continuation_asts.emplace_back(std::move(ast));
    }
    LUISA_INFO("Translated {} continuation(s) to AST.", continuation_asts.size());

    // 5. Build the state-machine scheduler kernel.
    auto num_conts = continuation_asts.size();
    auto frame_type = split_info.frame_type;
    auto scheduler_builder = FuncBuilder::define_kernel([&] {
        auto fb = FuncBuilder::current();
        fb->set_block_size(make_uint3(16u, 16u, 1u));
        auto frame = fb->local(frame_type);
        auto zero = fb->call(frame_type, CallOp::ZERO, {});
        fb->assign(frame, zero);

        auto cont0_func = Function{continuation_asts[0].get()};
        auto orig_args = original_function.arguments();
        auto orig_bindings = original_function.bound_arguments();
        luisa::vector<const Expression *> call_args;
        size_t orig_idx = 0;
        for (auto &arg : cont0_func.arguments()) {
            if (arg.type() == frame_type) {
                call_args.emplace_back(frame);
            } else if (arg.tag() == Variable::Tag::DISPATCH_ID ||
                       arg.tag() == Variable::Tag::THREAD_ID ||
                       arg.tag() == Variable::Tag::BLOCK_ID ||
                       arg.tag() == Variable::Tag::DISPATCH_SIZE) {
            } else {
                const Expression *expr = nullptr;
                if (orig_idx < orig_bindings.size()) {
                    auto &binding = orig_bindings[orig_idx];
                    luisa::visit(
                        [&]<typename T>(const T &b) {
                            if constexpr (std::is_same_v<T, luisa::monostate>) {
                                switch (arg.tag()) {
                                    case Variable::Tag::BUFFER: expr = fb->buffer(arg.type()); break;
                                    case Variable::Tag::TEXTURE: expr = fb->texture(arg.type()); break;
                                    case Variable::Tag::BINDLESS_ARRAY: expr = fb->bindless_array(); break;
                                    case Variable::Tag::ACCEL: expr = fb->accel(); break;
                                    default: expr = fb->argument(arg.type()); break;
                                }
                            } else if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                                expr = fb->buffer_binding(arg.type(), b.handle, b.offset, b.size);
                            } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                                expr = fb->texture_binding(arg.type(), b.handle, b.level);
                            } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                                expr = fb->bindless_array_binding(b.handle);
                            } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                                expr = fb->accel_binding(b.handle);
                            }
                        },
                        binding);
                } else {
                    switch (arg.tag()) {
                        case Variable::Tag::BUFFER: expr = fb->buffer(arg.type()); break;
                        case Variable::Tag::TEXTURE: expr = fb->texture(arg.type()); break;
                        case Variable::Tag::BINDLESS_ARRAY: expr = fb->bindless_array(); break;
                        case Variable::Tag::ACCEL: expr = fb->accel(); break;
                        default: expr = fb->argument(arg.type()); break;
                    }
                }
                call_args.emplace_back(expr);
                orig_idx++;
            }
        }

        fb->call(cont0_func, luisa::span{call_args});

        auto target_token_type = Type::of<uint>();

        // The old impl's scheduler: loop { switch(frame.target_token) { case i: node(i); } default: return; }
        // frame.target_token is field 0 of the frame struct.
        auto loop = fb->loop_();
        fb->push_scope(loop->body());
        {
            auto sw = fb->switch_(fb->member(target_token_type, frame, 0u));
            fb->push_scope(sw->body());
            for (size_t i = 1u; i < num_conts; i++) {
                auto case_val = fb->literal(target_token_type, static_cast<uint>(i));
                auto case_stmt = fb->case_(case_val);
                fb->push_scope(case_stmt->body());
                fb->call(Function{continuation_asts[i].get()}, luisa::span{call_args});
                fb->pop_scope(case_stmt->body());
            }
            auto default_stmt = fb->default_();
            fb->push_scope(default_stmt->body());
            fb->return_(nullptr);
            fb->pop_scope(default_stmt->body());
            fb->pop_scope(sw->body());
        }
        fb->pop_scope(loop->body());
    });

    // 6. Compile.
    auto raytracing_shader = device.compile(
        Kernel<2, Image<float>, Image<uint>, Accel, uint2>{scheduler_builder});

    // --- Standard rendering pipeline (same as path_tracing_ir.cpp) ---
    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        auto p = dispatch_id().xy();
        auto accum = accum_image.read(p);
        auto curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        auto coord = dispatch_id().xy();
        auto hdr = hdr_image.read(coord);
        auto ldr = linear_to_srgb(clamp(hdr.xyz() / hdr.w * scale, 0.0f, 1.0f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    auto make_sampler_shader = device.compile(make_sampler_kernel);

    static constexpr auto resolution = make_uint2(1024u);
    auto framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    auto accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    std::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    CommandList cmd_list;
    auto seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    cmd_list << clear_shader(accum_image).dispatch(resolution)
             << make_sampler_shader(seed_image).dispatch(resolution);

    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!opts.offline) {
        window = std::make_unique<Window>("path tracing (xir coroutine split)", resolution);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            }));
    }
    auto ldr_image = device.create_image<float>(
        (!opts.offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    auto last_time = 0.0;
    auto frame_count = 0u;
    Clock clock;
    bool infinite_render = !opts.offline;
    uint total_spp = opts.offline ? (opts.spp == 0u ? 1024u : opts.spp) : 0u;

    while (infinite_render || frame_count < total_spp) {
        cmd_list << raytracing_shader(framebuffer, seed_image, accel, resolution)
                        .dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer)
                        .dispatch(resolution);
        if (!opts.offline && swap_chain.has_value()) {
            cmd_list << hdr2ldr_shader(accum_image, ldr_image, 2.0f).dispatch(resolution);
            stream << cmd_list.commit()
                   << swap_chain->present(ldr_image);
            if (window->should_close()) { break; }
            window->poll_events();
        } else {
            stream << cmd_list.commit();
        }
        auto dt = clock.toc() - last_time;
        last_time = clock.toc();
        frame_count += spp_per_dispatch;
        LUISA_INFO("time: {} ms", dt);
    }
    stream << hdr2ldr_shader(accum_image, ldr_image, 2.0f).dispatch(resolution)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();

    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_path_tracing_split.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    if (opts.offline) {
        if (opts.compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution.x, resolution.y, 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    }
    return 0;
}
