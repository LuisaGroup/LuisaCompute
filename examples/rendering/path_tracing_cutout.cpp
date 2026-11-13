#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <iostream>
#include <string_view>

#include <stb/stb_image_write.h>

#include "common/reference_compare.h"
#include "common/path_tracing_sample_plan.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/stl/algorithm.h>

#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

struct Material {
    float3 albedo;
    float3 emission;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

enum class TraceMode {
    cutout_query,
    accept_query,
    opaque_query,
    direct,
    // Hardware opacity micromap ("OMM"): the cut-out of the two boxes is baked into
    // a micromap and discarded by the traversal itself, so this mode commits every
    // surface candidate the hardware reports (no software filter at all).
    omm,
};

[[nodiscard]] static luisa::optional<TraceMode> parse_trace_mode(
    luisa::string_view name) noexcept {
    if (name == "cutout-query") { return TraceMode::cutout_query; }
    if (name == "accept-query") { return TraceMode::accept_query; }
    if (name == "opaque-query") { return TraceMode::opaque_query; }
    if (name == "direct") { return TraceMode::direct; }
    if (name == "omm") { return TraceMode::omm; }
    return luisa::nullopt;
}

[[nodiscard]] static luisa::string_view trace_mode_name(
    TraceMode mode) noexcept {
    switch (mode) {
        case TraceMode::cutout_query: return "cutout-query";
        case TraceMode::accept_query: return "accept-query";
        case TraceMode::opaque_query: return "opaque-query";
        case TraceMode::direct: return "direct";
        case TraceMode::omm: return "omm";
    }
    return "invalid";
}

// ---------------------------------------------------------------------------
// Hardware opacity micromaps (--trace-mode omm)
// ---------------------------------------------------------------------------
// `filter_triangle_hit` below cuts the two boxes out with a *procedural,
// barycentric* stripe mask, and neither the Cornell Box OBJ nor `Mesh` carries a UV
// map (a mesh has no UV input at all), so the OMM path rebuilds those two boxes as
// **de-indexed** meshes - three vertices and three UVs per triangle - with
//     uv = {(0,0), (1,0), (0,1)} for triangle (v0, v1, v2).
// `TriangleHit::bary` is `(w1, w2)`, so in that frame `uv.x == bary.x` and
// `uv.y == bary.y`, and therefore
//     fract(10 * uv.x) < 0.5   (short box, was `fract(10 * h.bary.x) < .5`)
//     fract( 6 * uv.y) < 0.6   (tall box,  was `fract( 6 * h.bary.y) < .6`)
// are exactly the predicates the software filter evaluates - a mask texture with
// those values bakes to the same cut-out, modulo micro-triangle quantization.
struct OmmBox {
    // Accel instance index of the box (also its index in the material buffer).
    uint instance;
    // First de-indexed vertex of this box in the shared vertex buffer.
    uint first_vertex;
    uint triangle_count;
    // Stripes per triangle edge, and the axis they run along.
    uint stripe_frequency;
    bool stripes_along_v;
    float stripe_duty;
    luisa::string_view name;
};

// RGBA mask for one box: `channel = opaque ? 1 : 0`, replicated over all four
// channels so the bake may select any of them.
[[nodiscard]] static luisa::vector<float> make_omm_mask(
    const OmmBox &box, uint size) noexcept {
    luisa::vector<float> pixels(static_cast<size_t>(size) * size * 4u, 0.0f);
    for (auto y = 0u; y < size; y++) {
        auto v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
        for (auto x = 0u; x < size; x++) {
            auto u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
            auto t = box.stripes_along_v ? v : u;
            auto f = std::fmod(t * static_cast<float>(box.stripe_frequency), 1.0f);
            auto value = f < box.stripe_duty ? 1.0f : 0.0f;
            auto index = (static_cast<size_t>(y) * size + x) * 4u;
            for (auto c = 0u; c < 4u; c++) { pixels[index + c] = value; }
        }
    }
    return pixels;
}

// Host-side inverse of the micromap's space-filling curve, for the debug view: for
// every sample of a `grid x grid` barycentric raster, the micro triangle that
// covers it (or `omm_debug_outside_triangle` for samples outside the triangle).
// The bits themselves are then read *on the device* from `OpacityMicromap::data()`,
// which is what the debug view is meant to demonstrate.
static constexpr uint omm_debug_grid = 128u;
static constexpr uint omm_debug_outside_triangle = 0xffffffffu;

[[nodiscard]] static luisa::vector<uint> make_omm_debug_table(uint level) noexcept {
    luisa::vector<uint> table(static_cast<size_t>(omm_debug_grid) * omm_debug_grid,
                              omm_debug_outside_triangle);
    for (auto y = 0u; y < omm_debug_grid; y++) {
        auto v = (static_cast<float>(y) + 0.5f) / static_cast<float>(omm_debug_grid);
        for (auto x = 0u; x < omm_debug_grid; x++) {
            auto u = (static_cast<float>(x) + 0.5f) / static_cast<float>(omm_debug_grid);
            if (u + v > 1.0f) { continue; }
            table[static_cast<size_t>(y) * omm_debug_grid + x] =
                static_cast<uint>(OpacityMicromap::micro_triangle_index(level, make_float2(u, v)));
        }
    }
    return table;
}

// clang-format off
LUISA_STRUCT(Material, albedo, emission) {};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--offline] [--spp N] [--iterations N] [--max-registers N] [--max-spp-per-dispatch N] [--trace-mode cutout-query|accept-query|opaque-query|direct|omm] [--omm-level N] [--omm-conservative] [--omm-debug-view] [--ray-query-lowering pipeline|loop] [--capture-float4s N]. <backend>: cuda, dx, metal, metal4, vk, hip, fallback, simd", argv[0]);
        exit(1);
    }

    auto opts = luisa::ref::ExampleOptions::parse(argc, argv);
    if (!opts.valid()) {
        LUISA_WARNING("Invalid command line: {}", opts.error_message);
        return 1;
    }
    // Keep the example backend-neutral. The HIP backend owns its native
    // traversal occupancy policy; this override remains available for explicit
    // resource-sensitivity experiments.
    auto max_registers = 0u;
    auto capture_float4_count = 0u;
    auto trace_mode = TraceMode::cutout_query;
    auto enable_ray_query_pipeline = true;
    auto force_ray_query_pipeline = false;
    // OMM-only knobs; ignored (and inert) in every other trace mode.
    auto omm_level = 5u;
    auto omm_debug_view = false;
    // `filter_triangle_hit` tests the *hit point*, i.e. the micro triangle the hit
    // falls into, so the centroid rule is the one that reproduces it. The
    // conservative rule (cut only when all three corners are transparent) never lets
    // the cut-out grow past the mask and therefore cuts distinctly less.
    auto omm_conservative = false;
    for (auto i = 2; i < argc; i++) {
        auto option = luisa::string_view{argv[i]};
        if (option == "--max-registers") {
            if (i + 1 >= argc) {
                LUISA_WARNING("Missing value for {}.", option);
                return 1;
            }
            auto value = luisa::string_view{argv[++i]};
            auto parsed = luisa::ref::parse_uint32_option_value(value);
            if (!parsed) {
                LUISA_WARNING("Invalid value '{}' for {}.", value, option);
                return 1;
            }
            max_registers = *parsed;
        } else if (option == "--trace-mode") {
            if (i + 1 >= argc) {
                LUISA_WARNING("Missing value for {}.", option);
                return 1;
            }
            auto parsed = parse_trace_mode(argv[++i]);
            if (!parsed) {
                LUISA_WARNING("Invalid --trace-mode '{}'.", argv[i]);
                return 1;
            }
            trace_mode = *parsed;
        } else if (option == "--omm-level") {
            if (i + 1 >= argc) {
                LUISA_WARNING("Missing value for {}.", option);
                return 1;
            }
            auto value = luisa::string_view{argv[++i]};
            auto parsed = luisa::ref::parse_uint32_option_value(value);
            if (!parsed || *parsed > OpacityMicromap::max_representable_subdivision_level) {
                LUISA_WARNING(
                    "Invalid value '{}' for {}; expected an integer in "
                    "[0, {}].",
                    value, option,
                    OpacityMicromap::max_representable_subdivision_level);
                return 1;
            }
            omm_level = *parsed;
        } else if (option == "--omm-debug-view") {
            omm_debug_view = true;
        } else if (option == "--omm-conservative") {
            omm_conservative = true;
        } else if (option == "--ray-query-lowering") {
            if (i + 1 >= argc) {
                LUISA_WARNING("Missing value for {}.", option);
                return 1;
            }
            auto lowering = luisa::string_view{argv[++i]};
            if (lowering == "pipeline") {
                enable_ray_query_pipeline = true;
                force_ray_query_pipeline = true;
            } else if (lowering == "loop") {
                enable_ray_query_pipeline = false;
                force_ray_query_pipeline = false;
            } else {
                LUISA_WARNING(
                    "Invalid --ray-query-lowering '{}'; expected "
                    "'pipeline' or 'loop'.",
                    lowering);
                return 1;
            }
        } else if (option == "--capture-float4s") {
            if (i + 1 >= argc) {
                LUISA_WARNING("Missing value for {}.", option);
                return 1;
            }
            auto value = luisa::string_view{argv[++i]};
            auto parsed = luisa::ref::parse_uint32_option_value(value);
            if (!parsed || *parsed > 64u) {
                LUISA_WARNING(
                    "Invalid value '{}' for {}; expected an integer in "
                    "[0, 64].",
                    value, option);
                return 1;
            }
            capture_float4_count = *parsed;
        }
    }
    LUISA_INFO("Trace microbenchmark mode: {}; ray-query lowering: {}; "
               "mutable callback state: {} float4 ({} source byte(s); "
               "IFT input/output payload is reported by the XIR pass).",
               trace_mode_name(trace_mode),
               !enable_ray_query_pipeline ?
                   "stateful loop (forced)" :
               force_ray_query_pipeline ?
                   "pipeline/IFT (forced)" :
                   "automatic",
               capture_float4_count,
               capture_float4_count * static_cast<uint>(sizeof(float4)));

    Device device = context.create_device(argv[1]);

    // Hardware opacity micromaps are optional, so `omm` degrades to the software
    // cut-out on a backend that cannot build one instead of failing the run.
    // Nothing here runs for the default trace mode.
    static constexpr uint omm_mask_size = 256u;
    auto omm_max_level = 0u;
    if (trace_mode == TraceMode::omm) {
        if (!OpacityMicromap::is_supported(device)) {
            LUISA_WARNING_WITH_LOCATION(
                "Backend '{}' has no opacity micromap support "
                "(VK_KHR/EXT_opacity_micromap); falling back to "
                "--trace-mode cutout-query.",
                device.backend_name());
            trace_mode = TraceMode::cutout_query;
        } else {
            omm_max_level = OpacityMicromap::max_subdivision_level(
                device, OpacityMicromap::Format::TwoState);
            if (omm_level > omm_max_level) {
                LUISA_WARNING_WITH_LOCATION(
                    "This device supports two-state opacity micromaps up to "
                    "subdivision level {}; clamping --omm-level {} to it "
                    "(the stripe edges will be coarser than requested).",
                    omm_max_level, omm_level);
                omm_level = omm_max_level;
            }
            if (capture_float4_count != 0u) {
                LUISA_WARNING_WITH_LOCATION(
                    "{} mutable callback float4(s) perturb the software "
                    "cut-out through `filter_triangle_hit`'s capture bias, a "
                    "baked micromap cannot be perturbed that way, so `omm` and "
                    "`cutout-query` differ by design in this configuration.",
                    capture_float4_count);
            }
            LUISA_INFO("Opacity micromap: two-state, subdivision level {} "
                       "(device maximum {}), {}x{} mask, {} rule; the traversal "
                       "discards the cut-out, the shader commits every candidate.",
                       omm_level, omm_max_level, omm_mask_size, omm_mask_size,
                       omm_conservative ? "conservative (all three corners)" :
                                          "centroid");
        }
    }

    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        luisa::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (uint i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{
            p[i + 0u],
            p[i + 1u],
            p[i + 2u]});
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    // The instance layout of the Cornell Box: …, short box, tall box, light.
    auto shape_count = static_cast<uint>(obj_reader.GetShapes().size());
    auto light_inst = shape_count - 1u;
    auto tall_inst = shape_count - 2u;
    auto short_inst = shape_count - 3u;

    // OMM path: re-emit the two cut-out boxes as de-indexed triangles appended to
    // the *shared* vertex buffer. Appending (instead of giving the boxes their own
    // vertex buffer) keeps the shading path below valid, which reads positions
    // through `vertex_buffer` indexed by `hit.inst`'s triangle buffer. The UV map is
    // per vertex and per triangle: `uv = {(0,0), (1,0), (0,1)}` for `(v0, v1, v2)`,
    // which makes `uv.x` the barycentric weight of vertex 1 and `uv.y` the weight of
    // vertex 2, i.e. exactly `TriangleHit::bary`.
    //
    // `bake()` addresses the UV buffer with the triangle's vertex indices, exactly
    // like a vertex buffer (`uv[triangle.i0]`, …), so the UV array has to live in the
    // same index space as `vertices` - one entry per vertex, not one per box.
    luisa::vector<OmmBox> omm_boxes;
    luisa::vector<float2> vertex_uvs;
    Buffer<float2> uv_buffer;// only created in `omm` mode
    if (trace_mode == TraceMode::omm) {
        // The non-box vertices are never sampled by a bake; (0,0) keeps the array
        // parallel to `vertices`.
        vertex_uvs.resize(vertices.size(), make_float2(0.0f, 0.0f));
        auto &&shapes = obj_reader.GetShapes();
        auto add_box = [&](uint instance, uint frequency, bool along_v,
                           float duty, luisa::string_view name) {
            auto &&shape_indices = shapes[instance].mesh.indices;
            auto first_vertex = static_cast<uint>(vertices.size());
            for (auto k = static_cast<size_t>(0u); k < shape_indices.size(); k++) {
                auto v = static_cast<size_t>(shape_indices[k].vertex_index) * 3u;
                vertices.emplace_back(float3{p[v + 0u], p[v + 1u], p[v + 2u]});
                switch (k % 3u) {
                    case 0u: vertex_uvs.emplace_back(make_float2(0.0f, 0.0f)); break;
                    case 1u: vertex_uvs.emplace_back(make_float2(1.0f, 0.0f)); break;
                    default: vertex_uvs.emplace_back(make_float2(0.0f, 1.0f)); break;
                }
            }
            omm_boxes.emplace_back(OmmBox{
                .instance = instance,
                .first_vertex = first_vertex,
                .triangle_count = static_cast<uint>(shape_indices.size() / 3u),
                .stripe_frequency = frequency,
                .stripes_along_v = along_v,
                .stripe_duty = duty,
                .name = name});
            LUISA_INFO("OMM box '{}' (instance {}): {} de-indexed triangle(s), "
                       "{} stripes along {}, duty {}.",
                       name, instance, omm_boxes.back().triangle_count,
                       frequency, along_v ? "v" : "u", duty);
        };
        add_box(short_inst, 10u, false, 0.5f, "short box");
        add_box(tall_inst, 6u, true, 0.6f, "tall box");
    }

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(opts.offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    if (!vertex_uvs.empty()) {
        uv_buffer = device.create_buffer<float2>(vertex_uvs.size());
        stream << uv_buffer.copy_from(luisa::span{vertex_uvs}) << synchronize();
    }
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    // `omm` mode only: one mask texture and one micromap per cut-out box, parallel to
    // `omm_boxes`. They are declared here - before `accel` below - because a micromap
    // must outlive every BLAS built from the mesh that binds it.
    luisa::vector<Image<float>> omm_masks;
    luisa::vector<OpacityMicromap> omm_micromaps;
    auto omm_option = OpacityMicromap::Option{
        .format = OpacityMicromap::Format::TwoState,
        .subdivision_level = omm_level,
        .resolve_boundary_in_any_hit = false,
        .alpha_cutoff = 0.5f,
        // The rule that turns three mask samples into one bit: the centroid one cuts
        // the micro triangle the hit point falls into (which is what
        // `filter_triangle_hit` evaluates), the conservative one only cuts micro
        // triangles whose three corners are *all* transparent.
        .conservative = omm_conservative,
        .index_format = OpacityMicromap::IndexFormat::NoIndex};
    auto omm_box_of = [&](uint instance) noexcept -> OmmBox * {
        for (auto &box : omm_boxes) {
            if (box.instance == instance) { return &box; }
        }
        return nullptr;
    };
    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        std::vector<tinyobj::index_t> const &t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (tinyobj::index_t i : t) { indices.emplace_back(i.vertex_index); }
        auto box = omm_box_of(index);
        if (box != nullptr) {
            // De-indexed: every triangle owns its three vertices, which is what makes
            // a per-triangle UV parameterisation expressible at all.
            indices.clear();
            indices.reserve(3u * box->triangle_count);
            for (auto k = 0u; k < box->triangle_count; k++) {
                auto base = box->first_vertex + 3u * k;
                indices.emplace_back(base + 0u);
                indices.emplace_back(base + 1u);
                indices.emplace_back(base + 2u);
            }
        }
        Buffer<Triangle> &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        heap.emplace_on_update(index, triangle_buffer);
        if (box == nullptr) {
            Mesh &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
            stream << triangle_buffer.copy_from(luisa::span{indices})
                   << mesh.build();
        } else {
            auto mask_pixels = make_omm_mask(*box, omm_mask_size);
            auto &mask = omm_masks.emplace_back(device.create_image<float>(
                PixelStorage::FLOAT4, omm_mask_size, omm_mask_size));
            auto &micromap = omm_micromaps.emplace_back(
                device.create_opacity_micromap(box->triangle_count, omm_option));
            // One dispatch of the built-in baker: it samples `mask` at every micro
            // triangle of every triangle of `triangle_buffer` - addressing the UVs as
            // `uv_buffer[triangle.i0..i2]` - and packs the bits into
            // `micromap.data()`. Synchronise so the mask may be released with the
            // scope once the bits are final.
            stream << triangle_buffer.copy_from(luisa::span{indices})
                   << mask.copy_from(luisa::span{mask_pixels})
                   << micromap.bake(mask.view(0u), uv_buffer.view(),
                                    triangle_buffer.view())
                          .commit()
                   << synchronize();
            LUISA_INFO("Baked the '{}' cut-out into a micromap: {} triangle(s) x "
                       "{} micro triangle(s) = {} bytes of packed bits.",
                       box->name, box->triangle_count,
                       OpacityMicromap::micro_triangle_count(omm_level),
                       OpacityMicromap::data_byte_size(
                           OpacityMicromap::Format::TwoState, omm_level,
                           box->triangle_count));
            // The mesh carries the micromap *handle*; `MeshCutoutMask` refuses a
            // micromap without bits, and the traversal - not the shader - discards
            // the cut-out.
            Mesh &mesh = meshes.emplace_back(device.create_mesh(
                vertex_buffer, triangle_buffer, MeshCutoutMask{micromap}));
            stream << mesh.build();
        }
    }

    AccelOption accel_option{.hint = AccelOption::UsageHint::FAST_TRACE,
                             .allow_compaction = true,
                             .allow_update = true};
    Accel accel = device.create_accel(accel_option);
    auto mesh_count = static_cast<uint>(meshes.size());
    LUISA_ASSERT(mesh_count == shape_count, "The accel must have one instance per shape.");
    for (auto i = 0u; i < mesh_count; i++) {
        // Only the two alpha-cutout boxes need surface-candidate filtering.
        // Marking the room and light opaque lets ray queries auto-commit them.
        // In `omm` mode the boxes must stay non-opaque too: an opaque instance
        // override (`FORCE_OPAQUE`) would make the traversal ignore the micromap.
        auto opaque = trace_mode == TraceMode::opaque_query ||
                      trace_mode == TraceMode::direct ||
                      (i != short_inst && i != tall_inst);
        accel.emplace_back(meshes[i], make_float4x4(1.0f), 0xffu, opaque);
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    if (omm_debug_view && !omm_boxes.empty()) {
        // "The bits are the bindable surface": a micromap is not a texture and has no
        // descriptor type, so the only way to look at one from a shader is
        // `OpacityMicromap::data()`, an ordinary `BufferView<uint>`. This kernel
        // unpacks the two-state bits; the *spatial* mapping (which micro triangle
        // covers a barycentric sample) comes from a host table built with the public
        // `OpacityMicromap::micro_triangle_index()`, because the micromap's
        // space-filling curve has no closed form in the DSL.
        Kernel2D omm_debug_kernel = [](
                                        BufferUInt bits,
                                        BufferUInt coverage,
                                        UInt words_per_triangle,
                                        ImageFloat image) noexcept {
            set_block_size(8u, 8u, 1u);
            UInt2 p = dispatch_id().xy();
            UInt triangle = p.y / omm_debug_grid;
            UInt row = p.y % omm_debug_grid;
            UInt index = coverage.read(row * omm_debug_grid + p.x);
            Float3 color = make_float3(0.08f);// outside the triangle
            $if (index != omm_debug_outside_triangle) {
                UInt word = bits.read(triangle * words_per_triangle + (index >> 5u));
                UInt opaque = (word >> (index & 31u)) & 1u;
                color = ite(opaque == 0u,
                            make_float3(0.10f, 0.10f, 0.35f),// cut out
                            make_float3(0.90f, 0.90f, 0.85f));// opaque
            };
            image.write(p, make_float4(color, 1.0f));
        };
        auto omm_debug_shader = device.compile(omm_debug_kernel);
        auto coverage_table = make_omm_debug_table(omm_level);
        Buffer<uint> coverage_buffer = device.create_buffer<uint>(coverage_table.size());
        stream << coverage_buffer.copy_from(luisa::span{coverage_table})
               << synchronize();
        auto words_per_triangle = static_cast<uint>(
            (OpacityMicromap::micro_triangle_count(omm_level) + 31u) / 32u);
        for (auto b = 0u; b < omm_boxes.size(); b++) {
            auto &&box = omm_boxes[b];
            auto &&micromap = omm_micromaps[b];
            auto size = make_uint2(omm_debug_grid, omm_debug_grid * box.triangle_count);
            Image<float> omm_debug_image = device.create_image<float>(PixelStorage::BYTE4, size);
            luisa::vector<std::array<uint8_t, 4u>> host_debug(size.x * size.y);
            stream << omm_debug_shader(micromap.data(), coverage_buffer,
                                       words_per_triangle, omm_debug_image)
                          .dispatch(size)
                   << omm_debug_image.copy_to(luisa::span{host_debug})
                   << synchronize();
            auto filename = luisa::format(
                "test_path_tracing_cutout_omm_{}.png", box.instance);
            stbi_write_png(filename.c_str(),
                           static_cast<int>(size.x), static_cast<int>(size.y),
                           4, host_debug.data(), 0);
            LUISA_INFO("OMM debug view for the '{}': {} triangle(s) x {} micro "
                       "triangle(s) = {} bytes of packed bits, written to '{}'; "
                       "white = opaque, blue = cut out by the hardware.",
                       box.name, box.triangle_count,
                       OpacityMicromap::micro_triangle_count(omm_level),
                       OpacityMicromap::data_byte_size(
                           OpacityMicromap::Format::TwoState, omm_level,
                           box.triangle_count),
                       filename);
        }
    }

    luisa::vector<Material> materials;
    materials.reserve(accel.size());
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.45f, 0.11f, 0.52f), make_float3(0.0f)}); // short box
    materials.emplace_back(Material{make_float3(0.025f, 0.21f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light
    Buffer<Material> material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.copy_from(luisa::span{materials});

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return clamp(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                            12.92f * x,
                            x <= 0.00031308f),
                     0.0f, 1.0f);
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3 &normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr float fov = radians(27.8f);
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        Float3 direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    Callable filter_triangle_hit = [&](Var<TriangleHit> h,
                                       Float capture_bias) noexcept {
        Bool valid = def(true);
        $if (h.inst == tall_inst) {
            valid = fract(6.f * h.bary.y + capture_bias) < .6f;
        }
        $elif (h.inst == short_inst) {
            valid = fract(10.f * h.bary.x + capture_bias) < .5f;
        };
        return valid;
    };

    auto default_max_spp_per_dispatch =
        device.backend_name() == "metal" ||
                device.backend_name() == "fallback" ?
            1u :
            64u;
    auto max_spp_per_dispatch = opts.max_spp_per_dispatch.value_or(
        default_max_spp_per_dispatch);
    bool infinite_render = !opts.offline && opts.spp == 0u;
    auto sample_plan = luisa::ref::PathTracingSamplePassPlan{
        .total_spp = opts.offline ? (opts.spp == 0u ? luisa::ref::DEFAULT_PATH_TRACING_SPP : opts.spp) : opts.spp,
        .max_spp_per_dispatch = max_spp_per_dispatch,
        .infinite = infinite_render,
    };

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution, UInt dispatch_spp) noexcept {
        set_block_size(16u, 16u, 1u);
        auto make_capture_state = [&](Expr<Ray> query_ray) noexcept {
            luisa::vector<Float4> state;
            state.reserve(capture_float4_count);
            auto ray_value = def(query_ray);
            for (auto i = 0u; i < capture_float4_count; i++) {
                auto lane = static_cast<float>(i + 1u);
                state.emplace_back(def(make_float4(
                    ray_value->origin() +
                        ray_value->direction() * (lane * 0.0009765625f),
                    ray_value->t_min() + lane * 0.00390625f)));
            }
            return state;
        };
        auto update_capture_state = [&](luisa::vector<Float4> &state,
                                        Var<TriangleHit> hit) noexcept {
            Float checksum = def(0.0f);
            for (auto i = 0u; i < capture_float4_count; i++) {
                auto lane = static_cast<float>(i + 1u);
                auto candidate_value = make_float4(
                    hit.bary,
                    cast<float>(hit.prim & 255u),
                    cast<float>(hit.inst & 255u));
                state[i] = state[i] *
                               make_float4(0.96875f, 0.953125f,
                                           0.9375f, 0.921875f) +
                           candidate_value * (lane * 0.000244140625f);
                checksum += dot(
                    state[i],
                    make_float4(0.25f, 0.125f, 0.0625f,
                                0.03125f) /
                        lane);
            }
            return fract(abs(checksum) * 0.0009765625f) * 0.05f;
        };
        auto trace_closest = [&](Expr<Ray> query_ray) noexcept {
            if (trace_mode == TraceMode::direct) {
                auto surface_hit = accel.intersect(query_ray, {});
                return def<CommittedHit>(
                    surface_hit.inst, surface_hit.prim, surface_hit.bary,
                    ite(surface_hit->miss(),
                        static_cast<uint32_t>(HitType::Miss),
                        static_cast<uint32_t>(HitType::Surface)),
                    surface_hit.committed_ray_t);
            }
            if (trace_mode == TraceMode::opaque_query) {
                return accel.traverse(query_ray, {}).trace();
            }
            // `accept_query` and `omm` both commit every surface candidate the
            // traversal reports. The difference is *why*: `accept_query` has no
            // cut-out at all, while `omm` had the cut-out discarded by the micromap
            // *inside* the traversal. They deliberately share the code path, so
            // re-applying `filter_triangle_hit` here would cut twice with a
            // different (ray-dependent) predicate.
            if (trace_mode == TraceMode::accept_query ||
                trace_mode == TraceMode::omm) {
                return accel.traverse(query_ray, {})
                    .on_surface_candidate([](auto &candidate) noexcept {
                        candidate.commit();
                    })
                    .trace();
            }
            auto capture_state = make_capture_state(query_ray);
            return accel.traverse(query_ray, {})
                .on_surface_candidate([&](auto &candidate) noexcept {
                    auto candidate_hit = candidate.hit();
                    auto capture_bias = update_capture_state(
                        capture_state, candidate_hit);
                    $if (filter_triangle_hit(candidate_hit, capture_bias)) {
                        candidate.commit();
                    };
                })
                .trace();
        };
        auto trace_any = [&](Expr<Ray> query_ray) noexcept {
            if (trace_mode == TraceMode::direct) {
                return accel.intersect_any(query_ray, {});
            }
            if (trace_mode == TraceMode::opaque_query) {
                return !accel.traverse_any(query_ray, {}).trace()->miss();
            }
            if (trace_mode == TraceMode::accept_query ||
                trace_mode == TraceMode::omm) {
                return !accel.traverse_any(query_ray, {})
                            .on_surface_candidate([](auto &candidate) noexcept {
                                candidate.commit();
                            })
                            .trace()
                            ->miss();
            }
            auto capture_state = make_capture_state(query_ray);
            return !accel.traverse_any(query_ray, {})
                        .on_surface_candidate([&](auto &candidate) noexcept {
                            auto candidate_hit = candidate.hit();
                            auto capture_bias = update_capture_state(
                                capture_state, candidate_hit);
                            $if (filter_triangle_hit(
                                     candidate_hit, capture_bias)) {
                                candidate.commit();
                            };
                        })
                        .trace()
                        ->miss();
        };
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float3 radiance = def(make_float3(0.0f));
        $for (i, dispatch_spp) {
            Float rx = lcg(state);
            Float ry = lcg(state);
            Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
            Float light_area = length(cross(light_u, light_v));
            Float3 light_normal = normalize(cross(light_u, light_v));
            $for (depth, 10u) {

                // trace
                Var<CommittedHit> hit = trace_closest(ray);
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float4x4 m = accel.instance_transform(hit.inst);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                p = (m * make_float4(p, 1.f)).xyz();
                Float3 ng = normalize(cross(p1 - p0, p2 - p0));
                ng = (m * make_float4(ng, 0.f)).xyz();
                Float3 ns = faceforward(ng, ray->direction(), ng);
                Float cos_wo = dot(-ray->direction(), ns);
                Var<Material> material = material_buffer->read(hit.inst);

                // hit light
                $if (hit.inst == light_inst) {
                    $if (dot(light_normal, ray->direction()) <= 0.f) {// front
                        $if (depth == 0u) {
                            radiance += light_emission;
                        }
                        $else {
                            Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                            Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                            radiance += mis_weight * beta * light_emission;
                        };
                    };
                    $break;
                };

                // sample light
                Float ux_light = lcg(state);
                Float uy_light = lcg(state);
                Float3 p_light = light_position + ux_light * light_u + uy_light * light_v;
                Float3 pp = offset_ray_origin(p, ng);
                Float3 pp_light = offset_ray_origin(p_light, light_normal);
                Float d_light = distance(pp, pp_light);
                Float3 wi_light = normalize(pp_light - pp);
                Var<Ray> shadow_ray = make_ray(offset_ray_origin(pp, ng), wi_light, 0.f, d_light);
                Bool occluded = trace_any(shadow_ray);
                Float cos_wi_light = dot(wi_light, ns);
                Float cos_light = -dot(light_normal, wi_light);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                    Float pdf_bsdf = cos_wi_light * inv_pi;
                    Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    Float3 bsdf = material.albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };

                // sample BSDF
                Var<Onb> onb = make_onb(ns);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= material.albedo;// * cos_wi * inv_pi / pdf_bsdf => * 1.f

                // rr
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        radiance /= dispatch_spp.cast<float>();
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), dispatch_spp.cast<float>()));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float4 curr = curr_image.read(p);
        accum_image.write(p, accum + make_float4(curr.xyz() * curr.w, curr.w));
    };

    Callable aces_tonemapping = [](Float3 x) noexcept {
        static constexpr float a = 2.51f;
        static constexpr float b = 0.03f;
        static constexpr float c = 2.43f;
        static constexpr float d = 0.59f;
        static constexpr float e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = hdr.xyz() / hdr.w * scale;
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto clear_shader = device.compile(clear_kernel);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto accumulate_shader = device.compile(accumulate_kernel);
    ShaderOption raytracing_option{.max_registers = max_registers};
    raytracing_option.enable_ray_query_pipeline =
        enable_ray_query_pipeline;
    raytracing_option.force_ray_query_pipeline =
        force_ray_query_pipeline;
    Clock raytracing_compile_clock;
    auto raytracing_shader = device.compile(
        raytracing_kernel, raytracing_option);
    auto raytracing_compile_time_ms = raytracing_compile_clock.toc();
    LUISA_INFO("Ray-query shader compile time: {} ms.",
               raytracing_compile_time_ms);
    auto make_sampler_shader = device.compile(make_sampler_kernel);

    static constexpr uint2 resolution = make_uint2(1024u);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);

    luisa::unique_ptr<Window> window;
    luisa::optional<Swapchain> swap_chain;
    if (!opts.offline) {
        window = luisa::make_unique<Window>("path tracing", resolution);
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
    Image<float> ldr_image = device.create_image<float>(
        (!opts.offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    uint64_t frame_count = 0u;
    auto iteration_count = opts.offline ? opts.iterations : 1u;
    luisa::vector<double> rendering_times;
    rendering_times.reserve(iteration_count);
    for (auto iteration = 0u; iteration < iteration_count; iteration++) {
        CommandList initialize;
        initialize << clear_shader(accum_image).dispatch(resolution)
                   << make_sampler_shader(seed_image).dispatch(resolution);
        stream << initialize.commit()
               << synchronize();

        // Keep offline/reference runs reproducible while preserving a fresh
        // animation sequence for the interactive example. Resetting the seed
        // per offline iteration gives the performance comparison identical
        // rays and acceleration-structure updates.
        std::mt19937 rand{opts.offline ? 42u : std::random_device{}()};
        std::normal_distribution<float> dist{0.f, 1.f};
        frame_count = 0u;
        double last_time = 0.0;
        Clock clock;
        while (sample_plan.has_next(frame_count)) {
            auto dispatch_spp = sample_plan.next_dispatch_spp(frame_count);
            float4x4 t = translation(make_float3(0.f, dist(rand) * .03f + .1f, 0.f));
            accel.set_transform_on_update(tall_inst, t);
            CommandList commands;
            commands << accel.build(AccelBuildRequest::PREFER_UPDATE)
                     << raytracing_shader(framebuffer, seed_image, accel, resolution, dispatch_spp)
                            .dispatch(resolution)
                     << accumulate_shader(accum_image, framebuffer)
                            .dispatch(resolution);
            if (!opts.offline && swap_chain.has_value()) {
                commands << hdr2ldr_shader(accum_image, ldr_image, 1.0f, swap_chain->backend_storage() != PixelStorage::BYTE4).dispatch(resolution);
                stream << commands.commit()
                       << swap_chain->present(ldr_image);
                if (window->should_close()) { break; }
                window->poll_events();
            } else {
                stream << commands.commit();
            }
            auto now = clock.toc();
            LUISA_INFO("time: {} ms", now - last_time);
            last_time = now;
            frame_count += dispatch_spp;
        }
        stream << synchronize();
        auto rendering_time_ms = clock.toc();
        rendering_times.emplace_back(rendering_time_ms);
        LUISA_INFO(
            "Ray-query rendering iteration {}/{}: {} ms for {} spp; "
            "throughput: {} spp/s.",
            iteration + 1u, iteration_count, rendering_time_ms, frame_count,
            frame_count / rendering_time_ms * 1000.0);
    }
    stream << hdr2ldr_shader(accum_image, ldr_image, 1.0f, false).dispatch(resolution)
           << ldr_image.copy_to(luisa::span{host_image})
           << synchronize();

    auto sorted_rendering_times = rendering_times;
    luisa::sort(sorted_rendering_times.begin(), sorted_rendering_times.end());
    auto middle = sorted_rendering_times.size() / 2u;
    auto median_rendering_time_ms = sorted_rendering_times[middle];
    if (sorted_rendering_times.size() % 2u == 0u) {
        median_rendering_time_ms =
            (sorted_rendering_times[middle - 1u] + median_rendering_time_ms) *
            0.5;
    }
    auto sample_throughput = frame_count / median_rendering_time_ms * 1000.0;
    LUISA_INFO(
        "Ray-query rendering median: {} ms for {} spp; throughput: {} spp/s.",
        median_rendering_time_ms, frame_count, sample_throughput);
    LUISA_INFO("FPS: {}", sample_throughput);
    stbi_write_png("test_path_tracing_cutout.png", resolution.x, resolution.y, 4, host_image.data(), 0);
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
