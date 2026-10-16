// Offline validation of the ImGuiWindow rasterization mesh layout and shaders,
// without a swapchain (DX present is unavailable in this environment). Mirrors
// the GUIMeshVertex / GUIVarying packing and the vertex/pixel stages in
// src/gui/imgui_window.cpp. Backends without the raster extension are skipped,
// exactly as ImGuiWindow falls back to ray tracing there.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/vertex_attribute.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <array>
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

struct alignas(16u) GUIMeshVertex {
    float4 position;
    float4 normal;
    float4 tangent;
    float4 color;
};
static_assert(sizeof(GUIMeshVertex) == 64u);

struct GUIVarying {
    float4 position;
    float2 uv;
    float4 color;
    float2 clip_min;
    float2 clip_max;
    float tex_id;
    float2 screen;
};

LUISA_STRUCT(GUIMeshVertex, position, normal, tangent, color) {
    [[nodiscard]] auto pixel() const noexcept { return position.xy(); }
    [[nodiscard]] auto clip_min() const noexcept { return normal.xy(); }
    [[nodiscard]] auto clip_max() const noexcept { return make_float2(normal.z, normal.w); }
    [[nodiscard]] auto tex_uv() const noexcept { return tangent.xy(); }
    [[nodiscard]] auto tex_id() const noexcept { return tangent.z; }
    [[nodiscard]] auto rgba() const noexcept { return color; }
};
LUISA_STRUCT(GUIVarying, position, uv, color, clip_min, clip_max, tex_id, screen) {};

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    constexpr uint w = 64u, h = 64u;

    // Only backends that expose the raster extension can run this test. The
    // mirrored ImGuiWindow raster pipeline is unreachable without it (the window
    // falls back to ray tracing, see _rasterization_possible in
    // src/gui/imgui_window.cpp) and the raster entry points fail closed, so skip
    // here as well instead of requesting a pipeline the backend cannot create.
    if (device.extension<RasterExt>() == nullptr) {
        LUISA_INFO("Backend '{}' does not provide '{}'; "
                   "skipping the ImGui raster offline test.",
                   device.backend_name(), RasterExt::name);
        return 0;
    }
    auto is_vk = device.backend_name() == luisa::string_view{"vk"};

    MeshFormat mf;
    const VertexAttribute attrs[]{
        {VertexAttributeType::Position, PixelFormat::RGBA32F},
        {VertexAttributeType::Normal, PixelFormat::RGBA32F},
        {VertexAttributeType::Tangent, PixelFormat::RGBA32F},
        {VertexAttributeType::Color, PixelFormat::RGBA32F},
    };
    mf.emplace_vertex_stream(attrs);

    RasterStageKernel vert = [](Var<GUIMeshVertex> v, Float2 fb_size, Float y_flip) noexcept {
        Var<GUIVarying> o;
        auto p = v->pixel();
        auto ndc_x = p.x / fb_size.x * 2.f - 1.f;
        auto ndc_y = y_flip * (1.f - p.y / fb_size.y * 2.f);
        o.position = make_float4(ndc_x, ndc_y, 0.f, 1.f);
        o.uv = v->tex_uv();
        o.color = v->rgba();
        o.clip_min = v->clip_min();
        o.clip_max = v->clip_max();
        o.tex_id = v->tex_id();
        o.screen = p;
        return o;
    };
    RasterStageKernel pixel = [](Var<GUIVarying> i, BindlessVar texture_array) noexcept {
        $if (any(i.screen < i.clip_min - .5f) | any(i.screen > i.clip_max - .5f)) {
            raster_discard();
        };
        auto c = i.color;
        auto tex_id = i.tex_id.cast<uint>();
        $if(tex_id != 0u) {
            c *= texture_array->tex2d(tex_id).sample(i.uv);
        };
        return make_float4(c.xyz() * c.w, c.w);
    };
    RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};
    auto shader = [&] {
        if (is_vk) {
            device.compile_to(kernel, mf, "imgui_raster_offline");
            return device.load_raster_shader<float2, float, BindlessArray>("imgui_raster_offline");
        } else {
            return device.compile(kernel, mf);
        }
    }();
    expect(static_cast<bool>(shader)) << "raster shader compiled";

    // opaque red triangle covering pixels x[4,32] y[4,32]; scissor clip exposes
    // only the top-left 16x16 region, so pixels beyond it are discarded.
    auto vb = device.create_buffer<GUIMeshVertex>(3);
    auto mkv = [](float x, float y) {
        auto v = GUIMeshVertex{};
        v.position = make_float4(x, y, 0.f, 0.f);
        v.normal = make_float4(0.f, 0.f, 16.f, 16.f);// clip top-left 16x16
        v.tangent = make_float4(0.f, 0.f, 0.f, 0.f); // uv 0, tex_id 0
        v.color = make_float4(1.f, 0.f, 0.f, 1.f);   // opaque red
        return v;
    };
    std::array<GUIMeshVertex, 3> verts{mkv(4.f, 4.f), mkv(60.f, 4.f), mkv(4.f, 60.f)};
    auto fb = device.create_image<float>(PixelStorage::FLOAT4, w, h, 1u, false, true);
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto textures = device.create_bindless_array();
    auto clear = device.compile<2>([](ImageFloat img) noexcept {
        img.write(dispatch_id().xy(), make_float4(0.f, 0.f, 0.f, 1.f));
    });
    VertexBufferView vv{vb};
    auto fb2 = make_float2(w, h);
    RasterState state{};
    state.cull_mode = CullMode::None;
    std::array<float, w * h * 4u> pixels{};
    auto ch = [&](uint x, uint y, uint c) { return static_cast<uint>(pixels[(y * w + x) * 4u + c] * 255.f); };
    // The backend-correct flip puts the (top-left-origin) triangle at pixel
    // (8,8); the opposite flip moves it to the bottom (DX: +1 top, VK: -1
    // top — the two backends' rasterizer NDC y axes are inverted). The GUI
    // raster path is gated to DX (Vulkan's is AOT-only), where the correct
    // value is +1.
    auto top_flip = is_vk ? -1.f : 1.f;
    auto bot_flip = -top_flip;
    struct Res { uint nz; uint inside; uint clipped_x; uint clipped_y; } r_top, r_bot;
    auto probe = [&](float flip, Res &out) {
        auto m = luisa::vector<RasterMesh>{};
        m.emplace_back(luisa::span<VertexBufferView const>{&vv, 1u}, 3u, 1u, 0u);
        stream << vb.copy_from(luisa::span{verts.data(), verts.size()})
               << clear(fb).dispatch(w, h)
               << shader(fb2, flip, textures).draw(std::move(m), mf,
                                                   Viewport{0u, 0u, w, h}, state, nullptr, fb)
               << synchronize();
        stream << fb.copy_to(luisa::span{pixels.data(), pixels.size()}) << synchronize();
        out.nz = 0u;
        for (uint y = 0u; y < h; y++) for (uint x = 0u; x < w; x++) if (ch(x, y, 0u) > 20u) out.nz++;
        out.inside = ch(8u, 8u, 0u);
        out.clipped_x = ch(40u, 8u, 0u);
        out.clipped_y = ch(8u, 40u, 0u);
    };
    probe(top_flip, r_top);
    probe(bot_flip, r_bot);
    // top-left pixel is red only with the correct flip
    expect(r_top.inside > 200u) << "correct flip should render at (8,8)";
    expect(r_bot.inside < 30u) << "wrong flip must not render at (8,8)";
    // scissor (16x16 clip) discards pixels outside it regardless of orientation
    expect(r_top.clipped_x == 0u) << "scissor must clip +x";
    expect(r_top.clipped_y == 0u) << "scissor must clip +y";
    expect(r_top.nz > 0u && r_top.nz < 400u) << "clipped area smaller than full triangle";
    LUISA_INFO("raster offline PASSED: top_inside={} bot_inside={} nz={}", r_top.inside, r_bot.inside, r_top.nz);
    return 0;
}
