// Offline validation of the ImGuiWindow tiled compute 2D rasterizer without
// requiring a swapchain or ray-tracing support. Mirrors the GUIVertex layout,
// canonical watertight top-left edge evaluation, per-tile CSR triangle list,
// and front-to-back alpha compositing in src/gui/imgui_window.cpp.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/sugar.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

struct alignas(16u) GUIVertex {
    float px;
    float py;
    float pz;
    uint clip_idx;
    float2 uv;
    uint packed_color;
    uint tex_id;
};

[[nodiscard]] uint pack_rgba8(float r, float g, float b, float a) noexcept {
    auto to_u8 = [](float v) noexcept {
        return static_cast<uint>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f)) & 0xffu;
    };
    return to_u8(r) | (to_u8(g) << 8u) | (to_u8(b) << 16u) | (to_u8(a) << 24u);
}

}// namespace

LUISA_STRUCT(GUIVertex, px, py, pz, clip_idx, uv, packed_color, tex_id) {
    [[nodiscard]] auto color() const noexcept {
        auto r = (packed_color & 0xffu).cast<float>() / 255.f;
        auto g = ((packed_color >> 8u) & 0xffu).cast<float>() / 255.f;
        auto b = ((packed_color >> 16u) & 0xffu).cast<float>() / 255.f;
        auto a = ((packed_color >> 24u) & 0xffu).cast<float>() / 255.f;
        return make_float4(r, g, b, a);
    }
};

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    auto stream = device.create_stream(StreamTag::COMPUTE);

    constexpr uint w = 64u;
    constexpr uint h = 64u;
    constexpr uint tile_size = 16u;
    constexpr uint tiles_x = (w + tile_size - 1u) / tile_size;
    constexpr uint tiles_y = (h + tile_size - 1u) / tile_size;
    constexpr uint num_tiles = tiles_x * tiles_y;

    auto clear = device.compile<2>([](ImageFloat fb) noexcept {
        fb.write(dispatch_id().xy(), make_float4(0.f, 0.f, 0.f, 1.f));
    });

    auto compute_render = device.compile<2>(
        [tile_size](ImageFloat fb,
           UInt tx_count,
           BufferUInt tile_offsets,
           BufferUInt tile_tri_indices,
           BufferVar<GUIVertex> vertices,
           BindlessVar texture_array,
           BufferFloat4 clip_rects) noexcept {
            set_block_size(tile_size, tile_size, 1u);
            auto tid = dispatch_id().xy();
            auto tile_xy = tid / tile_size;
            auto tile_id = tile_xy.y * tx_count + tile_xy.x;
            auto range_begin = tile_offsets.read(tile_id);
            auto range_end = tile_offsets.read(tile_id + 1u);
            $if (range_begin == range_end) { $return(); };

            auto eval_edge = [](Float2 a, Float2 b, Float2 q, Float &out_e) noexcept {
                auto a_less = (a.y < b.y) | ((a.y == b.y) & (a.x < b.x));
                auto lo = select(b, a, a_less);
                auto hi = select(a, b, a_less);
                auto raw = (hi.x - lo.x) * (q.y - lo.y) - (hi.y - lo.y) * (q.x - lo.x);
                out_e = select(-raw, raw, a_less);
                auto is_top_left = (b.y < a.y) | ((b.y == a.y) & (b.x > a.x));
                return select(out_e > 0.f, out_e >= 0.f, is_top_left);
            };

            auto old = fb.read(tid).xyz();
            auto q = make_float2(tid) + make_float2(.5f, .5f);
            auto sample_sum = def(make_float3(0.f));
            auto beta = def(1.f);
            auto idx = def(range_end);
            $while (idx > range_begin & beta > 1e-3f) {
                idx -= 1u;
                auto tri_idx = tile_tri_indices.read(idx);
                auto base = tri_idx * 3u;
                auto v0 = vertices.read(base + 0u);
                auto clip = clip_rects.read(v0.clip_idx);
                $if (all(q >= clip.xy()) & all(q < clip.zw())) {
                    auto v1 = vertices.read(base + 1u);
                    auto v2 = vertices.read(base + 2u);
                    auto p0 = make_float2(v0.px, v0.py);
                    auto p1 = make_float2(v1.px, v1.py);
                    auto p2 = make_float2(v2.px, v2.py);
                    auto e0 = def(0.f);
                    auto e1 = def(0.f);
                    auto e2 = def(0.f);
                    auto inside = eval_edge(p1, p2, q, e0) &
                                  eval_edge(p2, p0, q, e1) &
                                  eval_edge(p0, p1, q, e2);
                    $if (inside) {
                        auto inv_sum = 1.f / max(e0 + e1 + e2, 1e-20f);
                        auto w0 = e0 * inv_sum;
                        auto w1 = e1 * inv_sum;
                        auto w2 = e2 * inv_sum;
                        auto c = w0 * v0->color() + w1 * v1->color() + w2 * v2->color();
                        auto tex_id = v0.tex_id;
                        $if (tex_id != 0u) {
                            auto uv = w0 * v0.uv + w1 * v1.uv + w2 * v2.uv;
                            c *= texture_array->tex2d(tex_id).sample(uv);
                        };
                        sample_sum += c.xyz() * beta * c.w;
                        beta *= 1.f - c.w;
                    };
                };
            };
            fb.write(tid, make_float4(sample_sum + beta * old, 1.f));
        });

    auto textures = device.create_bindless_array();
    auto tex = device.create_image<float>(PixelStorage::BYTE4, 2u, 2u);
    std::array<uint8_t, 16u> tex_bytes{
        0u, 255u, 0u, 255u,  0u, 255u, 0u, 255u,
        0u, 255u, 0u, 255u,  0u, 255u, 0u, 255u,
    };
    textures.emplace_on_update(1u, tex, Sampler::linear_point_edge());
    stream << tex.copy_from(tex_bytes.data()) << textures.update();

    std::vector<float4> clip_rects{
        make_float4(0.f, 0.f, 16.f, 16.f), // clip 0: top-left 16x16 scissor
        make_float4(0.f, 0.f, 64.f, 64.f), // clip 1: full image
    };

    std::vector<GUIVertex> verts;
    std::vector<uint4> tri_bounds;
    auto push_tri = [&](float2 p0, float2 p1, float2 p2,
                        uint clip_idx, uint col, uint tex_id) {
        auto area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if (area < 0.f) { std::swap(p1, p2); }
        auto mk = [&](float2 p, float2 uv) {
            return GUIVertex{p.x, p.y, 0.f, clip_idx, uv, col, tex_id};
        };
        verts.push_back(mk(p0, make_float2(0.5f, 0.5f)));
        verts.push_back(mk(p1, make_float2(0.5f, 0.5f)));
        verts.push_back(mk(p2, make_float2(0.5f, 0.5f)));
        auto clip = clip_rects[clip_idx];
        auto bmin = max(min(p0, min(p1, p2)), clip.xy());
        auto bmax = min(max(p0, max(p1, p2)), clip.zw());
        auto tx0 = static_cast<uint>(std::floor(bmin.x)) / tile_size;
        auto ty0 = static_cast<uint>(std::floor(bmin.y)) / tile_size;
        auto tx1 = std::min(tiles_x - 1u, static_cast<uint>(std::max(0.f, std::ceil(bmax.x) - 1.f)) / tile_size);
        auto ty1 = std::min(tiles_y - 1u, static_cast<uint>(std::max(0.f, std::ceil(bmax.y) - 1.f)) / tile_size);
        tri_bounds.push_back(make_uint4(tx0, ty0, tx1, ty1));
    };

    // 1. Opaque red triangle covering [4,60]x[4,60] clipped to top-left [0,16]x[0,16]
    push_tri(make_float2(4.f, 4.f), make_float2(60.f, 4.f), make_float2(4.f, 60.f),
             0u, pack_rgba8(1.f, 0.f, 0.f, 1.f), 0u);

    // 2. Semi-transparent blue quad [20,44]x[20,44] split along diagonal x == y
    //    Tests watertight shared diagonal: every pixel on x == y (center x+0.5 == y+0.5)
    //    must receive alpha = 0.5 exactly once (neither 0 nor 0.75).
    auto blue_half = pack_rgba8(0.f, 0.f, 1.f, 0.5f);
    push_tri(make_float2(20.f, 20.f), make_float2(44.f, 20.f), make_float2(44.f, 44.f), 1u, blue_half, 0u);
    push_tri(make_float2(20.f, 20.f), make_float2(44.f, 44.f), make_float2(20.f, 44.f), 1u, blue_half, 0u);

    // 3. Textured white quad [48,60]x[4,16] sampling green bindless texture #1
    auto white = pack_rgba8(1.f, 1.f, 1.f, 1.f);
    push_tri(make_float2(48.f, 4.f), make_float2(60.f, 4.f), make_float2(60.f, 16.f), 1u, white, 1u);
    push_tri(make_float2(48.f, 4.f), make_float2(60.f, 16.f), make_float2(48.f, 16.f), 1u, white, 1u);

    // 4. 20 stacked translucent layers at [4,16]x[48,60] (exceeds legacy depth-peel limit of 16)
    auto red_layer = pack_rgba8(1.f, 0.f, 0.f, 0.1f);
    for (int layer = 0; layer < 20; layer++) {
        push_tri(make_float2(4.f, 48.f), make_float2(16.f, 48.f), make_float2(16.f, 60.f), 1u, red_layer, 0u);
        push_tri(make_float2(4.f, 48.f), make_float2(16.f, 60.f), make_float2(4.f, 60.f), 1u, red_layer, 0u);
    }

    // Build CSR tile lists in submission order
    std::vector<uint> tile_counts(num_tiles, 0u);
    for (auto b : tri_bounds) {
        for (auto ty = b.y; ty <= b.w; ty++) {
            for (auto tx = b.x; tx <= b.z; tx++) {
                tile_counts[ty * tiles_x + tx]++;
            }
        }
    }
    std::vector<uint> tile_offsets(num_tiles + 1u, 0u);
    for (uint i = 0u; i < num_tiles; i++) {
        tile_offsets[i + 1u] = tile_offsets[i] + tile_counts[i];
    }
    std::vector<uint> tile_cursors(tile_offsets.begin(), tile_offsets.begin() + num_tiles);
    std::vector<uint> tile_tri_indices(tile_offsets.back(), 0u);
    for (uint tri_idx = 0u; tri_idx < static_cast<uint>(tri_bounds.size()); tri_idx++) {
        auto b = tri_bounds[tri_idx];
        for (auto ty = b.y; ty <= b.w; ty++) {
            for (auto tx = b.x; tx <= b.z; tx++) {
                tile_tri_indices[tile_cursors[ty * tiles_x + tx]++] = tri_idx;
            }
        }
    }

    auto vb = device.create_buffer<GUIVertex>(verts.size());
    auto cb = device.create_buffer<float4>(clip_rects.size());
    auto tob = device.create_buffer<uint>(tile_offsets.size());
    auto tib = device.create_buffer<uint>(tile_tri_indices.size());
    auto fb = device.create_image<float>(PixelStorage::FLOAT4, w, h);

    std::vector<float> pixels(w * h * 4u, 0.f);
    stream << vb.copy_from(verts.data())
           << cb.copy_from(clip_rects.data())
           << tob.copy_from(tile_offsets.data())
           << tib.copy_from(tile_tri_indices.data())
           << clear(fb).dispatch(w, h)
           << compute_render(fb, tiles_x, tob, tib, vb, textures, cb).dispatch(w, h)
           << fb.copy_to(pixels.data())
           << synchronize();

    auto px = [&](uint x, uint y, uint c) {
        return pixels[(y * w + x) * 4u + c];
    };

    // 1. Scissor-clipped red triangle
    expect(px(8u, 8u, 0u) > 0.99f) << "inside clipped triangle must be red";
    expect(px(40u, 8u, 0u) < 0.01f) << "scissor must clip +x";
    expect(px(8u, 40u, 0u) < 0.01f) << "scissor must clip +y";

    // 2. Watertight shared diagonal of semi-transparent blue quad (alpha = 128/255 ~ 0.502)
    float expected_half = static_cast<float>(std::lround(0.5f * 255.f)) / 255.f;
    for (uint d = 20u; d < 44u; d++) {
        auto diag_b = px(d, d, 2u);
        expect(std::abs(diag_b - expected_half) < 0.02f)
            << "shared diagonal must be watertight (no crack, no double blend)";
    }

    // 3. Bindless textured quad (white * green texture = green)
    expect(px(54u, 10u, 0u) < 0.01f && px(54u, 10u, 1u) > 0.99f && px(54u, 10u, 2u) < 0.01f)
        << "bindless texture sampling must modulate vertex color";

    // 4. 20 stacked layers at alpha ~= 26/255: expected = 1 - (1 - a)^20 ~= 0.883
    float layer_a = static_cast<float>(std::lround(0.1f * 255.f)) / 255.f;
    float expected_stack = 1.f - std::pow(1.f - layer_a, 20.f);
    expect(std::abs(px(10u, 54u, 0u) - expected_stack) < 0.02f)
        << "20 overlapping translucent layers must composite accurately without depth cap";

    LUISA_INFO("ImGui compute offline PASSED on backend '{}' (diag={:.3f}, stack20={:.3f}).",
               device.backend_name(), px(32u, 32u, 2u), px(10u, 54u, 0u));
    return 0;
}
