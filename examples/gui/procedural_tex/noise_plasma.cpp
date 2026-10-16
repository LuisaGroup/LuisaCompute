// Procedural Texture Generator -- diamond-square / plasma fractal implementation.
//
// Iterative midpoint displacement with two GPU ping-pong float buffers. Every
// destination point of every pass is a pure function of its own (coordinate,
// level, seed): bilinear interpolation of the previously written coarser grid
// plus a jitter that halves each level. No atomics, no partial writes, so
// repeated generate() calls are bit-exact (required by the headless test).
//
// Grid scheme: L = ceil(log2(max(w, h))) levels. At level l the logical grid
// is gw(l) x gh(l) with g(l) = max(1, w >> (L - l)) (and similarly for h), so
// level 1 is at most 2x2 and level L is the full w x h image grid. The fill
// pass writes the level-1 grid (the 4 corner control points -- for the
// square, power-of-two default resolution these are exactly the buffer
// corners); each refine pass fully rewrites the next grid from the previous
// one; the blit pass maps the finest grid to the image.

#include "noise_plasma.h"

#include <algorithm>
#include <cmath>

#include "noise_common.h"

namespace procedural_tex {

Plasma::Plasma(Device &device) {
    _device = &device;

    auto hash = make_hash01();

    // Fill pass: the 2x2 (or smaller) level-1 grid. Index is the linear point
    // id; x = id % gw, y = id / gw.
    Kernel1D fill = [&](BufferVar<float> dst, UInt gw, UInt gh, UInt w, UInt seed) noexcept {
        auto id = dispatch_id().x;
        auto x = id % gw;
        auto y = id / gw;
        Float v = hash(x + 53u * seed, y + 97u * seed);
        dst.write(y * w + x, v);
    };
    _fill_shader = device.compile<1>(fill);

    // Refine pass: one thread per destination point of the level-l grid
    // (dw x dh). Reads the (cw x ch) source grid, bilinearly interpolates at
    // the source-space position of the destination point, then adds the level
    // jitter. dst[y * w + x] (the destination grid is stored inside the
    // full-size ping-pong buffer at rows of stride w).
    Kernel1D refine = [&](BufferVar<float> src, BufferVar<float> dst, UInt cw, UInt ch,
                          UInt dw, UInt dh, UInt w, UInt level, UInt seed,
                          Float jitter_scale) noexcept {
        auto id = dispatch_id().x;
        auto x = id % dw;
        auto y = id / dw;
        // Source-space position of this destination point: the coarse cell
        // containing destination x spans source indices (x-1)/2 .. x/2, so the
        // source coordinate is (x-1)*0.5 (clamped into [0, cw-1]; for x = 0 the
        // unsigned wrap is harmless because the clamp discards it).
        Float sx = clamp(cast<float>(x - 1u) * 0.5f, 0.0f, cast<float>(cw - 1u));
        Float sy = clamp(cast<float>(y - 1u) * 0.5f, 0.0f, cast<float>(ch - 1u));
        UInt ix = cast<uint>(floor(sx));
        UInt iy = cast<uint>(floor(sy));
        Float fx = sx - cast<float>(ix);
        Float fy = sy - cast<float>(iy);
        UInt jx = min(ix + 1u, cw - 1u);
        UInt jy = min(iy + 1u, ch - 1u);
        Float v00 = src.read(iy * w + ix);
        Float v10 = src.read(iy * w + jx);
        Float v01 = src.read(jy * w + ix);
        Float v11 = src.read(jy * w + jx);
        Float v = lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
        // Deterministic jitter (spec formula):
        //   (hash(x + 137*level + 61*seed, y + 251*level + 29*seed) - 0.5) *
        //   roughness * pow(0.5, level)
        // The caller folds roughness * 0.5^level into jitter_scale.
        Float r = hash(x + 137u * level + 61u * seed, y + 251u * level + 29u * seed);
        v = v + (r - 0.5f) * jitter_scale;
        dst.write(y * w + x, v);
    };
    _refine_shader = device.compile<1>(refine);

    // Blit pass: clamp the finest grid to [0, 1] and write grayscale RGBA.
    Kernel2D blit = [&](ImageVar<float> image, BufferVar<float> grid, UInt w, UInt h) noexcept {
        auto coord = dispatch_id().xy();
        Float v = clamp(grid.read(coord.y * w + coord.x), 0.0f, 1.0f);
        image.write(coord, make_float4(v, v, v, 1.0f));
    };
    _blit_shader = device.compile<2>(blit);
}

#if ENABLE_DISPLAY
bool Plasma::draw_ui() {
    bool changed = false;
    ImGui::PushID(this);
    changed |= ImGui::SliderFloat("roughness", &_params.roughness, 0.1f, 0.9f, "%.2f");
    changed |= ImGui::SliderInt("seed", &_params.seed, 0, 255);
    ImGui::PopID();
    return changed;
}
#endif

void Plasma::generate(Image<float> &image, Stream &stream) {
    const auto w = static_cast<uint>(image.size().x);
    const auto h = static_cast<uint>(image.size().y);
    const auto size = w * h;
    const auto seed = static_cast<uint>(std::clamp(_params.seed, 0, 255));
    const auto roughness = std::clamp(_params.roughness, 0.1f, 0.9f);

    // (Re)allocate the ping-pong scratch when the resolution changes.
    if (_scratch_w != w || _scratch_h != h || !_a || !_b) {
        _a = _device->create_buffer<float>(size);
        _b = _device->create_buffer<float>(size);
        _scratch_w = w;
        _scratch_h = h;
    }

    // Number of refinement levels: L = ceil(log2(max(w, h))).
    const auto side = std::max(w, h);
    uint level_count = 0u;
    while ((1u << level_count) < side) {
        ++level_count;
    }
    // Grid dimensions at level l (1-based): g(l) = max(1, side >> (L - l));
    // clamped against the actual w/h so grids never exceed the buffer rows.
    auto grid_w = [&](uint l) { return std::min(w, std::max(1u, w >> (level_count - l))); };
    auto grid_h = [&](uint l) { return std::min(h, std::max(1u, h >> (level_count - l))); };

    // Pass 0: fill the level-1 grid (the corner control points).
    {
        auto gw = grid_w(1u);
        auto gh = grid_h(1u);
        stream << _fill_shader(_a, gw, gh, w, seed).dispatch(gw * gh);
    }

    // Levels 2..L: each fully rewrites the destination ping-pong buffer from
    // the source buffer written by the previous level, then swaps.
    Buffer<float> *src = &_a;
    Buffer<float> *dst = &_b;
    for (uint l = 2u; l <= level_count; ++l) {
        auto cw = grid_w(l - 1u);
        auto ch = grid_h(l - 1u);
        auto dw = grid_w(l);
        auto dh = grid_h(l);
        // Jitter amplitude: roughness * 0.5^l (halves each level).
        auto jitter_scale = roughness * std::pow(0.5f, static_cast<float>(l));
        stream << _refine_shader(*src, *dst, cw, ch, dw, dh, w, l, seed, jitter_scale)
                      .dispatch(dw * dh);
        std::swap(src, dst);
    }

    // Final pass: blit the finest grid to the image.
    stream << _blit_shader(image, *src, w, h).dispatch(make_uint2(w, h));
}

bool Plasma::validate(const luisa::vector<float4> &pixels, uint2 resolution) {
    // Generic invariants (finite, [0, 1], spatially varying).
    if (!ProceduralTexture::validate(pixels, resolution)) {
        return false;
    }
    // Plasma is a smooth fractal: adjacent pixels correlate strongly. A
    // per-pixel white-noise field passes the generic checks but has ~zero
    // row-to-row correlation, so require the average absolute difference of
    // vertically adjacent pixels to stay well below the full range.
    const auto w = resolution.x;
    const auto h = resolution.y;
    double diff_sum = 0.0;
    size_t count = 0u;
    for (auto y = 1u; y < h; ++y) {
        for (auto x = 0u; x < w; ++x) {
            diff_sum += std::abs(pixels[y * w + x].x - pixels[(y - 1u) * w + x].x);
            ++count;
        }
    }
    const auto avg_diff = static_cast<float>(diff_sum / static_cast<double>(count));
    if (avg_diff > 0.1f) {
        LUISA_WARNING("[plasma] not spatially smooth (mean vertical difference {})", avg_diff);
        return false;
    }
    return true;
}

luisa::unique_ptr<ProceduralTexture> create_plasma(Device &device) {
    return luisa::make_unique<Plasma>(device);
}

}// namespace procedural_tex
