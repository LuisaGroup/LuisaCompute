// Procedural Texture Generator -- diamond-square / plasma fractal.
//
// Iterative midpoint displacement with two GPU ping-pong float buffers. Each
// level doubles the resolution of the heightfield, sampling the previous
// (coarser) buffer bilinearly and adding a halving-amplitude jitter. Pass 0
// writes 4 corner control points; levels 1..L each fully rewrite their
// destination buffer; a final blit pass maps the finest buffer to the image.
//
// The kernel is a pure function of (x, y, level, seed): every new point's
// value depends only on the previously written coarser buffer and a hash of
// its own lattice index -- no atomics, no shared state, fully deterministic.
//
// This is the one algorithm in the catalog that needs GPU scratch (the
// ping-pong buffers) whose size depends on the image resolution; the scratch
// is allocated lazily on the first generate() and recreated whenever the
// resolution changes. All dispatches go through the caller-provided stream
// without an explicit synchronize -- the driver handles read-back.
#pragma once

#include "procedural_texture.h"

namespace procedural_tex {

struct PlasmaParams {
    float roughness = 0.55f; // jitter amplitude at level 1 (slider 0.1..0.9)
    int seed = 0;            // hash seed (slider 0..255)
};

class Plasma final : public ProceduralTexture {
public:
    explicit Plasma(Device &device);

    [[nodiscard]] luisa::string_view name() const noexcept override { return "plasma"; }

#if ENABLE_DISPLAY
    [[nodiscard]] bool draw_ui() override;
#endif

    void generate(Image<float> &image, Stream &stream) override;

    // Additionally require the plasma cloud to be spatially smooth (adjacent
    // rows correlate) -- a per-pixel hash would pass the generic spatially-
    // varying test but fail this one.
    [[nodiscard]] bool validate(const luisa::vector<float4> &pixels, uint2 resolution) override;

private:
    PlasmaParams _params;
    Device *_device{nullptr}; // captured in ctor; generate() has no Device&

    // Scratch: two ping-pong float grids of w*h (recreated on resolution change).
    Buffer<float> _a;
    Buffer<float> _b;
    uint _scratch_w{0u};
    uint _scratch_h{0u};

    // Pass kernels. 1D (one thread per pixel) for fill/refine (buffer I/O);
    // 2D for blit (image write).
    using FillShaderType = Shader1D<Buffer<float>, uint, uint, uint, uint>;
    using RefineShaderType = Shader1D<Buffer<float>, Buffer<float>, uint, uint, uint, uint, uint, uint, uint, float>;
    using BlitShaderType = Shader2D<Image<float>, Buffer<float>, uint, uint>;

    FillShaderType _fill_shader;
    RefineShaderType _refine_shader;
    BlitShaderType _blit_shader;
};

[[nodiscard]] luisa::unique_ptr<ProceduralTexture> create_plasma(Device &device);

}// namespace procedural_tex
