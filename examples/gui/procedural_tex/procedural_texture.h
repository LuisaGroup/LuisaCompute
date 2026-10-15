// Procedural Texture Generator -- algorithm interface.
// Each procedural-noise algorithm lives in its own .h/.cpp pair (noise_*.h /
// noise_*.cpp): the header declares the parameter struct and the factory, the
// .cpp implements the DSL kernel(s) and compiles them through Device::compile.
// This header defines the shared interface the GUI driver (main.cpp) and the
// headless self-test consume.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/vstl/memory.h>

#ifndef ENABLE_DISPLAY
#ifdef LUISA_ENABLE_GUI
#define ENABLE_DISPLAY 1
#endif
#endif

#if ENABLE_DISPLAY
#include <imgui.h>
#endif

namespace procedural_tex {

using namespace luisa;
using namespace luisa::compute;

// Base class for one procedural-texture algorithm.
//
// Lifecycle (driven by main.cpp):
//   1. The factory creates the object and compiles its kernel(s) immediately
//      (Device::compile) -- compilation happens once, never per frame.
//   2. generate() is called lazily: once after creation and afterwards only
//      when draw_ui() reported a parameter change.
//   3. In headless mode draw_ui() is never called; generate() + validate()
//      run directly against the default parameters.
class ProceduralTexture {
public:
    virtual ~ProceduralTexture() = default;

    // Human-readable name shown in the ImGui window title and the logs.
    [[nodiscard]] virtual luisa::string_view name() const noexcept = 0;

#if ENABLE_DISPLAY
    // Draw the ImGui parameter widgets of this algorithm (called inside its
    // window, between ImGui::Begin/End). Must return true exactly when a
    // parameter changed since the previous call, so the driver can regenerate
    // the texture lazily instead of every frame.
    [[nodiscard]] virtual bool draw_ui() = 0;
#else
    [[nodiscard]] virtual bool draw_ui() { return false; }
#endif

    // Regenerate the texture. Must write RGBA float4 with RGB in [0, 1] to every
    // pixel of `image`. `image.size()` equals the resolution passed to the
    // factory. Implementations pass their parameters as kernel arguments, so a
    // parameter change only re-dispatches, never recompiles.
    virtual void generate(Image<float> &image, Stream &stream) = 0;

    // Optional per-algorithm validation on the generated pixels (headless
    // self-test). The default implementation checks the generic invariants:
    // every channel is finite and inside [0, 1], and the image is not a
    // constant (a procedural texture must vary over space).
    [[nodiscard]] virtual bool validate(const luisa::vector<float4> &pixels, uint2 resolution);
};

// Create every supported algorithm. Factories live in noise_*.cpp; the list
// itself lives in noise_all.cpp. Kernels are compiled inside the factories.
[[nodiscard]] luisa::vector<luisa::unique_ptr<ProceduralTexture>> create_all_procedural_textures(Device &device);

}// namespace procedural_tex
