#pragma once
// OCR engine -- safe dispatch adapters (Phase 2).
//
// EVERY OCR kernel launch goes through these adapters. They exist because a
// flattened LaunchPlan1D carries a 2D grid (total_x, total_y), but
// compute::ShaderInvoke<1>::dispatch(uint) only sets the X dimension --
// dispatching a flattened plan through that overload would silently drop all
// y-rows of work. The adapters make that misuse impossible:
//
// - dispatch_1d(invoke, plan): Kernel1D launches. Empty plans (zero work)
//   return nullptr and the caller appends nothing. Flattened plans use the
//   multi-dispatch span form with ONE uint3{total_x, total_y, 1} entry, which
//   vk/cuda/dx encode as a single 2D-grid dispatch; kernels recover the linear
//   id with ocr_linear_id() (dispatch_y()*dispatch_size().x + dispatch_x()).
// - dispatch_2d(invoke, plan): Kernel2D launches (GEMM family). Same empty
//   skip; the plan's padded totals always cover every requested element.
//
// Command lifetime / copy semantics:
// - The invoke is consumed by value (moved in); after the call it must not be
//   reused. The returned luisa::unique_ptr<Command> owns the encoded command;
//   append it to a Stream or CommandList immediately. Buffer arguments are
//   encoded BY HANDLE at invoke-construction time -- the referenced buffers
//   (and any host spans behind copy commands) must stay alive until the
//   owning stream completes. The adapters copy nothing and retain nothing.
// - The uint3 span passed to the runtime is copied into the command encoder
//   (ComputeDispatchCmdEncoder::set_dispatch_sizes memcpy's it), so the
//   stack-local entry here is safe.
#include <memory>
#include <utility>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/shader.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Append-safe: `stream << dispatch_1d(shader(args...), plan)` is NOT valid for
// nullptr; use the OCR_DISPATCH_1D / OCR_DISPATCH_2D macros below, which skip
// empty plans, or check for nullptr explicitly.
[[nodiscard]] inline luisa::unique_ptr<compute::Command> dispatch_1d(
    compute::detail::ShaderInvoke<1> &&invoke, const LaunchPlan1D &plan) noexcept {
    if (plan.empty()) { return nullptr; }
    if (!plan.is_2d()) {
        return std::move(invoke).dispatch(plan.total_x);
    }
    // One multi-dispatch entry describing the flattened 2D grid. The runtime
    // copies the span, so this stack local is safe.
    const compute::uint3 entry = compute::make_uint3(plan.total_x, plan.total_y, 1u);
    return std::move(invoke).dispatch(luisa::span<const compute::uint3>{&entry, 1u});
}

[[nodiscard]] inline luisa::unique_ptr<compute::Command> dispatch_2d(
    compute::detail::ShaderInvoke<2> &&invoke, const LaunchPlan2D &plan) noexcept {
    if (plan.empty()) { return nullptr; }
    return std::move(invoke).dispatch(plan.total_x, plan.total_y);
}

// Stream-append helpers: skip empty plans without branching at every site.
// Usage: OCR_DISPATCH_1D(stream, shader(args...), plan);
#define OCR_DISPATCH_1D(stream, invoke_expr, plan)                       \
    do {                                                                 \
        if (auto _ocr_cmd = ::luisa::ocr::dispatch_1d((invoke_expr), (plan))) { \
            (stream) << std::move(_ocr_cmd);                             \
        }                                                                \
    } while (false)

#define OCR_DISPATCH_2D(stream, invoke_expr, plan)                       \
    do {                                                                 \
        if (auto _ocr_cmd = ::luisa::ocr::dispatch_2d((invoke_expr), (plan))) { \
            (stream) << std::move(_ocr_cmd);                             \
        }                                                                \
    } while (false)

}// namespace luisa::ocr