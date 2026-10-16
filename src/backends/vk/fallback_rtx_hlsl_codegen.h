#pragma once

// The compatibility HLSL route of the Vulkan backend, in software
// (fallback) ray-tracing mode.
//
// A device that answers ray tracing with the software fallback compiles its
// ray-tracing shaders through `lc::hlsl::CodegenUtility` - the same entry point
// the DX backend and the established Vulkan HLSL-to-SPIR-V route use - because
// the traversal of a fallback acceleration structure lives in the shared HLSL
// builtin `src/backends/common/hlsl/builtin/fallback_rtx_header.bytes`: it
// declares the shader's acceleration-structure argument as the fallback's two
// storage-buffer slots (src/backends/common/rtx/fallback_rtx.h) and maps
// `RAY_TRACING_TRACE_CLOSEST` / `RAY_TRACING_TRACE_ANY` onto
// `_FallbackTraceClosest` / `_FallbackTraceAny`.  The native XIR-to-SPIR-V
// route cannot express that traversal, which is exactly why the fallback raises
// `UserComputeHlslFallbackReason::FALLBACK_RTX`.
//
// The HLSL codegen has to be told it is compiling in that mode; it takes the
// flag as a trailing `bool fallback_rtx` of `CodegenUtility::Codegen(...)`.
// This header detects that parameter at compile time, so a tree whose HLSL
// codegen does not carry the fallback traversal yet still builds - the shader
// is then refused with an actionable message instead of being compiled into a
// hardware traversal that a fallback device cannot execute (which would be a
// silent wrong result).

#include "../common/hlsl/hlsl_codegen.h"

#include <luisa/core/logging.h>

#include <type_traits>
#include <utility>

namespace lc::vk::detail {

// Whether `CodegenUtility::Codegen` accepts the trailing fallback flag.
template<typename Utility, typename = void>
struct hlsl_codegen_supports_fallback_rtx : std::false_type {};

template<typename Utility>
struct hlsl_codegen_supports_fallback_rtx<
    Utility,
    std::void_t<decltype(std::declval<Utility &>().Codegen(
        std::declval<Function>(), std::declval<luisa::string_view>(),
        std::declval<uint>(), true, false, false, false, true))>>
    : std::true_type {};

[[nodiscard]] inline constexpr bool hlsl_codegen_supports_fallback_rtx_v =
    hlsl_codegen_supports_fallback_rtx<hlsl::CodegenUtility>::value;

// The compute-shader entry of the compatibility route.  With `fallback_rtx`
// set, a codegen that does not have the software traversal fails closed.
template<typename Utility>
[[nodiscard]] hlsl::CodegenResult codegen_compat_compute(
    Utility &util, Function kernel, luisa::string_view native_code,
    uint custom_mask, bool enable_debug_info, bool enable_fast_math,
    bool fallback_rtx) noexcept {
    if constexpr (hlsl_codegen_supports_fallback_rtx<Utility>::value) {
        return util.Codegen(
            kernel, native_code, custom_mask, true, false,
            enable_debug_info, enable_fast_math,
            /*fallback_rtx=*/fallback_rtx);
    } else {
        if (fallback_rtx) {
            LUISA_ERROR(
                "Vulkan shader '{}' traces rays through the software fallback "
                "acceleration structure "
                "(VulkanDeviceConfigExt::use_fallback_rtx()), but the HLSL "
                "codegen of this build has no software traversal (see "
                "src/backends/common/hlsl/builtin/fallback_rtx_header.bytes). "
                "Disable the fallback on this device, or rebuild the Vulkan "
                "backend with the HLSL fallback-RTX traversal.",
                kernel.name());
        }
        return util.Codegen(
            kernel, native_code, custom_mask, true, false,
            enable_debug_info, enable_fast_math);
    }
}

}// namespace lc::vk::detail
