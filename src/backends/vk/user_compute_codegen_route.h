#pragma once

#include <cstdint>

#include <luisa/ast/op.h>
#include <luisa/core/stl/string.h>

namespace lc::vk::detail {

// Native XIR-to-SPIR-V is the normal Vulkan user-compute path. These are the
// explicitly unsupported AST features that still require the compatibility
// HLSL path; internal Vulkan builtins are outside this routing contract.
enum class UserComputeHlslFallbackReason : uint32_t {
    NATIVE_INCLUDE = 1u << 0u,
    PRINTING = 1u << 1u,
    COOPERATIVE_OPERATIONS = 1u << 2u,
    ASYNC_COPY = 1u << 3u,
    MOTION_BLUR = 1u << 4u,
};

using UserComputeHlslFallbackReasonMask = uint32_t;

struct UserComputeCodegenRequirements {
    bool native_include{};
    bool printing{};
    bool cooperative_operations{};
    bool async_copy{};
    bool motion_blur{};
};

struct UserComputeCodegenRoute {
    UserComputeHlslFallbackReasonMask hlsl_fallback_reasons{};

    [[nodiscard]] constexpr bool uses_native_xir_spirv() const noexcept {
        return hlsl_fallback_reasons == 0u;
    }

    [[nodiscard]] constexpr bool requires_hlsl_fallback() const noexcept {
        return !uses_native_xir_spirv();
    }

    [[nodiscard]] constexpr bool contains(
        UserComputeHlslFallbackReason reason) const noexcept {
        return (hlsl_fallback_reasons & static_cast<uint32_t>(reason)) != 0u;
    }
};

enum class RequiredNativeXirSpirvStatus : uint8_t {
    SATISFIED,
    NATIVE_CODEGEN_UNAVAILABLE,
    HLSL_FALLBACK_REQUIRED,
};

struct RequiredNativeXirSpirvPlan {
    RequiredNativeXirSpirvStatus status{
        RequiredNativeXirSpirvStatus::SATISFIED};

    [[nodiscard]] constexpr bool satisfied() const noexcept {
        return status == RequiredNativeXirSpirvStatus::SATISFIED;
    }
};

[[nodiscard]] constexpr RequiredNativeXirSpirvPlan
plan_required_native_xir_spirv(
    bool required, bool native_codegen_compiled,
    UserComputeCodegenRoute route = {}) noexcept {
    if (!required) { return {}; }
    if (!native_codegen_compiled) {
        return {.status =
                    RequiredNativeXirSpirvStatus::NATIVE_CODEGEN_UNAVAILABLE};
    }
    if (route.requires_hlsl_fallback()) {
        return {.status =
                    RequiredNativeXirSpirvStatus::HLSL_FALLBACK_REQUIRED};
    }
    return {};
}

[[nodiscard]] constexpr UserComputeCodegenRoute
plan_user_compute_codegen_route(
    UserComputeCodegenRequirements requirements) noexcept {
    auto reasons = UserComputeHlslFallbackReasonMask{};
    auto add = [&](bool enabled,
                   UserComputeHlslFallbackReason reason) constexpr noexcept {
        if (enabled) { reasons |= static_cast<uint32_t>(reason); }
    };
    add(requirements.native_include,
        UserComputeHlslFallbackReason::NATIVE_INCLUDE);
    add(requirements.printing,
        UserComputeHlslFallbackReason::PRINTING);
    add(requirements.cooperative_operations,
        UserComputeHlslFallbackReason::COOPERATIVE_OPERATIONS);
    add(requirements.async_copy,
        UserComputeHlslFallbackReason::ASYNC_COPY);
    add(requirements.motion_blur,
        UserComputeHlslFallbackReason::MOTION_BLUR);
    return {.hlsl_fallback_reasons = reasons};
}

[[nodiscard]] constexpr luisa::string_view
user_compute_hlsl_fallback_reason_name(
    UserComputeHlslFallbackReason reason) noexcept {
    using namespace std::string_view_literals;
    switch (reason) {
        case UserComputeHlslFallbackReason::NATIVE_INCLUDE:
            return "native include"sv;
        case UserComputeHlslFallbackReason::PRINTING:
            return "printing"sv;
        case UserComputeHlslFallbackReason::COOPERATIVE_OPERATIONS:
            return "cooperative operations"sv;
        case UserComputeHlslFallbackReason::ASYNC_COPY:
            return "async copy"sv;
        case UserComputeHlslFallbackReason::MOTION_BLUR:
            return "motion blur"sv;
    }
    return "unknown"sv;
}

}// namespace lc::vk::detail
