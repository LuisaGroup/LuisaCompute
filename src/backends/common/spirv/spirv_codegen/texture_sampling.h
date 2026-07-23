#pragma once

#include <cstdint>
#include <optional>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/op.h>

namespace luisa::compute::xir {
class Value;
class Function;
class Instruction;
}// namespace luisa::compute::xir
namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace lc::spirv {

// The immutable sampler descriptor heap is address-major and filter-minor.
// XIR defines both configured-sampler selectors as uint32. Constant selectors
// outside this domain are handoff errors; dynamic selectors are saturated to
// [0, 3] before indexing so invalid runtime enum values cannot escape the fixed
// 4 x 4 heap and every valid value is unchanged.
inline constexpr uint32_t spirv_configured_sampler_selector_count = 4u;
inline constexpr uint32_t spirv_configured_sampler_selector_max =
    spirv_configured_sampler_selector_count - 1u;
inline constexpr uint32_t spirv_configured_sampler_heap_size =
    spirv_configured_sampler_selector_count *
    spirv_configured_sampler_selector_count;

struct SpirvTextureSampleOpInfo {
    bool valid{false};
    bool direct{false};
    bool is_2d{false};
    bool explicit_lod{false};
    bool gradients{false};
    bool lod_clamp{false};
    bool sampler_operands{false};
};

namespace detail {

enum class SpirvTextureSampleSource : uint8_t {
    DIRECT,
    BINDLESS,
};

enum class SpirvTextureSampleDimension : uint8_t {
    D2,
    D3,
};

enum class SpirvTextureSampleLod : uint8_t {
    IMPLICIT,
    EXPLICIT,
    GRADIENT,
    GRADIENT_WITH_CLAMP,
};

enum class SpirvTextureSamplerSource : uint8_t {
    CONFIGURED_OPERANDS,
    BINDLESS_SLOT,
};

[[nodiscard]] constexpr SpirvTextureSampleOpInfo
make_texture_sample_op_info(
    SpirvTextureSampleSource source,
    SpirvTextureSampleDimension dimension,
    SpirvTextureSampleLod lod,
    SpirvTextureSamplerSource sampler) noexcept {
    return {
        .valid = true,
        .direct = source == SpirvTextureSampleSource::DIRECT,
        .is_2d = dimension == SpirvTextureSampleDimension::D2,
        .explicit_lod = lod == SpirvTextureSampleLod::EXPLICIT,
        .gradients = lod == SpirvTextureSampleLod::GRADIENT ||
                     lod == SpirvTextureSampleLod::GRADIENT_WITH_CLAMP,
        .lod_clamp =
            lod == SpirvTextureSampleLod::GRADIENT_WITH_CLAMP,
        .sampler_operands =
            sampler ==
            SpirvTextureSamplerSource::CONFIGURED_OPERANDS};
}

}// namespace detail

enum class SpirvSamplerFilterPlan : uint8_t {
    SUPPORTED,
    INVALID_SELECTOR,
    REQUIRES_ANISOTROPY,
};

[[nodiscard]] constexpr SpirvSamplerFilterPlan
plan_spirv_sampler_filter(
    bool selector_is_constant, uint32_t selector,
    bool sampler_anisotropy_enabled) noexcept {
    if (selector_is_constant &&
        selector >= spirv_configured_sampler_selector_count) {
        return SpirvSamplerFilterPlan::INVALID_SELECTOR;
    }
    // A dynamic selector can evaluate to the ANISOTROPIC entry (3). On a
    // target without samplerAnisotropy, accepting it would silently reach a
    // valid-but-semantically-different linear placeholder.
    if (!sampler_anisotropy_enabled &&
        (!selector_is_constant || selector == 3u)) {
        return SpirvSamplerFilterPlan::REQUIRES_ANISOTROPY;
    }
    return SpirvSamplerFilterPlan::SUPPORTED;
}

// One operation-shape table shared by validation and emission. Keeping the
// operand interpretation here prevents the dialect from accepting a sampling
// form that the emitter decodes differently.
[[nodiscard]] constexpr SpirvTextureSampleOpInfo
spirv_texture_sample_op_info(
    luisa::compute::xir::ResourceQueryOp op) noexcept {
    namespace xir = luisa::compute::xir;
    using enum detail::SpirvTextureSampleDimension;
    using enum detail::SpirvTextureSampleLod;
    using enum detail::SpirvTextureSampleSource;
    using enum detail::SpirvTextureSamplerSource;
    using detail::make_texture_sample_op_info;
    switch (op) {
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE:
            return make_texture_sample_op_info(
                DIRECT, D2, IMPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
            return make_texture_sample_op_info(
                DIRECT, D2, EXPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
            return make_texture_sample_op_info(
                DIRECT, D2, GRADIENT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
            return make_texture_sample_op_info(
                DIRECT, D2, GRADIENT_WITH_CLAMP,
                CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE:
            return make_texture_sample_op_info(
                DIRECT, D3, IMPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
            return make_texture_sample_op_info(
                DIRECT, D3, EXPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
            return make_texture_sample_op_info(
                DIRECT, D3, GRADIENT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            return make_texture_sample_op_info(
                DIRECT, D3, GRADIENT_WITH_CLAMP,
                CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
            return make_texture_sample_op_info(
                BINDLESS, D2, IMPLICIT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
            return make_texture_sample_op_info(
                BINDLESS, D2, EXPLICIT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
            return make_texture_sample_op_info(
                BINDLESS, D2, GRADIENT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
            return make_texture_sample_op_info(
                BINDLESS, D2, GRADIENT_WITH_CLAMP,
                BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D2, IMPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D2, EXPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D2, GRADIENT,
                CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D2, GRADIENT_WITH_CLAMP,
                CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
            return make_texture_sample_op_info(
                BINDLESS, D3, IMPLICIT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
            return make_texture_sample_op_info(
                BINDLESS, D3, EXPLICIT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
            return make_texture_sample_op_info(
                BINDLESS, D3, GRADIENT, BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
            return make_texture_sample_op_info(
                BINDLESS, D3, GRADIENT_WITH_CLAMP,
                BINDLESS_SLOT);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D3, IMPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D3, EXPLICIT, CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D3, GRADIENT,
                CONFIGURED_OPERANDS);
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
            return make_texture_sample_op_info(
                BINDLESS, D3, GRADIENT_WITH_CLAMP,
                CONFIGURED_OPERANDS);
        default: return {};
    }
}

struct SpirvSamplerSelectorDecodeResult {
    std::optional<uint32_t> value;
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostic.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Dynamic uint32 selectors have no value and no diagnostic. Constants retain
// their exact uint32 value so range validation cannot accidentally wrap a bad
// value into the fixed 16-entry sampler table.
[[nodiscard]] SpirvSamplerSelectorDecodeResult
decode_spirv_sampler_selector_constant(
    const luisa::compute::xir::Value *value) noexcept;

[[nodiscard]] bool spirv_sampler_selector_type_supported(
    const luisa::compute::Type *type) noexcept;

struct SpirvSamplerTargetDiagnostic {
    const luisa::compute::xir::Function *function{nullptr};
    const luisa::compute::xir::Instruction *instruction{nullptr};
    luisa::string message;
};

struct SpirvSamplerTargetValidationResult {
    luisa::vector<SpirvSamplerTargetDiagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.empty();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return succeeded();
    }
};

// Validates the target-dependent half of the configured-sampler contract over
// the exact kernel-reachable function graph. Type/constant range checks remain
// in the target-independent dialect validator.
[[nodiscard]] SpirvSamplerTargetValidationResult
validate_spirv_sampler_target_contract(
    luisa::span<const luisa::compute::xir::Function *const> functions,
    bool sampler_anisotropy_enabled) noexcept;

}// namespace lc::spirv
