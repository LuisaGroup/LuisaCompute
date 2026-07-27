#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <luisa/ast/function.h>
#include <luisa/ast/statement.h>
#include <luisa/core/stl/unordered_map.h>

namespace lc::vk::detail {

constexpr uint32_t sampler_filter_count = 4u;
constexpr uint32_t sampler_address_count = 4u;
constexpr uint32_t sampler_heap_size =
    sampler_filter_count * sampler_address_count;
constexpr uint32_t anisotropic_sampler_filter = 3u;

[[nodiscard]] constexpr uint32_t sampler_heap_index(
    uint32_t filter, uint32_t address) noexcept {
    // The shader ABI is address-major and filter-minor. This intentionally
    // differs from Sampler::code(), whose public serialization is
    // filter-major and address-minor.
    return address * sampler_filter_count + filter;
}

struct SamplerAnisotropySupport {
    bool physical_device_feature{false};
    bool imported_device{false};
    float max_sampler_anisotropy{1.0f};
};

struct SamplerAnisotropyPlan {
    bool enabled{false};
    float max_anisotropy{1.0f};
};

[[nodiscard]] constexpr auto plan_sampler_anisotropy(
    SamplerAnisotropySupport support,
    float requested_max_anisotropy = 16.0f) noexcept {
    // Vulkan does not expose the feature chain used to create an imported
    // VkDevice. Physical-device support alone therefore cannot authorize an
    // anisotropic sampler on an imported logical device.
    auto enabled = !support.imported_device &&
                   support.physical_device_feature &&
                   support.max_sampler_anisotropy >= 1.0f;
    if (!enabled) {
        return SamplerAnisotropyPlan{};
    }
    auto requested = requested_max_anisotropy < 1.0f ?
                         1.0f :
                         requested_max_anisotropy;
    return SamplerAnisotropyPlan{
        .enabled = true,
        .max_anisotropy = requested < support.max_sampler_anisotropy ?
                              requested :
                              support.max_sampler_anisotropy};
}

[[nodiscard]] constexpr bool sampler_requirement_is_supported(
    bool requires_anisotropy, bool anisotropy_enabled) noexcept {
    return !requires_anisotropy || anisotropy_enabled;
}

enum class ExplicitSamplerFilter : uint8_t {
    NON_ANISOTROPIC,
    ANISOTROPIC,
    DYNAMIC,
    INVALID,
};

[[nodiscard]] constexpr bool call_has_explicit_sampler_filter(
    luisa::compute::CallOp op) noexcept {
    using luisa::compute::CallOp;
    switch (op) {
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case CallOp::UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case CallOp::TYPED_UNIFORM_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case CallOp::TYPED_BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case CallOp::TEXTURE2D_SAMPLE:
        case CallOp::TEXTURE2D_SAMPLE_LEVEL:
        case CallOp::TEXTURE2D_SAMPLE_GRAD:
        case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case CallOp::TEXTURE3D_SAMPLE:
        case CallOp::TEXTURE3D_SAMPLE_LEVEL:
        case CallOp::TEXTURE3D_SAMPLE_GRAD:
        case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            return true;
        default:
            return false;
    }
}

namespace sampler_anisotropy_detail {

struct ConstantFilter {
    bool known{false};
    bool valid{true};
    uint64_t value{0u};
};

template<typename T>
[[nodiscard]] inline ConstantFilter filter_from_scalar(T value) noexcept {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if constexpr (std::is_signed_v<T>) {
            if (value < 0) {
                return {.known = true, .valid = false};
            }
        }
        return {.known = true,
                .valid = true,
                .value = static_cast<uint64_t>(value)};
    } else {
        return {.known = true, .valid = false};
    }
}

template<typename T>
[[nodiscard]] inline ConstantFilter filter_from_bytes(
    const std::byte *data) noexcept {
    T value{};
    std::memcpy(&value, data, sizeof(value));
    return filter_from_scalar(value);
}

[[nodiscard]] inline ConstantFilter constant_filter(
    const luisa::compute::Expression *expression) noexcept {
    using namespace luisa::compute;
    if (expression == nullptr) {
        return {.known = true, .valid = false};
    }
    if (expression->tag() == Expression::Tag::LITERAL) {
        auto literal = static_cast<const LiteralExpr *>(expression);
        return luisa::visit(
            []<typename T>(const T &value) noexcept {
                return filter_from_scalar(value);
            },
            literal->value());
    }
    if (expression->tag() != Expression::Tag::CONSTANT) {
        return {};
    }
    auto constant = static_cast<const ConstantExpr *>(expression)->data();
    switch (constant.type()->tag()) {
        case Type::Tag::INT8:
            return filter_from_bytes<int8_t>(constant.raw());
        case Type::Tag::UINT8:
            return filter_from_bytes<uint8_t>(constant.raw());
        case Type::Tag::INT16:
            return filter_from_bytes<int16_t>(constant.raw());
        case Type::Tag::UINT16:
            return filter_from_bytes<uint16_t>(constant.raw());
        case Type::Tag::INT32:
            return filter_from_bytes<int32_t>(constant.raw());
        case Type::Tag::UINT32:
            return filter_from_bytes<uint32_t>(constant.raw());
        case Type::Tag::INT64:
            return filter_from_bytes<int64_t>(constant.raw());
        case Type::Tag::UINT64:
            return filter_from_bytes<uint64_t>(constant.raw());
        default:
            return {.known = true, .valid = false};
    }
}

}// namespace sampler_anisotropy_detail

[[nodiscard]] inline ExplicitSamplerFilter classify_explicit_sampler_filter(
    const luisa::compute::Expression *expression) noexcept {
    auto constant = sampler_anisotropy_detail::constant_filter(expression);
    if (!constant.known) {
        return ExplicitSamplerFilter::DYNAMIC;
    }
    if (!constant.valid || constant.value >= sampler_filter_count) {
        return ExplicitSamplerFilter::INVALID;
    }
    return constant.value == anisotropic_sampler_filter ?
               ExplicitSamplerFilter::ANISOTROPIC :
               ExplicitSamplerFilter::NON_ANISOTROPIC;
}

struct SamplerUsage {
    bool uses_explicit_sampler{false};
    bool requires_anisotropy{false};
    bool has_dynamic_filter{false};
    bool has_invalid_filter{false};
};

[[nodiscard]] inline SamplerUsage analyze_sampler_usage(
    luisa::compute::Function root) noexcept {
    using namespace luisa::compute;
    SamplerUsage usage{};
    luisa::unordered_set<
        const luisa::compute::detail::FunctionBuilder *> visited;
    auto visit_function = [&](auto &&self, Function function) noexcept -> void {
        if (!function || !visited.emplace(function.builder()).second) {
            return;
        }
        traverse_expressions<true>(
            function.body(),
            [&](const Expression *expression) noexcept {
                if (expression->tag() != Expression::Tag::CALL) {
                    return;
                }
                auto call = static_cast<const CallExpr *>(expression);
                if (!call->is_builtin() ||
                    !call_has_explicit_sampler_filter(call->op())) {
                    return;
                }
                usage.uses_explicit_sampler = true;
                auto arguments = call->arguments();
                if (arguments.size() < 2u) {
                    usage.has_invalid_filter = true;
                    return;
                }
                switch (classify_explicit_sampler_filter(
                    arguments[arguments.size() - 2u])) {
                    case ExplicitSamplerFilter::NON_ANISOTROPIC:
                        break;
                    case ExplicitSamplerFilter::ANISOTROPIC:
                        usage.requires_anisotropy = true;
                        break;
                    case ExplicitSamplerFilter::DYNAMIC:
                        usage.requires_anisotropy = true;
                        usage.has_dynamic_filter = true;
                        break;
                    case ExplicitSamplerFilter::INVALID:
                        usage.has_invalid_filter = true;
                        break;
                }
            },
            [](const Statement *) noexcept {},
            [](const Statement *) noexcept {});
        for (auto &&callable : function.custom_callables()) {
            self(self, Function{callable.get()});
        }
    };
    visit_function(visit_function, root);
    return usage;
}

}// namespace lc::vk::detail
