#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

namespace {

[[nodiscard]] const Type *raster_payload_member(
    const Type *payload_type, size_t member_index) noexcept {
    if (payload_type == nullptr) { return nullptr; }
    return payload_type->is_structure() ?
               payload_type->members()[member_index] :
               payload_type;
}

[[nodiscard]] bool is_float_raster_varying(
    const Type *type) noexcept {
    if (type == nullptr || !type->is_scalar_or_vector()) {
        return false;
    }
    auto element = type->is_vector() ? type->element() : type;
    return element->is_float16() || element->is_float32();
}

}// namespace

bool resolve_raster_interpolation(
    const Type *payload_type,
    size_t member_index,
    RasterInterpolation &interpolation,
    luisa::string &reason) noexcept {
    if (payload_type == nullptr || member_index == 0u ||
        (payload_type->is_structure() &&
         member_index >= payload_type->members().size())) {
        reason = "invalid raster varying member index";
        return false;
    }
    auto member = raster_payload_member(payload_type, member_index);
    interpolation = RasterInterpolation::DEFAULT;
    if (payload_type->is_structure()) {
        auto attributes = payload_type->member_attributes();
        if (!attributes.empty()) {
            auto &attribute = attributes[member_index];
            if (attribute.key == "position") {
                reason = "raster payload contains more than one position member";
                return false;
            }
            if (attribute.key == raster_interpolation_attribute_key &&
                !parse_raster_interpolation(attribute, interpolation)) {
                reason = "raster varying has invalid interpolation mode '" +
                         attribute.value + "'";
                return false;
            }
        }
    }
    auto floating = is_float_raster_varying(member);
    if (interpolation == RasterInterpolation::DEFAULT) {
        interpolation = floating ?
                            RasterInterpolation::CENTER_PERSPECTIVE :
                            RasterInterpolation::FLAT;
    }
    if (interpolation != RasterInterpolation::FLAT && !floating) {
        reason = "perspective, centroid, and sample interpolation require a floating-point varying";
        return false;
    }
    reason.clear();
    return true;
}

bool validate_raster_interpolation(
    const Type *payload_type,
    luisa::string &reason) noexcept {
    if (payload_type == nullptr || !payload_type->is_structure()) {
        reason.clear();
        return true;
    }
    auto attributes = payload_type->member_attributes();
    if (!attributes.empty()) {
        auto &position = attributes.front();
        if (!position.key.empty() && position.key != "position") {
            reason = "raster payload member zero must be the position";
            return false;
        }
        if (!position.value.empty()) {
            reason = "raster position cannot carry an interpolation mode";
            return false;
        }
    }
    for (auto member_index = 1u;
         member_index < payload_type->members().size();
         ++member_index) {
        auto interpolation = RasterInterpolation::DEFAULT;
        if (!resolve_raster_interpolation(
                payload_type, member_index,
                interpolation, reason)) {
            return false;
        }
    }
    reason.clear();
    return true;
}

}// namespace luisa::compute::metal::detail
