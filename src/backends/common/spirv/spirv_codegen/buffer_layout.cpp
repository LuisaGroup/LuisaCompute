#include "buffer_layout.h"

#include <algorithm>

#include <luisa/ast/type.h>
#include <luisa/core/mathematics.h>

namespace lc::spirv {

using luisa::compute::Type;

namespace {

[[nodiscard]] SpirvTypedBufferLayout fail(
    SpirvTypedBufferLayoutStatus status, const Type *type,
    size_t byte_offset = 0u) noexcept {
    return {
        .status = status,
        .byte_offset = byte_offset,
        .offending_type = type};
}

[[nodiscard]] SpirvTypedBufferLayout plan_type(
    const Type *type) noexcept {
    if (type == nullptr) {
        return fail(
            SpirvTypedBufferLayoutStatus::UNSUPPORTED_TYPE, type);
    }
    if (type->is_bool()) {
        // OpTypeBool is a logical type and cannot inhabit StorageBuffer.
        return fail(
            SpirvTypedBufferLayoutStatus::LOGICAL_BOOL, type);
    }
    if (type->is_scalar()) {
        return {
            .status = SpirvTypedBufferLayoutStatus::COMPATIBLE,
            .base_alignment = type->size()};
    }
    if (type->is_vector()) {
        auto *element = type->element();
        if (element == nullptr || !element->is_scalar() ||
            element->is_bool()) {
            return fail(
                element != nullptr && element->is_bool() ?
                    SpirvTypedBufferLayoutStatus::LOGICAL_BOOL :
                    SpirvTypedBufferLayoutStatus::UNSUPPORTED_TYPE,
                type);
        }
        auto component_count = type->dimension() == 2u ? 2u : 4u;
        return {
            .status = SpirvTypedBufferLayoutStatus::COMPATIBLE,
            .base_alignment = element->size() * component_count};
    }
    if (type->is_matrix()) {
        auto *element = type->element();
        if (element == nullptr || !element->is_float32()) {
            return fail(
                SpirvTypedBufferLayoutStatus::UNSUPPORTED_TYPE, type);
        }
        auto *column = Type::vector(element, type->dimension());
        auto column_layout = plan_type(column);
        if (!column_layout) { return column_layout; }
        if (column->size() % column_layout.base_alignment != 0u) {
            return fail(
                SpirvTypedBufferLayoutStatus::INVALID_MATRIX_STRIDE,
                type, column->size());
        }
        return {
            .status = SpirvTypedBufferLayoutStatus::COMPATIBLE,
            .base_alignment = column_layout.base_alignment};
    }
    if (type->is_array()) {
        auto *element = type->element();
        auto element_layout = plan_type(element);
        if (!element_layout) { return element_layout; }
        if (element->size() % element_layout.base_alignment != 0u) {
            return fail(
                SpirvTypedBufferLayoutStatus::INVALID_ARRAY_STRIDE,
                element, element->size());
        }
        return {
            .status = SpirvTypedBufferLayoutStatus::COMPATIBLE,
            .base_alignment = element_layout.base_alignment};
    }
    if (type->is_structure()) {
        auto base_alignment = size_t{1u};
        auto byte_offset = size_t{0u};
        for (auto *member : type->members()) {
            auto member_layout = plan_type(member);
            if (!member_layout) { return member_layout; }
            byte_offset = luisa::align(byte_offset, member->alignment());
            if (byte_offset % member_layout.base_alignment != 0u) {
                return fail(
                    SpirvTypedBufferLayoutStatus::MISALIGNED_STRUCT_MEMBER,
                    member, byte_offset);
            }
            base_alignment = std::max(
                base_alignment, member_layout.base_alignment);
            byte_offset += member->size();
        }
        if (type->size() % base_alignment != 0u) {
            return fail(
                SpirvTypedBufferLayoutStatus::INVALID_STRUCT_STRIDE,
                type, type->size());
        }
        return {
            .status = SpirvTypedBufferLayoutStatus::COMPATIBLE,
            .base_alignment = base_alignment};
    }
    return fail(
        SpirvTypedBufferLayoutStatus::UNSUPPORTED_TYPE, type);
}

}// namespace

SpirvTypedBufferLayout plan_spirv_typed_buffer_layout(
    const Type *element_type) noexcept {
    auto layout = plan_type(element_type);
    if (layout &&
        element_type->size() % layout.base_alignment != 0u) {
        return fail(
            SpirvTypedBufferLayoutStatus::INVALID_RUNTIME_ARRAY_STRIDE,
            element_type, element_type->size());
    }
    return layout;
}

}// namespace lc::spirv
