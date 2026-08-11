#include "simd_bindless_array.h"

#include <luisa/core/logging.h>

#include "simd_buffer.h"

namespace luisa::compute::simd {

SIMDBindlessArray::SIMDBindlessArray(
    size_t size, BindlessSlotType type) noexcept
    : _slots(size), _type{type} {}

void SIMDBindlessArray::_update_buffer(
    size_t slot_index,
    const BindlessArrayUpdateCommand::ModifiedBuffer &buffer) noexcept {
    LUISA_ASSERT(
        slot_index < _slots.size(),
        "Bindless buffer slot {} exceeds SIMD array size {}.",
        slot_index, _slots.size());
    LUISA_ASSERT(
        _type == BindlessSlotType::MULTIPLE ||
            _type == BindlessSlotType::BUFFER_ONLY,
        "Cannot update a buffer in a texture-only SIMD bindless array.");
    auto &slot = _slots[slot_index];
    using Operation = BindlessArrayUpdateCommand::Operation;
    switch (buffer.op) {
        case Operation::NONE: break;
        case Operation::REMOVE:
            slot.buffer = {};
            break;
        case Operation::EMPLACE: {
            auto *resource = reinterpret_cast<SIMDBuffer *>(buffer.handle);
            LUISA_ASSERT(
                resource != nullptr,
                "Cannot bind a null buffer to SIMD bindless slot {}.",
                slot_index);
            LUISA_ASSERT(
                buffer.offset_bytes <= resource->size(),
                "Bindless buffer offset {} exceeds buffer size {}.",
                buffer.offset_bytes, resource->size());
            auto remaining = resource->view_with_offset(buffer.offset_bytes);
            auto size = buffer.size_bytes ==
                    BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                remaining.size_bytes :
                buffer.size_bytes;
            LUISA_ASSERT(
                size > 0u && size <= remaining.size_bytes,
                "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                buffer.offset_bytes, buffer.offset_bytes + size,
                buffer.offset_bytes + remaining.size_bytes);
            slot.buffer = resource->view(buffer.offset_bytes, size);
            break;
        }
    }
}

void SIMDBindlessArray::_reject_texture_update(
    const BindlessArrayUpdateCommand::ModifiedTexture &texture) noexcept {
    if (texture.op != BindlessArrayUpdateCommand::Operation::NONE) {
        LUISA_ERROR_WITH_LOCATION(
            "Bindless texture slots are not implemented by the SIMD "
            "backend yet.");
    }
}

void SIMDBindlessArray::update(
    const BindlessArrayUpdateCommand &command) noexcept {
    command.visit_modifications(
        [&](const auto &modifications) noexcept {
            for (auto &&modification : modifications) {
                if constexpr (requires { modification.buffer; }) {
                    _update_buffer(
                        modification.slot, modification.buffer);
                }
                if constexpr (requires { modification.tex2d; }) {
                    _reject_texture_update(modification.tex2d);
                }
                if constexpr (requires { modification.tex3d; }) {
                    _reject_texture_update(modification.tex3d);
                }
            }
        });
}

SIMDHostBindlessArrayView SIMDBindlessArray::host_view() const noexcept {
    return {.slots = _slots.data(), .size = _slots.size()};
}

void *SIMDBindlessArray::native_handle() noexcept {
    return _slots.data();
}

}// namespace luisa::compute::simd
