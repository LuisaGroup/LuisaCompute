#include "simd_bindless_array.h"

#include <bit>

#include <luisa/core/logging.h>

#include "simd_buffer.h"
#include "simd_texture.h"

static_assert(
    offsetof(luisa::compute::simd::SIMDHostBindlessSlot, texture3d) -
        offsetof(luisa::compute::simd::SIMDHostBindlessSlot, texture2d) ==
    sizeof(luisa::compute::simd::SIMDHostBindlessTextureSlot));
static_assert(
    luisa::compute::simd::simd_bindless_linear_point_mirror_sampler_code ==
    ((static_cast<uint32_t>(
          luisa::compute::Sampler::Filter::LINEAR_POINT)
      << 2u) |
     static_cast<uint32_t>(
         luisa::compute::Sampler::Address::MIRROR)));

namespace {

[[nodiscard]] constexpr uint64_t lane_mask(
    uint32_t lane_count) noexcept {
    return lane_count >= 64u ? ~uint64_t{0u} :
                               (uint64_t{1u} << lane_count) - 1u;
}

[[nodiscard]] const luisa::compute::simd::SIMDHostBindlessTextureSlot &
texture_slot(
    const luisa::compute::simd::SIMDHostBindlessSlot &slot,
    uint32_t dimension) noexcept {
    return dimension == 2u ? slot.texture2d : slot.texture3d;
}

}// namespace

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

void SIMDBindlessArray::_update_texture(
    size_t slot_index,
    const BindlessArrayUpdateCommand::ModifiedTexture &texture,
    uint32_t dimension) noexcept {
    LUISA_ASSERT(
        slot_index < _slots.size(),
        "Bindless texture slot {} exceeds SIMD array size {}.",
        slot_index, _slots.size());
    LUISA_ASSERT(
        dimension == 2u || dimension == 3u,
        "Invalid SIMD bindless texture dimension {}.", dimension);
    auto expected_type = dimension == 2u ?
                             BindlessSlotType::TEXTURE2D_ONLY :
                             BindlessSlotType::TEXTURE3D_ONLY;
    LUISA_ASSERT(
        _type == BindlessSlotType::MULTIPLE || _type == expected_type,
        "Cannot update a {}D texture in this SIMD bindless array.",
        dimension);
    auto &descriptor = dimension == 2u ?
                           _slots[slot_index].texture2d :
                           _slots[slot_index].texture3d;
    using Operation = BindlessArrayUpdateCommand::Operation;
    switch (texture.op) {
        case Operation::NONE: break;
        case Operation::REMOVE:
            descriptor = {};
            break;
        case Operation::EMPLACE: {
            auto *resource = reinterpret_cast<SIMDTexture *>(
                texture.handle);
            LUISA_ASSERT(
                resource != nullptr,
                "Cannot bind a null {}D texture to SIMD bindless slot {}.",
                dimension, slot_index);
            LUISA_ASSERT(
                resource->dimension() == dimension,
                "SIMD bindless slot expects a {}D texture, got {}D.",
                dimension, resource->dimension());
            auto size = resource->size(0u);
            LUISA_ASSERT(
                size.x <= simd_bindless_texture_extent_mask &&
                    size.y <= simd_bindless_texture_extent_mask &&
                    size.z <= simd_bindless_texture_extent_mask,
                "SIMD bindless texture extent ({}, {}, {}) exceeds the "
                "20-bit packet descriptor limit.",
                size.x, size.y, size.z);
            auto base_view = resource->view(0u);
            descriptor = {
                .texture = resource,
                .byte1_mip0 = base_view.storage() == PixelStorage::BYTE1 ?
                                  base_view.data() :
                                  nullptr,
                .metadata = simd_bindless_texture_metadata(
                    texture.sampler.code(), size.x, size.y, size.z),
            };
            break;
        }
    }
}

void SIMDBindlessArray::_sample_texture(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *sampler_codes,
    const float *u, const float *v, const float *w,
    const float *levels, float *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    LUISA_ASSERT(
        slots != nullptr && slot_indices != nullptr &&
            u != nullptr && v != nullptr && w != nullptr &&
            values != nullptr,
        "SIMD bindless texture sample received a null packet field.");
    LUISA_ASSERT(
        dimension == 2u || dimension == 3u,
        "Invalid SIMD bindless sample dimension {}.", dimension);
    for (auto remaining = active_mask_bits; remaining != 0u;) {
        auto seed = static_cast<uint32_t>(std::countr_zero(remaining));
        auto seed_slot_index = slot_indices[seed];
        LUISA_ASSERT(
            seed_slot_index < slot_count,
            "SIMD bindless texture slot {} is out of range (size {}).",
            seed_slot_index, slot_count);
        auto &seed_descriptor = texture_slot(
            slots[seed_slot_index], dimension);
        auto *texture = static_cast<SIMDTexture *>(
            seed_descriptor.texture);
        LUISA_ASSERT(
            texture != nullptr,
            "SIMD bindless {}D texture slot {} is unbound.",
            dimension, seed_slot_index);
        auto sampler_code = sampler_codes == nullptr ?
                                simd_bindless_texture_sampler(seed_descriptor) :
                                sampler_codes[seed];
        LUISA_ASSERT(
            sampler_code < 16u,
            "Invalid SIMD bindless sampler code {}.", sampler_code);
        auto group = uint64_t{0u};
        for (auto candidates = remaining; candidates != 0u;
             candidates &= candidates - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(candidates));
            auto slot_index = slot_indices[lane];
            LUISA_ASSERT(
                slot_index < slot_count,
                "SIMD bindless texture slot {} is out of range (size {}).",
                slot_index, slot_count);
            auto &descriptor = texture_slot(slots[slot_index], dimension);
            auto lane_sampler = sampler_codes == nullptr ?
                                    simd_bindless_texture_sampler(descriptor) :
                                    sampler_codes[lane];
            if (descriptor.texture == texture &&
                lane_sampler == sampler_code) {
                group |= uint64_t{1u} << lane;
            }
        }
        texture->sample_float_packet(
            Sampler::decode(sampler_code), 0u, lane_count, group,
            u, v, w, levels, values);
        remaining &= ~group;
    }
}

void SIMDBindlessArray::_read_texture(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    const uint32_t *levels, float *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    LUISA_ASSERT(
        slots != nullptr && slot_indices != nullptr &&
            x != nullptr && y != nullptr && z != nullptr &&
            values != nullptr,
        "SIMD bindless texture read received a null packet field.");
    LUISA_ASSERT(
        dimension == 2u || dimension == 3u,
        "Invalid SIMD bindless read dimension {}.", dimension);
    for (auto remaining = active_mask_bits; remaining != 0u;) {
        auto seed = static_cast<uint32_t>(std::countr_zero(remaining));
        auto seed_slot_index = slot_indices[seed];
        LUISA_ASSERT(
            seed_slot_index < slot_count,
            "SIMD bindless texture slot {} is out of range (size {}).",
            seed_slot_index, slot_count);
        auto &seed_descriptor = texture_slot(
            slots[seed_slot_index], dimension);
        auto *texture = static_cast<SIMDTexture *>(
            seed_descriptor.texture);
        LUISA_ASSERT(
            texture != nullptr,
            "SIMD bindless {}D texture slot {} is unbound.",
            dimension, seed_slot_index);
        auto level = levels == nullptr ? 0u : levels[seed];
        auto group = uint64_t{0u};
        for (auto candidates = remaining; candidates != 0u;
             candidates &= candidates - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(candidates));
            auto slot_index = slot_indices[lane];
            LUISA_ASSERT(
                slot_index < slot_count,
                "SIMD bindless texture slot {} is out of range (size {}).",
                slot_index, slot_count);
            auto &descriptor = texture_slot(slots[slot_index], dimension);
            auto lane_level = levels == nullptr ? 0u : levels[lane];
            if (descriptor.texture == texture && lane_level == level) {
                group |= uint64_t{1u} << lane;
            }
        }
        if (level < texture->mip_levels()) {
            auto view = texture->host_view(level);
            view.read_float(
                view.texture, level, lane_count, group,
                x, y, z, values);
        }
        remaining &= ~group;
    }
}

void SIMDBindlessArray::_size_texture(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *levels, uint32_t *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    LUISA_ASSERT(
        slots != nullptr && slot_indices != nullptr && values != nullptr,
        "SIMD bindless texture size received a null packet field.");
    LUISA_ASSERT(
        dimension == 2u || dimension == 3u,
        "Invalid SIMD bindless size dimension {}.", dimension);
    for (auto remaining = active_mask_bits; remaining != 0u;) {
        auto seed = static_cast<uint32_t>(std::countr_zero(remaining));
        auto seed_slot_index = slot_indices[seed];
        LUISA_ASSERT(
            seed_slot_index < slot_count,
            "SIMD bindless texture slot {} is out of range (size {}).",
            seed_slot_index, slot_count);
        auto &seed_descriptor = texture_slot(
            slots[seed_slot_index], dimension);
        auto *texture = static_cast<SIMDTexture *>(
            seed_descriptor.texture);
        LUISA_ASSERT(
            texture != nullptr,
            "SIMD bindless {}D texture slot {} is unbound.",
            dimension, seed_slot_index);
        auto level = levels == nullptr ? 0u : levels[seed];
        auto group = uint64_t{0u};
        for (auto candidates = remaining; candidates != 0u;
             candidates &= candidates - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(candidates));
            auto slot_index = slot_indices[lane];
            LUISA_ASSERT(
                slot_index < slot_count,
                "SIMD bindless texture slot {} is out of range (size {}).",
                slot_index, slot_count);
            auto &descriptor = texture_slot(slots[slot_index], dimension);
            auto lane_level = levels == nullptr ? 0u : levels[lane];
            if (descriptor.texture == texture && lane_level == level) {
                group |= uint64_t{1u} << lane;
            }
        }
        auto size = texture->size(level);
        for (auto candidates = group; candidates != 0u;
             candidates &= candidates - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(candidates));
            for (auto axis = 0u; axis < dimension; axis++) {
                values[axis * lane_count + lane] = size[axis];
            }
        }
        remaining &= ~group;
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
                    _update_texture(
                        modification.slot, modification.tex2d, 2u);
                }
                if constexpr (requires { modification.tex3d; }) {
                    _update_texture(
                        modification.slot, modification.tex3d, 3u);
                }
            }
        });
}

SIMDHostBindlessArrayView SIMDBindlessArray::host_view() const noexcept {
    return {
        .slots = _slots.data(),
        .size = _slots.size(),
        .sample_texture = _sample_texture,
        .read_texture = _read_texture,
        .size_texture = _size_texture,
    };
}

void *SIMDBindlessArray::native_handle() noexcept {
    return _slots.data();
}

}// namespace luisa::compute::simd
