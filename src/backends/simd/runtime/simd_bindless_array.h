#pragma once

#include <cstddef>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>

#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute::simd {

class SIMDBindlessArray {

private:
    luisa::vector<SIMDHostBindlessSlot> _slots;
    BindlessSlotType _type;

private:
    void _update_buffer(
        size_t slot,
        const BindlessArrayUpdateCommand::ModifiedBuffer &buffer) noexcept;
    void _update_texture(
        size_t slot,
        const BindlessArrayUpdateCommand::ModifiedTexture &texture,
        uint32_t dimension) noexcept;
    static void _sample_texture(
        const SIMDHostBindlessSlot *slots, size_t slot_count,
        uint32_t dimension, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *slot_indices,
        const uint32_t *sampler_codes,
        const float *u, const float *v, const float *w,
        const float *levels, float *values) noexcept;
    static void _read_texture(
        const SIMDHostBindlessSlot *slots, size_t slot_count,
        uint32_t dimension, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *slot_indices,
        const uint32_t *x, const uint32_t *y, const uint32_t *z,
        const uint32_t *levels, float *values) noexcept;
    static void _size_texture(
        const SIMDHostBindlessSlot *slots, size_t slot_count,
        uint32_t dimension, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *slot_indices,
        const uint32_t *levels, uint32_t *values) noexcept;

public:
    SIMDBindlessArray(size_t size, BindlessSlotType type) noexcept;
    ~SIMDBindlessArray() noexcept = default;

    void update(const BindlessArrayUpdateCommand &command) noexcept;
    [[nodiscard]] SIMDHostBindlessArrayView host_view() const noexcept;
    [[nodiscard]] void *native_handle() noexcept;
};

}// namespace luisa::compute::simd
