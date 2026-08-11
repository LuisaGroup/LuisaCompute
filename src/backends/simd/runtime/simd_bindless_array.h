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
    static void _reject_texture_update(
        const BindlessArrayUpdateCommand::ModifiedTexture &texture) noexcept;

public:
    SIMDBindlessArray(size_t size, BindlessSlotType type) noexcept;
    ~SIMDBindlessArray() noexcept = default;

    void update(const BindlessArrayUpdateCommand &command) noexcept;
    [[nodiscard]] SIMDHostBindlessArrayView host_view() const noexcept;
    [[nodiscard]] void *native_handle() noexcept;
};

}// namespace luisa::compute::simd
