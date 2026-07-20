#pragma once

#include <luisa/backends/ext/pinned_memory_ext.hpp>

namespace luisa::compute::hip {

class HIPDevice;

class HIPPinnedMemoryExt final : public PinnedMemoryExt {

private:
    HIPDevice *_device;

protected:
    [[nodiscard]] BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept override;
    [[nodiscard]] BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept override;

public:
    explicit HIPPinnedMemoryExt(HIPDevice *device) noexcept
        : _device{device} {}
    [[nodiscard]] DeviceInterface *device() const noexcept override;
};

}// namespace luisa::compute::hip
