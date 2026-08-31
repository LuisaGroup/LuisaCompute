#pragma once

#include <luisa/backends/ext/pinned_memory_ext.hpp>

namespace luisa::compute::metal {

class MetalDevice;

class MetalPinnedMemoryExt final : public PinnedMemoryExt {

private:
    MetalDevice *_device;

protected:
    [[nodiscard]] BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept override;

    [[nodiscard]] BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept override;

public:
    explicit MetalPinnedMemoryExt(MetalDevice *device) noexcept;
    [[nodiscard]] DeviceInterface *device() const noexcept override;
};

}// namespace luisa::compute::metal
