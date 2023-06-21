//
// Created by Mike Smith on 2023/6/22.
//

#pragma once

#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute {

struct PinnedMemoryOption {
    bool write_combined{false};
};

class PinnedMemoryExt : public DeviceExtension {

protected:
    [[nodiscard]] virtual BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept = 0;

    [[nodiscard]] virtual BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept = 0;

public:
    [[nodiscard]] virtual DeviceInterface *device() const noexcept = 0;

    template<typename T>
    [[nodiscard]] auto pin_host_memory(T *host_ptr, size_t elem_count,
                                       PinnedMemoryOption option = {}) noexcept {
        auto elem_type = Type::of<T>();
        auto info = _pin_host_memory(elem_type, elem_count, host_ptr, option);
        return Buffer<T>{device(), info};
    }

    template<typename T>
    [[nodiscard]] auto allocate_pinned_memory(size_t elem_count,
                                              PinnedMemoryOption option = {}) noexcept {
        auto elem_type = Type::of<T>();
        auto info = _allocate_pinned_host_memory(elem_type, elem_count, option);
        return Buffer<T>{device(), info};
    }
};

}// namespace luisa::compute
