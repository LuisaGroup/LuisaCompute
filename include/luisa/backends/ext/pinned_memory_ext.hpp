#pragma once

#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute {

struct PinnedMemoryOption {
    bool write_combined{false};
};

class PinnedMemoryExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "PinnedMemoryExt";

protected:
    [[nodiscard]] virtual BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept = 0;

    [[nodiscard]] virtual BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept = 0;

public:
    [[nodiscard]] virtual DeviceInterface *device() const noexcept = 0;

    // Pin and register the host memory as a device-accessible
    //  buffer, and return
    //  - first: a Buffer owns the pinned memory resource that
    //     *unpins* and *unregisters* the memory when destructed.
    //  - second: a BufferView with the offset and size for
    //     access to the input host memory region.
    // Note: the page-aligned host accessible pointer can be
    //  obtained by calling Buffer::native_handle(), which
    //  is not necessarily the same as the input host_ptr, if
    //  the input host_ptr is not page-aligned.
    template<typename T>
    [[nodiscard]] auto pin_host_memory(T *host_ptr, size_t elem_count,
                                       PinnedMemoryOption option = {}) noexcept {
        auto elem_type = Type::of<T>();
        auto info = _pin_host_memory(elem_type, elem_count, host_ptr, option);
        auto buffer = Buffer<T>{device(), info};
        auto offset_bytes = static_cast<std::byte *>(host_ptr) -
                            static_cast<std::byte *>(buffer.native_handle());
        auto view = buffer.view(offset_bytes / info.element_stride, elem_count);
        return std::pair{std::move(buffer), std::move(view)};
    }

    // Allocate a pinned memory buffer on host
    // Note: the host accessible pointer can be obtained by
    //  calling Buffer::native_handle().
    template<typename T>
    [[nodiscard]] auto allocate_pinned_memory(size_t elem_count,
                                              PinnedMemoryOption option = {}) noexcept {
        auto elem_type = Type::of<T>();
        auto info = _allocate_pinned_host_memory(elem_type, elem_count, option);
        return Buffer<T>{device(), info};
    }
};

}// namespace luisa::compute
