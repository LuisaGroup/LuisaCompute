//
// Created by Mike on 12/27/2023.
//

#pragma once

#include <luisa/backends/ext/pinned_memory_ext.hpp>

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAPinnedMemoryExt final : public PinnedMemoryExt {

private:
    CUDADevice *_device;

public:
    explicit CUDAPinnedMemoryExt(CUDADevice *device) noexcept;

protected:
    [[nodiscard]] BufferCreationInfo _pin_host_memory(const Type *elem_type, size_t elem_count,
                                                      void *host_ptr, const PinnedMemoryOption &option) noexcept override;
    [[nodiscard]] BufferCreationInfo _allocate_pinned_memory(const Type *elem_type, size_t elem_count,
                                                             const PinnedMemoryOption &option) noexcept override;

public:
    [[nodiscard]] DeviceInterface *device() const noexcept override;
};

}// namespace luisa::compute::cuda
