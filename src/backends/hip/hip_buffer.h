//
// Created by mike on 12/25/25.
//

#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPBuffer {

private:
    enum class Ownership : uint8_t {
        DEVICE,
        HOST_ALLOCATION,
        REGISTERED_HOST,
        EXTERNAL_DEVICE,
        EXTERNAL_HOST
    };

private:
    hipDeviceptr_t _device_ptr{};
    void *_host_ptr{};
    size_t _size_bytes{};
    size_t _indirect_capacity{};
    Ownership _ownership{Ownership::EXTERNAL_DEVICE};

public:
    struct Binding {
        hipDeviceptr_t ptr;
        uint64_t size_bytes;
    };

    struct alignas(16) IndirectHeader {
        uint32_t size;
        uint32_t _padding[3];
    };

    struct alignas(16) IndirectDispatch {
        uint32_t block_size[3];
        uint32_t _padding;
        uint32_t dispatch_size_and_kernel_id[4];
    };

    struct IndirectBinding {
        hipDeviceptr_t ptr;
        uint64_t offset_and_capacity;
    };

    static_assert(sizeof(IndirectHeader) == 16u);
    static_assert(sizeof(IndirectDispatch) == 32u);
    static_assert(sizeof(IndirectBinding) == 16u);

public:
    HIPBuffer() noexcept;
    ~HIPBuffer() noexcept;
    [[nodiscard]] static HIPBuffer *create_device_buffer(size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *create_indirect_buffer(size_t capacity) noexcept;
    [[nodiscard]] static HIPBuffer *create_host_buffer(size_t size_bytes, bool write_combined = false) noexcept;
    [[nodiscard]] static HIPBuffer *register_host_buffer(void *host_ptr, size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *import_external_device_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *import_external_host_buffer(void *external_ptr, size_t size_bytes) noexcept;
    static void destroy(HIPBuffer *buffer) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _device_ptr; }
    [[nodiscard]] auto native_handle() const noexcept { return _host_ptr == nullptr ? _device_ptr : _host_ptr; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] auto is_indirect() const noexcept { return _indirect_capacity != 0u; }
    [[nodiscard]] auto indirect_capacity() const noexcept { return _indirect_capacity; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
    [[nodiscard]] IndirectBinding indirect_binding(size_t offset, size_t size) const noexcept;
};

}// namespace luisa::compute::hip
