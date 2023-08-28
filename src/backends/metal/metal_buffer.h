#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/runtime/dispatch_buffer.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalBufferBase {
public:
    MetalBufferBase() noexcept = default;
    virtual ~MetalBufferBase() noexcept = default;
    MetalBufferBase(MetalBufferBase &&) noexcept = delete;
    MetalBufferBase(const MetalBufferBase &) noexcept = delete;
    MetalBufferBase &operator=(MetalBufferBase &&) noexcept = delete;
    MetalBufferBase &operator=(const MetalBufferBase &) noexcept = delete;
    virtual void set_name(luisa::string_view name) noexcept = 0;
    [[nodiscard]] virtual bool is_indirect() const noexcept { return false; }
};

class MetalBuffer : public MetalBufferBase {

private:
    MTL::Buffer *_handle;

public:
    struct Binding {
        uint64_t address;
        size_t size;
    };

public:
    MetalBuffer(MTL::Device *device, size_t size) noexcept;
    ~MetalBuffer() noexcept override;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
    void set_name(luisa::string_view name) noexcept override;
};

class MetalIndirectDispatchBuffer : public MetalBufferBase {

private:
    MTL::Buffer *_dispatch_buffer;
    MTL::IndirectCommandBuffer *_command_buffer;
    size_t _capacity;

public:
    struct Binding {
        uint64_t address;
        uint offset;
        uint capacity;
    };

    struct alignas(16) Header {
        uint count;
    };

    struct alignas(16) Dispatch {
        uint3 block_size;
        uint4 dispatch_size_and_kernel_id;
    };

public:
    MetalIndirectDispatchBuffer(MTL::Device *device, size_t capacity) noexcept;
    ~MetalIndirectDispatchBuffer() noexcept override;
    [[nodiscard]] auto dispatch_buffer() const noexcept { return _dispatch_buffer; }
    [[nodiscard]] auto command_buffer() const noexcept { return _command_buffer; }
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] bool is_indirect() const noexcept override { return true; }
    [[nodiscard]] Binding binding(size_t offset, size_t count) const noexcept;
    void set_name(luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::metal

