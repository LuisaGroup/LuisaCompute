//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <core/stl/string.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalBuffer {

private:
    MTL::Buffer *_handle;

public:
    struct Binding {
        uint64_t address;
        size_t size;
    };

public:
    MetalBuffer(MTL::Device *device, size_t size) noexcept;
    ~MetalBuffer() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
