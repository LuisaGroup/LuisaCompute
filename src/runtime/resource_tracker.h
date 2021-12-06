//
// Created by Mike on 2021/12/6.
//

#pragma once

#include <core/basic_types.h>
#include <core/allocator.h>

namespace luisa::compute {

class ResourceTracker {

private:
    luisa::unordered_map<uint64_t, size_t> _buffer_ref_count;
    luisa::unordered_map<uint64_t, size_t> _texture_ref_count;
    luisa::vector<uint64_t> _buffers_to_remove;
    luisa::vector<uint64_t> _textures_to_remove;

public:
    void retain_buffer(uint64_t handle) noexcept;
    void release_buffer(uint64_t handle) noexcept;
    void retain_texture(uint64_t handle) noexcept;
    void release_texture(uint64_t handle) noexcept;
    void commit() noexcept;
    [[nodiscard]] bool uses_buffer(uint64_t handle) const noexcept;
    [[nodiscard]] bool uses_texture(uint64_t handle) const noexcept;
};

}
