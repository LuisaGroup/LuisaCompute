//
// Created by Mike on 2021/12/6.
//

#pragma once

#include <core/stl.h>
#include <core/hash.h>

namespace luisa::compute {

class LC_RUNTIME_API ResourceTracker {

private:
    luisa::unordered_map<uint64_t, size_t, Hash64> _buffer_ref_count;
    luisa::unordered_map<uint64_t, size_t, Hash64> _texture_ref_count;
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

    template<typename F>
    void traverse_buffers(F &&f) const noexcept {
        for (auto [handle, _] : _buffer_ref_count) {
            std::invoke(std::forward<F>(f), handle);
        }
    }

    template<typename F>
    void traverse_textures(F &&f) const noexcept {
        for (auto [handle, _] : _texture_ref_count) {
            std::invoke(std::forward<F>(f), handle);
        }
    }
};

}
