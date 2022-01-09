//
// Created by Mike on 2021/12/6.
//

#include <core/logging.h>
#include <runtime/resource_tracker.h>

namespace luisa::compute {

void ResourceTracker::retain_buffer(uint64_t handle) noexcept {
    if (auto [iter, is_first] = _buffer_ref_count.try_emplace(handle, 1u); !is_first) {
        iter->second++;
    }
}

void ResourceTracker::release_buffer(uint64_t handle) noexcept {
    _buffers_to_remove.emplace_back(handle);
}

void ResourceTracker::retain_texture(uint64_t handle) noexcept {
    if (auto [iter, is_first] = _texture_ref_count.try_emplace(handle, 1u); !is_first) {
        iter->second++;
    }
}

void ResourceTracker::release_texture(uint64_t handle) noexcept {
    _textures_to_remove.emplace_back(handle);
}

void ResourceTracker::commit() noexcept {
    constexpr auto do_remove = [](auto &m, auto h) noexcept {
        if (auto iter = m.find(h); iter != m.end()) {
            if (--iter->second == 0u) { m.erase(iter); }
        }
    };
    for (auto handle : _buffers_to_remove) {
        do_remove(_buffer_ref_count, handle);
    }
    for (auto handle : _textures_to_remove) {
        do_remove(_texture_ref_count, handle);
    }
}

bool ResourceTracker::uses_buffer(uint64_t handle) const noexcept {
    return _buffer_ref_count.count(handle) != 0u;
}

bool ResourceTracker::uses_texture(uint64_t handle) const noexcept {
    return _texture_ref_count.count(handle) != 0u;
}

}
