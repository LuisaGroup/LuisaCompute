//
// Created by Mike on 2021/12/6.
//

#include <core/logging.h>
#include <runtime/resource_tracker.h>

namespace luisa::compute {

void ResourceTracker::retain(uint64_t handle) noexcept {
    if (auto [iter, is_first] = _ref_count.try_emplace(handle, 1u); !is_first) {
        iter->second++;
    }
}

void ResourceTracker::release(uint64_t handle) noexcept {
    _remove_queue.emplace_back(handle);
}

void ResourceTracker::commit() noexcept {
    constexpr auto do_remove = [](auto &m, auto h) noexcept {
        if (auto iter = m.find(h); iter != m.end()) {
            if (--iter->second == 0u) { m.erase(iter); }
        }
    };
    for (auto handle : _remove_queue) {
        do_remove(_ref_count, handle);
    }
}

bool ResourceTracker::uses(uint64_t handle) const noexcept {
    return _ref_count.count(handle) != 0u;
}

}// namespace luisa::compute
