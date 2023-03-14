//
// Created by Mike on 3/14/2023.
//

#include <backends/common/resource_tracker.h>

namespace luisa::compute {

ResourceTracker::ResourceTracker(size_t reserved_size) noexcept {
    _resource_references.reserve(reserved_size);
    _remove_list.reserve(reserved_size);
}

void ResourceTracker::retain(uint64_t handle) noexcept {
    _resource_references.try_emplace(handle, 0u).first->second++;
}

void ResourceTracker::release(uint64_t handle) noexcept {
    _remove_list.emplace_back(handle);
}

bool ResourceTracker::contains(uint64_t handle) const noexcept {
    return _resource_references.contains(handle);
}

}// namespace luisa::compute
