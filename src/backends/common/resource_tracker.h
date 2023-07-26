#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute {

class ResourceTracker {

private:
    luisa::vector<uint64_t> _remove_list;
    luisa::unordered_map<uint64_t, size_t> _resource_references;

public:
    explicit ResourceTracker(size_t reserved_size) noexcept {
        _resource_references.reserve(reserved_size);
        _remove_list.reserve(reserved_size);
    }
    ~ResourceTracker() noexcept = default;
    ResourceTracker(ResourceTracker &&) noexcept = default;
    ResourceTracker(const ResourceTracker &) noexcept = delete;
    ResourceTracker &operator=(ResourceTracker &&) noexcept = default;
    ResourceTracker &operator=(const ResourceTracker &) noexcept = delete;

    void retain(uint64_t handle) noexcept {
        _resource_references.try_emplace(handle, 0u).first->second++;
    }
    void release(uint64_t handle) noexcept {
        _remove_list.emplace_back(handle);
    }
    [[nodiscard]] auto contains(uint64_t handle) const noexcept {
        return _resource_references.contains(handle);
    }

    template<typename Destroy>
    void commit(Destroy &&destroy) noexcept {
        for (auto handle : _remove_list) {
            if (auto it = _resource_references.find(handle);
                it != _resource_references.end()) {
                if (--it->second == 0u) {
                    destroy(handle);
                    _resource_references.erase(it);
                }
            }
        }
        _remove_list.clear();
    }

    void commit() noexcept {
        commit([](auto) noexcept {});
    }

    template<typename F>
    void traverse(F &&f) const noexcept {
        for (auto [handle, ref] : _resource_references) { f(handle); }
    }
};

}// namespace luisa::compute

