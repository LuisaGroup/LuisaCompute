//
// Created by Mike on 3/14/2023.
//

#include <core/stl/vector.h>
#include <core/stl/unordered_map.h>

namespace luisa::compute {

class ResourceTracker {

private:
    luisa::vector<uint64_t> _remove_list;
    luisa::unordered_map<uint64_t, size_t> _resource_references;

public:
    explicit ResourceTracker(size_t reserved_size) noexcept;
    ~ResourceTracker() noexcept = default;
    ResourceTracker(ResourceTracker &&) noexcept = default;
    ResourceTracker(const ResourceTracker &) noexcept = delete;
    ResourceTracker &operator=(ResourceTracker &&) noexcept = default;
    ResourceTracker &operator=(const ResourceTracker &) noexcept = delete;

    void retain(uint64_t handle) noexcept;
    void release(uint64_t handle) noexcept;
    [[nodiscard]] bool contains(uint64_t handle) const noexcept;

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

    template<typename F>
    void traverse(F &&f) const noexcept {
        for (auto [handle, ref] : _resource_references) { f(handle); }
    }
};

}// namespace luisa::compute
