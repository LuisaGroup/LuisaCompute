//
// Created by Mike on 2021/12/6.
//

#pragma once
#include <core/stl/hash.h>
#include <core/stl/unordered_map.h>
#include <core/stl/vector.h>

namespace luisa::compute {

class LC_RUNTIME_API ResourceTracker {

private:
    luisa::unordered_map<uint64_t, size_t> _ref_count;
    luisa::vector<uint64_t> _remove_queue;

public:
    void retain(uint64_t handle) noexcept;
    void release(uint64_t handle) noexcept;
    [[nodiscard]] bool uses(uint64_t handle) const noexcept;
    void commit() noexcept;

    template<typename F>
    void traverse(F &&f) const noexcept {
        for (auto [handle, _] : _ref_count) {
            std::invoke(std::forward<F>(f), handle);
        }
    }
};

}// namespace luisa::compute
