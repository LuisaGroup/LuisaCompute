//
// Created by Mike on 2021/11/13.
//

#pragma once

#include <mutex>
#include <optional>

#include <core/stl.h>

namespace luisa {

// TODO: support allocator & comparator
template<typename Key, typename Value>
class LRUCache {

private:
    luisa::lru_cache<Key, Value> _cache;
    std::mutex _mutex;

public:
    explicit LRUCache(size_t cap) noexcept : _cache{cap} {}
    LRUCache(LRUCache &&) noexcept = delete;
    LRUCache(const LRUCache &) noexcept = delete;
    LRUCache &operator=(LRUCache &&) noexcept = delete;
    LRUCache &operator=(const LRUCache &) noexcept = delete;

    [[nodiscard]] static auto create(size_t capacity) noexcept {
        return luisa::make_unique<LRUCache>(capacity);
    }

    [[nodiscard]] auto fetch(const Key &key) noexcept -> luisa::optional<Value> {
        std::scoped_lock lock{_mutex};
        auto value = _cache.at(key);
        _cache.touch(key);
        return value;
    }

    void update(const Key &key, Value value) noexcept {
        std::scoped_lock lock{_mutex};
        _cache.emplace(key, std::move(value));
    }
};

}// namespace luisa
