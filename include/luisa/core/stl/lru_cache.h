#pragma once

#include <mutex>
#include <EASTL/bonus/lru_cache.h>
#include <luisa/core/stl/hash_fwd.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>

namespace luisa {

template<typename Key, typename Value, typename Hash = luisa::hash<Key>>
using lru_cache = eastl::lru_cache<
    Key, Value,
    EASTLAllocatorType,
    eastl::list<Key, EASTLAllocatorType>,
    eastl::unordered_map<
        Key,
        eastl::pair<Value, typename eastl::list<Key, EASTLAllocatorType>::iterator>,
        luisa::hash<Key>,
        std::equal_to<Key>,
        EASTLAllocatorType>>;

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

    template<typename F>
    void set_delete_callback(F &&f) noexcept {
        _cache.setDeleteCallback(std::forward<F>(f));
    }

    [[nodiscard]] auto fetch(const Key &key) noexcept -> luisa::optional<Value> {
        std::lock_guard lock{_mutex};
        auto value = _cache.at(key);
        _cache.touch(key);
        return value;
    }

    void update(const Key &key, Value value) noexcept {
        std::lock_guard lock{_mutex};
        _cache.emplace(key, std::move(value));
    }
};

}// namespace luisa
