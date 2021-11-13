//
// Created by Mike on 2021/11/13.
//

#pragma once

#include <mutex>
#include <optional>

#include <core/allocator.h>

namespace luisa {

template<typename Key, typename Value, typename Hash = luisa::Hash64>
class LRUCache {

private:
    luisa::unordered_map<Key, uint64_t, Hash> _key_to_timepoint;
    luisa::map<uint64_t, std::pair<Key, Value>> _timepoint_to_key_and_value;
    size_t _capacity;
    uint64_t _current_timepoint{0u};
    std::mutex _mutex;

public:
    explicit LRUCache(size_t cap) noexcept : _capacity{cap} {}
    LRUCache(LRUCache &&) noexcept = delete;
    LRUCache(const LRUCache &) noexcept = delete;
    LRUCache &operator=(LRUCache &&) noexcept = delete;
    LRUCache &operator=(const LRUCache &) noexcept = delete;

    [[nodiscard]] static auto create(size_t capacity) noexcept {
        return luisa::make_unique<LRUCache>(capacity);
    }

    [[nodiscard]] auto fetch(const Key &key) noexcept -> std::optional<Value> {
        std::scoped_lock lock{_mutex};
        auto timepoint_iter = _key_to_timepoint.find(key);
        // not in cache
        if (timepoint_iter == _key_to_timepoint.end()) {
            return std::nullopt;
        }
        // in cache, update timepoint
        auto old_timepoint = timepoint_iter->second;
        auto new_timepoint = ++_current_timepoint;
        timepoint_iter->second = new_timepoint;
        auto iter = _timepoint_to_key_and_value.find(old_timepoint);
        auto value = std::move(iter->second.second);
        auto copy = value;
        _timepoint_to_key_and_value.erase(iter);
        _timepoint_to_key_and_value.emplace(
            new_timepoint,
            std::make_pair(key, std::move(value)));
        return copy;
    }

    void update(const Key &key, Value value) noexcept {
        std::scoped_lock lock{_mutex};
        // already in the cache, just update
        auto item = std::make_pair(key, std::move(value));
        if (auto timepoint_iter = _key_to_timepoint.find(key);
            timepoint_iter != _key_to_timepoint.end()) {
            // update timepoint
            auto old_timepoint = timepoint_iter->second;
            auto new_timepoint = ++_current_timepoint;
            timepoint_iter->second = new_timepoint;
            // update the cache item
            _timepoint_to_key_and_value.erase(old_timepoint);
            _timepoint_to_key_and_value.emplace(new_timepoint, item);
        } else {
            // remove the least recently used item if cache exceeds the limit
            if (_key_to_timepoint.size() >= _capacity) {
                // note: map is sorted, so begin() is the least recently used
                auto lru_iter = _timepoint_to_key_and_value.begin();
                auto lru_key = lru_iter->second.first;
                _timepoint_to_key_and_value.erase(lru_iter);
                _key_to_timepoint.erase(lru_key);
            }
            // emplace the new item
            auto timepoint = ++_current_timepoint;
            _key_to_timepoint.emplace(key, timepoint);
            _timepoint_to_key_and_value.emplace(timepoint, item);
        }
    }
};

}// namespace luisa
