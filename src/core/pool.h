//
// Created by Mike Smith on 2021/10/29.
//

#pragma once

#include <array>
#include <mutex>

#include <core/logging.h>
#include <core/spin_mutex.h>
#include <core/thread_safety.h>
#include <core/stl.h>

namespace luisa {

/**
 * @brief Pool class
 * 
 * @tparam T type
 * @tparam thread_safe whether the pool is thread-safe
 */
template<typename T, bool thread_safe = true>
class Pool : public thread_safety<conditional_mutex_t<true>> {

public:
    static constexpr auto block_size = 64u;

private:
    luisa::vector<T *> _blocks;
    luisa::vector<T *> _available_objects;

private:
    void _enlarge() noexcept {
        auto size = sizeof(T) * block_size;
        auto p = static_cast<T *>(detail::allocator_allocate(size, alignof(T)));
        if (_blocks.empty()) { _available_objects.reserve(block_size); }
        _blocks.emplace_back(p);
        p += block_size;
        for (auto i = 0u; i < block_size; i++) {
            _available_objects.emplace_back(--p);
        }
    }

public:
    /**
     * @brief Construct a new Pool object.
     * default constructor 
     */
    Pool() noexcept = default;
    Pool(Pool &&) noexcept = default;
    Pool(const Pool &) noexcept = delete;
    Pool &operator=(Pool &&) noexcept = default;
    Pool &operator=(const Pool &) noexcept = default;

    /**
     * @brief Destroy the Pool object.
     * detect leaking
     */
    ~Pool() noexcept {
        if (!_blocks.empty()) {
            if (auto available = _available_objects.size(),
                expected = _blocks.size() * block_size;
                available != expected) {
                LUISA_WARNING_WITH_LOCATION(
                    "Leaks detected in pool: "
                    "expected {} objects but got {}.",
                    expected, available);
            }
            for (auto b : _blocks) {
                detail::allocator_deallocate(b, alignof(T));
            }
        }
    }

    /**
     * @brief Construct a T object in pool.
     * Thread safe. Construct using space in pool.
     * @tparam Args construct parameters of T
     * @param args construct parameters of T
     * @return pointer to constructed object
     */
    template<typename... Args>
    [[nodiscard]] auto create(Args &&...args) noexcept {
        auto p = with_lock([this] {
            if (_available_objects.empty()) { _enlarge(); }
            auto p = _available_objects.back();
            _available_objects.pop_back();
            return p;
        });
        return std::construct_at(p, std::forward<Args>(args)...);
    }

    /**
     * @brief Recycle a T object to pool.
     * Will call T's destruct function. Thread safe. Object's address will be saved in pool.
     * If object is not construct by pool, may cause leak warning.
     * @param object object to be recycled
     */
    void recycle(T *object) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) { object->~T(); }
        with_lock([this, object] { _available_objects.emplace_back(object); });
    }
};

}// namespace luisa
