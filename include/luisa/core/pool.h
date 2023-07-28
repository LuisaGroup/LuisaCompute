#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/thread_safety.h>

namespace luisa {

namespace detail {
void LC_CORE_API memory_pool_check_memory_leak(size_t expected, size_t actual) noexcept;
}

/**
 * @brief Pool class
 * 
 * @tparam T type
 * @tparam thread_safe whether the pool is thread-safe
 */
template<typename T, bool thread_safe = true, bool check_recycle = !std::is_trivially_destructible_v<T>>
class Pool : public thread_safety<conditional_mutex_t<true, luisa::spin_mutex>> {

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
            if constexpr (check_recycle) {
                detail::memory_pool_check_memory_leak(
                    _blocks.size() * block_size,
                    _available_objects.size());
            }
            for (auto b : _blocks) {
                detail::allocator_deallocate(b, alignof(T));
            }
        }
    }

    [[nodiscard]] auto allocate() noexcept {
        return with_lock([this] {
            if (_available_objects.empty()) { _enlarge(); }
            auto p = _available_objects.back();
            _available_objects.pop_back();
            return p;
        });
    }

    void deallocate(T *object) noexcept {
        with_lock([this, object] {
            _available_objects.emplace_back(object);
        });
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
        auto p = allocate();
        return std::construct_at(p, std::forward<Args>(args)...);
    }

    /**
     * @brief Recycle a T object to pool.
     * Will call T's destruct function. Thread safe. Object's address will be saved in pool.
     * If object is not construct by pool, may cause leak warning.
     * @param object object to be recycled
     */
    void destroy(T *object) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) { object->~T(); }
        deallocate(object);
    }
};

}// namespace luisa

