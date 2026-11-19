#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <new>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/thread_safety.h>

namespace luisa {

namespace detail {
void LUISA_CORE_API memory_pool_check_memory_leak(size_t expected, size_t actual) noexcept;
}// namespace detail

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

    void _dispose() noexcept {
        if (!_blocks.empty()) {
            if constexpr (check_recycle) {
                detail::memory_pool_check_memory_leak(
                    _blocks.size() * block_size,
                    _available_objects.size());
            }
            for (auto b : _blocks) {
                detail::allocator_deallocate(b, alignof(T));
            }
            _blocks.clear();
        }
    }

public:
    /**
     * @brief Construct a new Pool object.
     * default constructor 
     */
    Pool() noexcept = default;
    Pool(Pool &&rhs) noexcept
        : _blocks{std::move(rhs._blocks)},
          _available_objects{std::move(rhs._available_objects)} {
    }
    Pool(const Pool &) noexcept = delete;

    Pool &operator=(Pool &&rhs) noexcept {
        _dispose();
        _blocks = std::move(rhs._blocks);
        _available_objects = std::move(rhs._available_objects);
        return *this;
    }
    Pool &operator=(const Pool &) noexcept = default;

    /**
     * @brief Destroy the Pool object.
     * detect leaking
     */
    ~Pool() noexcept {
        _dispose();
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
    requires(luisa::is_constructible_v<T, Args&&...>)
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

/**
 * @brief Lock-free object pool, usable from any number of threads at once.
 *
 * @tparam T type
 * @tparam check_recycle whether the destructor reports slots that were never returned
 *
 * Objects are handed out from blocks of `block_size` slots, and a slot is claimed by a
 * single compare-and-swap on the owning block's occupancy mask. Nothing blocks and
 * nothing is allocated on the fast path, and the only shared cache line a thread touches
 * is the mask of the block that happens to serve it, so concurrent `allocate` /
 * `deallocate` calls do not serialise on one lock word the way `Pool` does.
 *
 * The scheme is free of ABA, not merely unlikely to hit it: a slot is identified by a bit
 * index inside a block, and a block is only ever *appended* to the pool's block list, so
 * no pointer that a thread may have observed is recycled into another role while that
 * thread sits between its load and its compare-and-swap. Reuse is published through the
 * mask -- release when a slot comes back, acquire when it is claimed -- which orders the
 * previous occupant's teardown before the next occupant is constructed over it.
 *
 * Blocks are returned to the process only by the destructor, and a block whose slots all
 * came back is reused instead of dropped, so the footprint follows the high-water mark of
 * concurrently live objects rather than the total number ever handed out.
 */
template<typename T, bool check_recycle = !std::is_trivially_destructible_v<T>>
class ConcurrentPool {

public:
    static constexpr auto block_size = 64u;

private:
    static_assert(block_size == 64u, "The occupancy mask is a single 64-bit word.");
    static_assert(std::is_nothrow_destructible_v<T>);

    struct Block;

    struct Slot {
        Block *block{nullptr};// owning block, needed on the free path
        alignas(T) unsigned char storage[sizeof(T)];
    };

    struct Block {
        /// Append-only successor: an older block is never unlinked while the pool lives,
        /// which is what makes an observed block pointer safe to dereference later.
        Block *next{nullptr};
        /// Bit `i` is set while slot `i` is live.
        std::atomic<uint64_t> occupied{0u};
        Slot slots[block_size];
    };

private:
    std::atomic<Block *> _blocks{nullptr};

private:
    void _append_block() noexcept {
        auto block = static_cast<Block *>(
            detail::allocator_allocate(sizeof(Block), alignof(Block)));
        ::new (static_cast<void *>(block)) Block{};
        for (auto &slot : block->slots) { slot.block = block; }
        auto head = _blocks.load(std::memory_order_relaxed);
        do { block->next = head; } while (!_blocks.compare_exchange_weak(
            head, block,
            std::memory_order_release,
            std::memory_order_relaxed));
    }

public:
    ConcurrentPool() noexcept = default;
    ConcurrentPool(ConcurrentPool &&) noexcept = delete;
    ConcurrentPool(const ConcurrentPool &) noexcept = delete;
    ConcurrentPool &operator=(ConcurrentPool &&) noexcept = delete;
    ConcurrentPool &operator=(const ConcurrentPool &) noexcept = delete;

    ~ConcurrentPool() noexcept {
        auto block = _blocks.load(std::memory_order_relaxed);
        if constexpr (check_recycle) {
            size_t capacity = 0u;
            size_t available = 0u;
            for (auto b = block; b != nullptr; b = b->next) {
                capacity += block_size;
                available += static_cast<size_t>(
                    std::popcount(~b->occupied.load(std::memory_order_relaxed)));
            }
            detail::memory_pool_check_memory_leak(capacity, available);
        }
        while (block != nullptr) {
            auto next = block->next;
            detail::allocator_deallocate(block, alignof(Block));
            block = next;
        }
    }

    /// Number of blocks the pool owns, i.e. its capacity divided by `block_size`.
    /// Blocks are never released while the pool lives, so this is the peak number
    /// of concurrently live objects, rounded up to whole blocks.
    [[nodiscard]] auto num_blocks() const noexcept {
        size_t count = 0u;
        for (auto block = _blocks.load(std::memory_order_acquire);
             block != nullptr; block = block->next) {
            count++;
        }
        return count;
    }

    [[nodiscard]] T *allocate() noexcept {
        constexpr auto all_occupied = ~uint64_t{0u};
        for (;;) {
            for (auto block = _blocks.load(std::memory_order_acquire);
                 block != nullptr; block = block->next) {
                auto mask = block->occupied.load(std::memory_order_relaxed);
                while (mask != all_occupied) {
                    // `~mask` is non-zero here, so `countr_zero` names a free slot.
                    auto index = static_cast<size_t>(std::countr_zero(~mask));
                    auto bit = uint64_t{1u} << index;
                    if (block->occupied.compare_exchange_weak(
                            mask, mask | bit,
                            std::memory_order_acquire,
                            std::memory_order_relaxed)) {
                        return reinterpret_cast<T *>(block->slots[index].storage);
                    }
                    // `mask` is refreshed on failure; a full block exits the loop.
                }
            }
            // Every block is full, or the pool is still empty: append another one. Two
            // threads may both append, which only costs one spare block.
            _append_block();
        }
    }

    void deallocate(T *object) noexcept {
        auto slot = reinterpret_cast<Slot *>(
            reinterpret_cast<unsigned char *>(object) - offsetof(Slot, storage));
        auto block = slot->block;
        auto index = static_cast<size_t>(slot - block->slots);
        auto bit = uint64_t{1u} << index;
        auto previous = block->occupied.fetch_and(~bit, std::memory_order_release);
        LUISA_DEBUG_ASSERT(previous & bit,
                           "Deallocating an object that is not allocated in the pool.");
    }

    /**
     * @brief Construct a T object in pool.
     * Thread safe. Construct using space in pool.
     * @tparam Args construct parameters of T
     * @param args construct parameters of T
     * @return pointer to constructed object
     */
    template<typename... Args>
        requires(luisa::is_constructible_v<T, Args &&...>)
    [[nodiscard]] auto create(Args &&...args) noexcept {
        return std::construct_at(allocate(), std::forward<Args>(args)...);
    }

    /**
     * @brief Recycle a T object to pool.
     * Will call T's destructor first. Thread safe.
     * @param object object to be recycled
     */
    void destroy(T *object) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) { object->~T(); }
        deallocate(object);
    }
};

}// namespace luisa
