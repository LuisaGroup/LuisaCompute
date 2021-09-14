//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <bit>
#include <span>
#include <array>
#include <vector>
#include <memory>
#include <concepts>

#include <core/clock.h>
#include <core/platform.h>
#include <core/concepts.h>
#include <core/logging.h>
#include <core/mathematics.h>
#include <core/spin_mutex.h>

namespace luisa {

class Arena : public concepts::Noncopyable {

public:
    struct Link {
        std::byte *data;
        Link *next;
        Link(std::byte *data, Link *next) noexcept : data{data}, next{next} {}
    };
    static constexpr auto block_size = static_cast<size_t>(64ul * 1024ul) - sizeof(Link);

private:
    Link *_head{nullptr};
    uint64_t _current_address{0ul};
    size_t _total{0ul};
    spin_mutex _mutex;

public:
    explicit Arena() noexcept = default;
    Arena(Arena &&) noexcept = delete;
    Arena &operator=(Arena &&) noexcept = delete;
    ~Arena() noexcept;
    [[nodiscard]] static Arena &global(bool is_thread_local = false) noexcept;

    template<typename T = std::byte, size_t alignment = alignof(T)>
    [[nodiscard]] auto allocate(size_t n = 1u) {

        static_assert(std::is_trivially_destructible_v<T>);

        static constexpr auto size = sizeof(T);
        auto byte_size = n * size;

        auto do_allocate = [this, byte_size] {
            auto aligned_p = reinterpret_cast<std::byte *>(
                (_current_address + alignment - 1u) / alignment * alignment);
            if (_head == nullptr || aligned_p + byte_size > _head->data + block_size) [[unlikely]] {
                static constexpr auto alloc_alignment = std::max(alignment, static_cast<size_t>(16u));
                static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
                auto alloc_size = std::max(block_size, byte_size);
                static constexpr auto link_alignment = alignof(Link);
                auto link_offset = (alloc_size + link_alignment - 1u) / link_alignment * link_alignment;
                auto alloc_size_with_link = link_offset + sizeof(Link);
                auto storage = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size_with_link));
                if (storage == nullptr) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION(
                        "Failed to allocate memory with size {} and alignment {}.",
                        alloc_size_with_link, alloc_alignment);
                }
                _head = luisa::construct_at(reinterpret_cast<Link *>(storage + link_offset), storage, _head);
                _total += alloc_size_with_link;
                aligned_p = _head->data;
            }
            _current_address = reinterpret_cast<uint64_t>(aligned_p + byte_size);
            return reinterpret_cast<T *>(aligned_p);
        };

        std::scoped_lock lock{_mutex};
        return do_allocate();
    }

    template<typename T, typename... Args>
    [[nodiscard]] T *create(Args &&...args) {
        static_assert(std::is_trivially_destructible_v<T>);
        return luisa::construct_at(allocate<T>(1u), std::forward<Args>(args)...);
    }
};

template<typename T>
class ArenaVector : public concepts::Noncopyable {

    static_assert(std::is_trivially_destructible_v<T>);

private:
    Arena &_arena;
    T *_data{nullptr};
    size_t _capacity{0u};
    size_t _size{0u};

public:
    explicit ArenaVector(Arena &arena, size_t capacity = 16u) noexcept
        : _arena{arena},
          _data{arena.allocate<T>(capacity)},
          _capacity{capacity} {}

    template<typename U>
        requires concepts::container<U> && concepts::constructible<T, typename std::remove_cvref_t<U>::value_type>
        ArenaVector(Arena &arena, U &&span, size_t capacity = 0u)
            : ArenaVector{arena, std::max(span.size(), capacity)} {
            std::uninitialized_copy_n(span.begin(), span.size(), _data);
            _size = span.size();
        }

        template<typename U>
            requires concepts::constructible<T, U>
        explicit ArenaVector(Arena &arena, std::initializer_list<U> init, size_t capacity = 0u)
            : ArenaVector{arena, std::max(init.size(), capacity)} {
            std::uninitialized_copy_n(init.begin(), init.size(), _data);
            _size = init.size();
        }

        ArenaVector(ArenaVector &&) noexcept = default;
        ArenaVector &operator=(ArenaVector &&) noexcept = default;

        [[nodiscard]] auto empty() const noexcept { return _size == 0u; }
        [[nodiscard]] auto capacity() const noexcept { return _capacity; }
        [[nodiscard]] auto size() const noexcept { return _size; }

        [[nodiscard]] T *data() noexcept { return _data; }
        [[nodiscard]] const T *data() const noexcept { return _data; }

        [[nodiscard]] T &operator[](size_t i) noexcept { return _data[i]; }
        [[nodiscard]] const T &operator[](size_t i) const noexcept { return _data[i]; }

        template<typename... Args>
        T &emplace_back(Args &&...args) {
            if (_size == _capacity) {
                _capacity = next_pow2(_capacity * 2u);
                LUISA_VERBOSE_WITH_LOCATION(
                    "Capacity of ArenaVector exceeded, reallocating for {} ({} bytes).",
                    _capacity, _capacity * sizeof(T));
                auto new_data = _arena.allocate<T>(_capacity);
                std::uninitialized_move_n(_data, _size, new_data);
                _data = new_data;
            }
            return *luisa::construct_at(_data + _size++, std::forward<Args>(args)...);
        }

        void pop_back() noexcept { _size--; /* trivially destructible */ }

        [[nodiscard]] T &front() noexcept { return _data[0]; }
        [[nodiscard]] const T &front() const noexcept { return _data[0]; }
        [[nodiscard]] T &back() noexcept { return _data[_size - 1u]; }
        [[nodiscard]] const T &back() const noexcept { return _data[_size - 1u]; }

        [[nodiscard]] T *begin() noexcept { return _data; }
        [[nodiscard]] T *end() noexcept { return _data + _size; }
        [[nodiscard]] const T *cbegin() const noexcept { return _data; }
        [[nodiscard]] const T *cend() const noexcept { return _data + _size; }
};

template<typename T>
    requires concepts::span_convertible<T>
    ArenaVector(Arena &, T &&)
->ArenaVector<typename std::remove_cvref_t<T>::value_type>;

template<typename T>
ArenaVector(Arena &, std::initializer_list<T>) -> ArenaVector<T>;

struct ArenaString : public std::string_view {
    ArenaString(Arena &arena, std::string_view s) noexcept
        : std::string_view{std::strncpy(arena.allocate<char>(s.size()), s.data(), s.size()), s.size()} {}
};

template<typename T>
class ArenaPool : concepts::Noncopyable {

    struct Node {
        T object;
        Node *next;
        [[nodiscard]] static auto of(T *data) noexcept {
            return reinterpret_cast<Node *>(data);
        }
    };

private:
    static_assert(std::is_trivially_destructible_v<T>);
    Arena &_arena;
    Node *_head{nullptr};
    spin_mutex _mutex;
    size_t _count{0u};
    size_t _total{0u};

public:
    explicit ArenaPool(Arena &arena) noexcept : _arena{arena} {}
    ArenaPool(ArenaPool &&) noexcept = delete;
    ArenaPool &operator=(ArenaPool &&) noexcept = delete;

    template<typename... Args>
    [[nodiscard]] auto create(Args &&...args) {
        Node *node = nullptr;
        [[maybe_unused]] auto [count, total] = [this, &node] {
            std::scoped_lock lock{_mutex};
            if (_head == nullptr) {// empty pool
                _total++;
                node = _arena.allocate<Node>();
            } else {
                node = _head;
                _head = _head->next;
                _count--;
            }
            return std::make_pair(_count, _total);
        }();
        auto object = luisa::construct_at(&node->object, std::forward<Args>(args)...);
        LUISA_VERBOSE_WITH_LOCATION(
            "Created pool object at address {} (available = {}, total = {}).",
            fmt::ptr(object), count, total);
        return object;
    }

    void recycle(T *object) noexcept {
        auto node = Node::of(object);
        [[maybe_unused]] auto [count, total] = [node, this] {
            std::scoped_lock lock{_mutex};
            node->next = _head;
            _head = node;
            _count++;
            return std::make_pair(_count, _total);
        }();
        LUISA_VERBOSE_WITH_LOCATION(
            "Recycled pool object at address {} (available = {}, total = {}).",
            fmt::ptr(object), count, total);
    }
};

}// namespace luisa
