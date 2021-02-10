//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <span>
#include <vector>
#include <memory>

#include <core/concepts.h>
#include <core/logging.h>

namespace luisa {

#ifdef _MSC_VER
[[nodiscard]] inline auto aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}
#endif

class Arena : public Noncopyable {

public:
    static constexpr auto block_size = static_cast<size_t>(256ul * 1024ul);

private:
    std::vector<std::byte *> _blocks;
    uint64_t _ptr{0ul};

public:
    Arena() noexcept = default;
    Arena(Arena &&) noexcept = default;
    Arena &operator=(Arena &&) noexcept = default;
    ~Arena() noexcept {
        for (auto p : _blocks) { free(p); }
    }

    template<typename T, typename... Args>
    [[nodiscard]] T *create(Args &&...args) {
        auto memory = allocate<T>(1u);
        return new (memory.data()) T(std::forward<Args>(args)...);
    }

    template<typename T = std::byte, size_t alignment = alignof(T)>
    [[nodiscard]] std::span<T> allocate(size_t n) {

        static_assert(std::is_trivially_destructible_v<T>);
        static constexpr auto size = sizeof(T);

        auto byte_size = n * size;
        auto aligned_p = reinterpret_cast<std::byte *>((_ptr + alignment - 1u) / alignment * alignment);
        if (_blocks.empty() || aligned_p + byte_size > _blocks.back() + block_size) {
            static constexpr auto alloc_alignment = std::max(alignment, sizeof(void *));
            static_assert((alloc_alignment & (alloc_alignment - 1u)) == 0, "Alignment should be power of two.");
            auto alloc_size = (std::max(block_size, byte_size) + alloc_alignment - 1u) / alloc_alignment * alloc_alignment;
            aligned_p = static_cast<std::byte *>(aligned_alloc(alloc_alignment, alloc_size));
            if (aligned_p == nullptr) { LUISA_ERROR_WITH_LOCATION("Failed to allocate memory: size = {}, alignment = {}, count = {}", size, alignment, n); }
            _blocks.emplace_back(aligned_p);
        }
        _ptr = reinterpret_cast<uint64_t>(aligned_p + byte_size);
        return {reinterpret_cast<T *>(aligned_p), n};
    }
};

template<typename T, size_t capacity>
class FixedVector : public Noncopyable {

    static_assert(std::is_trivially_destructible_v<T>);

private:
    T *_data{nullptr};
    size_t _size{0u};

public:
    explicit FixedVector(Arena &arena) noexcept : _data{arena.allocate<T>(capacity).data()} {}

    FixedVector(Arena &arena, std::span<T> span) noexcept : FixedVector{arena} {
        if (span.size() > capacity) {
            LUISA_ERROR_WITH_LOCATION("Capacity of ArenaVector exceeded: {} out of {} required.", span.size(), capacity);
        }
        std::copy(span.begin(), span.end(), _data);
    }

    explicit FixedVector(Arena &arena, std::initializer_list<T> init) noexcept : FixedVector{arena} {
        if (init.size() > capacity) {
            LUISA_ERROR_WITH_LOCATION("Capacity of ArenaVector exceeded: {} out of {} required.", init.size(), capacity);
        }
        std::copy(init.begin(), init.end(), _data);
    }

    FixedVector(FixedVector &&) noexcept = default;
    FixedVector &operator=(FixedVector &&) noexcept = default;

    [[nodiscard]] auto empty() const noexcept { return _size == 0u; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    [[nodiscard]] T *data() noexcept { return _data; }
    [[nodiscard]] const T *data() const noexcept { return _data; }

    [[nodiscard]] T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] const T &operator[](size_t i) const noexcept { return _data[i]; }

    template<typename... Args>
    T &emplace_back(Args &&...args) noexcept {
        if (_size == capacity) {
            LUISA_ERROR_WITH_LOCATION("Capacity of ArenaVector exceeded: {}.", capacity);
        }
        return new (_data + _size++) T{std::forward<Args>(args)...};
    }

    T &push_back(T v) noexcept { return emplace_back(std::move(v)); }
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

struct FixedString : public std::string_view {

    FixedString(Arena &arena, std::string_view s) noexcept
        : std::string_view{std::strncpy(arena.allocate<char>(s.size() + 1).data(), s.data(), s.size()), s.size()} {}

    [[nodiscard]] const char *c_str() const noexcept { return data(); }
};

}// namespace luisa
