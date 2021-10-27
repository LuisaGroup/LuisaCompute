//
// Created by Mike Smith on 2021/9/13.
//

#pragma once

#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace luisa {

using std::construct_at;
using std::destroy_at;

namespace detail {
void *allocator_allocate(size_t size, size_t alignment) noexcept;
void allocator_deallocate(void *p, size_t alignment) noexcept;
void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace detail

template<typename T = std::byte>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    template<typename U>
    constexpr explicit allocator(allocator<U>) noexcept {}
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        return static_cast<T *>(detail::allocator_allocate(sizeof(T) * n, alignof(T)));
    }
    void deallocate(T *p, size_t) const noexcept {
        detail::allocator_deallocate(p, alignof(T));
    }
    template<typename R>
    [[nodiscard]] constexpr auto operator==(allocator<R>) const noexcept -> bool {
        return std::is_same_v<T, R>;
    }
};

template<typename T>
[[nodiscard]] inline auto allocate(size_t n = 1u) noexcept {
    return allocator<T>{}.allocate(n);
}

template<typename T>
inline void deallocate(T *p) noexcept {
    allocator<T>{}.deallocate(p, 0u);
}

template<typename T, typename... Args>
[[nodiscard]] inline auto new_with_allocator(Args &&...args) noexcept {
    return construct_at(allocate<T>(), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    if (p != nullptr) {
        destroy_at(p);
        deallocate(p);
    }
}

struct deleter {
    void operator()(auto p) const noexcept {
        delete_with_allocator(p);
    }
};

template<typename T>
using unique_ptr = std::unique_ptr<T, deleter>;

using std::shared_ptr;
using std::weak_ptr;

template<typename T, typename... Args>
[[nodiscard]] auto make_unique(Args &&...args) noexcept {
    return unique_ptr<T>{new_with_allocator<T>(std::forward<Args>(args)...)};
}

template<typename T, typename... Args>
[[nodiscard]] auto make_shared(Args &&...args) noexcept {
    return shared_ptr<T>{
        new_with_allocator<T>(std::forward<Args>(args)...),
        deleter{}, allocator{}};
}

using string = std::basic_string<char, std::char_traits<char>, allocator<char>>;

template<typename T>
using vector = std::vector<T, allocator<T>>;

template<typename Key, typename Value, typename Pred = std::less<>>
using map = std::map<Key, Value, Pred, allocator<std::pair<const Key, Value>>>;

template<typename Key, typename Pred = std::less<>>
using set = std::set<Key, Pred, allocator<Key>>;

template<typename Key, typename Value, typename Hash = std::hash<Key>, typename Pred = std::equal_to<>>
using unordered_map = std::unordered_map<Key, Value, Hash, Pred, allocator<std::pair<const Key, Value>>>;

template<typename Key, typename Hash = std::hash<Key>, typename Pred = std::equal_to<>>
using unordered_set = std::unordered_set<Key, Hash, Pred, allocator<Key>>;

}// namespace luisa
