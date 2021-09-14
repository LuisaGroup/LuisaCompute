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

#include <core/platform.h>

namespace luisa {

namespace detail {
void *allocator_allocate(size_t size, size_t alignment) noexcept;
void allocator_deallocate(void *p, size_t alignment) noexcept;
}// namespace detail

template<typename T = void>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    constexpr explicit allocator(allocator<>) noexcept;
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        return static_cast<T *>(detail::allocator_allocate(sizeof(T) * n, alignof(T)));
    }
    void deallocate(T *p, size_t) const noexcept {
        detail::allocator_deallocate(p, alignof(T));
    }
};

template<>
struct allocator<void> {};

template<typename T>
constexpr allocator<T>::allocator(allocator<>) noexcept {}

template<typename T, typename... Args>
[[nodiscard]] inline decltype(auto) new_with_allocator(Args &&...args) noexcept {
    return construct_at(allocator<T>{}.allocate(1u), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    destroy_at(p);
    allocator<T>{}.deallocate(p, 1u);
}

namespace detail {
template<typename T>
struct UniquePtrDeleterWithAllocator {
    void operator()(T *p) const noexcept { allocator<T>{}.deallocate(p, 1u); }
};
}// namespace detail

template<typename T>
using unique_ptr = std::unique_ptr<T, detail::UniquePtrDeleterWithAllocator<T>>;

using std::shared_ptr;
using std::weak_ptr;

template<typename T, typename... Args>
[[nodiscard]] auto make_unique(Args &&...args) noexcept {
    return unique_ptr<T>{new_with_allocator<T>(std::forward<Args>(args)...)};
}

template<typename T, typename... Args>
[[nodiscard]] auto make_shared(Args &&...args) noexcept {
    return std::shared_ptr<T>{
        new_with_allocator<T>(std::forward<Args>(args)...),
        detail::UniquePtrDeleterWithAllocator<T>{},
        allocator{}};
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
