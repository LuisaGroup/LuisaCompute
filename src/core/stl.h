//
// Created by Mike Smith on 2021/9/13.
//

#pragma once

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/fmt/fmt.h>

#include <EASTL/bit.h>
#include <EASTL/span.h>
#include <EASTL/list.h>
#include <EASTL/slist.h>
#include <EASTL/deque.h>
#include <EASTL/queue.h>
#include <EASTL/memory.h>
#include <EASTL/vector.h>
#include <EASTL/variant.h>
#include <EASTL/optional.h>
#include <EASTL/bitvector.h>
#include <EASTL/fixed_map.h>
#include <EASTL/fixed_set.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/shared_ptr.h>
#include <EASTL/functional.h>
#include <EASTL/vector_map.h>
#include <EASTL/vector_set.h>
#include <EASTL/shared_array.h>
#include <EASTL/fixed_vector.h>
#include <EASTL/fixed_hash_map.h>
#include <EASTL/fixed_hash_set.h>
#include <EASTL/vector_multimap.h>
#include <EASTL/vector_multiset.h>
#include <EASTL/bonus/lru_cache.h>
#include <EASTL/bonus/ring_buffer.h>

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <core/dll_export.h>
#include <core/hash.h>

namespace luisa {

namespace detail {
LC_CORE_API void *allocator_allocate(size_t size, size_t alignment) noexcept;
LC_CORE_API void allocator_deallocate(void *p, size_t alignment) noexcept;
LC_CORE_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace detail

[[nodiscard]] inline auto align(size_t s, size_t a) noexcept {
    return (s + a - 1u) / a * a;
}

template<typename T = std::byte>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    explicit constexpr allocator(const char *) noexcept {}
    template<typename U>
    constexpr allocator(allocator<U>) noexcept {}
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        return static_cast<T *>(detail::allocator_allocate(sizeof(T) * n, alignof(T)));
    }
    [[nodiscard]] auto allocate(std::size_t n, size_t alignment, size_t) const noexcept {
        assert(alignment >= alignof(T));
        return static_cast<T *>(detail::allocator_allocate(sizeof(T) * n, alignment));
    }
    void deallocate(T *p, size_t) const noexcept {
        detail::allocator_deallocate(p, alignof(T));
    }
    void deallocate(void *p, size_t) const noexcept {
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
    return std::construct_at(allocate<T>(), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    if (p != nullptr) {
        std::destroy_at(p);
        deallocate(p);
    }
}

using string = std::basic_string<char, std::char_traits<char>, allocator<char>>;
using std::string_view;

using eastl::const_pointer_cast;
using eastl::dynamic_pointer_cast;
using eastl::enable_shared_from_this;
using eastl::function;
using eastl::make_shared;
using eastl::make_unique;
using eastl::reinterpret_pointer_cast;
using eastl::shared_ptr;
using eastl::span;
using eastl::static_pointer_cast;
using eastl::unique_ptr;
using eastl::weak_ptr;

// FIXME: eastl is not compatible with std-style allocators
template<typename T>
using shared_array = eastl::shared_array<T>;

template<typename T>
using vector = eastl::vector<T>;

template<typename T>
using deque = eastl::deque<T>;

template<typename T, typename Container = luisa::deque<T>>
using queue = eastl::queue<T, Container>;

template<typename T>
using forward_list = eastl::slist<T>;

using eastl::bitvector;
using eastl::fixed_hash_map;
using eastl::fixed_hash_multimap;
using eastl::fixed_map;
using eastl::fixed_multimap;
using eastl::fixed_multiset;
using eastl::fixed_set;
using eastl::ring_buffer;

template<typename T, size_t n, bool allow_overflow = true>
using fixed_vector = eastl::fixed_vector<T, n, allow_overflow, eastl::allocator>;

using eastl::lru_cache;

using eastl::vector_map;
using eastl::vector_multimap;
using eastl::vector_multiset;
using eastl::vector_set;

using eastl::make_optional;
using eastl::monostate;
using eastl::move_only_function;
using eastl::nullopt;
using eastl::optional;
using eastl::variant;
using eastl::variant_alternative_t;
using eastl::variant_size_v;

template<typename T = void>
struct equal_to {
    [[nodiscard]] bool operator()(const T &lhs, const T &rhs) const noexcept { return lhs == rhs; }
};

template<>
struct equal_to<void> {
    using is_transparent = void;
    template<typename T1, typename T2>
    [[nodiscard]] bool operator()(T1 &&lhs, T2 &&rhs) const noexcept {
        return std::forward<T1>(lhs) == std::forward<T2>(rhs);
    }
};

#define LUISA_COMPUTE_USE_ROBIN_MAP

#ifdef LUISA_COMPUTE_USE_ROBIN_MAP
template<typename K, typename V,
         typename Hash = hash<K>,
         typename Eq = equal_to<>,
         typename Allocator = luisa::allocator<std::pair<const K, V>>>
using unordered_map = tsl::robin_map<K, V, Hash, Eq, Allocator>;

template<typename K,
         typename Hash = hash<K>,
         typename Eq = equal_to<>,
         typename Allocator = luisa::allocator<K>>
using unordered_set = tsl::robin_set<K, Hash, Eq, Allocator>;
#else
template<typename K, typename V,
         typename Hash = hash<K>,
         typename Eq = equal_to<>,
         typename Allocator = luisa::allocator<std::pair<const K, V>>>
using unordered_map = std::unordered_map<K, V, Hash, Eq, Allocator>;

template<typename K,
         typename Hash = hash<K>,
         typename Eq = equal_to<>,
         typename Allocator = luisa::allocator<K>>
using unordered_set = std::unordered_set<K, Hash, Eq, Allocator>;
#endif

// FIXME: EASTL does not support `contains()` and
//  does not implement `try_emplace()` correctly.
//  So we use the std ones to work around.
template<typename Key,
         typename Compare = std::less<>,
         typename Allocator = luisa::allocator<Key>>
using set = std::set<Key, Compare, Allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>,
         typename Allocator = luisa::allocator<std::pair<const Key, Value>>>
using map = std::map<Key, Value, Compare, Allocator>;

template<typename Key,
         typename Compare = std::less<>,
         typename Allocator = luisa::allocator<Key>>
using multiset = std::multiset<Key, Compare, Allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>,
         typename Allocator = luisa::allocator<std::pair<const Key, Value>>>
using multimap = std::multimap<Key, Value, Compare, Allocator>;

using eastl::bit_cast;
using eastl::get;
using eastl::get_if;
using eastl::holds_alternative;
using eastl::visit;

struct default_sentinel_t {};
inline constexpr default_sentinel_t default_sentinel{};

namespace detail {

template<typename F>
class LazyConstructor {
private:
    mutable F _ctor;

public:
    explicit LazyConstructor(F _ctor) noexcept : _ctor{_ctor} {}
    [[nodiscard]] operator auto() const noexcept { return _ctor(); }
};

}// namespace detail

template<typename F>
[[nodiscard]] auto lazy_construct(F ctor) noexcept {
    return detail::LazyConstructor<F>(ctor);
}

#ifndef FMT_STRING
#define FMT_STRING(...) __VA_ARGS__
#endif

template<typename FMT, typename... Args>
[[nodiscard]] inline auto format(FMT &&f, Args &&...args) noexcept {
    using memory_buffer = fmt::basic_memory_buffer<char, fmt::inline_buffer_size, luisa::allocator<char>>;
    memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), std::forward<FMT>(f), std::forward<Args>(args)...);
    return luisa::string{buffer.data(), buffer.size()};
}

[[nodiscard]] inline auto hash_to_string(uint64_t hash) noexcept {
    return luisa::format("{:016X}", hash);
}

}// namespace luisa
