//
// Created by Mike Smith on 2021/9/13.
//

#pragma once

#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

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
#include <EASTL/fixed_hash_map.h>
#include <EASTL/fixed_hash_set.h>
#include <EASTL/vector_multimap.h>
#include <EASTL/vector_multiset.h>
#include <EASTL/bonus/lru_cache.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/container/node_hash_map.h>
#include <absl/container/node_hash_set.h>

#include <core/dll_export.h>
#include <core/hash.h>

namespace luisa {

namespace detail {
LC_CORE_API void *allocator_allocate(size_t size, size_t alignment) noexcept;
LC_CORE_API void allocator_deallocate(void *p, size_t alignment) noexcept;
LC_CORE_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace detail

template<typename T = std::byte>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    template<typename U>
    constexpr allocator(allocator<U>) noexcept {}
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
    return std::construct_at(allocate<T>(), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    if (p != nullptr) {
        std::destroy_at(p);
        deallocate(p);
    }
}

using eastl::const_pointer_cast;
using eastl::dynamic_pointer_cast;
using eastl::enable_shared_from_this;
using eastl::function;
using eastl::make_shared;
using eastl::make_unique;
using eastl::reinterpret_pointer_cast;
using eastl::shared_array;
using eastl::shared_ptr;
using eastl::static_pointer_cast;
using eastl::unique_ptr;
using eastl::weak_ptr;

using string = std::basic_string<char, std::char_traits<char>, allocator<char>>;
using std::string_view;

using eastl::span;
using eastl::vector;

using eastl::deque;
using eastl::list;
using eastl::make_optional;
using eastl::monostate;
using eastl::move_only_function;
using eastl::nullopt;
using eastl::optional;
using eastl::queue;
using eastl::slist;
using eastl::variant;
using eastl::variant_alternative_t;
using eastl::variant_size_v;

#define LUISA_COMPUTE_USE_ABSEIL_HASH_TABLES

#ifdef LUISA_COMPUTE_USE_ABSEIL_HASH_TABLES
template<typename K, typename V,
         typename Hash = Hash64,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<std::pair<const K, V>>>
using unordered_map = absl::flat_hash_map<K, V, Hash, Eq, Allocator>;
template<typename K,
         typename Hash = Hash64,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<const K>>
using unordered_set = absl::flat_hash_set<K, Hash, Eq, Allocator>;
#else
using std::unordered_map;
using std::unordered_set;
#endif

template<typename K, typename V,
         typename Compare = std::less<K>,
         typename Alloc = luisa::allocator<std::pair<const K, V>>>
using map = absl::btree_map<K, V, Compare, Alloc>;

template<typename K, typename V,
         typename Compare = std::less<K>,
         typename Alloc = luisa::allocator<std::pair<const K, V>>>
using multimap = absl::btree_multimap<K, V, Compare, Alloc>;

template<typename K,
         typename Compare = std::less<K>,
         typename Alloc = luisa::allocator<K>>
using set = absl::btree_set<K, Compare, Alloc>;

template<typename K,
         typename Compare = std::less<K>,
         typename Alloc = luisa::allocator<K>>
using multiset = absl::btree_multiset<K, Compare, Alloc>;

using eastl::bit_cast;
using eastl::get;
using eastl::get_if;
using eastl::holds_alternative;
using eastl::visit;

using eastl::fixed_hash_map;
using eastl::fixed_hash_multimap;
using eastl::fixed_map;
using eastl::fixed_multimap;
using eastl::fixed_multiset;
using eastl::fixed_set;

using eastl::bitvector;
using eastl::lru_cache;

using eastl::vector_map;
using eastl::vector_multimap;
using eastl::vector_multiset;
using eastl::vector_set;

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
