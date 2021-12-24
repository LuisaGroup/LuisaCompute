//
// Created by Mike Smith on 2021/9/13.
//

#pragma once

#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <span>
#include <string>

#include <EASTL/memory.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/shared_ptr.h>
#include <EASTL/span.h>
#include <EASTL/vector.h>
#include <EASTL/map.h>
#include <EASTL/set.h>
#include <EASTL/deque.h>
#include <EASTL/queue.h>
#include <EASTL/unordered_map.h>
#include <EASTL/unordered_set.h>
#include <EASTL/functional.h>
#include <EASTL/variant.h>
#include <EASTL/optional.h>

namespace luisa {

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

using eastl::function;
using eastl::unique_ptr;
using eastl::shared_ptr;
using eastl::weak_ptr;
using eastl::enable_shared_from_this;
using eastl::make_unique;
using eastl::make_shared;
using eastl::const_pointer_cast;
using eastl::reinterpret_pointer_cast;
using eastl::static_pointer_cast;
using eastl::dynamic_pointer_cast;

using string = std::basic_string<char, std::char_traits<char>, allocator<char>>;

using eastl::vector;
using eastl::span;

using eastl::deque;
using eastl::queue;
using eastl::map;
using eastl::set;
using eastl::multimap;
using eastl::multiset;
using eastl::unordered_map;
using eastl::unordered_set;

using eastl::variant;
using eastl::optional;
using eastl::monostate;
using eastl::visit;
using eastl::holds_alternative;

namespace detail {

template<typename F>
class LazyConstructor {
private:
    mutable F _ctor;

public:
    explicit LazyConstructor(F _ctor) noexcept: _ctor{_ctor} {}
    [[nodiscard]] operator auto() const noexcept { return _ctor(); }
};

}

template<typename F>
[[nodiscard]] auto lazy_construct(F ctor) noexcept {
    return detail::LazyConstructor<F>(ctor);
}

}// namespace luisa
