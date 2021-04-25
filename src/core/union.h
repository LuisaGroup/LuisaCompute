//
// Created by Mike Smith on 2021/2/9.
//

#pragma once

#include <memory>
#include <type_traits>
#include <algorithm>

#include <core/platform.h>
#include <core/concepts.h>
#include <core/logging.h>
#include <core/memory.h>

namespace luisa {

namespace detail {

template<typename Tuple, typename U, int index = 0>
struct IndexOfImpl {

    template<typename T>
    static constexpr auto always_false = false;

    static_assert(always_false<U>);
};

template<typename U, int index>
struct IndexOfImpl<std::tuple<>, U, index> {
    static constexpr auto value = -1;
};

template<typename First, typename... Others, typename U, int index>
struct IndexOfImpl<std::tuple<First, Others...>, U, index> {
    static constexpr auto value = IndexOfImpl<std::tuple<Others...>, U, index + 1>::value;
};

template<typename... Others, typename U, int index>
struct IndexOfImpl<std::tuple<U, Others...>, U, index> {
    static constexpr auto value = index;
};

}// namespace detail

template<typename... T>
class Union {

public:
    static_assert(std::conjunction_v<std::is_trivially_destructible<T>...>);
    static constexpr auto alignment_bytes = std::max({alignof(T)...});
    static constexpr auto size_bytes = std::max(alignment_bytes, std::max({sizeof(T)...}));
    static constexpr auto alternative_count = sizeof...(T);

    using Alternatives = std::tuple<T...>;

    template<typename U>
    static constexpr int index_of = detail::IndexOfImpl<Alternatives, U>::value;

    template<typename U>
    static constexpr bool contains = (index_of<U> != -1);

private:
    std::aligned_storage_t<size_bytes, alignment_bytes> _storage;
    int _index{-1};

#define LUISA_UNION_MAKE_DISPATCH_IMPL(CONST)                             \
    template<int current, typename F>                                     \
    LUISA_FORCE_INLINE void _dispatch_impl##CONST(F &&f) CONST noexcept { \
        if constexpr (current != alternative_count) {                     \
            if (current == _index) {                                      \
                using U = std::tuple_element_t<current, Alternatives>;    \
                f(*std::launder(reinterpret_cast<CONST U *>(&_storage))); \
            } else {                                                      \
                _dispatch_impl##CONST<current + 1>(std::forward<F>(f));   \
            }                                                             \
        }                                                                 \
    }
    LUISA_UNION_MAKE_DISPATCH_IMPL()
    LUISA_UNION_MAKE_DISPATCH_IMPL(const)
#undef LUISA_UNION_MAKE_DISPATCH_IMPL

    template<typename... A, std::enable_if_t<std::conjunction_v<std::is_move_constructible<A>...>, int> = 0>
    void _copy_or_move(Union<A...> &&rhs) noexcept {
        static_assert(std::is_same_v<Union<A...>, Union>);
        if (!rhs.empty()) {
            rhs.dispatch([p = &_storage](auto &v) mutable noexcept {
                using U = std::remove_cvref_t<decltype(v)>;
                luisa::construct_at(reinterpret_cast<U *>(p), std::move(v));
            });
        }
        _index = rhs._index;
        rhs.clear();
    }

    template<typename... A, std::enable_if_t<std::conjunction_v<std::is_copy_constructible<A>...>, int> = 0>
    void _copy_or_move(const Union<A...> &rhs) noexcept {
        static_assert(std::is_same_v<Union<A...>, Union>);
        rhs.dispatch([this](auto &v) noexcept {
            using U = std::remove_cvref_t<decltype(v)>;
            luisa::construct_at(reinterpret_cast<U *>(&_storage), v);
        });
        _index = rhs._index;
    }

public:
    Union() noexcept = default;

    template<typename U, std::enable_if_t<std::is_same_v<std::remove_cvref_t<U>, Union>, int> = 0>
    Union(U &&another) noexcept { _copy_or_move(std::forward<U>(another)); }

    template<typename U, std::enable_if_t<contains<U>, int> = 0>
    explicit Union(U u) { emplace(std::move(u)); }

    Union &operator=(Union &&) noexcept = delete;
    Union &operator=(const Union &) noexcept = delete;

    template<typename U>
    void assign(U &&rhs) noexcept { _copy_or_move(std::forward<U>(rhs)); }

    template<typename U, std::enable_if_t<contains<std::remove_cvref_t<U>>, int> = 0>
    decltype(auto) emplace(U &&u) {
        using UU = std::remove_cvref_t<U>;
        _index = index_of<UU>;
        return *luisa::construct_at(reinterpret_cast<UU *>(&_storage), std::forward<U>(u));
    }

    template<typename U, typename... Args, std::enable_if_t<contains<std::remove_cvref_t<U>>, int> = 0>
    U &emplace(Args &&...args) {
        _index = index_of<U>;
        return *luisa::construct_at(reinterpret_cast<U *>(&_storage), std::forward<Args>(args)...);
    }

    void clear() noexcept { _index = -1; /* trivially destructible */ }

    [[nodiscard]] auto empty() const noexcept { return _index == -1; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    template<typename U, std::enable_if_t<contains<U>, int> = 0>
    [[nodiscard]] const U &as() const noexcept {
        if (auto required_index = index_of<U>; _index != required_index) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid type #{} required from union holding type #{}.", required_index, _index);
        }
        return *std::launder(reinterpret_cast<const U *>(&_storage));
    }

    template<typename U>
    [[nodiscard]] bool holds() const noexcept {
        if constexpr (contains<U>) { return index_of<U> == _index; }
        return false;
    }

    template<typename F, std::enable_if_t<std::conjunction_v<std::is_invocable<F, T &>...>, int> = 0>
    inline void dispatch(F &&f) noexcept {
        if (!empty()) { _dispatch_impl<0>(std::forward<F>(f)); }
    }

    template<typename F,
             std::enable_if_t<
                 std::conjunction_v<std::negation<std::conjunction<std::is_invocable<F, T &>...>>,
                                    std::is_invocable<F, const T &>...>,
                 int> = 0>
    inline void dispatch(F &&f) const noexcept {
        if (!empty()) { _dispatch_implconst<0>(std::forward<F>(f)); }
    }
};

}// namespace luisa
