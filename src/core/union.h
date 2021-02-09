//
// Created by Mike Smith on 2021/2/9.
//

#pragma once

#include <memory>
#include <type_traits>
#include <algorithm>

#include <core/concepts.h>
#include <core/logging.h>

namespace luisa {

namespace detail {

template<typename Tuple, typename U, int index>
struct IndexOfImpl {
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
    static constexpr auto type_count = sizeof...(T);

    using Types = std::tuple<T...>;

    template<typename U>
    static constexpr int index_of = detail::IndexOfImpl<Types, U, 0>::value;

    template<typename U>
    static constexpr bool contains = (index_of<U> != -1);

private:
    std::aligned_storage_t<size_bytes, alignment_bytes> _storage;
    int _index{-1};

    template<int current, typename F>
    void _visit_impl(F &&f) const noexcept {
        if constexpr (current != type_count) {
            if (current == _index) {
                f(*reinterpret_cast<const std::tuple_element_t<current, Types> *>(&_storage));
            } else {
                _visit_impl<current + 1>(std::forward<F>(f));
            }
        }
    }

public:
    Union() noexcept = default;
    Union(Union &&) noexcept = default;
    Union(const Union &) noexcept = default;
    Union &operator=(Union &&) noexcept = default;
    Union &operator=(const Union &u) noexcept = default;

    template<typename U, std::enable_if_t<contains<U>, int> = 0>
    explicit Union(U u) noexcept { emplace(std::move(u)); }

    template<typename U, std::enable_if_t<contains<std::remove_cvref_t<U>>, int> = 0>
    void emplace(U &&u) noexcept {
        using UU = std::remove_cvref_t<U>;
        _index = index_of<UU>;
        new (&_storage) UU{std::forward<U>(u)};
    }

    void clear() noexcept { _index = -1; }

    [[nodiscard]] auto empty() const noexcept { return _index == -1; }
    [[nodiscard]] auto index() const noexcept { return _index; }

    template<typename U, std::enable_if_t<contains<U>, int> = 0>
    [[nodiscard]] const U &as() const noexcept {
        if (auto required_index = index_of<U>; _index != required_index) {
            LUISA_ERROR_WITH_LOCATION("Invalid type #{} required from union holding type #{}.", required_index, _index);
        }
        return *reinterpret_cast<const U *>(&_storage);
    }

    template<typename U, std::enable_if_t<contains<U>, int> = 0>
    [[nodiscard]] bool is() const noexcept { return index_of<U> == _index; }

    template<typename F>
    void visit(F &&f) const noexcept {
        if (!empty()) { _visit_impl<0>(std::forward<F>(f)); }
    }
    
};

}// namespace luisa
