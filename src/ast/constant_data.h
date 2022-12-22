//
// Created by Mike Smith on 2021/3/6.
//

#pragma once

#include <core/stl/variant.h>
#include <core/stl/vector.h>
#include <core/basic_types.h>
#include <core/concepts.h>

namespace luisa::compute {

namespace detail {

template<template<typename> typename C, typename T, bool constify>
struct constant_data_view {
    static_assert(always_false_v<T>);
};

template<template<typename> typename C, typename... T>
struct constant_data_view<C, std::tuple<T...>, true> {
    using type = luisa::variant<C<const T>...>;
};

template<template<typename> typename C, typename... T>
struct constant_data_view<C, std::tuple<T...>, false> {
    using type = luisa::variant<C<T>...>;
};

template<template<typename> typename C, typename T, bool constify>
using constant_data_view_t = typename constant_data_view<C, T, constify>::type;

template<typename T>
using to_span_t = luisa::span<T>;

template<typename T>
using to_vector_t = luisa::vector<T>;

}// namespace detail

/// Constant data
class LC_AST_API ConstantData {

public:
    using View = detail::constant_data_view_t<detail::to_span_t, basic_types, true>;

protected:
    View _view;
    uint64_t _hash{};

    ConstantData(View v, uint64_t hash) noexcept
        : _view{v}, _hash{hash} {}

public:
    ConstantData() noexcept = default;
    /**
     * @brief Construct ConstantData from given data
     * 
     * @param data must belong to basic_types
     * @return ConstantData 
     */
    [[nodiscard]] static ConstantData create(View data) noexcept;
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    [[nodiscard]] auto view() const noexcept { return _view; }
};

}// namespace luisa::compute
