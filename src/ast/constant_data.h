//
// Created by Mike Smith on 2021/3/6.
//

#pragma once

#include <variant>
#include <memory>
#include <span>
#include <mutex>
#include <vector>
#include <array>

#include <core/basic_types.h>
#include <core/concepts.h>

#include <serialize/key_value_pair.h>

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

}// namespace detail

class AstSerializer;

/// Constant data
class ConstantData {

public:
    friend class AstSerializer;
    using View = detail::constant_data_view_t<luisa::span, basic_types, true>;

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

    template<typename S>
    void save(S& s) {
        s.serialize(MAKE_NAME_PAIR(_hash), MAKE_NAME_PAIR(_view));
    }

    template<typename S>
    void load(S& s) {
        detail::constant_data_view_t<luisa::vector, basic_types, false> data;
        s.serialize(MAKE_NAME_PAIR(_hash), KeyValuePair{"_view", data});
        *this = luisa::visit([](auto &&v) noexcept { return create(v); }, data);
    }
};

}// namespace luisa::compute
