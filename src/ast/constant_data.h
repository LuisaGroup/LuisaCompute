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

namespace luisa::compute {

namespace detail {

template<typename T>
struct constant_data_view {
    static_assert(always_false_v<T>);
};

template<typename... T>
struct constant_data_view<std::tuple<T...>> {
    using type = luisa::variant<luisa::span<const T>...>;
};

template<typename T>
using constant_data_view_t = typename constant_data_view<T>::type;

}// namespace detail

class AstSerializer;

/// Constant data
class ConstantData {

public:
    friend class AstSerializer;
    using View = detail::constant_data_view_t<basic_types>;

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
