//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(detail::Expr<Src> s) noexcept { return s.template as<Dest>(); }

[[nodiscard]] inline auto thread_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->thread_id()};
}

[[nodiscard]] inline auto block_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->block_id()};
}

[[nodiscard]] inline auto dispatch_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->dispatch_id()};
}

[[nodiscard]] inline auto launch_size() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->launch_size()};
}

[[nodiscard]] inline auto block_size() noexcept {
    return FunctionBuilder::current()->block_size();
}

inline void set_block_size(uint x, uint y = 1u, uint z = 1u) noexcept {
    FunctionBuilder::current()->set_block_size(
        uint3{std::max(x, 1u), std::max(y, 1u), std::max(z, 1u)});
}

template<typename... T>
[[nodiscard]] inline auto multiple(T &&...v) noexcept {
    return std::make_tuple(detail::Expr{v}...);
}

// math functions

// atomic functions

// sync functions

}
