//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <luisa/dsl/ref.h>
#include <luisa/dsl/arg.h>

namespace luisa::compute {

namespace detail {

template<typename T>
void apply_default_initializer(Ref<T> var) noexcept {
    if constexpr (luisa::is_basic_v<T>) {
        dsl::assign(var, T{});
    } else {
        auto impl = [&var]<size_t... i>(std::index_sequence<i...>) noexcept {
            (apply_default_initializer(detail::Ref{var.template get<i>()}), ...);
        };
        impl(std::make_index_sequence<std::tuple_size_v<struct_member_tuple_t<T>>>{});
    }
}

}// namespace detail

/// Class of variable
template<typename T>
struct Var : public detail::Ref<T> {

    static_assert(std::is_trivially_destructible_v<T>);

    /// Construct from expression
    explicit Var(const Expression *expr) noexcept
        : detail::Ref<T>{expr} {}

    // for local variables of basic or array types
    /// Construct a local variable of basic or array types
    Var() noexcept
        : detail::Ref<T>{detail::FunctionBuilder::current()->local(Type::of<T>())} {
        // No more necessary. Backends now guarantee variable initialization.
        // detail::apply_default_initializer(detail::Ref{*this});
    }

    /// Assign members from args
    template<typename... Args, size_t... i>
    Var(std::tuple<Args...> args, std::index_sequence<i...>) noexcept : Var{} {
        (dsl::assign(this->template get<i>(), std::get<i>(args)), ...);
    }

    /// Assign members
    template<typename... Args>
    Var(std::tuple<Args...> args) noexcept
        : Var{args, std::index_sequence_for<Args...>{}} {}

    /// Assign from a single argument
    template<typename Arg>
        requires concepts::different<std::remove_cvref_t<Arg>, Var<T>> &&
                 std::negation_v<std::is_pointer<std::remove_cvref_t<Arg>>>
    Var(Arg &&arg) noexcept : Var{} {
        using member_tuple = struct_member_tuple_t<T>;
        if constexpr (std::tuple_size_v<member_tuple> > 1u ||
                      std::is_same_v<expr_value_t<Arg>, T>) {
            dsl::assign(*this, std::forward<Arg>(arg));
        } else {
            dsl::assign(this->template get<0u>(), std::forward<Arg>(arg));
        }
    }

    /// Assign from list
    template<typename First, typename Second, typename... Other>
    Var(First &&first, Second &&second, Other &&...other) noexcept
        : Var{std::make_tuple(
              Expr{std::forward<First>(first)},
              Expr{std::forward<Second>(second)},
              Expr{std::forward<Other>(other)}...)} {}

    // create as function arguments, for internal use only
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Ref<T>{detail::FunctionBuilder::current()->argument(Type::of<T>())} {}
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : detail::Ref<T>{detail::FunctionBuilder::current()->reference(Type::of<T>())} {}

    Var(Var &&) noexcept = default;
    Var(const Var &another) noexcept : Var{Expr{another}} {}
    void operator=(Var &&rhs) & noexcept { detail::Ref<T>::operator=(std::move(rhs)); }
    void operator=(const Var &rhs) & noexcept { detail::Ref<T>::operator=(rhs); }
};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

template<typename... T>
Var(std::tuple<T...>) -> Var<std::tuple<expr_value_t<T>...>>;

template<typename T, size_t N>
using ArrayVar = Var<std::array<expr_value_t<T>, N>>;

using Int = Var<int>;
using Int2 = Var<int2>;
using Int3 = Var<int3>;
using Int4 = Var<int4>;
using UInt = Var<uint>;
using UInt2 = Var<uint2>;
using UInt3 = Var<uint3>;
using UInt4 = Var<uint4>;
using Float = Var<float>;
using Float2 = Var<float2>;
using Float3 = Var<float3>;
using Float4 = Var<float4>;
using Bool = Var<bool>;
using Bool2 = Var<bool2>;
using Bool3 = Var<bool3>;
using Bool4 = Var<bool4>;
using Float2x2 = Var<float2x2>;
using Float3x3 = Var<float3x3>;
using Float4x4 = Var<float4x4>;

template<size_t N>
using ArrayInt = ArrayVar<int, N>;
template<size_t N>
using ArrayInt2 = ArrayVar<int2, N>;
template<size_t N>
using ArrayInt3 = ArrayVar<int3, N>;
template<size_t N>
using ArrayInt4 = ArrayVar<int4, N>;
template<size_t N>
using ArrayUInt = ArrayVar<uint, N>;
template<size_t N>
using ArrayUInt2 = ArrayVar<uint2, N>;
template<size_t N>
using ArrayUInt3 = ArrayVar<uint3, N>;
template<size_t N>
using ArrayUInt4 = ArrayVar<uint4, N>;
template<size_t N>
using ArrayFloat = ArrayVar<float, N>;
template<size_t N>
using ArrayFloat2 = ArrayVar<float2, N>;
template<size_t N>
using ArrayFloat3 = ArrayVar<float3, N>;
template<size_t N>
using ArrayFloat4 = ArrayVar<float4, N>;
template<size_t N>
using ArrayBool = ArrayVar<bool, N>;
template<size_t N>
using ArrayBool2 = ArrayVar<bool2, N>;
template<size_t N>
using ArrayBool3 = ArrayVar<bool3, N>;
template<size_t N>
using ArrayBool4 = ArrayVar<bool4, N>;
template<size_t N>
using ArrayFloat2x2 = ArrayVar<float2x2, N>;
template<size_t N>
using ArrayFloat3x3 = ArrayVar<float3x3, N>;
template<size_t N>
using ArrayFloat4x4 = ArrayVar<float4x4, N>;

}// namespace luisa::compute

