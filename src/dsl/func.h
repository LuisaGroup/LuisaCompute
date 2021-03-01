//
// Created by Mike Smith on 2021/2/28.
//

#pragma once

#include <dsl/var.h>

namespace luisa::compute::dsl {

template<typename T>
class KernelFunc {
    static_assert(always_false_v<T>);
};

template<typename... Args>
class KernelFunc<void(Args...)> {

private:
    Function _function;

public:
    template<typename Def>
    requires concepts::InvocableRet<void, Def, Var<Args>...>
    KernelFunc(Def &&def) noexcept
        : _function{FunctionBuilder::define_kernel([&def] {
              def(detail::create_argument<Args>()...);
          })} {}

    // TODO: integration into runtime...
    [[nodiscard]] auto function() const noexcept { return _function; }
};

template<typename T>
class CallableFunc {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class CallableFunc<Ret(Args...)> {

private:
    Function _function;

public:
    template<typename Def>
    requires concepts::Invocable<Def, Var<Args>...>
    CallableFunc(Def &&def) noexcept
        : _function{FunctionBuilder::define_callable([&def] {
              if constexpr (std::is_same_v<Ret, void>) {
                  def(detail::create_argument<Args>()...);
              } else {
                  Var<Ret> ret = def(detail::create_argument<Args>()...);
                  FunctionBuilder::current()->return_(ret.expression());
              }
          })} {}

    auto operator()(detail::Expr<Args>... args) const noexcept {
        if constexpr (std::is_same_v<Ret, void>) {
            auto expr = FunctionBuilder::current()->call(
                nullptr,
                fmt::format("custom_{}", _function.uid()),
                {args.expression()...});
            FunctionBuilder::current()->void_(expr);
        } else {
            return detail::Expr<Ret>{FunctionBuilder::current()->call(
                Type::of<Ret>(),
                fmt::format("custom_{}", _function.uid()),
                {args.expression()...})};
        }
    }
};

namespace detail {

template<typename T>
struct function {
    using type = typename function<
        std::remove_cvref_t<decltype(std::function{std::declval<T>()})>>::type;
};

template<typename R, typename... A>
using function_declaration_t = R(A...);

template<typename... Args>
struct function<std::function<void(Args...)>> {
    using type = function_declaration_t<void, var_value_t<Args>...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Var<Ret>(Args...)>> {
    using type = function_declaration_t<Ret, var_value_t<Args>...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Expr<Ret>(Args...)>> {
    using type = function_declaration_t<Ret, var_value_t<Args>...>;
};

template<typename T>
using function_t = typename function<T>::type;

}// namespace detail

template<typename T>
KernelFunc(T &&) -> KernelFunc<detail::function_t<T>>;

template<typename T>
CallableFunc(T &&) -> CallableFunc<detail::function_t<T>>;

}// namespace luisa::compute::dsl
