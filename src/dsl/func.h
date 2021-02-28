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

    static_assert(std::conjunction_v<is_var<Args>...>);

private:
    FunctionBuilder _builder;

public:
    template<typename Def>
    requires concepts::InvocableRet<void, Def, Args...>
    KernelFunc(Def &&def) noexcept : _builder{Function::Tag::KERNEL} {
        _builder.define([&def] {
            def(detail::create_argument<Args>()...);
        });
    }

    KernelFunc(KernelFunc &&) noexcept = default;
    KernelFunc &operator=(KernelFunc &&) noexcept = delete;

    // TODO: integration into runtime...
    [[nodiscard]] auto function() const noexcept { return Function{_builder}; }
};

template<typename T>
class DeviceFunc {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class DeviceFunc<Ret(Args...)> {
    
    static_assert(std::conjunction_v<is_var<Ret>, is_var<Args>...>);

private:


public:

};

namespace detail {

template<typename T>
struct function {
    using type = typename function<
        std::remove_cvref_t<decltype(std::function{std::declval<T>()})>>::type;
};

template<typename Ret, typename... Args>
struct function<std::function<Ret(Args...)>> {
    using type = Ret(Args...);
};

template<typename T>
using function_t = typename function<T>::type;

}// namespace detail

template<typename T>
KernelFunc(T &&) -> KernelFunc<detail::function_t<T>>;

template<typename T>
DeviceFunc(T &&) -> DeviceFunc<detail::function_t<T>>;

}// namespace luisa::compute::dsl
