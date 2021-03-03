//
// Created by Mike Smith on 2021/2/28.
//

#pragma once

#include <runtime/command.h>
#include <dsl/var.h>

namespace luisa::compute::dsl {

namespace detail {

template<typename T>
struct callable_invoke_argument {
    static_assert(always_false_v<T>);
};

template<typename T>
struct callable_invoke_argument<Var<T>> {
    using type = Expr<T>;
};

template<typename T>
struct callable_invoke_argument<BufferView<T>> {
    using type = BufferView<T>;
};

template<typename T>
using callable_invoke_argument_t = typename callable_invoke_argument<T>::type;

template<typename T>
struct kernel_invoke_argument {
    static_assert(always_false_v<T>);
};

template<typename T>
struct kernel_invoke_argument<Var<T>> {
    using type = const T &;
};

template<typename T>
struct kernel_invoke_argument<BufferView<T>> {
    using type = BufferView<T>;
};

template<typename T>
using kernel_invoke_argument_t = typename kernel_invoke_argument<T>::type;

template<typename T>
struct is_argument : std::false_type {};

template<typename T>
struct is_argument<Var<T>> : std::true_type {};

template<typename T>
struct is_argument<BufferView<T>> : std::true_type {};

template<typename T>
constexpr auto is_argument_v = is_argument<T>::value;

template<typename... T>
concept Arguments = std::conjunction_v<is_argument<T>...>;

}// namespace detail

template<typename T>
class Kernel {
    static_assert(always_false_v<T>);
};

namespace detail {

struct KernelInvoke : public concepts::Noncopyable {

    uint32_t fid;
    KernelArgumentEncoder encoder;

    explicit KernelInvoke(uint32_t function_uid) noexcept
        : fid{function_uid} {}
    KernelInvoke(KernelInvoke &&) noexcept = default;
    KernelInvoke &operator=(KernelInvoke &&) noexcept = delete;
    
    template<typename T>
    KernelInvoke &operator <<(BufferView<T> buffer) noexcept {
        encoder.encode_buffer(buffer.handle(), buffer.offset_bytes());
        return *this;
    }
    
    template<typename T>
    KernelInvoke &operator <<(T data) noexcept {
        encoder.encode_uniform(&data, sizeof(T), alignof(T));
        return *this;
    }

    [[nodiscard]] auto parallelize(uint3 dispatch_size, uint3 block_size = uint3{8u}) &&noexcept {
        return KernelLaunchCommand{fid, std::move(encoder), dispatch_size, block_size};
    }

    [[nodiscard]] auto parallelize(uint2 dispatch_size, uint2 block_size = uint2{16u, 16u}) &&noexcept {
        return KernelLaunchCommand{fid, std::move(encoder), uint3{dispatch_size, 1u}, uint3{block_size, 1u}};
    }

    [[nodiscard]] auto parallelize(uint32_t dispatch_size, uint32_t block_size = 256u) &&noexcept {
        return KernelLaunchCommand{fid, std::move(encoder), uint3{dispatch_size, 1u, 1u}, uint3{block_size, 1u, 1u}};
    }
};

}// namespace detail

template<typename... Args>
class Kernel<void(Args...)> {

    static_assert(std::conjunction_v<detail::is_argument<Args>...>);

private:
    Function _function;

public:
    Kernel(Kernel &&) noexcept = default;
    Kernel(const Kernel &) noexcept = default;
    
    template<typename Def>
    requires concepts::InvocableRet<void, Def, Args...>
    Kernel(Def &&def) noexcept
        : _function{FunctionBuilder::define_kernel([&def] {
              def(Args{detail::ArgumentCreation{}}...);
          })} {}

    [[nodiscard]] auto operator()(detail::kernel_invoke_argument_t<Args>... args) const noexcept {
        detail::KernelInvoke invoke{_function.uid()};
        (invoke << ... << args);
        return invoke;
    }
};

template<typename T>
class Callable {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {

private:
    Function _function;

public:
    Callable(Callable &&) noexcept = default;
    Callable(const Callable &) noexcept = default;
    
    template<typename Def>
    requires concepts::Invocable<Def, Args...>
    Callable(Def &&def) noexcept
        : _function{FunctionBuilder::define_callable([&def] {
              if constexpr (std::is_same_v<Ret, void>) {
                  def(Args{detail::ArgumentCreation{}}...);
              } else {
                  Var ret{def(Args{detail::ArgumentCreation{}}...)};
                  FunctionBuilder::current()->return_(ret.expression());
              }
          })} {}

    auto operator()(detail::callable_invoke_argument_t<Args>... args) const noexcept {
        if constexpr (std::is_same_v<Ret, void>) {
            auto expr = FunctionBuilder::current()->call(
                nullptr,
                fmt::format("custom_{}", _function.uid()),
                {args.expression()...});
            FunctionBuilder::current()->void_(expr);
        } else {
            using RetT = typename Ret::ValueType;
            return detail::Expr<RetT>{FunctionBuilder::current()->call(
                Type::of<RetT>(),
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
    using type = function_declaration_t<void, Args...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Var<Ret>(Args...)>> {
    using type = function_declaration_t<Ret, Args...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Expr<Ret>(Args...)>> {
    using type = function_declaration_t<Var<Ret>, Args...>;
};

template<typename T>
struct function<Kernel<T>> {
    using type = T;
};

template<typename T>
struct function<Callable<T>> {
    using type = T;
};

template<typename T>
using function_t = typename function<T>::type;

}// namespace detail

template<typename T>
Kernel(T &&) -> Kernel<detail::function_t<T>>;

template<typename T>
Callable(T &&) -> Callable<detail::function_t<T>>;

}// namespace luisa::compute::dsl
