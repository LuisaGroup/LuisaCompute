//
// Created by Mike Smith on 2021/2/28.
//

#pragma once

#include <runtime/command.h>
#include <dsl/var.h>

namespace luisa::compute::dsl {

namespace detail {

template<typename T>
struct definition_to_prototype {
    static_assert(always_false_v<T>, "Invalid type in function definition.");
};

template<typename T>
struct definition_to_prototype<Var<T>> {
    using type = T;
};

template<typename T>
struct definition_to_prototype<BufferView<T>> {
    using type = Buffer<T>;
};

template<typename T>
struct prototype_to_creation {
    using type = Var<T>;
};

template<typename T>
struct prototype_to_creation<Buffer<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_creation<BufferView<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_kernel_invocation {
    using type = T;
};

template<typename T>
struct prototype_to_kernel_invocation<Buffer<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_kernel_invocation<BufferView<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_callable_invocation {
    using type = Expr<T>;
};

template<typename T>
struct prototype_to_callable_invocation<Buffer<T>> {
    using type = BufferView<T>;
};

template<typename T>
struct prototype_to_callable_invocation<BufferView<T>> {
    using type = BufferView<T>;
};

template<typename T>
using definition_to_prototype_t = typename definition_to_prototype<T>::type;

template<typename T>
using prototype_to_creation_t = typename prototype_to_creation<T>::type;

template<typename T>
using prototype_to_kernel_invocation_t = typename prototype_to_kernel_invocation<T>::type;

template<typename T>
using prototype_to_callable_invocation_t = typename prototype_to_callable_invocation<T>::type;

}// namespace detail

template<typename T>
class Kernel {
    static_assert(always_false_v<T>);
};

namespace detail {

class KernelInvoke {

private:
    uint32_t _function_uid;
    KernelArgumentEncoder _encoder;

public:
    explicit KernelInvoke(uint32_t function_uid) noexcept : _function_uid{function_uid} {}

    template<typename T>
    KernelInvoke &operator<<(BufferView<T> buffer) noexcept {
        _encoder.encode_buffer(buffer.handle(), buffer.offset_bytes());
        return *this;
    }

    template<typename T>
    KernelInvoke &operator<<(T data) noexcept {
        _encoder.encode_uniform(&data, sizeof(T), alignof(T));
        return *this;
    }

    [[nodiscard]] auto parallelize(uint3 dispatch_size, uint3 block_size = uint3{8u}) &&noexcept {
        return KernelLaunchCommand::create(
            _function_uid, std::move(_encoder),
            dispatch_size, block_size);
    }

    [[nodiscard]] auto parallelize(uint2 dispatch_size, uint2 block_size = uint2{16u, 16u}) &&noexcept {
        return KernelLaunchCommand::create(
            _function_uid, std::move(_encoder),
            uint3{dispatch_size, 1u}, uint3{block_size, 1u});
    }

    [[nodiscard]] auto parallelize(uint32_t dispatch_size, uint32_t block_size = 256u) &&noexcept {
        return KernelLaunchCommand::create(
            _function_uid, std::move(_encoder),
            uint3{dispatch_size, 1u, 1u}, uint3{block_size, 1u, 1u});
    }
};

}// namespace detail

template<typename... Args>
class Kernel<void(Args...)> {

private:
    Function _function;

public:
    Kernel(Kernel &&) noexcept = default;
    Kernel(const Kernel &) noexcept = default;

    template<typename Def>
    requires concepts::invocable_with_return<void, Def, detail::prototype_to_creation_t<Args>...>
    Kernel(Def &&def) noexcept
        : _function{FunctionBuilder::define_kernel([&def] {
              def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...);
          })} {}

    [[nodiscard]] auto operator()(detail::prototype_to_kernel_invocation_t<Args>... args) const noexcept {
        detail::KernelInvoke invoke{_function.uid()};
        (invoke << ... << args);
        return invoke;
    }
};

namespace detail {

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename... T, size_t... i>
[[nodiscard]] inline auto tuple_to_var_impl(std::tuple<T...> tuple, std::index_sequence<i...>) noexcept {
    return Var<std::tuple<expr_value_t<T>...>>{std::get<i>(tuple)...};
}

template<typename... T>
[[nodiscard]] inline auto tuple_to_var(std::tuple<T...> tuple) noexcept {
    return tuple_to_var_impl(std::move(tuple), std::index_sequence_for<T...>{});
}

template<typename... T, size_t... i>
[[nodiscard]] inline auto var_to_tuple_impl(Expr<std::tuple<T...>> v, std::index_sequence<i...>) noexcept {
    return std::tuple<Expr<T>...>{v.template member<i>()...};
}

template<typename... T>
[[nodiscard]] inline auto var_to_tuple(Expr<std::tuple<T...>> v) noexcept {
    return var_to_tuple_impl(v, std::index_sequence_for<T...>{});
}

}// namespace detail

template<typename T>
class Callable {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {

    static_assert(std::negation_v<is_buffer_or_view<Ret>>,
                  "Callables may not return buffers (or their views).");

private:
    Function _function;

public:
    Callable(Callable &&) noexcept = default;
    Callable(const Callable &) noexcept = default;

    template<typename Def>
    requires concepts::invocable<Def, detail::prototype_to_creation_t<Args>...>
    Callable(Def &&def) noexcept
        : _function{FunctionBuilder::define_callable([&def] {
              if constexpr (std::is_same_v<Ret, void>) {
                  def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...);
              } else if constexpr (detail::is_tuple_v<Ret>) {
                  auto ret = detail::tuple_to_var(def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...));
                  FunctionBuilder::current()->return_(ret.expression());
              } else {
                  Var<Ret> ret{def(detail::prototype_to_creation_t<Args>{detail::ArgumentCreation{}}...)};
                  FunctionBuilder::current()->return_(ret.expression());
              }
          })} {}

    auto operator()(detail::prototype_to_callable_invocation_t<Args>... args) const noexcept {
        if constexpr (std::is_same_v<Ret, void>) {
            auto expr = FunctionBuilder::current()->call(
                nullptr,
                fmt::format("custom_{}", _function.uid()),
                {args.expression()...});
            FunctionBuilder::current()->void_(expr);
        } else if constexpr (detail::is_tuple_v<Ret>) {
            Var ret = detail::Expr<Ret>{FunctionBuilder::current()->call(
                Type::of<Ret>(),
                fmt::format("custom_{}", _function.uid()),
                {args.expression()...})};
            return detail::var_to_tuple(ret);
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
using function_signature_t = R(A...);

template<typename... Args>
struct function<std::function<void(Args...)>> {
    using type = function_signature_t<void, definition_to_prototype_t<Args>...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Var<Ret>(Args...)>> {
    using type = function_signature_t<Ret, definition_to_prototype_t<Args>...>;
};

template<typename Ret, typename... Args>
struct function<std::function<Expr<Ret>(Args...)>> {
    using type = function_signature_t<Ret, definition_to_prototype_t<Args>...>;
};

template<typename... Ret, typename... Args>
struct function<std::function<std::tuple<Ret...>(Args...)>> {
    using type = function_signature_t<std::tuple<expr_value_t<Ret>...>, definition_to_prototype_t<Args>...>;
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

namespace luisa::compute::dsl::detail {

struct KernelBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Kernel{std::forward<F>(def)}; }
};

struct CallableBuilder {
    template<typename F>
    [[nodiscard]] auto operator%(F &&def) const noexcept { return Callable{std::forward<F>(def)}; }
};

}// namespace luisa::compute::dsl::detail

#define LUISA_KERNEL ::luisa::compute::dsl::detail::KernelBuilder{} % [&]
#define LUISA_CALLABLE ::luisa::compute::dsl::detail::CallableBuilder{} % [&]
