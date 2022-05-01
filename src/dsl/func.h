//
// Created by Mike Smith on 2021/2/28.
//

#pragma once

#include <type_traits>

#include <core/stl.h>
#include <runtime/command.h>
#include <runtime/device.h>
#include <runtime/shader.h>
#include <dsl/arg.h>
#include <dsl/var.h>

namespace luisa::compute {

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
struct definition_to_prototype<const Var<T> &> {
    using type = T;
};

template<typename T>
struct definition_to_prototype<Var<T> &> {
    using type = T &;
};

template<typename T>
struct prototype_to_creation_tag {
    using type = ArgumentCreation;
};

template<typename T>
struct prototype_to_creation_tag<const T &> {
    using type = ArgumentCreation;
};

template<typename T>
struct prototype_to_creation_tag<T &> {
    using type = ReferenceArgumentCreation;
};

template<typename T>
struct prototype_to_creation {
    using type = Var<T>;
};

template<typename T>
struct prototype_to_creation<T &> {
    using type = Var<T> &;
};

template<typename T>
struct prototype_to_creation<const T &> {
    using type = const Var<T> &;
};

template<typename T>
struct prototype_to_callable_invocation {
    using type = Expr<T>;
};

template<typename T>
struct prototype_to_callable_invocation<const T &> {
    using type = Expr<T>;
};

template<typename T>
struct prototype_to_callable_invocation<T &> {
    using type = Ref<T>;
};

template<typename T>
using definition_to_prototype_t = typename definition_to_prototype<T>::type;

template<typename T>
using prototype_to_creation_tag_t = typename prototype_to_creation_tag<T>::type;

template<typename T>
using prototype_to_creation_t = typename prototype_to_creation<T>::type;

template<typename T>
using prototype_to_callable_invocation_t = typename prototype_to_callable_invocation<T>::type;

/// Kernel default block size
template<size_t N>
[[nodiscard]] constexpr auto kernel_default_block_size() {
    if constexpr (N == 1) {
        return uint3{256u, 1u, 1u};
    } else if constexpr (N == 2) {
        return uint3{16u, 16u, 1u};
    } else if constexpr (N == 3) {
        return uint3{8u, 8u, 8u};
    } else {
        static_assert(always_false_v<std::integral_constant<size_t, N>>);
    }
}

template<typename NextVar, typename... OtherVars, typename NextTag, typename... OtherTags, typename... T>
[[nodiscard]] std::tuple<T..., NextVar, OtherVars...> create_argument_definitions_impl(
    std::tuple<T...> tuple, std::tuple<NextVar, OtherVars...> *, std::tuple<NextTag, OtherTags...> *) noexcept;

/// Create argument definitions
template<typename VarTuple, typename TagTuple, typename T>
[[nodiscard]] inline auto create_argument_definitions(T tuple) noexcept {
    if constexpr (std::tuple_size_v<VarTuple> == 0) {
        return std::move(tuple);// ensure move ctor
    } else {
        return create_argument_definitions_impl(
            std::move(tuple),
            static_cast<VarTuple *>(nullptr),
            static_cast<TagTuple *>(nullptr));
    }
}

/// Append an element in a tuple
template<typename... T, typename A>
[[nodiscard]] inline auto tuple_append(std::tuple<T...> tuple, A &&arg) noexcept {
    auto append = []<typename TT, typename AA, size_t... i>(TT tuple, AA &&arg, std::index_sequence<i...>) noexcept {
        return std::make_tuple(std::move(std::get<i>(tuple))..., std::forward<AA>(arg));
    };
    return append(std::move(tuple), std::forward<A>(arg), std::index_sequence_for<T...>{});
}

template<typename NextVar, typename... OtherVars, typename NextTag, typename... OtherTags, typename... T>
[[nodiscard]] inline std::tuple<T..., NextVar, OtherVars...> create_argument_definitions_impl(
    std::tuple<T...> tuple, std::tuple<NextVar, OtherVars...> *, std::tuple<NextTag, OtherTags...> *) noexcept {
    return create_argument_definitions<std::tuple<OtherVars...>, std::tuple<OtherTags...>>(
        tuple_append(std::move(tuple), NextVar{NextTag{}}));
}

}// namespace detail

template<typename T>
struct is_kernel : std::false_type {};

template<typename T>
struct is_callable : std::false_type {};

template<size_t N, typename... Args>
class Kernel;

template<typename...>
struct Kernel1D;

template<typename...>
struct Kernel2D;

template<typename...>
struct Kernel3D;

template<size_t N, typename... Args>
struct is_kernel<Kernel<N, Args...>> : std::true_type {};

template<typename... Args>
struct is_kernel<Kernel1D<Args...>> : std::true_type {};

template<typename... Args>
struct is_kernel<Kernel2D<Args...>> : std::true_type {};

template<typename... Args>
struct is_kernel<Kernel3D<Args...>> : std::true_type {};

/**
 * @brief Class of kernel function.
 * 
 * To create a kernel, user needs to provide a function. 
 * The function will be called during construction of Kernel object.
 * All operations inside the provided function will be recorded by FunctionBuilder.
 * After calling the function, the function is changed to AST represented by FunctionBuilder.
 * When compiling the kernel, the AST will be sent to backend and translated to specific backend code.
 * 
 * @tparam N = 1, 2, 3. KernelND
 * @tparam Args args of kernel function
 */
template<size_t N, typename... Args>
class Kernel {

    static_assert(N == 1u || N == 2u || N == 3u);
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);
    static_assert(std::negation_v<std::disjunction<std::is_reference<Args>...>>);

    template<typename...>
    friend struct Kernel1D;

    template<typename...>
    friend struct Kernel2D;

    template<typename...>
    friend struct Kernel3D;

private:
    using SharedFunctionBuilder = luisa::shared_ptr<const detail::FunctionBuilder>;
    SharedFunctionBuilder _builder{nullptr};
    explicit Kernel(SharedFunctionBuilder builder) noexcept : _builder{std::move(builder)} {}
    mutable luisa::map<const Device::Interface *, luisa::shared_ptr<Shader<N, Args...>>> _compiled_shaders;

public:
    /**
     * @brief Create Kernel object from function.
     * 
     * Def must be a callable function and not a kernel.
     * This function will be called during construction.
     * 
     * @param def definition of kernel function 
     */
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    Kernel(Def &&def) noexcept {
        static_assert(std::is_invocable_r_v<void, Def, detail::prototype_to_creation_t<Args>...>);
        _builder = detail::FunctionBuilder::define_kernel([&def] {
            detail::FunctionBuilder::current()->set_block_size(detail::kernel_default_block_size<N>());
            []<size_t... i>(auto &&def, std::index_sequence<i...>) noexcept {
                using arg_tuple = std::tuple<Args...>;
                using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                std::invoke(std::forward<decltype(def)>(def),
                            static_cast<detail::prototype_to_creation_t<
                                std::tuple_element_t<i, arg_tuple>> &&>(std::get<i>(args))...);
            }
            (std::forward<Def>(def), std::index_sequence_for<Args...>{});
        });
    }
    [[nodiscard]] const auto &function() const noexcept { return _builder; }

    /**
     * @brief Convenient interface that compiles the kernel and then launches it
     * @param device the device to run on
     * @param args kernel arguments
     * @return shader dispatch delegate
     */
    [[nodiscard]] decltype(auto) operator()(Device &device, detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        auto impl = device.impl();
        auto [iter, _] = _compiled_shaders.try_emplace(impl, nullptr);
        if (iter->second == nullptr) {
            iter->second = luisa::make_shared<Shader<N, Args...>>(device.compile(*this));
        }
        return (*iter->second)(args...);
    }

    template<typename S>
    void serialize(S &s) {
        auto builder = luisa::const_pointer_cast<detail::FunctionBuilder>(_builder);
        s.serialize(MAKE_NAME_PAIR(builder));
    }
};

#define LUISA_KERNEL_BASE(N)                                     \
public                                                           \
    Kernel<N, Args...> {                                         \
        using Kernel<N, Args...>::Kernel;                        \
        Kernel##N##D(Kernel<N, Args...> k) noexcept              \
            : Kernel<N, Args...>{std::move(k._builder)} {}       \
        Kernel##N##D &operator=(Kernel<N, Args...> k) noexcept { \
            this->_builder = std::move(k._builder);              \
            return *this;                                        \
        }                                                        \
    }

/// 1D kernel. Kernel<1, Args...>
template<typename... Args>
struct Kernel1D : LUISA_KERNEL_BASE(1);

/// 2D kernel. Kernel<2, Args...>
template<typename... Args>
struct Kernel2D : LUISA_KERNEL_BASE(2);

/// 3D kernel. Kernel<3, Args...>
template<typename... Args>
struct Kernel3D : LUISA_KERNEL_BASE(3);

/// 1D kernel. Kernel<1, Args...>
template<typename... Args>
struct Kernel1D<void(Args...)> : LUISA_KERNEL_BASE(1);

/// 2D kernel. Kernel<2, Args...>
template<typename... Args>
struct Kernel2D<void(Args...)> : LUISA_KERNEL_BASE(2);

/// 3D kernel. Kernel<3, Args...>
template<typename... Args>
struct Kernel3D<void(Args...)> : LUISA_KERNEL_BASE(3);

#undef LUISA_KERNEL_BASE

namespace detail {

/// Callable invoke
class CallableInvoke {

public:
    static constexpr auto max_argument_count = 64u;

private:
    std::array<const Expression *, max_argument_count> _args{};
    size_t _arg_count{0u};

public:
    CallableInvoke() noexcept = default;
    /// Add an argument.
    template<typename T>
    CallableInvoke &operator<<(Expr<T> arg) noexcept {
        if (_arg_count == max_argument_count) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Too many arguments for callable.");
        }
        _args[_arg_count++] = arg.expression();
        return *this;
    }
    /// Add an argument.
    template<typename T>
    decltype(auto) operator<<(Ref<T> arg) noexcept {
        return (*this << Expr{arg});
    }
    [[nodiscard]] auto args() const noexcept { return luisa::span{_args.data(), _arg_count}; }
};

}// namespace detail

/// Callable class. Callable<T> is not allowed, unless T is a function type.
template<typename T>
class Callable {
    static_assert(always_false_v<T>);
};

template<typename T>
struct is_callable<Callable<T>> : std::true_type {};

/// Callable class with a function type as template parameter.
template<typename Ret, typename... Args>
class Callable<Ret(Args...)> {

    static_assert(
        std::negation_v<std::disjunction<
            is_buffer_or_view<Ret>,
            is_image_or_view<Ret>,
            is_volume_or_view<Ret>>>,
        "Callables may not return buffers, "
        "images or volumes (or their views).");
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _builder;

public:
    /**
     * @brief Construct a Callable object.
     * 
     * The function provided will be called and recorded during construction.
     * 
     * @param f the function of callable.
     */
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    Callable(Def &&f) noexcept
        : _builder{detail::FunctionBuilder::define_callable([&f] {
              static_assert(std::is_invocable_v<Def, detail::prototype_to_creation_t<Args>...>);
              auto create = []<size_t... i>(auto &&def, std::index_sequence<i...>) noexcept {
                  using arg_tuple = std::tuple<Args...>;
                  using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                  using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                  auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                  static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                  return std::invoke(std::forward<decltype(def)>(def),
                                     static_cast<detail::prototype_to_creation_t<
                                         std::tuple_element_t<i, arg_tuple>> &&>(std::get<i>(args))...);
              };
              if constexpr (std::is_same_v<Ret, void>) {
                  create(std::forward<Def>(f), std::index_sequence_for<Args...>{});
              } else {
                  auto ret = def<Ret>(create(std::forward<Def>(f), std::index_sequence_for<Args...>{}));
                  detail::FunctionBuilder::current()->return_(ret.expression());
              }
          })} {}

    /// Call the callable.
    auto operator()(detail::prototype_to_callable_invocation_t<Args>... args) const noexcept {
        detail::CallableInvoke invoke;
        static_cast<void>((invoke << ... << args));
        if constexpr (std::is_same_v<Ret, void>) {
            detail::FunctionBuilder::current()->call(
                _builder->function(), invoke.args());
        } else {
            return def<Ret>(
                detail::FunctionBuilder::current()->call(
                    Type::of<Ret>(), _builder->function(), invoke.args()));
        }
    }
};

namespace detail {

template<typename R, typename... Args>
using function_signature = R(Args...);

template<typename>
struct canonical_signature;

template<typename Ret, typename... Args>
struct canonical_signature<Ret(Args...)> {
    using type = function_signature<Ret, Args...>;
};

template<typename Ret, typename... Args>
struct canonical_signature<Ret (*)(Args...)>
    : canonical_signature<Ret(Args...)> {};

template<typename F>
struct canonical_signature
    : canonical_signature<decltype(&F::operator())> {};

#define LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(...)               \
    template<typename Ret, typename Cls, typename... Args>        \
    struct canonical_signature<Ret (Cls::*)(Args...) __VA_ARGS__> \
        : canonical_signature<Ret(Args...)> {};
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE()
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(const)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(volatile)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(const volatile)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(noexcept)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(const noexcept)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(volatile noexcept)
LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE(const volatile noexcept)
#undef LUISA_MAKE_FUNCTOR_CANONICAL_SIGNATURE

template<typename T>
using canonical_signature_t = typename canonical_signature<T>::type;

template<typename T>
struct dsl_function {
    using type = typename dsl_function<
        canonical_signature_t<
            std::remove_cvref_t<T>>>::type;
};

template<typename... Args>
struct dsl_function<function_signature<void, Args...>> {
    using type = function_signature<
        void,
        definition_to_prototype_t<Args>...>;
};

template<typename Ret, typename... Args>
struct dsl_function<function_signature<Ret, Args...>> {
    using type = function_signature<
        expr_value_t<Ret>,
        definition_to_prototype_t<Args>...>;
};

template<typename... Ret, typename... Args>
struct dsl_function<function_signature<std::tuple<Ret...>, Args...>> {
    using type = function_signature<
        std::tuple<expr_value_t<Ret>...>,
        definition_to_prototype_t<Args>...>;
};

template<typename RA, typename RB, typename... Args>
struct dsl_function<function_signature<std::pair<RA, RB>, Args...>> {
    using type = function_signature<
        std::tuple<expr_value_t<RA>, expr_value_t<RB>>,
        definition_to_prototype_t<Args>...>;
};

template<typename T>
struct dsl_function<Kernel1D<T>> {
    using type = T;
};

template<typename T>
struct dsl_function<Kernel2D<T>> {
    using type = T;
};

template<typename T>
struct dsl_function<Kernel3D<T>> {
    using type = T;
};

template<typename T>
struct dsl_function<Callable<T>> {
    using type = T;
};

template<typename T>
using dsl_function_t = typename dsl_function<T>::type;

}// namespace detail

template<typename T>
Kernel1D(T &&) -> Kernel1D<detail::dsl_function_t<std::remove_cvref_t<T>>>;

template<typename T>
Kernel2D(T &&) -> Kernel2D<detail::dsl_function_t<std::remove_cvref_t<T>>>;

template<typename T>
Kernel3D(T &&) -> Kernel3D<detail::dsl_function_t<std::remove_cvref_t<T>>>;

template<typename T>
Callable(T &&) -> Callable<detail::dsl_function_t<std::remove_cvref_t<T>>>;

}// namespace luisa::compute
