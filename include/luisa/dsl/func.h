#pragma once

#include <type_traits>
#include <luisa/core/stl/memory.h>
#include <luisa/ast/external_function.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/arg.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/resource.h>

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

class FunctionBuilder;

[[nodiscard]] LC_DSL_API luisa::shared_ptr<const FunctionBuilder>
transform_function(Function callable) noexcept;

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
    explicit Kernel(SharedFunctionBuilder builder) noexcept
        : _builder{detail::transform_function(builder->function())} {}

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
        auto ast = detail::FunctionBuilder::define_kernel([&def] {
            detail::FunctionBuilder::current()->set_block_size(detail::kernel_default_block_size<N>());
            []<size_t... i>(auto &&def, std::index_sequence<i...>) noexcept {
                using arg_tuple = std::tuple<Args...>;
                using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                luisa::invoke(std::forward<decltype(def)>(def),
                              static_cast<detail::prototype_to_creation_t<
                                  std::tuple_element_t<i, arg_tuple>> &&>(std::get<i>(args))...);
            }(std::forward<Def>(def), std::index_sequence_for<Args...>{});
        });
        _builder = detail::transform_function(ast->function());
    }
    [[nodiscard]] const auto &function() const noexcept { return _builder; }
};

#define LUISA_KERNEL_BASE(N)                                     \
public                                                           \
    Kernel<N, Args...> {                                         \
        using Kernel<N, Args...>::Kernel;                        \
        Kernel##N##D(Kernel<N, Args...> k) noexcept              \
            : Kernel<N, Args...> { std::move(k._builder) }       \
        {}                                                       \
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

private:
    static LC_DSL_API void _error_too_many_arguments() noexcept;

public:
    CallableInvoke() noexcept = default;
    /// Add an argument.
    template<typename T>
    CallableInvoke &operator<<(Expr<T> arg) noexcept {
        if constexpr (requires { typename Expr<T>::is_binding_group; }) {
            callable_encode_binding_group(*this, arg);
        } else if constexpr (is_soa_expr_v<T>) {
            callable_encode_soa(*this, arg);
        } else {
            if (_arg_count == max_argument_count) [[unlikely]] {
                _error_too_many_arguments();
            }
            _args[_arg_count++] = arg.expression();
        }
        return *this;
    }
    /// Add an argument.
    template<typename T>
    decltype(auto) operator<<(Ref<T> arg) noexcept {
        return (*this << Expr{arg});
    }
    [[nodiscard]] auto args() const noexcept {
        return luisa::span{_args.data(), _arg_count};
    }
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
    friend class CallableLibrary;
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
    explicit Callable(luisa::shared_ptr<const detail::FunctionBuilder> builder) noexcept : _builder{std::move(builder)} {}
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
    Callable(Def &&f) noexcept {
        auto ast = detail::FunctionBuilder::define_callable([&f] {
            static_assert(std::is_invocable_v<Def, detail::prototype_to_creation_t<Args>...>);
            auto create = []<size_t... i>(auto &&def, std::index_sequence<i...>) noexcept {
                using arg_tuple = std::tuple<Args...>;
                using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                return luisa::invoke(std::forward<decltype(def)>(def),
                                     static_cast<detail::prototype_to_creation_t<
                                         std::tuple_element_t<i, arg_tuple>> &&>(std::get<i>(args))...);
            };
            if constexpr (std::is_same_v<Ret, void>) {
                create(std::forward<Def>(f), std::index_sequence_for<Args...>{});
                detail::FunctionBuilder::current()->return_(nullptr);// to check if any previous $return called with non-void types
            } else {
                auto ret = def<Ret>(create(std::forward<Def>(f), std::index_sequence_for<Args...>{}));
                detail::FunctionBuilder::current()->return_(ret.expression());
            }
        });
        _builder = detail::transform_function(ast->function());
    }

    /// Get the underlying AST
    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }

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

// TODO: External callable
template<typename T>
class ExternalCallable {
    static_assert(always_false_v<T>);
};

template<typename Ret, typename... Args>
class ExternalCallable<Ret(Args...)> {

    static_assert(
        std::negation_v<std::disjunction<
            is_buffer_or_view<Ret>,
            is_image_or_view<Ret>,
            is_volume_or_view<Ret>>>,
        "Callables may not return buffers, "
        "images or volumes (or their views).");
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

private:
    luisa::shared_ptr<ExternalFunction> _func;

private:
    // FIXME: support for resources
    template<typename T>
    struct UsageOf {
        static constexpr auto value = Usage::READ;
    };

    template<typename T>
    struct UsageOf<T &> {
        static constexpr auto value = Usage::READ_WRITE;
    };

    template<typename T>
    struct UsageOf<const T &> {
        static constexpr auto value = Usage::READ;
    };

public:
    ExternalCallable(luisa::string name) noexcept
        : _func{luisa::make_shared<ExternalFunction>(
              std::move(name), Type::of<Ret>(),
              luisa::vector<const Type *>{Type::of<Args>()...},
              luisa::vector<Usage>{UsageOf<Args>::value...})} {}

    auto operator()(detail::prototype_to_callable_invocation_t<Args>... args) const noexcept {
        detail::CallableInvoke invoke;
        static_cast<void>((invoke << ... << args));
        if constexpr (std::is_same_v<Ret, void>) {
            detail::FunctionBuilder::current()->call(_func, invoke.args());
        } else {
            return def<Ret>(
                detail::FunctionBuilder::current()->call(
                    Type::of<Ret>(), _func, invoke.args()));
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
