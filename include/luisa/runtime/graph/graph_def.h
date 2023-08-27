#pragma once
#include <luisa/dsl/arg.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/resource.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_builder.h>
#include <iostream>

namespace luisa::compute::graph {
namespace detail {
/// Append an element in a tuple
template<typename... T, typename A>
[[nodiscard]] inline auto tuple_append(std::tuple<T...> tuple, A &&arg) noexcept {
    auto append = []<typename TT, typename AA, size_t... i>(TT tuple, AA &&arg, std::index_sequence<i...>) noexcept {
        return std::make_tuple(std::move(std::get<i>(tuple))..., std::forward<AA>(arg));
    };
    return append(std::move(tuple), std::forward<A>(arg), std::index_sequence_for<T...>{});
}

template<typename... Args>
auto print_types_with_index() {
    auto print = []<size_t... I>(std::index_sequence<I...>) {
        ((std::cout << "[" << I << "] " << typeid(Args).name() << ";\n"),
         ...);
    };
    std::cout << "{\n";
    print(std::index_sequence_for<Args...>{});
    std::cout << "}\n";
}

template<typename... Args>
auto print_args_with_index(Args &&...args) {
    auto print = []<size_t... I>(std::index_sequence<I...>, Args &&...args) {
        ((std::cout << "[" << I << "] " << typeid(args).name() << ";\n"),
         ...);
    };
    std::cout << "{\n";
    print(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
    std::cout << "}\n";
}

template<typename F, typename... Args>
auto for_each_type_with_index(F &&func) {
    []<typename Fn, size_t... I>(Fn &&f, std::index_sequence<I...>) {
        (f(I), ...);
    }(std::forward<F>(func), std::index_sequence_for<Args...>{});
}

template<typename F, typename... Args>
auto for_each_arg_with_index(F &&func, Args &&...args) {
    []<typename Fn, size_t... I>(Fn &&f, std::index_sequence<I...>, Args &&...args) {
        (f(I, std::forward<Args>(args)), ...);
    }(std::forward<F>(func), std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
}

}// namespace detail

//Args: prototype, e.g. BufferView<int>
template<typename... Args>
class GraphDefBase {
    template<typename...>
    friend class GraphDef;

    template<typename T>
    using U = unique_ptr<T>;
public:
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    GraphDefBase(Def &&def) noexcept {
        static_assert(std::conjunction_v<std::is_base_of<GraphVarBase, Args>...>);
        _builder = GraphBuilder::build([&] {
            // debug print:
            detail::print_types_with_index<Args...>();
            GraphBuilder::set_var_count(sizeof...(Args));
            []<typename Fn, size_t... I>(Fn &&fn, std::index_sequence<I...>) {
                fn(GraphBuilder::define_graph_var<Args, I>()...);
            }(std::forward<Def>(def), std::index_sequence_for<Args...>{});
        });
    }
    // TODO: to make private, now public just for test
    U<GraphBuilder> _builder = nullptr;
};

template<typename... Args>
class GraphDef : public GraphDefBase<Args...> {
public:
    using GraphDefBase<Args...>::GraphDefBase;
    GraphDef(GraphDefBase<Args...> g) noexcept : GraphDefBase<Args...>{} {}
    GraphDef &operator=(GraphDefBase<Args...> g) noexcept { return *this; }
};

template<typename... Args>
class GraphDef<void(Args...)> : public GraphDefBase<Args...> {
public:
    using GraphDefBase<Args...>::GraphDefBase;
    GraphDef(GraphDefBase<Args...> g) noexcept : GraphDefBase<Args...>{} {}
    GraphDef &operator=(GraphDefBase<Args...> g) noexcept { return *this; }
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
        canonical_signature_t<std::remove_cvref_t<T>>>::type;
};

template<typename... Args>
struct dsl_function<function_signature<void, Args...>> {
    using type = function_signature<
        void,
        std::remove_cvref_t<Args>...>;
};

//template<typename Ret, typename... Args>
//struct dsl_function<function_signature<Ret, Args...>> {
//    using type = function_signature<
//        expr_value_t<Ret>,
//        definition_to_prototype_t<Args>...>;
//};

//template<typename... Ret, typename... Args>
//struct dsl_function<function_signature<std::tuple<Ret...>, Args...>> {
//    using type = function_signature<
//        std::tuple<expr_value_t<Ret>...>,
//        definition_to_prototype_t<Args>...>;
//};

//template<typename RA, typename RB, typename... Args>
//struct dsl_function<function_signature<std::pair<RA, RB>, Args...>> {
//    using type = function_signature<
//        std::tuple<expr_value_t<RA>, expr_value_t<RB>>,
//        definition_to_prototype_t<Args>...>;
//};

template<typename T>
struct dsl_function<GraphDef<T>> {
    using type = T;
};

template<typename T>
using dsl_function_t = typename dsl_function<T>::type;

}// namespace detail

template<typename T>
GraphDef(T &&) -> GraphDef<detail::dsl_function_t<std::remove_cvref_t<T>>>;
}// namespace luisa::compute::graph