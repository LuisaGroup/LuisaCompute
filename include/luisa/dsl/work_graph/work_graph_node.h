#pragma once

#include <luisa/dsl/func.h>
#include <luisa/dsl/work_graph/work_graph_types.h>

namespace luisa::compute {

template<typename InputRecord, typename T>
class WorkGraphNode {
    static_assert(always_false_v<T>);
};

template<typename InputRecord, typename Ret, typename... Args>
class WorkGraphNode<InputRecord, Ret(Args...)> {
    static_assert(std::is_void_v<Ret>, "work graph nodes must have void return type");
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

    // verify that Args[0] == Var<InputRecord>
    static constexpr bool InputRecordEmpty = std::is_empty_v<InputRecord>;

    static_assert(
        InputRecordEmpty || sizeof...(Args) >= 1,
        "must be at least one argument for input record (unless input record type is empty)"
    );

    template<typename InputRecord, typename... Args>
    requires (std::is_empty_v<InputRecord> || std::is_same_v<Var<InputRecord>, std::tuple_element_t<0, std::tuple<Args...>>>)
    struct FirstArgumentIsInputRecord {
        static constexpr bool value = true;
    };

    static_assert(
        FirstArgumentIsInputRecord<InputRecord, Args...>::value,
        "first argument must have type Var<InputRecord> (unless input record type is empty)"
    );

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _builder;
    WorkGraphLaunchType _launch_type;

public:
    /**
     * @brief Construct a WorkGraphNode object.
     *
     * The function provided will be called and recorded during construction.
     *
     * @param f the function of callable.
     */
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    WorkGraphNode(WorkGraphLaunchType launch_type, Def &&f) noexcept : _launch_type(launch_type) {
        static_assert(std::is_invocable_r_v<void, Def, detail::prototype_to_creation_t<Args>...>);
        _builder = detail::FunctionBuilder::define_work_graph_node([&f] {
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
    }
    /// Get the underlying AST
    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }
};

namespace detail {
template<typename InputRecord, typename T>
struct dsl_function<WorkGraphNode<InputRecord, T>> {
    using type = T;
};
}// namespace detail

// I can't seem to get CTAD to work, so just have a helper
template<typename InputRecord, typename T>
auto make_work_graph_node(WorkGraphLaunchType launch_type, T&& t) {
    return WorkGraphNode<InputRecord, detail::dsl_function_t<std::remove_cvref_t<T>>>(launch_type, std::forward<T>(t));
}

}// namespace luisa::compute
