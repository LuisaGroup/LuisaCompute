#pragma once

#include <luisa/dsl/func.h>
#include <luisa/dsl/work_graph/work_graph_types.h>

namespace luisa::compute {

class WorkGraphEmptyRecord {};

template<typename InputRecord, typename T>
class WorkGraphNodeKernel {
    static_assert(always_false_v<T>);
};

template<typename InputRecord, typename Ret, typename... Args>
class WorkGraphNodeKernel<InputRecord, Ret(Args...)> {
    static_assert(std::is_void_v<Ret>, "work graph nodes must have void return type");
    static_assert(std::negation_v<std::disjunction<std::is_pointer<Args>...>>);

    // verify that Args[0] == Var<InputRecord>
    static constexpr bool InputRecordEmpty = std::is_same_v<InputRecord, WorkGraphEmptyRecord>;

    static_assert(
        InputRecordEmpty || sizeof...(Args) >= 1,
        "must be at least one argument for input record (unless input record type is empty)"
    );

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _builder;

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
    WorkGraphNodeKernel(Def &&f) noexcept {
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

            // return type is always void
            create(std::forward<Def>(f), std::index_sequence_for<Args...>{});
            detail::FunctionBuilder::current()->return_(nullptr);// to check if any previous $return called with non-void types
        });
    }
    /// Get the underlying AST
    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }
};

namespace detail {
template<typename InputRecord, typename T>
struct dsl_function<WorkGraphNodeKernel<InputRecord, T>> {
    using type = T;
};
}// namespace detail

template<typename T>
struct FirstArgumentOrEmptyRecord {
    static_assert(always_false_v<T>);
    using type = void;
};

template<typename Ret, typename... Args>
requires (sizeof...(Args) == 0)
struct FirstArgumentOrEmptyRecord<Ret(Args...)> {
    using type = WorkGraphEmptyRecord;
};


template<typename Ret, typename... Args>
requires (sizeof...(Args) >= 1)
struct FirstArgumentOrEmptyRecord<Ret(Args...)> {
    using type = std::tuple_element_t<0, std::tuple<Args...>>;
};

template<typename T>
WorkGraphNodeKernel(T &&) -> WorkGraphNodeKernel<
    typename FirstArgumentOrEmptyRecord<detail::dsl_function_t<std::remove_cvref_t<T>>>::type,
    detail::dsl_function_t<std::remove_cvref_t<T>>
>;

}// namespace luisa::compute
