#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/func.h>

namespace luisa::compute {

template<typename T>
class Coroutine {
    static_assert(always_false_v<T>, "Coroutine requires a function signature type");
};

template<typename T>
struct is_coroutine : std::false_type {};

template<typename T>
struct is_coroutine<Coroutine<T>> : std::true_type {};

namespace detail {

/// Result of the two-phase coroutine compilation pipeline.
struct CoroutineCompileResult {
    coro::CoroGraph graph;
    CoroFrameDesc frame_desc;
    luisa::vector<luisa::shared_ptr<const FunctionBuilder>> subroutines;
};

/// Run the full coroutine compilation pipeline on an AST function.
/// Phase 1 (DSL recording) is already done; this performs Phase 2:
///   AST → XIR → coro-cfg-distill → coro-split → coro-materialize →
///   coro-reg2mem → extract CoroGraph + CoroFrameDesc + continuations.
/// @throw std::runtime_error on any pipeline failure.
[[nodiscard]] LUISA_CORO_API CoroutineCompileResult
compile_coroutine_pipeline(
    const luisa::shared_ptr<const FunctionBuilder> &builder);

} // namespace detail

template<typename Ret, typename... Args>
class Coroutine<Ret(Args...)> {
    // Phase 1: Ret must be void
    static_assert(std::is_same_v<Ret, void>, "Coroutine return type must be void in Phase 1");

    using SharedFunctionBuilder = luisa::shared_ptr<const detail::FunctionBuilder>;
    SharedFunctionBuilder _builder{nullptr};

    // Phase 2: compiled pipeline results
    coro::CoroGraph _graph;
    CoroFrameDesc _frame_desc;
    luisa::vector<SharedFunctionBuilder> _subroutines;

public:
    /// Construct a Coroutine from a definition lambda and compile eagerly.
    /// Follows the same argument-creation pattern as Callable.
    /// Phase 1: record DSL operations into AST.
    /// Phase 2: run full compilation pipeline (AST→XIR→passes).
    /// @throw std::runtime_error if compilation fails.
    template<typename Def>
        requires std::negation_v<is_callable<std::remove_cvref_t<Def>>> &&
                 std::negation_v<is_kernel<std::remove_cvref_t<Def>>>
    Coroutine(Def &&def) {
        _builder = detail::FunctionBuilder::define_coroutine([&def] {
            static_assert(std::is_invocable_v<Def, detail::prototype_to_creation_t<Args>...>);
            auto create = []<size_t... i>(auto &&d, std::index_sequence<i...>) noexcept {
                using var_tuple = std::tuple<Var<std::remove_cvref_t<Args>>...>;
                using tag_tuple = std::tuple<detail::prototype_to_creation_tag_t<Args>...>;
                auto args = detail::create_argument_definitions<var_tuple, tag_tuple>(std::tuple<>{});
                static_assert(std::tuple_size_v<decltype(args)> == sizeof...(Args));
                luisa::invoke(
                    std::forward<decltype(d)>(d),
                    static_cast<detail::prototype_to_creation_t<
                        std::tuple_element_t<i, std::tuple<Args...>>> &&>(std::get<i>(args))...);
            };
            create(std::forward<Def>(def), std::index_sequence_for<Args...>{});
        });

        // Phase 2: eager compilation
        auto result = detail::compile_coroutine_pipeline(_builder);
        _graph = std::move(result.graph);
        _frame_desc = std::move(result.frame_desc);
        _subroutines = std::move(result.subroutines);
    }

    /// Access the underlying AST function (pre-compilation entry)
    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }

    /// @return the compiled coroutine state-transition graph
    [[nodiscard]] const coro::CoroGraph &graph() const noexcept { return _graph; }

    /// @return the compiled frame layout descriptor
    [[nodiscard]] const CoroFrameDesc &frame_desc() const noexcept { return _frame_desc; }

    /// @return the entry callable (scope 0).
    ///         If the pipeline produced translated subroutines, returns the
    ///         translated scope-0 callable; otherwise falls back to the
    ///         original AST builder.
    [[nodiscard]] SharedFunctionBuilder entry() const noexcept {
        return _subroutines.empty() ? _builder : _subroutines[0u];
    }

    /// @return a subroutine by scope index (0 = entry, 1+ = continuations).
    [[nodiscard]] SharedFunctionBuilder operator[](size_t index) const noexcept {
        return index < _subroutines.size() ? _subroutines[index] : nullptr;
    }

    /// @return the number of compiled subroutines (entry + continuations).
    [[nodiscard]] size_t subroutine_count() const noexcept { return _subroutines.size(); }
};

/// CTAD deduction guide — deduces Coroutine<void(Args...)> from a lambda
template<typename Def>
Coroutine(Def) -> Coroutine<detail::dsl_function_t<std::remove_cvref_t<Def>>>;

/// Minimal Generator<R(Args...)> wrapper.
/// Stores a Coroutine and will provide iteration API in later phases.
template<typename R, typename... Args>
class Generator {
    Coroutine<void(Args...)> _coroutine;

public:
    template<typename Def>
    explicit Generator(Def &&def) : _coroutine{std::forward<Def>(def)} {}

    [[nodiscard]] auto function() const noexcept { return _coroutine.function(); }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _coroutine.function_builder(); }
};

} // namespace luisa::compute
