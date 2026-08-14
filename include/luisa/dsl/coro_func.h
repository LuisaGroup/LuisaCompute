//
// Created by Mike on 2024/5/8.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/concepts.h>
#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/func.h>

namespace luisa::compute {

template<typename T>
class Coroutine {
    static_assert(always_false_v<T>, "Coroutine requires a function signature type (e.g. Coroutine<void(int, float)>).");
};

namespace detail {

/// Coroutine awaiter — returned by Coroutine::operator().
/// Holds a move-only callback that runs all subroutines chained on a frame.
class CoroAwaiter : public concepts::Noncopyable {

public:
    using Await = luisa::move_only_function<void()>;
    explicit CoroAwaiter(Await await) noexcept
        : _await{std::move(await)} {}
    void await() && noexcept { _await(); }

private:
    Await _await;
};

/// Run all subroutines of a coroutine in token-chained order on a frame.
/// @param frame      The CoroFrame to run on.
/// @param node_count Total number of subroutines in the graph.
/// @param node       Callback invoked for each target token; receives
///                   (token, frame) and should call the matching subroutine.
LUISA_CORO_API void coroutine_chained_await_impl(
    CoroFrame &frame, size_t node_count,
    luisa::move_only_function<void(size_t, CoroFrame &)> node) noexcept;

/// Result of the two-phase coroutine compilation pipeline.
struct CoroutineCompileResult {
    coro::CoroGraph graph;
    CoroFrameDesc frame_desc;
    luisa::vector<luisa::shared_ptr<const FunctionBuilder>> subroutines;
    luisa::vector<uint32_t> trigger_tokens;
    // For every continuation, maps each argument after the frame argument to
    // its position in the source coroutine signature. The map is strictly
    // increasing and may be a proper subset after continuation-local dead
    // argument projection.
    luisa::vector<luisa::vector<size_t>>
        subroutine_source_argument_indices;
    // The enclosing coroutine transaction owns exactly its input and output
    // full-XIR verifier boundaries. Composed passes expose their counts so a
    // regression cannot silently reintroduce nested full verification.
    size_t boundary_verifier_count{0u};
    size_t nested_pass_boundary_verifier_count{0u};
};

/// Run the full coroutine compilation pipeline on an AST function.
/// Phase 1 (DSL recording) is already done; this performs Phase 2:
///   AST → XIR → coro-cfg-distill → coro-split → coro-materialize →
///   coro-reg2mem → extract CoroGraph + CoroFrameDesc + continuations.
/// @throw std::runtime_error on any pipeline failure.
[[nodiscard]] LUISA_CORO_API CoroutineCompileResult
compile_coroutine_pipeline(
    const luisa::shared_ptr<const FunctionBuilder> &builder);

}// namespace detail

#if defined(__GNUC__) && __GNUC__ >= 16
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtemplate-body"
#endif
template<typename Ret, typename... Args>
class Coroutine<Ret(Args...)> {

    static_assert(std::is_same_v<Ret, void>,
                  "Coroutine function must return void.");

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _builder{nullptr};
    coro::CoroGraph _graph;
    CoroFrameDesc _frame_desc;
    luisa::vector<luisa::shared_ptr<const detail::FunctionBuilder>> _subroutines;
    struct WrappedSubroutine {
        luisa::shared_ptr<const detail::FunctionBuilder> builder;
        luisa::vector<size_t> call_argument_indices;
    };
    luisa::vector<WrappedSubroutine> _wrapped_subroutines;
    luisa::vector<uint32_t> _trigger_tokens;
    Function _coro_func;

public:
    class Subroutine {

    private:
        luisa::shared_ptr<const detail::FunctionBuilder> _builder;
        Function _f;
        luisa::vector<size_t> _call_argument_indices;

    private:
        friend class Coroutine;
        explicit Subroutine(Function function) noexcept : _f{function} {}
        explicit Subroutine(
            luisa::shared_ptr<const detail::FunctionBuilder> builder,
            luisa::vector<size_t> call_argument_indices = {}) noexcept
            : _builder{std::move(builder)},
              _f{_builder ? _builder->function() : Function{}},
              _call_argument_indices{std::move(call_argument_indices)} {}

    public:
        explicit operator bool() const noexcept { return static_cast<bool>(_f); }

        void operator()(CoroFrame &frame, const Var<std::remove_cvref_t<Args>> &...args) const noexcept {
            luisa::vector<const Expression *> source_call_args;
            source_call_args.reserve(sizeof...(Args));
            (source_call_args.emplace_back(
                 detail::extract_expression(args)),
             ...);
            luisa::vector<const Expression *> call_args;
            call_args.reserve(1u + _call_argument_indices.size());
            call_args.emplace_back(frame.expression());
            for (auto index : _call_argument_indices) {
                LUISA_ASSERT(
                    index < source_call_args.size(),
                    "Invalid projected coroutine call argument index {} "
                    "(source argument count {}).",
                    index, source_call_args.size());
                call_args.emplace_back(source_call_args[index]);
            }
            detail::FunctionBuilder::current()->call(
                _f, luisa::span<const Expression *const>{call_args});
        }
    };

private:
    /// Create a wrapper callable that fills in captured resource bindings
    /// so the scheduler only needs to pass user-facing args.
    [[nodiscard]] static auto _make_subroutine_wrapper(
        Function coroutine, Function cc,
        luisa::span<const size_t> source_argument_indices) noexcept {
        using FB = luisa::compute::detail::FunctionBuilder;
        LUISA_ASSERT(
            cc.arguments().size() ==
                1u + source_argument_indices.size(),
            "Invalid projected coroutine continuation signature "
            "(expected {}, got {}).",
            1u + source_argument_indices.size(),
            cc.arguments().size());
        LUISA_ASSERT(coroutine.arguments().size() ==
                         coroutine.bound_arguments().size(),
                     "Invalid capture list size (expected {}, got {}).",
                     coroutine.arguments().size(),
                     coroutine.bound_arguments().size());
        luisa::vector<size_t> source_to_call_argument(
            coroutine.arguments().size(), static_cast<size_t>(-1));
        size_t call_argument_count = 0u;
        for (size_t source_index = 0u;
             source_index < coroutine.arguments().size(); ++source_index) {
            if (luisa::holds_alternative<luisa::monostate>(
                    coroutine.bound_arguments()[source_index])) {
                source_to_call_argument[source_index] =
                    call_argument_count++;
            }
        }
        LUISA_ASSERT(
            call_argument_count == sizeof...(Args),
            "Coroutine source ABI has {} unbound argument(s), but its C++ "
            "signature has {}.",
            call_argument_count, sizeof...(Args));
        luisa::vector<size_t> projected_call_argument_indices;
        auto builder = FB::define_callable([&] {
            luisa::vector<const Expression *> args;
            args.reserve(1u + source_argument_indices.size());
            auto fb = FB::current();
            args.emplace_back(fb->reference(cc.arguments().front().type()));
            size_t previous_source_index = 0u;
            bool first_source_index = true;
            for (size_t projected_index = 0u;
                 projected_index < source_argument_indices.size();
                 ++projected_index) {
                auto source_index =
                    source_argument_indices[projected_index];
                LUISA_ASSERT(
                    source_index < coroutine.arguments().size() &&
                        (first_source_index ||
                         source_index > previous_source_index),
                    "Coroutine source argument projection must be a "
                    "strictly increasing in-range sequence.");
                first_source_index = false;
                previous_source_index = source_index;
                auto cc_arg = cc.arguments()[projected_index + 1u];
                auto b = coroutine.bound_arguments()[source_index];
                auto internal_arg = luisa::visit(
                    [&]<typename T>(T b) noexcept -> const Expression * {
                        if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                            return fb->buffer_binding(cc_arg.type(), b.handle, b.offset, b.size);
                        } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                            return fb->texture_binding(cc_arg.type(), b.handle, b.level);
                        } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                            return fb->bindless_array_binding(b.handle);
                        } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                            return fb->accel_binding(b.handle);
                        } else {
                            static_assert(std::is_same_v<T, luisa::monostate>);
                            switch (cc_arg.tag()) {
                                case Variable::Tag::REFERENCE: return fb->reference(cc_arg.type());
                                case Variable::Tag::BUFFER: return fb->buffer(cc_arg.type());
                                case Variable::Tag::TEXTURE: return fb->texture(cc_arg.type());
                                case Variable::Tag::BINDLESS_ARRAY: return fb->bindless_array();
                                case Variable::Tag::ACCEL: return fb->accel();
                                default: return fb->argument(cc_arg.type());
                            }
                        }
                    },
                    b);
                args.emplace_back(internal_arg);
                if (luisa::holds_alternative<luisa::monostate>(b)) {
                    auto call_index =
                        source_to_call_argument[source_index];
                    LUISA_ASSERT(
                        call_index != static_cast<size_t>(-1),
                        "Projected unbound coroutine argument has no "
                        "source call position.");
                    projected_call_argument_indices.emplace_back(
                        call_index);
                }
                auto usage = cc.variable_usage(cc_arg.uid());
                if (usage != Usage::NONE) {
                    internal_arg->mark(usage);
                }
            }
            fb->call(cc, args);
        });
        return WrappedSubroutine{
            .builder = std::move(builder),
            .call_argument_indices =
                std::move(projected_call_argument_indices)};
    }

    [[nodiscard]] Subroutine _get_wrapped_subroutine(size_t index) const noexcept {
        if (index >= _wrapped_subroutines.size()) { return Subroutine{Function{}}; }
        auto &&wrapped = _wrapped_subroutines[index];
        return Subroutine{wrapped.builder,
                          wrapped.call_argument_indices};
    }

public:
    /// Construct a Coroutine from a definition lambda and compile eagerly.
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

        auto result = detail::compile_coroutine_pipeline(_builder);
        _graph = std::move(result.graph);
        _frame_desc = std::move(result.frame_desc);
        _subroutines = std::move(result.subroutines);
        _trigger_tokens = std::move(result.trigger_tokens);
        LUISA_ASSERT(
            result.subroutine_source_argument_indices.size() ==
                _subroutines.size(),
            "Coroutine lowering returned {} argument projection(s) for {} "
            "subroutine(s).",
            result.subroutine_source_argument_indices.size(),
            _subroutines.size());
        _coro_func = _builder->function();
        _wrapped_subroutines.reserve(_subroutines.size());
        for (size_t index = 0u; index < _subroutines.size(); ++index) {
            auto &&subroutine = _subroutines[index];
            _wrapped_subroutines.emplace_back(
                _make_subroutine_wrapper(
                    _coro_func, subroutine->function(),
                    result.subroutine_source_argument_indices[index]));
        }
    }

    [[nodiscard]] auto function() const noexcept { return Function{_builder.get()}; }
    [[nodiscard]] auto const &function_builder() const & noexcept { return _builder; }
    [[nodiscard]] auto &&function_builder() && noexcept { return std::move(_builder); }

    [[nodiscard]] const coro::CoroGraph &graph() const noexcept { return _graph; }
    [[nodiscard]] auto shared_graph() const noexcept { return luisa::shared_ptr<const coro::CoroGraph>{}; }

    [[nodiscard]] const CoroFrameDesc &frame() const noexcept { return _frame_desc; }
    [[nodiscard]] const CoroFrameDesc &frame_desc() const noexcept { return _frame_desc; }
    [[nodiscard]] auto shared_frame() const noexcept { return luisa::shared_ptr<const CoroFrameDesc>{}; }

    [[nodiscard]] Subroutine entry() const noexcept {
        return _get_wrapped_subroutine(0u);
    }

    [[nodiscard]] Subroutine operator[](size_t index) const noexcept {
        return _get_wrapped_subroutine(index);
    }

    [[nodiscard]] Subroutine subroutine(size_t token) const noexcept {
        auto *node = _graph.node_by_token(token);
        return node ? _get_wrapped_subroutine(node->index) : Subroutine{Function{}};
    }

    [[nodiscard]] Subroutine subroutine(luisa::string_view name) const noexcept {
        auto *node = _graph.node_by_name(name);
        return node ? _get_wrapped_subroutine(node->index) : Subroutine{Function{}};
    }

    [[nodiscard]] size_t subroutine_count() const noexcept { return _subroutines.size(); }

    [[nodiscard]] uint32_t trigger_token(size_t index) const noexcept {
        return index < _trigger_tokens.size() ? _trigger_tokens[index] : 0u;
    }

    [[nodiscard]] CoroFrame instantiate() const noexcept {
        return CoroFrame::create(&_frame_desc);
    }

    [[nodiscard]] CoroFrame instantiate(Expr<uint3> coro_id, Expr<uint3> dispatch_size) const noexcept {
        auto frame = CoroFrame::create(&_frame_desc);
        frame.coro_id = coro_id;
        frame.dispatch_size_x = dispatch_size.x;
        frame.dispatch_size_y = dispatch_size.y;
        frame.dispatch_size_z = dispatch_size.z;
        return frame;
    }

    [[nodiscard]] CoroFrame instantiate(Expr<uint3> coro_id) const noexcept {
        return instantiate(coro_id, dispatch_size());
    }

private:
    [[nodiscard]] auto _await(luisa::optional<Expr<uint3>> coro_id,
                              const Var<std::remove_cvref_t<Args>> &...args) const noexcept {
        return detail::CoroAwaiter{[=, coro_id = std::move(coro_id), this]() noexcept {
            auto frame = coro_id ? instantiate(*coro_id) : instantiate();
            detail::coroutine_chained_await_impl(
                frame, subroutine_count(),
                [&](size_t token, CoroFrame &f) noexcept {
                    subroutine(token)(f, args...);
                });
        }};
    }

public:
    [[nodiscard]] auto operator()(const Var<std::remove_cvref_t<Args>> &...args) const noexcept {
        return _await(luisa::nullopt, args...);
    }

    [[nodiscard]] auto operator()(Expr<uint3> coro_id,
                                  const Var<std::remove_cvref_t<Args>> &...args) const noexcept {
        return _await(luisa::make_optional(coro_id), args...);
    }
};

#if defined(__GNUC__) && __GNUC__ >= 16
#pragma GCC diagnostic pop
#endif

/// CTAD deduction guide — deduces Coroutine<void(Args...)> from a lambda.
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

}// namespace luisa::compute
