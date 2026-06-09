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
    Function _coro_func;
    mutable luisa::vector<luisa::shared_ptr<const detail::FunctionBuilder>> _wrapped_subroutines;

public:
    class Subroutine {

    private:
        Function _f;

    private:
        friend class Coroutine;
        explicit Subroutine(Function function) noexcept : _f{function} {}

    public:
        explicit operator bool() const noexcept { return static_cast<bool>(_f); }

        void operator()(CoroFrame &frame, const Var<std::remove_cvref_t<Args>> &...args) const noexcept {
            const Expression *call_args[1u + sizeof...(Args)];
            call_args[0] = frame.expression();
            size_t ai = 1u;
            ((call_args[ai++] = detail::extract_expression(args)), ...);
            detail::FunctionBuilder::current()->call(
                _f, luisa::span<const Expression *const>{call_args, 1u + sizeof...(Args)});
        }
    };

private:
    /// Create a wrapper callable that fills in captured resource bindings
    /// so the scheduler only needs to pass user-facing args.
    [[nodiscard]] static auto _make_subroutine_wrapper(Function coroutine, Function cc) noexcept {
        using FB = luisa::compute::detail::FunctionBuilder;
        return FB::define_callable([&] {
            luisa::vector<const Expression *> args;
            args.reserve(1u + coroutine.arguments().size());
            LUISA_ASSERT(coroutine.arguments().size() == coroutine.bound_arguments().size(),
                         "Invalid capture list size (expected {}, got {}).",
                         coroutine.arguments().size(), coroutine.bound_arguments().size());
            auto fb = FB::current();
            args.emplace_back(fb->reference(cc.arguments().front().type()));
            for (auto arg_i = 0u; arg_i < coroutine.arguments().size(); arg_i++) {
                auto cc_arg = cc.arguments()[arg_i + 1u];
                auto b = coroutine.bound_arguments()[arg_i];
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
            }
            fb->call(cc, args);
        });
    }

    [[nodiscard]] Subroutine _get_wrapped_subroutine(size_t index) const noexcept {
        if (index >= _subroutines.size()) { return Subroutine{{}}; }
        if (!_wrapped_subroutines[index]) {
            bool has_captures = false;
            for (auto &b : _coro_func.bound_arguments()) {
                if (!luisa::holds_alternative<luisa::monostate>(b)) { has_captures = true; break; }
            }
            _wrapped_subroutines[index] = has_captures
                ? _make_subroutine_wrapper(_coro_func, _subroutines[index]->function())
                : _subroutines[index];
        }
        return Subroutine{_wrapped_subroutines[index]->function()};
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
        _coro_func = _builder->function();
        _wrapped_subroutines.resize(_subroutines.size());
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
        return node ? _get_wrapped_subroutine(node->index) : Subroutine{{}};
    }

    [[nodiscard]] Subroutine subroutine(luisa::string_view name) const noexcept {
        auto *node = _graph.node_by_name(name);
        return node ? _get_wrapped_subroutine(node->index) : Subroutine{{}};
    }

    [[nodiscard]] size_t subroutine_count() const noexcept { return _subroutines.size(); }

    [[nodiscard]] CoroFrame instantiate() const noexcept {
        return CoroFrame::create(&_frame_desc);
    }

    [[nodiscard]] CoroFrame instantiate(Expr<uint3> coro_id) const noexcept {
        auto frame = CoroFrame::create(&_frame_desc);
        frame.coro_id = coro_id;
        return frame;
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
