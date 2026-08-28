#pragma once

#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::dsl_detail {
[[nodiscard]] LUISA_DSL_API luisa::string format_source_location(const char *file, int line) noexcept;
}// namespace luisa::compute::dsl_detail

#ifndef LUISA_COMPUTE_DESUGAR

#include <luisa/dsl/syntax.h>

namespace luisa::compute {
inline namespace dsl {

inline void suspend_impl() {
    detail::FunctionBuilder::current()->suspend_();
}

inline void suspend_impl(uint32_t token) {
    detail::FunctionBuilder::current()->suspend_(token);
}

inline void suspend_impl(const char *name) {
    detail::FunctionBuilder::current()->suspend_(luisa::string{name});
}

inline void suspend_impl(uint32_t token, const char *name) {
    detail::FunctionBuilder::current()->suspend_(token, luisa::string{name});
}

template<typename T>
[[nodiscard]] inline CoroFrameExport coro_frame_export(
    luisa::string name, T &&value) noexcept {
    return CoroFrameExport{
        .value = detail::extract_expression(std::forward<T>(value)),
        .name = std::move(name)};
}

namespace coro_suspend_detail {

template<bool IsAnnotation>
using RecordedCoroSuspendExtensionBase =
    std::conditional_t<IsAnnotation, CoroSuspendAnnotation,
                       CoroSuspendExtension>;

// Keeps the complete source-side extension alive until FunctionBuilder owns
// the suspend statement. `freeze` is the only place where expression pointers
// become owner indices, so bindings remain part of the ordinary AST/XIR
// dataflow instead of acquiring a parallel storage model.
template<bool IsAnnotation>
class RecordedCoroSuspendExtension final
    : public RecordedCoroSuspendExtensionBase<IsAnnotation> {
private:
    luisa::string _schema;
    uint32_t _version;
    CoroSuspendFallback _fallback;
    luisa::vector<CoroSuspendBinding> _bindings;
    luisa::vector<const Expression *> _values;
    luisa::vector<CoroSuspendAttribute> _attributes;

public:
    RecordedCoroSuspendExtension(
        luisa::string schema, uint32_t version,
        CoroSuspendFallback fallback,
        luisa::vector<CoroSuspendBinding> bindings,
        luisa::vector<const Expression *> values,
        luisa::vector<CoroSuspendAttribute> attributes) noexcept
        : _schema{std::move(schema)}, _version{version},
          _fallback{fallback}, _bindings{std::move(bindings)},
          _values{std::move(values)},
          _attributes{std::move(attributes)} {
        LUISA_ASSERT(!_schema.empty(),
                     "Coroutine suspend extension schema must be non-empty.");
        LUISA_ASSERT(_version != 0u,
                     "Coroutine suspend extension '{}' has reserved version 0.",
                     _schema);
        LUISA_ASSERT(_bindings.size() == _values.size(),
                     "Coroutine suspend extension '{}' has {} bindings but "
                     "{} values.",
                     _schema, _bindings.size(), _values.size());
    }

    [[nodiscard]] luisa::string_view schema() const noexcept override {
        return _schema;
    }
    [[nodiscard]] uint32_t version() const noexcept override {
        return _version;
    }
    [[nodiscard]] CoroSuspendFallback fallback() const noexcept override {
        return _fallback;
    }
    [[nodiscard]] luisa::span<const CoroSuspendBinding>
    bindings() const noexcept override {
        return _bindings;
    }
    [[nodiscard]] luisa::span<const CoroSuspendAttribute>
    attributes() const noexcept override {
        return _attributes;
    }
    [[nodiscard]] CoroSuspendExtensionPtr clone() const noexcept override {
        return luisa::make_unique<RecordedCoroSuspendExtension>(
            _schema, _version, _fallback, _bindings, _values, _attributes);
    }
    [[nodiscard]] CoroSuspendExtensionPtr freeze(
        CoroSuspendExtensionRecorder &recorder) && noexcept override {
        for (size_t i = 0u; i < _bindings.size(); ++i) {
            _bindings[i].index = recorder.bind(_bindings[i], _values[i]);
        }
        if constexpr (IsAnnotation) {
            return make_coro_suspend_annotation_data(
                std::move(_schema), _version, _fallback,
                std::move(_bindings), std::move(_attributes));
        } else {
            return make_coro_suspend_extension_data(
                std::move(_schema), _version, _fallback,
                std::move(_bindings), std::move(_attributes));
        }
    }
};

}// namespace coro_suspend_detail

/// Fluent source-side declaration for a semantic coroutine stage or an
/// ignorable annotation. The resulting extension still owns its complete
/// schema, fallback policy, bindings, and attributes after AST/XIR lowering.
template<bool IsAnnotation>
class CoroSuspendExtensionBuilder {
private:
    luisa::string _schema;
    uint32_t _version;
    CoroSuspendFallback _fallback;
    luisa::vector<CoroSuspendBinding> _bindings;
    luisa::vector<const Expression *> _values;
    luisa::vector<CoroSuspendAttribute> _attributes;

private:
    template<typename T>
    void _bind(luisa::string name, T &&value,
               CoroSuspendBindingAccess access,
               CoroSuspendBindingLifetime lifetime) noexcept {
        auto *expression =
            detail::extract_expression(std::forward<T>(value));
        LUISA_ASSERT(expression != nullptr && expression->type() != nullptr,
                     "Coroutine suspend binding '{}' must be a typed value.",
                     name);
        LUISA_ASSERT(_values.size() <
                         static_cast<size_t>(
                             std::numeric_limits<uint32_t>::max()),
                     "Coroutine suspend extension '{}' exceeds the uint32 "
                     "binding ABI.",
                     _schema);
        _bindings.emplace_back(CoroSuspendBinding{
            .name = std::move(name),
            .access = access,
            .lifetime = lifetime,
            .index = static_cast<uint32_t>(_values.size())});
        _values.emplace_back(expression);
    }

public:
    CoroSuspendExtensionBuilder(
        luisa::string schema, uint32_t version,
        CoroSuspendFallback fallback) noexcept
        : _schema{std::move(schema)}, _version{version},
          _fallback{fallback} {}

    CoroSuspendExtensionBuilder(
        const CoroSuspendExtensionBuilder &) = delete;
    CoroSuspendExtensionBuilder(
        CoroSuspendExtensionBuilder &&) noexcept = default;
    CoroSuspendExtensionBuilder &operator=(
        const CoroSuspendExtensionBuilder &) = delete;
    CoroSuspendExtensionBuilder &operator=(
        CoroSuspendExtensionBuilder &&) noexcept = default;

    CoroSuspendExtensionBuilder &fallback(
        CoroSuspendFallback value) noexcept {
        _fallback = value;
        return *this;
    }

    template<typename T>
    CoroSuspendExtensionBuilder &read(
        luisa::string name, T &&value,
        CoroSuspendBindingLifetime lifetime =
            CoroSuspendBindingLifetime::queued) noexcept {
        _bind(std::move(name), std::forward<T>(value),
              CoroSuspendBindingAccess::read, lifetime);
        return *this;
    }

    template<typename T>
    CoroSuspendExtensionBuilder &write(
        luisa::string name, T &&value,
        CoroSuspendBindingLifetime lifetime =
            CoroSuspendBindingLifetime::resumed) noexcept {
        _bind(std::move(name), std::forward<T>(value),
              CoroSuspendBindingAccess::write, lifetime);
        return *this;
    }

    template<typename T>
    CoroSuspendExtensionBuilder &read_write(
        luisa::string name, T &&value,
        CoroSuspendBindingLifetime lifetime =
            CoroSuspendBindingLifetime::resumed) noexcept {
        _bind(std::move(name), std::forward<T>(value),
              CoroSuspendBindingAccess::read_write, lifetime);
        return *this;
    }

    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, CoroSuspendAttributeValue value) noexcept {
        _attributes.emplace_back(CoroSuspendAttribute{
            .name = std::move(name), .value = std::move(value)});
        return *this;
    }

    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, const char *value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              luisa::string{value}});
    }

    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, luisa::string value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              std::move(value)});
    }

    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, luisa::string_view value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              luisa::string{value}});
    }

    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, bool value) noexcept {
        return attribute(std::move(name),
                         CoroSuspendAttributeValue{value});
    }

    template<std::signed_integral T>
        requires(!std::same_as<T, bool>)
    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, T value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              static_cast<int64_t>(value)});
    }

    template<std::unsigned_integral T>
        requires(!std::same_as<T, bool>)
    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, T value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              static_cast<uint64_t>(value)});
    }

    template<std::floating_point T>
    CoroSuspendExtensionBuilder &attribute(
        luisa::string name, T value) noexcept {
        return attribute(std::move(name), CoroSuspendAttributeValue{
                                              static_cast<double>(value)});
    }

    [[nodiscard]] CoroSuspendExtensionPtr build() && noexcept {
        return luisa::make_unique<
            coro_suspend_detail::RecordedCoroSuspendExtension<IsAnnotation>>(
            std::move(_schema), _version, _fallback,
            std::move(_bindings), std::move(_values),
            std::move(_attributes));
    }
};

[[nodiscard]] inline auto coro_stage(
    luisa::string schema, uint32_t version = 1u) noexcept {
    return CoroSuspendExtensionBuilder<false>{
        std::move(schema), version, CoroSuspendFallback::reject};
}

[[nodiscard]] inline auto coro_annotation(
    luisa::string schema, uint32_t version = 1u) noexcept {
    return CoroSuspendExtensionBuilder<true>{
        std::move(schema), version, CoroSuspendFallback::ignore};
}

class SortCoroSuspendAnnotation final : public CoroSuspendAnnotation {
private:
    const Expression *_key;
    luisa::vector<CoroSuspendBinding> _bindings;
    luisa::vector<CoroSuspendAttribute> _attributes;

public:
    SortCoroSuspendAnnotation(
        const Expression *key, uint32_t range) noexcept
        : _key{key},
          _bindings{{.name = "key",
                     .access = CoroSuspendBindingAccess::read,
                     .lifetime = CoroSuspendBindingLifetime::queued,
                     .index = 0u}},
          _attributes{{.name = "range",
                       .value = static_cast<uint64_t>(range)}} {}

    [[nodiscard]] luisa::string_view schema() const noexcept override {
        return "luisa.coro.schedule.sort";
    }
    [[nodiscard]] uint32_t version() const noexcept override { return 1u; }
    [[nodiscard]] CoroSuspendFallback fallback() const noexcept override {
        return CoroSuspendFallback::ignore;
    }
    [[nodiscard]] luisa::span<const CoroSuspendBinding>
    bindings() const noexcept override {
        return _bindings;
    }
    [[nodiscard]] luisa::span<const CoroSuspendAttribute>
    attributes() const noexcept override {
        return _attributes;
    }
    [[nodiscard]] CoroSuspendExtensionPtr clone() const noexcept override {
        return luisa::make_unique<SortCoroSuspendAnnotation>(
            _key,
            static_cast<uint32_t>(
                luisa::get<uint64_t>(_attributes.front().value)));
    }
    [[nodiscard]] CoroSuspendExtensionPtr freeze(
        CoroSuspendExtensionRecorder &recorder) && noexcept override {
        auto bindings = _bindings;
        bindings.front().index = recorder.bind(
            bindings.front(), _key);
        return make_coro_suspend_annotation_data(
            luisa::string{schema()}, version(), fallback(),
            std::move(bindings), std::move(_attributes));
    }
};

template<typename T>
[[nodiscard]] inline CoroSuspendExtensionPtr coro_sort_by(
    T &&key, uint32_t range) noexcept {
    auto *expression =
        detail::extract_expression(std::forward<T>(key));
    LUISA_ASSERT(expression != nullptr &&
                     expression->type() == Type::of<uint>(),
                 "Coroutine sort key must be uint.");
    LUISA_ASSERT(range != 0u,
                 "Coroutine sort key range must be positive.");
    return luisa::make_unique<SortCoroSuspendAnnotation>(
        expression, range);
}

template<typename T>
inline constexpr bool is_coro_suspend_argument_v =
    std::same_as<std::remove_cvref_t<T>, CoroFrameExport> ||
    std::same_as<std::remove_cvref_t<T>, CoroSuspendExtensionPtr> ||
    std::same_as<std::remove_cvref_t<T>,
                 CoroSuspendExtensionBuilder<false>> ||
    std::same_as<std::remove_cvref_t<T>,
                 CoroSuspendExtensionBuilder<true>>;

template<typename... Args>
    requires(sizeof...(Args) != 0u &&
             (is_coro_suspend_argument_v<Args> && ...))
inline void suspend_impl(const char *name, Args &&...args) {
    constexpr auto frame_export_count =
        (static_cast<size_t>(
             std::same_as<std::remove_cvref_t<Args>, CoroFrameExport>) +
         ... + 0u);
    luisa::vector<CoroFrameExport> frame_exports;
    luisa::vector<CoroSuspendExtensionPtr> extensions;
    frame_exports.reserve(frame_export_count);
    extensions.reserve(sizeof...(Args) - frame_export_count);
    auto add = [&]<typename A>(A &&arg) noexcept {
        if constexpr (std::same_as<std::remove_cvref_t<A>,
                                   CoroFrameExport>) {
            frame_exports.emplace_back(std::forward<A>(arg));
        } else if constexpr (std::same_as<std::remove_cvref_t<A>,
                                          CoroSuspendExtensionPtr>) {
            extensions.emplace_back(std::forward<A>(arg));
        } else {
            extensions.emplace_back(std::move(arg).build());
        }
    };
    (add(std::forward<Args>(args)), ...);
    detail::FunctionBuilder::current()->suspend_(
        luisa::string{name}, std::move(frame_exports),
        std::move(extensions));
}

template<typename... Args>
    requires(sizeof...(Args) != 0u &&
             (is_coro_suspend_argument_v<Args> && ...))
inline void suspend_impl(uint32_t token, const char *name,
                         Args &&...args) {
    constexpr auto frame_export_count =
        (static_cast<size_t>(
             std::same_as<std::remove_cvref_t<Args>, CoroFrameExport>) +
         ... + 0u);
    luisa::vector<CoroFrameExport> frame_exports;
    luisa::vector<CoroSuspendExtensionPtr> extensions;
    frame_exports.reserve(frame_export_count);
    extensions.reserve(sizeof...(Args) - frame_export_count);
    auto add = [&]<typename A>(A &&arg) noexcept {
        if constexpr (std::same_as<std::remove_cvref_t<A>,
                                   CoroFrameExport>) {
            frame_exports.emplace_back(std::forward<A>(arg));
        } else if constexpr (std::same_as<std::remove_cvref_t<A>,
                                          CoroSuspendExtensionPtr>) {
            extensions.emplace_back(std::forward<A>(arg));
        } else {
            extensions.emplace_back(std::move(arg).build());
        }
    };
    (add(std::forward<Args>(args)), ...);
    detail::FunctionBuilder::current()->suspend_(
        token, luisa::string{name}, std::move(frame_exports),
        std::move(extensions));
}

}
}// namespace luisa::compute::dsl

#define $ ::luisa::compute::Var

#define $thread_id ::luisa::compute::thread_id()
#define $thread_x ::luisa::compute::thread_x()
#define $thread_y ::luisa::compute::thread_y()
#define $thread_z ::luisa::compute::thread_z()
#define $block_id ::luisa::compute::block_id()
#define $block_x ::luisa::compute::block_x()
#define $block_y ::luisa::compute::block_y()
#define $block_z ::luisa::compute::block_z()
#define $dispatch_id ::luisa::compute::dispatch_id()
#define $dispatch_x ::luisa::compute::dispatch_x()
#define $dispatch_y ::luisa::compute::dispatch_y()
#define $dispatch_z ::luisa::compute::dispatch_z()
#define $dispatch_size ::luisa::compute::dispatch_size()
#define $dispatch_size_x ::luisa::compute::dispatch_size_x()
#define $dispatch_size_y ::luisa::compute::dispatch_size_y()
#define $dispatch_size_z ::luisa::compute::dispatch_size_z()
#define $block_size ::luisa::compute::block_size()
#define $block_size_x ::luisa::compute::block_size_x()
#define $block_size_y ::luisa::compute::block_size_y()
#define $block_size_z ::luisa::compute::block_size_z()

#define $int $<int>
#define $uint $<::luisa::uint>
#define $float $<float>
#define $bool $<bool>
#define $short $<short>
#define $ushort $<::luisa::ushort>
#define $slong $<::luisa::slong>
#define $ulong $<::luisa::ulong>
#define $half $<::luisa::half>

#define $int2 $<::luisa::int2>
#define $uint2 $<::luisa::uint2>
#define $float2 $<::luisa::float2>
#define $bool2 $<::luisa::bool2>
#define $short2 $<::luisa::short2>
#define $ushort2 $<::luisa::ushort2>
#define $slong2 $<::luisa::slong2>
#define $ulong2 $<::luisa::ulong2>
#define $half2 $<::luisa::half2>

#define $int3 $<::luisa::int3>
#define $uint3 $<::luisa::uint3>
#define $float3 $<::luisa::float3>
#define $bool3 $<::luisa::bool3>
#define $short3 $<::luisa::short3>
#define $ushort3 $<::luisa::ushort3>
#define $slong3 $<::luisa::slong3>
#define $ulong3 $<::luisa::ulong3>
#define $half3 $<::luisa::half3>

#define $int4 $<::luisa::int4>
#define $uint4 $<::luisa::uint4>
#define $float4 $<::luisa::float4>
#define $bool4 $<::luisa::bool4>
#define $short4 $<::luisa::short4>
#define $ushort4 $<::luisa::ushort4>
#define $slong4 $<::luisa::slong4>
#define $ulong4 $<::luisa::ulong4>
#define $half4 $<::luisa::half4>

#define $float2x2 $<::luisa::float2x2>
#define $float3x3 $<::luisa::float3x3>
#define $float4x4 $<::luisa::float4x4>

#define $array ::luisa::compute::ArrayVar
#define $constant ::luisa::compute::Constant
#define $shared ::luisa::compute::Shared
#define $buffer ::luisa::compute::BufferVar
#define $image ::luisa::compute::ImageVar
#define $volume ::luisa::compute::VolumeVar
#define $atomic ::luisa::compute::AtomicVar
#define $bindless ::luisa::compute::BindlessVar
#define $accel ::luisa::compute::AccelVar

#define $outline                                                                    \
    ::luisa::compute::detail::outliner_with_comment(                                \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept -> void
#define $outline_with_name(function_name)                                           \
    ::luisa::compute::detail::outliner_with_comment(                                \
        function_name,                                                              \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept -> void

#define $lambda(...)                                                              \
    (::luisa::compute::Lambda{                                                    \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        ([&] __VA_ARGS__)})

#define $break ::luisa::compute::dsl::break_()
#define $continue ::luisa::compute::dsl::continue_()
#define $return(...) ::luisa::compute::dsl::return_(__VA_ARGS__)
#define $unreachable ::luisa::compute::dsl::unreachable()

#define $if(...)                                                                  \
    ::luisa::compute::detail::IfStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept -> void
#define $else \
    / [&]() noexcept -> void
#define $elif(...) \
    *([&] { return __VA_ARGS__; }) % [&]() noexcept -> void

#define $loop                                                                       \
    ::luisa::compute::detail::LoopStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept -> void

#define $while(...)                                                                 \
    ::luisa::compute::detail::WhileStmtBuilder::create_with_comment(                \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) / \
        [&]() noexcept { return (__VA_ARGS__); } %                                  \
        [&]() noexcept -> void

#define $autodiff                                                                   \
    ::luisa::compute::detail::AutoDiffStmtBuilder::create_with_comment(             \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept -> void

#define $switch(...)                                                              \
    ::luisa::compute::detail::SwitchStmtBuilder::create_with_comment(             \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept -> void
#define $case(...)                                                                \
    ::luisa::compute::detail::SwitchCaseStmtBuilder::create_with_comment(         \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept -> void
#define $default                                                                    \
    ::luisa::compute::detail::SwitchDefaultStmtBuilder::create_with_comment(        \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept -> void

#define $for(x, ...)                                                                   \
    for (auto x : ::luisa::compute::dynamic_range_with_comment(                        \
             ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
             __VA_ARGS__))                                                             \
    ::luisa::compute::detail::StmtBodyInvoke{} % [&]() noexcept -> void

#define $comment(...) \
    ::luisa::compute::detail::comment(__VA_ARGS__)
#define $comment_with_location(...)                                                                \
    $comment(luisa::string{__VA_ARGS__}                                                            \
                 .append(" [")                                                                     \
                 .append(::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) \
                 .append("]"))

#define LUISA_COMPUTE_DSL_DEVICE_DEBUG_WATCH_ADD(x)                                  \
    static_assert(::luisa::compute::is_var_v<decltype(x)>,                           \
                  "Only DSL variables are allowed for evaluation in device debug."); \
    using _device_debug_type_##x = ::luisa::compute::expr_value_t<decltype(x)>;      \
    _device_debug_watches.emplace_back(::luisa::compute::detail::extract_expression(x));

#define LUISA_COMPUTE_DSL_DEVICE_DEBUG_WATCH_EVAL(x)       \
    auto x = *static_cast<const _device_debug_type_##x *>( \
        _device_debug_eval(_device_debug_ctx, _device_debug_watch_index++));

// device debug
#define LUISA_COMPUTE_DSL_DEVICE_DEBUG_IMPL(TRAP_FUNC, ...)                          \
    do {                                                                             \
        auto dispatch_id = ::luisa::compute::dispatch_id();                          \
        ::luisa::vector<const ::luisa::compute::Expression *> _device_debug_watches; \
        _device_debug_watches.reserve(                                               \
            ([](auto &&...args) noexcept { return sizeof...(args); })(               \
                dispatch_id __VA_OPT__(, ) __VA_ARGS__));                            \
        LUISA_MAP(LUISA_COMPUTE_DSL_DEVICE_DEBUG_WATCH_ADD,                          \
                  dispatch_id __VA_OPT__(, ) __VA_ARGS__)                            \
        using Eval = ::luisa::compute::DebugBreakStmt::Evaluator;                    \
        ::luisa::compute::detail::FunctionBuilder::current()->debug_break_(          \
            [](void *_device_debug_ctx, Eval *_device_debug_eval) noexcept {         \
                auto _device_debug_watch_index = static_cast<size_t>(0);             \
                LUISA_MAP(LUISA_COMPUTE_DSL_DEVICE_DEBUG_WATCH_EVAL,                 \
                          dispatch_id __VA_OPT__(, ) __VA_ARGS__)                    \
                [dispatch_id __VA_OPT__(, ) __VA_ARGS__] {                           \
                    TRAP_FUNC;                                                       \
                }();                                                                 \
            },                                                                       \
            std::move(_device_debug_watches));                                       \
    } while (false)

#define LUISA_COMPUTE_DSL_DEVICE_DEBUG_IMPL_REVERSE(TRAP_FUNC, ...) \
    LUISA_COMPUTE_DSL_DEVICE_DEBUG_IMPL(TRAP_FUNC __VA_OPT__(, ) LUISA_REVERSE(__VA_ARGS__))

#define $debug_break(...) \
    LUISA_COMPUTE_DSL_DEVICE_DEBUG_IMPL(LUISA_DEBUG_TRAP(), __VA_ARGS__)

#define $debug_break_on(...) \
    LUISA_COMPUTE_DSL_DEVICE_DEBUG_IMPL_REVERSE(LUISA_REVERSE(__VA_ARGS__))

#define $suspend(...) ::luisa::compute::dsl::suspend_impl(__VA_ARGS__)

#define $clc_work_stealing_1d(body) \
    ::luisa::compute::cluster_launch_control_work_stealing_1d([&](auto bx) noexcept -> void body)

#endif
