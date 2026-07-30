#pragma once

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
    ::luisa::compute::detail::LoopStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) / \
        [&]() noexcept -> void {                                                    \
        $if (!(__VA_ARGS__)) { $break; };                                           \
    } % [&]() noexcept -> void

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
