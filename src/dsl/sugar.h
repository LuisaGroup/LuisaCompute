//
// Created by Mike Smith on 2021/3/7.
//

#pragma once
#ifndef LC_DISABLE_DSL
#ifndef LUISA_COMPUTE_DESUGAR

#include <dsl/syntax.h>

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
#define $uint $<uint>
#define $float $<float>
#define $bool $<bool>

#define $int2 $<int2>
#define $uint2 $<uint2>
#define $float2 $<float2>
#define $bool2 $<bool2>

#define $int3 $<int3>
#define $uint3 $<uint3>
#define $float3 $<float3>
#define $bool3 $<bool3>

#define $int4 $<int4>
#define $uint4 $<uint4>
#define $float4 $<float4>
#define $bool4 $<bool4>

#define $array ::luisa::compute::ArrayVar
#define $constant ::luisa::compute::Constant
#define $shared ::luisa::compute::Shared
#define $buffer ::luisa::compute::BufferVar
#define $image ::luisa::compute::ImageVar
#define $volume ::luisa::compute::VolumeVar
#define $atomic ::luisa::compute::AtomicVar
#define $bindless ::luisa::compute::BindlessVar
#define $accel ::luisa::compute::AccelVar

#define $break ::luisa::compute::break_()
#define $continue ::luisa::compute::continue_()
#define $return(...) ::luisa::compute::return_(__VA_ARGS__)

#define $if(...) ::luisa::compute::detail::IfStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $else / [&]() noexcept
#define $elif(...) *([&] { return __VA_ARGS__; }) % [&]() noexcept
#define $loop ::luisa::compute::detail::LoopStmtBuilder{} % [&]() noexcept
#define $while(...) ::luisa::compute::detail::LoopStmtBuilder{} / [&]() noexcept { \
    $if(!(__VA_ARGS__)) { $break; };                                               \
} % [&]() noexcept

#define $switch(...) ::luisa::compute::detail::SwitchStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $case(...) ::luisa::compute::detail::SwitchCaseStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $default ::luisa::compute::detail::SwitchDefaultStmtBuilder{} % [&]() noexcept

#define $for(x, ...)                                    \
    for (auto x : ::luisa::compute::dynamic_range(__VA_ARGS__)) \
    ::luisa::compute::detail::ForStmtBodyInvoke{} % [&]() noexcept

#define $comment(...) \
    ::luisa::compute::comment(__VA_ARGS__)
#define $comment_with_location(...) \
    $comment(luisa::format(FMT_STRING("{} [{}:{}]"), std::string_view{__VA_ARGS__}, __FILE__, __LINE__))

#endif
#endif