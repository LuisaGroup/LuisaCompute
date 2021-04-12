//
// Created by Mike Smith on 2021/3/7.
//

#pragma once

#include <dsl/syntax.h>

#define $ ::luisa::compute::Var

#define $thread_id ::luisa::compute::thread_id()
#define $block_id ::luisa::compute::block_id()
#define $dispatch_id ::luisa::compute::dispatch_id()
#define $launch_size ::luisa::compute::launch_size()
#define $block_size ::luisa::compute::block_size()

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

#define $break ::luisa::compute::break_()
#define $continue ::luisa::compute::continue_()

#define $if(...) ::luisa::compute::detail::IfStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $else / [&]() noexcept
#define $elif(...) / ::luisa::compute::detail::Expr{__VA_ARGS__} % [&]() noexcept

#define $while(...) ::luisa::compute::detail::WhileStmtBuilder{__VA_ARGS__} % [&]() noexcept

#define $switch(...) ::luisa::compute::detail::SwitchStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $case(...) ::luisa::compute::detail::SwitchCaseStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $default ::luisa::compute::detail::SwitchDefaultStmtBuilder{} % [&]() noexcept

#define $for(...) for (auto __VA_ARGS__
#define $range(...) ::luisa::compute::range(__VA_ARGS__))
