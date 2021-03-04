//
// Created by Mike Smith on 2021/2/27.
//

#pragma once

#include <dsl/buffer.h>
#include <dsl/func.h>
#include <dsl/constant.h>
#include <dsl/shared.h>
#include <dsl/struct.h>
#include <dsl/var.h>
#include <dsl/stmt.h>

#ifndef LUISA_DISABLE_SYNTAX_SUGAR

#define $var ::luisa::compute::dsl::Var

#define $char $var<char>
#define $uchar $var<uchar>
#define $short $var<short>
#define $ushort $var<ushort>
#define $int $var<int>
#define $uint $var<uint>
#define $float $var<float>
#define $bool $var<bool>

#define $char2 $var<char2>
#define $uchar2 $var<uchar2>
#define $short2 $var<short2>
#define $ushort2 $var<ushort2>
#define $int2 $var<int2>
#define $uint2 $var<uint2>
#define $float2 $var<float2>
#define $bool2 $var<bool2>

#define $char3 $var<char3>
#define $uchar3 $var<uchar3>
#define $short3 $var<short3>
#define $ushort3 $var<ushort3>
#define $int3 $var<int3>
#define $uint3 $var<uint3>
#define $float3 $var<float3>
#define $bool3 $var<bool3>

#define $char4 $var<char4>
#define $uchar4 $var<uchar4>
#define $short4 $var<short4>
#define $ushort4 $var<ushort4>
#define $int4 $var<int4>
#define $uint4 $var<uint4>
#define $float4 $var<float4>
#define $bool4 $var<bool4>

#define $array ::luisa::compute::dsl::VarArray
#define $constant ::luisa::compute::dsl::Constant
#define $shared ::luisa::compute::dsl::Shared
#define $buffer ::luisa::compute::dsl::BufferView

#define $break ::luisa::compute::dsl::break_()
#define $continue ::luisa::compute::dsl::continue_()

#define $if(...) ::luisa::compute::dsl::detail::IfStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $else / [&]() noexcept
#define $elif(...) / ::luisa::compute::dsl::detail::Expr{__VA_ARGS__} % [&]() noexcept

#define $while(...) ::luisa::compute::dsl::detail::WhileStmtBuilder{__VA_ARGS__} % [&]() noexcept

#define $switch(...) ::luisa::compute::dsl::detail::SwitchStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $case(...) ::luisa::compute::dsl::detail::SwitchCaseStmtBuilder{__VA_ARGS__} % [&]() noexcept
#define $default ::luisa::compute::dsl::detail::SwitchDefaultStmtBuilder{} % [&]() noexcept

#endif
