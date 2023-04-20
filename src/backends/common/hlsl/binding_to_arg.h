#pragma once
#include <vstl/common.h>
#include <ast/function.h>
#include <runtime/rhi/argument.h>
namespace lc::hlsl {
using namespace luisa::compute;
vstd::vector<Argument> binding_to_arg(vstd::span<const Function::Binding> bindings);
}// namespace lc::hlsl