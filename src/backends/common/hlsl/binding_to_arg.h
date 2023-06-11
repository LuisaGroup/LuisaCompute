#pragma once
#include <luisa/vstl/common.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/argument.h>
namespace lc::hlsl {
using namespace luisa::compute;
vstd::vector<Argument> binding_to_arg(vstd::span<const Function::Binding> bindings);
}// namespace lc::hlsl
