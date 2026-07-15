#pragma once

#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/xir/module.h>

namespace luisa::compute::spirv {

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module>;

}// namespace luisa::compute::spirv
