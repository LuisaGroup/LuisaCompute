#pragma once

#include <luisa/ast/function.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/xir/module.h>

namespace luisa::compute::metal {

[[nodiscard]] luisa::unique_ptr<xir::Module>
metal_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept;

[[nodiscard]] luisa::unique_ptr<xir::Module>
metal_translate_raster_ast_to_xir(
    Function stage_function, xir::RasterStage stage,
    const ShaderOption &option) noexcept;

}// namespace luisa::compute::metal
