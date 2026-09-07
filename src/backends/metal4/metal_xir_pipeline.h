#pragma once

#include <cstddef>
#include <limits>

#include <luisa/ast/function.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/xir/module.h>

namespace luisa::compute::metal {

struct MetalRayQueryPipelinePolicy {
    bool enabled{true};
    size_t max_captured_payload_bytes{
        std::numeric_limits<size_t>::max()};
};

[[nodiscard]] luisa::unique_ptr<xir::Module>
metal_translate_ast_to_xir(
    Function kernel, const ShaderOption &option,
    MetalRayQueryPipelinePolicy ray_query_policy = {}) noexcept;

[[nodiscard]] luisa::unique_ptr<xir::Module>
metal_translate_raster_ast_to_xir(
    Function stage_function, xir::RasterStage stage,
    const ShaderOption &option) noexcept;

}// namespace luisa::compute::metal
