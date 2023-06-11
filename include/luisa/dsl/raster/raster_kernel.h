#pragma once

#include <luisa/ast/function.h>
#include <luisa/dsl/raster/raster_func.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/raster/app_data.h>
LUISA_STRUCT(luisa::compute::AppData, position, normal, tangent, color, uv, vertex_id, instance_id){};

namespace luisa::compute {
template<typename VertCallable, typename PixelCallable>
class RasterKernel;
LC_DSL_API void check_vert_ret_type(Type const *type);
template<typename VertRet, typename... VertArgs, typename PixelRet, typename... PixelArgs>
class RasterKernel<RasterStageKernel<VertRet(AppData, VertArgs...)>, RasterStageKernel<PixelRet(VertRet, PixelArgs...)>> {

public:
    using VertexKernel = RasterStageKernel<VertRet(AppData, VertArgs...)>;
    using PixelKernel = RasterStageKernel<PixelRet(VertRet, PixelArgs...)>;
    using RasterShaderType = RasterShader<VertArgs..., PixelArgs...>;

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _vert;
    luisa::shared_ptr<const detail::FunctionBuilder> _pixel;

public:
    RasterKernel(VertexKernel const &vert, PixelKernel const &pixel) noexcept
        : _vert{vert.function_builder()},
          _pixel{pixel.function_builder()} {
        // Structure's first element must be float4 as position
        check_vert_ret_type(Type::template of<VertRet>());
    }
    RasterKernel(RasterKernel const &) = default;
    RasterKernel(RasterKernel &&) = default;
    RasterKernel &operator=(RasterKernel const &) = default;
    RasterKernel &operator=(RasterKernel &&) = default;
    [[nodiscard]] auto vert() const noexcept { return Function{_vert.get()}; }
    [[nodiscard]] auto pixel() const noexcept { return Function{_pixel.get()}; }
};

}// namespace luisa::compute
