#pragma once

#include <ast/function.h>
#include <dsl/func.h>
#include <dsl/struct.h>
#include <runtime/raster/app_data.h>
LUISA_STRUCT(luisa::compute::AppData, position, normal, tangent, color, uv, vertex_id, instance_id){};

namespace luisa::compute {
template<typename VertCallable, typename PixelCallable>
class RasterKernel;
LC_DSL_API void check_vert_ret_type(Type const *type);
template<typename VertRet, typename... VertArgs, typename PixelRet, typename... PixelArgs>
class RasterKernel<Callable<VertRet(AppData, VertArgs...)>, Callable<PixelRet(VertRet, PixelArgs...)>> {

public:
    using VertexKernel = Callable<VertRet(AppData, VertArgs...)>;
    using PixelKernel = Callable<PixelRet(VertRet, PixelArgs...)>;
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
    RasterKernel(RasterKernel const &) = delete;
    RasterKernel(RasterKernel &&) = default;
    RasterKernel &operator=(RasterKernel const &) = delete;
    RasterKernel &operator=(RasterKernel &&) = delete;
    [[nodiscard]] auto vert() const noexcept { return Function{_vert.get()}; }
    [[nodiscard]] auto pixel() const noexcept { return Function{_pixel.get()}; }
};

}// namespace luisa::compute