#pragma once

#include <ast/function.h>

namespace luisa::compute {

template<typename VertCallable, typename PixelCallable>
class RasterKernel;

template<typename... Args>
class RasterShader;

template<typename VertRet, typename... VertArgs, typename PixelRet, typename... PixelArgs>
class RasterKernel<Callable<VertRet(VertArgs...)>, Callable<PixelRet(VertRet, PixelArgs...)>> {

public:
    using VertexKernel = Callable<VertRet(VertArgs...)>;
    using PixelKernel = Callable<PixelRet(VertRet, PixelArgs...)>;
    using RasterShaderType = RasterShader<VertArgs..., PixelArgs...>;

private:
    luisa::shared_ptr<const detail::FunctionBuilder> _vert;
    luisa::shared_ptr<const detail::FunctionBuilder> _pixel;

public:
    RasterKernel(VertexKernel const &vert, PixelKernel const &pixel) noexcept {
        Type const *v2pType = Type::template of<VertRet>();
        // Structure's first element must be float4 as position
        assert((v2pType->is_vector() && v2pType->element()->tag() == Type::Tag::FLOAT32 && v2pType->dimension() == 4) || (v2pType->is_structure() && v2pType->members().size() >= 1 && v2pType->members()[0]->is_vector() && v2pType->members()[0]->element()->tag() == Type::Tag::FLOA32 && v2pType->members()[0]->dimension() == 4));
        _vert = vert.function_builder();
        _pixel = pixel.function_builder();
    }
    auto const &vert() const noexcept {
        return _vert;
    }
    auto const &pixel() const noexcept {
        return _pixel;
    }
};

}// namespace luisa::compute
