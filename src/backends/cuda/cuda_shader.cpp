//
// Created by Mike on 4/4/2023.
//

#include <core/logging.h>
#include <backends/cuda/cuda_shader.h>

namespace luisa::compute::cuda {

CUDAShader::CUDAShader(luisa::vector<Usage> arg_usages) noexcept
    : _argument_usages{std::move(arg_usages)} {}

Usage CUDAShader::argument_usage(size_t i) const noexcept {
    LUISA_ASSERT(i < _argument_usages.size(),
                 "Invalid argument index {} for shader with {} argument(s).",
                 i, _argument_usages.size());
    return _argument_usages[i];
}

}// namespace luisa::compute::cuda
