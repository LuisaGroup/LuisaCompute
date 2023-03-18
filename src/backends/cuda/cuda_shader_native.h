//
// Created by Mike on 3/18/2023.
//

#pragma once

#include <cuda.h>
#include <core/stl/string.h>
#include <backends/cuda/cuda_shader.h>

namespace luisa::compute::cuda {

class CUDAShaderNative final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    luisa::string _entry;

public:
    CUDAShaderNative(const char *ptx, size_t ptx_size, const char *entry) noexcept;
    ~CUDAShaderNative() noexcept override;
    void launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;
};

}// namespace luisa::compute::cuda
