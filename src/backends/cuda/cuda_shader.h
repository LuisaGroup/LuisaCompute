//
// Created by Mike on 2021/12/4.
//

#pragma once

#include <span>
#include <memory>

#include <core/basic_types.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACommandEncoder;

struct CUDAShader {
    CUDAShader() noexcept = default;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    virtual void launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept = 0;
};

}// namespace luisa::compute::cuda
