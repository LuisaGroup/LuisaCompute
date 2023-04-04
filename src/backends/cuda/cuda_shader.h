//
// Created by Mike on 2021/12/4.
//

#pragma once

#include <span>
#include <memory>

#include <core/basic_types.h>
#include <ast/usage.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAShader {

private:
    luisa::vector<Usage> _argument_usages;

public:
    explicit CUDAShader(luisa::vector<Usage> arg_usages) noexcept;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    virtual void launch(CUDACommandEncoder &encoder,
                        ShaderDispatchCommand *command) const noexcept = 0;
    [[nodiscard]] Usage argument_usage(size_t i) const noexcept;
};

}// namespace luisa::compute::cuda
