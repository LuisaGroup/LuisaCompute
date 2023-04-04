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
    luisa::string _name;

private:
    virtual void _launch(CUDACommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept = 0;

public:
    explicit CUDAShader(luisa::vector<Usage> arg_usages) noexcept;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    [[nodiscard]] Usage argument_usage(size_t i) const noexcept;
    void launch(CUDACommandEncoder &encoder,
                ShaderDispatchCommand *command) const noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda
