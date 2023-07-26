#pragma once

#include <span>
#include <memory>

#include <luisa/core/basic_types.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/ast/usage.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAShader {

private:
    luisa::vector<Usage> _argument_usages;
    luisa::string _name;
    mutable spin_mutex _name_mutex;

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
    [[nodiscard]] virtual void *handle() const noexcept = 0;
    void launch(CUDACommandEncoder &encoder,
                ShaderDispatchCommand *command) const noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

