#pragma once

#include <span>
#include <memory>

#include <luisa/core/basic_types.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/ast/usage.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}// namespace luisa::compute

namespace luisa::compute::cuda {

class CUDACommandEncoder;
class CUDAShaderPrinter;

class CUDAShader {

private:
    luisa::unique_ptr<CUDAShaderPrinter> _printer;
    luisa::vector<Usage> _argument_usages;
    luisa::string _name;
    mutable spin_mutex _name_mutex;

private:
    virtual void _launch(CUDACommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept = 0;

public:
    static void _patch_ptx_version(luisa::vector<std::byte> &ptx) noexcept;

public:
    CUDAShader(luisa::unique_ptr<CUDAShaderPrinter> printer,
               luisa::vector<Usage> arg_usages) noexcept;
    virtual ~CUDAShader() noexcept;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    [[nodiscard]] Usage argument_usage(size_t i) const noexcept;
    [[nodiscard]] auto printer() const noexcept { return _printer.get(); }
    [[nodiscard]] virtual void *handle() const noexcept = 0;
    [[nodiscard]] virtual bool is_graph_compatible() const noexcept { return false; }
    void launch(CUDACommandEncoder &encoder,
                ShaderDispatchCommand *command) const noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda
