#pragma once

#include <span>
#include <memory>

#include <luisa/core/basic_types.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
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

public:
    // Cross-backend introspection API (used by e.g. the Vulkan backend's
    // VK_NV_cuda_kernel_launch interop to import DSL-compiled kernels).
    // All accessors are header-inline or pure virtual so they are safe to
    // call across DLL boundaries without exporting additional symbols.
    //
    // Usages of all kernel arguments (uniforms included), in argument order.
    [[nodiscard]] luisa::span<const Usage> argument_usages() const noexcept { return _argument_usages; }
    [[nodiscard]] size_t argument_count() const noexcept { return _argument_usages.size(); }
    // The loaded module image: the (possibly version-patched) PTX text, or
    // the linked cubin when the module was linked with cudadevrt. Either way
    // this is exactly the image this shader was loaded from. Empty for
    // shaders that do not own an importable image (e.g. OptiX pipelines).
    [[nodiscard]] virtual luisa::span<const std::byte> module_image() const noexcept = 0;
    // The __global__ entry point name ("kernel_main" for compute DSL kernels).
    [[nodiscard]] virtual luisa::string_view entry() const noexcept = 0;
    // The compiled block dimension (DSL set_block_size()).
    [[nodiscard]] virtual uint3 block_size() const noexcept = 0;
    // Number of bound (captured) arguments encoded before command arguments.
    [[nodiscard]] virtual size_t bound_argument_count() const noexcept = 0;
};

}// namespace luisa::compute::cuda
