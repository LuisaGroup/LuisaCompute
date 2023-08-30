#pragma once

#include <cuda.h>
#include <luisa/core/stl/string.h>
#include "cuda_shader.h"

namespace luisa::compute::graph {
class KernelNodeCmdEncoder;
}// namespace luisa::compute::graph

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAShaderNative final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    CUfunction _indirect_function{};
    luisa::string _entry;
    uint _block_size[3];
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;

private:
    void _launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;

public:
    CUDAShaderNative(CUDADevice *device,
                     const char *ptx, size_t ptx_size,
                     const char *entry, uint3 block_size,
                     luisa::vector<Usage> argument_usages,
                     luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~CUDAShaderNative() noexcept override;
    [[nodiscard]] void *handle() const noexcept override { return _function; }
    virtual void encode_kernel_node_parms(luisa::function<void(CUDA_KERNEL_NODE_PARAMS *)> func, luisa::compute::graph::KernelNodeCmdEncoder *encoder) noexcept override;
};

}// namespace luisa::compute::cuda
