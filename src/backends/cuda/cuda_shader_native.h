#pragma once

#include <cuda.h>
#include <luisa/core/stl/string.h>
#include "cuda_shader.h"

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
    CUDAShaderNative(CUDADevice *device, luisa::vector<std::byte> ptx,
                     const char *entry, const CUDAShaderMetadata &metadata,
                     luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~CUDAShaderNative() noexcept override;
    [[nodiscard]] bool is_graph_compatible() const noexcept override { return true; }
    [[nodiscard]] void *handle() const noexcept override { return _function; }
};

}// namespace luisa::compute::cuda
