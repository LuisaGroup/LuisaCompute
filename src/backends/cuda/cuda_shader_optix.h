//
// Created by Mike on 3/18/2023.
//

#include <core/spin_mutex.h>
#include <core/stl/string.h>
#include <core/stl/unordered_map.h>
#include <backends/cuda/cuda_shader.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAShaderOptiX final : public CUDAShader {

private:
    size_t _argument_buffer_size{};
    optix::Module _module{};
    optix::ProgramGroup _program_group_rg{};
    optix::ProgramGroup _program_group_ch_closest{};
    optix::ProgramGroup _program_group_ch_query{};
    optix::ProgramGroup _program_group_miss_closest{};
    optix::ProgramGroup _program_group_miss_any{};
    optix::ProgramGroup _program_group_miss_query{};
    optix::Pipeline _pipeline{};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    CUdeviceptr _sbt_buffer{};

private:
    void _launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;

public:
    CUDAShaderOptiX(optix::DeviceContext optix_ctx,
                    const char *ptx, size_t ptx_size,
                    const char *entry, bool enable_debug,
                    luisa::vector<Usage> argument_usages,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~CUDAShaderOptiX() noexcept override;
};

}// namespace luisa::compute::cuda
