//
// Created by Mike on 3/18/2023.
//

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include "cuda_shader.h"
#include "optix_api.h"

namespace luisa::compute::cuda {

class CUDACommandEncoder;
class CUDAIndirectDispatchOptiX;

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
    friend class CUDAIndirectDispatchOptiX;
    [[nodiscard]] optix::ShaderBindingTable _make_sbt() const noexcept;
    void _do_launch(CUstream stream, CUdeviceptr argument_buffer, uint3 dispatch_size) const noexcept;
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

