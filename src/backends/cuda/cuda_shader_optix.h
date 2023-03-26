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
    CUDADevice *_device;
    size_t _argument_buffer_size{};
    optix::Module _module{};
    optix::ProgramGroup _program_group_rg{};
    optix::ProgramGroup _program_group_ch_closest{};
    optix::ProgramGroup _program_group_ch_any{};
    optix::ProgramGroup _program_group_miss{};
    optix::Pipeline _pipeline{};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    mutable CUdeviceptr _sbt_buffer{};
    mutable luisa::spin_mutex _mutex;
    mutable CUevent _sbt_event{};
    mutable optix::ShaderBindingTable _sbt{};
    mutable luisa::unordered_set<uint64_t> _sbt_recorded_streams;

private:
    void _prepare_sbt(CUDACommandEncoder &encoder) const noexcept;

public:
    CUDAShaderOptiX(CUDADevice *device,
                    const char *ptx, size_t ptx_size,
                    const char *entry, bool enable_debug,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept;
    ~CUDAShaderOptiX() noexcept override;
    void launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;
};

}// namespace luisa::compute::cuda
