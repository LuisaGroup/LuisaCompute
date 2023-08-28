#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include "cuda_shader.h"
#include "optix_api.h"

namespace luisa::compute::cuda {

class CUDACommandEncoder;
struct CUDAShaderMetadata;

class CUDAShaderOptiX final : public CUDAShader {

public:
    struct IndirectParameters {
        CUDAIndirectDispatchBuffer::Header header;
        /*[[no_unique_address]]*/ CUDAIndirectDispatchBuffer::Dispatch dispatches[];
    };

private:
    size_t _argument_buffer_size{};
    optix::Module _module{};
    optix::ProgramGroup _program_group_rg{};
    optix::ProgramGroup _program_group_ray_query{};
    optix::Pipeline _pipeline{};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    CUdeviceptr _sbt_buffer{};

private:
    [[nodiscard]] optix::ShaderBindingTable _make_sbt() const noexcept;
    void _do_launch(CUstream stream, CUdeviceptr argument_buffer, uint3 dispatch_size) const noexcept;
    void _do_launch_indirect(CUstream stream, CUdeviceptr argument_buffer,
                             size_t dispatch_offset, size_t dispatch_count,
                             const IndirectParameters *indirect_buffer_device,
                             const IndirectParameters *indirect_params_readback) const noexcept;
    void _launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;

public:
    CUDAShaderOptiX(optix::DeviceContext optix_ctx,
                    const char *ptx, size_t ptx_size, const char *entry,
                    const CUDAShaderMetadata &metadata,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~CUDAShaderOptiX() noexcept override;
    [[nodiscard]] void *handle() const noexcept override { return _pipeline; }
};

}// namespace luisa::compute::cuda

