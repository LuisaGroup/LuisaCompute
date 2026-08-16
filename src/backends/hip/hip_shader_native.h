#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include "hip_shader.h"
#include "hip_shader_metadata.h"

namespace luisa::compute::hip {

class HIPDevice;

class HIPShaderNative final : public HIPShader {

private:
    hipModule_t _module{};
    hipFunction_t _function{};
    luisa::string _entry;
    uint _block_size[3];
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    HIPDevice *_device{nullptr};
    bool _requires_global_rt_stack{false};
    bool _uses_static_global_rt_stack{false};

private:
    void _launch(HIPCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;
    void _load_code_object(
        luisa::span<const std::byte> code_object,
        const HIPShaderMetadata &metadata,
        bool ray_tracing) noexcept;

public:
    HIPShaderNative(HIPDevice *device, luisa::string code,
                    const char *entry, const HIPShaderMetadata &metadata,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    HIPShaderNative(HIPDevice *device, luisa::string code,
                    const char *entry, const HIPShaderMetadata &metadata,
                    hiprtContext hiprt_ctx,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    HIPShaderNative(HIPDevice *device,
                    luisa::span<const std::byte> code_object,
                    const char *entry, const HIPShaderMetadata &metadata,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    HIPShaderNative(HIPDevice *device,
                    luisa::span<const std::byte> code_object,
                    const char *entry, const HIPShaderMetadata &metadata,
                    hiprtContext hiprt_ctx,
                    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~HIPShaderNative() noexcept override;
    [[nodiscard]] void *handle() const noexcept override { return _function; }
};

[[nodiscard]] luisa::vector<std::byte> hip_link_llvm_bitcode(
    luisa::string_view bitcode, const char *entry) noexcept;

}// namespace luisa::compute::hip
