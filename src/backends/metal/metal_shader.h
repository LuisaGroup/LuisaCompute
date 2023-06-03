//
// Created by Mike Smith on 2023/5/14.
//

#pragma once

#include <runtime/rhi/command.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalCommandEncoder;

struct MetalShaderHandle {
    NS::SharedPtr<MTL::ComputePipelineState> entry;
    NS::SharedPtr<MTL::ComputePipelineState> indirect_entry;
};

class MetalShader {

public:
    using Argument = ShaderDispatchCommand::Argument;

private:
    MetalShaderHandle _handle;
    luisa::vector<Usage> _argument_usages;
    luisa::vector<Argument> _bound_arguments;
    uint _block_size[3];
    mutable spin_mutex _name_mutex;
    NS::String *_name{nullptr};
    NS::String *_indirect_name{nullptr};
    MTL::ComputePipelineState *_prepare_indirect;

public:
    MetalShader(MetalDevice *device,
                MetalShaderHandle handle,
                luisa::vector<Usage> argument_usages,
                luisa::vector<Argument> bound_arguments,
                uint3 block_size) noexcept;
    ~MetalShader() noexcept;
    void launch(MetalCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept;
    [[nodiscard]] Usage argument_usage(uint index) const noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
