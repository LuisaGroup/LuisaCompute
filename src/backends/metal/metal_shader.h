//
// Created by Mike Smith on 2023/5/14.
//

#pragma once

#include <runtime/rhi/command.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalShader {

public:
    using Argument = ShaderDispatchCommand::Argument;

private:
    NS::SharedPtr<MTL::ComputePipelineState> _handle;
    luisa::vector<Usage> _argument_usages;
    luisa::vector<Argument> _bound_arguments;
    uint3 _block_size;
    NS::String *_name{nullptr};

public:
    MetalShader(NS::SharedPtr<MTL::ComputePipelineState> handle,
                luisa::vector<Usage> argument_usages,
                luisa::vector<Argument> bound_arguments,
                uint3 block_size) noexcept;
    ~MetalShader() noexcept;
    void launch(MetalCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept;
    [[nodiscard]] Usage argument_usage(uint index) const noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
