#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/command.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalDevice;
class MetalCommandEncoder;
class MetalShaderPrinter;

struct MetalShaderHandle {
    NS::SharedPtr<MTL::ComputePipelineState> entry;
    NS::SharedPtr<MTL::ComputePipelineState> indirect_entry;
};

enum class MetalShaderBinding : uint8_t { ARGUMENT_BUFFER,
                                          DIRECT_BUFFERS };

class MetalShader {

public:
    using Argument = ShaderDispatchCommand::Argument;

private:
    MetalShaderHandle _handle;
    luisa::vector<Usage> _argument_usages{};
    luisa::vector<uint8_t> _argument_sampled{};
    luisa::vector<Argument> _bound_arguments{};
    MetalShaderBinding _binding{MetalShaderBinding::ARGUMENT_BUFFER};
    // Indexed by Metal buffer slot; values index the original Runtime args.
    luisa::vector<uint32_t> _buffer_arguments;
    uint _block_size[3];
    uint64_t _source_checksum{};
    size_t _source_size_bytes{};
    size_t _source_line_count{};
    double _codegen_ms{};
    double _compile_ms{};
    mutable spin_mutex _name_mutex;
    NS::String *_name{nullptr};
    NS::String *_indirect_name{nullptr};
    MTL::ComputePipelineState *_prepare_indirect;
    luisa::unique_ptr<MetalShaderPrinter> _printer{nullptr};

public:
    MetalShader(MetalDevice *device,
                MetalShaderHandle handle,
                luisa::vector<Usage> argument_usages,
                luisa::vector<uint8_t> argument_sampled,
                luisa::vector<Argument> bound_arguments,
                luisa::span<const std::pair<luisa::string, luisa::string>> print_formats,
                uint3 block_size,
                uint64_t source_checksum,
                size_t source_size_bytes,
                size_t source_line_count,
                double codegen_ms,
                double compile_ms,
                MetalShaderBinding binding = MetalShaderBinding::ARGUMENT_BUFFER,
                luisa::vector<uint32_t> buffer_arguments = {}) noexcept;
    ~MetalShader() noexcept;
    void launch(MetalCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept;
    [[nodiscard]] Usage argument_usage(uint index) const noexcept;
    [[nodiscard]] auto pso() const noexcept { return _handle.entry.get(); }
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
