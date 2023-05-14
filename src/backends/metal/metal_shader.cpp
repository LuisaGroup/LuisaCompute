//
// Created by Mike Smith on 2023/5/14.
//

#include <core/logging.h>
#include <backends/metal/metal_shader.h>

namespace luisa::compute::metal {

MetalShader::MetalShader(NS::SharedPtr<MTL::ComputePipelineState> handle,
                         luisa::vector<Usage> argument_usages,
                         luisa::vector<Argument> bound_arguments,
                         uint3 block_size) noexcept
    : _handle{std::move(handle)},
      _argument_usages{std::move(argument_usages)},
      _bound_arguments{std::move(bound_arguments)},
      _block_size{block_size} {}

MetalShader::~MetalShader() noexcept {
    if (_name != nullptr) { _name->release(); }
}

Usage MetalShader::argument_usage(uint index) const noexcept {
#ifndef NDEBUG
    LUISA_ASSERT(index < _argument_usages.size(),
                 "Argument index out of range.");
#endif
    return _argument_usages[index];
}

void MetalShader::set_name(luisa::string_view name) noexcept {
    if (_name != nullptr) {
        _name->release();
        _name = nullptr;
    }
    if (!name.empty()) {
        _name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
    }
}

void MetalShader::launch(MetalCommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept {
    LUISA_ERROR_WITH_LOCATION("Metal shader dispatch not implemented.");
}

}// namespace luisa::compute::metal
