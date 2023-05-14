//
// Created by Mike Smith on 2023/5/14.
//

#include <core/logging.h>
#include <backends/metal/metal_buffer.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_accel.h>
#include <backends/metal/metal_bindless_array.h>
#include <backends/metal/metal_shader.h>

namespace luisa::compute::metal {

MetalShader::MetalShader(NS::SharedPtr<MTL::ComputePipelineState> handle,
                         luisa::vector<Usage> argument_usages,
                         luisa::vector<Argument> bound_arguments,
                         uint3 block_size) noexcept
    : _handle{std::move(handle)},
      _argument_usages{std::move(argument_usages)},
      _bound_arguments{std::move(bound_arguments)},
      _block_size{block_size.x, block_size.y, block_size.z} {}

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

    static constexpr auto argument_buffer_size = 65536u;
    static constexpr auto argument_alignment = 16u;
    static thread_local std::array<std::byte, argument_buffer_size> argument_buffer;

    auto compute_encoder = encoder.command_buffer()->computeCommandEncoder(
        MTL::DispatchTypeConcurrent);
    if (_name != nullptr) { compute_encoder->setLabel(_name); }
    compute_encoder->setComputePipelineState(_handle.get());

    // encode arguments
    auto copy = [offset = 0u](const void *ptr, size_t size) mutable noexcept {
        offset = align(offset, argument_alignment);
        LUISA_ASSERT(offset + size <= argument_buffer_size,
                     "Argument buffer overflow.");
        std::memcpy(argument_buffer.data() + offset, ptr, size);
        return offset += size;
    };

    auto mtl_usage = [](Usage usage) noexcept {
        auto u = 0u;
        if (to_underlying(usage) & to_underlying(Usage::READ)) { u |= MTL::ResourceUsageRead; }
        if (to_underlying(usage) & to_underlying(Usage::WRITE)) { u |= MTL::ResourceUsageWrite; }
        return u;
    };

    auto index = 0u;
    auto encode = [&](Argument arg) noexcept {
        auto usage = mtl_usage(_argument_usages[index++]);
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                auto buffer = reinterpret_cast<const MetalBuffer *>(arg.buffer.handle);
                auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                copy(&binding, sizeof(binding));
                if (usage != 0u) { compute_encoder->useResource(buffer->handle(), usage); }
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto texture = reinterpret_cast<const MetalTexture *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                copy(&binding, sizeof(binding));
                if (usage != 0u) { compute_encoder->useResource(texture->handle(), usage); }
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<MetalBindlessArray *>(arg.bindless_array.handle);
                auto binding = array->binding();
                copy(&binding, sizeof(binding));
                if (usage != 0u) { array->mark_resource_usages(compute_encoder); }
                break;
            }
            case Argument::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                copy(&binding, sizeof(binding));
                if (usage != 0u) { accel->mark_resource_usages(encoder, compute_encoder); }
                break;
            }
            case Argument::Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                copy(uniform.data(), uniform.size());
                break;
            }
        }
    };
    for (auto arg : _bound_arguments) { encode(arg); }
    for (auto arg : command->arguments()) { encode(arg); }

    // encode dispatch size
    auto dispatch_size = command->dispatch_size();
    auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    auto blocks = (dispatch_size + block_size - 1u) / block_size;
    auto size = copy(&dispatch_size, sizeof(dispatch_size));

    // set argument buffer
    compute_encoder->setBytes(argument_buffer.data(), size, 0u);

    // dispatch
    compute_encoder->dispatchThreadgroups(
        MTL::Size{blocks.x, blocks.y, blocks.z},
        MTL::Size{block_size.x, block_size.y, block_size.z});
    compute_encoder->endEncoding();
}

}// namespace luisa::compute::metal
