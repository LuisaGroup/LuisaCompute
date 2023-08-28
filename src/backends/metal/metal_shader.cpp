#include <luisa/core/logging.h>
#include "metal_device.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_accel.h"
#include "metal_bindless_array.h"
#include "metal_command_encoder.h"
#include "metal_shader.h"

namespace luisa::compute::metal {

MetalShader::MetalShader(MetalDevice *device,
                         MetalShaderHandle handle,
                         luisa::vector<Usage> argument_usages,
                         luisa::vector<Argument> bound_arguments,
                         uint3 block_size) noexcept
    : _handle{std::move(handle)},
      _argument_usages{std::move(argument_usages)},
      _bound_arguments{std::move(bound_arguments)},
      _block_size{block_size.x, block_size.y, block_size.z},
      _prepare_indirect{device->builtin_prepare_indirect_dispatches()} {}

MetalShader::~MetalShader() noexcept {
    if (_name) { _name->release(); }
    if (_indirect_name) { _indirect_name->release(); }
}

Usage MetalShader::argument_usage(uint index) const noexcept {
#ifndef NDEBUG
    LUISA_ASSERT(index < _argument_usages.size(),
                 "Argument index out of range.");
#endif
    return _argument_usages[index];
}

void MetalShader::set_name(luisa::string_view name) noexcept {
    std::scoped_lock lock{_name_mutex};
    if (_name) {
        _name->release();
        _name = nullptr;
    }
    if (_indirect_name) {
        _indirect_name->release();
        _indirect_name = nullptr;
    }
    if (!name.empty()) {
        _name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        auto indirect = luisa::format("{} (indirect)", name);
        _indirect_name = NS::String::alloc()->init(
            const_cast<char *>(indirect.data()), indirect.size(),
            NS::UTF8StringEncoding, false);
    }
}

void MetalShader::launch(MetalCommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept {

    static constexpr auto argument_buffer_size = 65536u;
    static constexpr auto argument_alignment = 16u;
    static thread_local std::array<std::byte, argument_buffer_size> argument_buffer;

    // encode arguments
    auto argument_offset = static_cast<size_t>(0u);
    auto copy = [&argument_offset](const void *ptr, size_t size) mutable noexcept {
        argument_offset = luisa::align(argument_offset, argument_alignment);
        LUISA_ASSERT(argument_offset + size <= argument_buffer_size,
                     "Argument buffer overflow.");
        std::memcpy(argument_buffer.data() + argument_offset, ptr, size);
        return argument_offset += size;
    };

    auto encode = [&](Argument arg) mutable noexcept {
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                if (reinterpret_cast<const MetalBufferBase *>(arg.buffer.handle)->is_indirect()) {
                    auto buffer = reinterpret_cast<const MetalIndirectDispatchBuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    copy(&binding, sizeof(binding));
                } else {
                    auto buffer = reinterpret_cast<const MetalBuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    copy(&binding, sizeof(binding));
                }
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto texture = reinterpret_cast<const MetalTexture *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                copy(&binding, sizeof(binding));
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<MetalBindlessArray *>(arg.bindless_array.handle);
                auto binding = array->binding();
                copy(&binding, sizeof(binding));
                break;
            }
            case Argument::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                copy(&binding, sizeof(binding));
                break;
            }
            case Argument::Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                copy(uniform.data(), uniform.size());
                break;
            }
        }
    };

    auto mtl_usage = [](Usage usage) noexcept {
        auto u = 0u;
        if (to_underlying(usage) & to_underlying(Usage::READ)) { u |= MTL::ResourceUsageRead; }
        if (to_underlying(usage) & to_underlying(Usage::WRITE)) { u |= MTL::ResourceUsageWrite; }
        return u;
    };

    auto mark_usage = [&, index = 0u](MTL::ComputeCommandEncoder *compute_encoder, Argument arg) mutable noexcept {
        auto usage = mtl_usage(_argument_usages[index++]);
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                if (reinterpret_cast<const MetalBufferBase *>(arg.buffer.handle)->is_indirect()) {
                    auto buffer = reinterpret_cast<const MetalIndirectDispatchBuffer *>(arg.buffer.handle);
                    if (usage != 0u) { compute_encoder->useResource(buffer->dispatch_buffer(), usage); }
                } else {
                    auto buffer = reinterpret_cast<const MetalBuffer *>(arg.buffer.handle);
                    if (usage != 0u) { compute_encoder->useResource(buffer->handle(), usage); }
                }
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto texture = reinterpret_cast<const MetalTexture *>(arg.texture.handle);
                if (usage != 0u) { compute_encoder->useResource(texture->handle(arg.texture.level), usage); }
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<MetalBindlessArray *>(arg.bindless_array.handle);
                if (usage != 0u) { array->mark_resource_usages(compute_encoder); }
                break;
            }
            case Argument::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(arg.accel.handle);
                if (usage != 0u) { accel->mark_resource_usages(encoder, compute_encoder, usage); }
                break;
            }
            default: break;
        }
    };

    if (command->is_indirect()) {

        auto indirect = command->indirect_dispatch();
        auto indirect_buffer = reinterpret_cast<MetalIndirectDispatchBuffer *>(indirect.handle);
        auto indirect_binding = indirect_buffer->binding(indirect.offset, indirect.max_dispatch_size);

        for (auto arg : _bound_arguments) { encode(arg); }
        for (auto arg : command->arguments()) { encode(arg); }
        auto argument_size = luisa::align(argument_offset, argument_alignment);

        // update indirect command buffer
        {
            auto command_encoder = encoder.command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent);
            {
                std::scoped_lock lock{_name_mutex};
                if (_indirect_name) { command_encoder->setLabel(_indirect_name); }
            }
            struct ICB {
                uint64_t dispatch_buffer;
                uint command_buffer_offset;
                uint command_buffer_capacity;
                MTL::ResourceID command_buffer;
                MTL::ResourceID pipeline_state;
            };
            ICB icb{.dispatch_buffer = indirect_binding.address,
                    .command_buffer_offset = indirect_binding.offset,
                    .command_buffer_capacity = indirect_binding.capacity,
                    .command_buffer = indirect_buffer->command_buffer()->gpuResourceID(),
                    .pipeline_state = _handle.indirect_entry->gpuResourceID()};
            command_encoder->setComputePipelineState(_prepare_indirect);
            command_encoder->setBytes(&icb, sizeof(icb), 0u);
            command_encoder->useResource(indirect_buffer->dispatch_buffer(), MTL::ResourceUsageRead);
            command_encoder->useResource(indirect_buffer->command_buffer(), MTL::ResourceUsageWrite);
            command_encoder->setBytes(argument_buffer.data(), argument_size, 1u);
            constexpr auto block_size = MetalDevice::prepare_indirect_dispatches_block_size;
            auto block_count = (indirect_binding.capacity - indirect_binding.offset + block_size - 1u) / block_size;
            command_encoder->dispatchThreadgroups(MTL::Size{block_count, 1u, 1u}, MTL::Size{block_size, 1u, 1u});
            command_encoder->endEncoding();

            // TODO: is this necessary?
            // auto blit_encoder = encoder.command_buffer()->blitCommandEncoder();
            // blit_encoder->optimizeIndirectCommandBuffer(indirect_buffer->command_buffer(),
            //                                             NS::Range{0u, indirect_buffer->capacity()});
            // blit_encoder->endEncoding();
        }

        // dispatch indirect
        auto compute_encoder = encoder.command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        {
            std::scoped_lock lock{_name_mutex};
            if (_name) { compute_encoder->setLabel(_name); }
        }
        compute_encoder->executeCommandsInBuffer(indirect_buffer->command_buffer(),
                                                 NS::Range::Make(indirect_binding.offset,
                                                                 indirect_binding.capacity - indirect_binding.offset));
        for (auto arg : _bound_arguments) { mark_usage(compute_encoder, arg); }
        for (auto arg : command->arguments()) { mark_usage(compute_encoder, arg); }
        compute_encoder->endEncoding();

    } else {

        auto compute_encoder = encoder.command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        {
            std::scoped_lock lock{_name_mutex};
            if (_name) { compute_encoder->setLabel(_name); }
        }
        compute_encoder->setComputePipelineState(_handle.entry.get());

        for (auto arg : _bound_arguments) {
            encode(arg);
            mark_usage(compute_encoder, arg);
        }
        for (auto arg : command->arguments()) {
            encode(arg);
            mark_usage(compute_encoder, arg);
        }

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
}

}// namespace luisa::compute::metal

