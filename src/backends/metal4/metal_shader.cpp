#include <luisa/core/logging.h>
#include <algorithm>
#include <cstdlib>
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
                         luisa::vector<uint8_t> argument_sampled,
                         luisa::vector<Argument> bound_arguments,
                         luisa::span<const std::pair<luisa::string, luisa::string>> print_formats,
                         uint3 block_size,
                         uint64_t source_checksum,
                         size_t source_size_bytes,
                         size_t source_line_count,
                         double codegen_ms,
                         double compile_ms) noexcept
    : _handle{std::move(handle)},
      _argument_usages{std::move(argument_usages)},
      _argument_sampled{std::move(argument_sampled)},
      _bound_arguments{std::move(bound_arguments)},
      _block_size{block_size.x, block_size.y, block_size.z},
      _source_checksum{source_checksum},
      _source_size_bytes{source_size_bytes},
      _source_line_count{source_line_count},
      _codegen_ms{codegen_ms},
      _compile_ms{compile_ms},
      _prepare_indirect{device->builtin_prepare_indirect_dispatches()} {
    static_cast<void>(print_formats);
}

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
        auto name_copy = luisa::string{name};
        _name = NS::String::alloc()->init(
            name_copy.c_str(), NS::UTF8StringEncoding);
        auto indirect = luisa::format("{} (indirect)", name);
        _indirect_name = NS::String::alloc()->init(
            indirect.c_str(), NS::UTF8StringEncoding);
        if (std::getenv("LUISA_METAL_SHADER_INFO") != nullptr) {
            auto *pipeline = _handle.entry.get();
            LUISA_INFO(
                "Metal shader info: stage='{}' cache_key='metal_kernel_{:016x}' "
                "source_bytes={} source_lines={} codegen_ms={:.3f} "
                "compile_ms={:.3f} block={}x{}x{} thread_width={} "
                "max_threads_per_threadgroup={} static_threadgroup_bytes={}.",
                name, _source_checksum, _source_size_bytes,
                _source_line_count, _codegen_ms, _compile_ms,
                _block_size[0], _block_size[1], _block_size[2],
                pipeline->threadExecutionWidth(),
                pipeline->maxTotalThreadsPerThreadgroup(),
                pipeline->staticThreadgroupMemoryLength());
        }
    }
}

void MetalShader::launch(MetalCommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept {

    static const auto profile_command_buffer =
        std::getenv("LUISA_METAL_COMMAND_BUFFER_PROFILE") != nullptr;
    if (profile_command_buffer) {
        std::scoped_lock lock{_name_mutex};
        if (_name) { encoder.command_buffer()->setLabel(_name); }
    }

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

    auto encode = [&](Argument arg, bool split_sampled_texture) mutable noexcept {
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
                auto texture = reinterpret_cast<const MetalTextureBase *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                copy(&binding, sizeof(binding));
                if (split_sampled_texture) { copy(&binding, sizeof(binding)); }
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

    auto split_sampled_texture = [&](size_t index, Argument arg) noexcept {
        return arg.tag == Argument::Tag::TEXTURE &&
               _argument_sampled[index] &&
               (to_underlying(_argument_usages[index]) &
                to_underlying(Usage::WRITE)) != 0u;
    };

    auto mark_usage = [&](Argument arg, size_t index) noexcept {
        auto argument_usage = _argument_usages[index];
        switch (arg.tag) {
            case Argument::Tag::BUFFER: {
                if (reinterpret_cast<const MetalBufferBase *>(arg.buffer.handle)->is_indirect()) {
                    auto buffer = reinterpret_cast<const MetalIndirectDispatchBuffer *>(arg.buffer.handle);
                    encoder.use_resource(buffer->dispatch_buffer());
                    encoder.use_resource(buffer->command_buffer());
                } else {
                    auto buffer = reinterpret_cast<const MetalBuffer *>(arg.buffer.handle);
                    encoder.use_resource(buffer->handle());
                }
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto texture = reinterpret_cast<const MetalTextureBase *>(arg.texture.handle);
                LUISA_ASSERT(
                    texture->kind() != MetalTextureBase::Kind::DEPTH ||
                        (to_underlying(argument_usage) &
                         to_underlying(Usage::WRITE)) == 0u,
                    "Metal depth textures cannot be written by compute shaders.");
                encoder.use_resource(texture->handle(arg.texture.level));
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<MetalBindlessArray *>(arg.bindless_array.handle);
                array->mark_resource_usages(encoder);
                break;
            }
            case Argument::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(arg.accel.handle);
                accel->mark_resource_usages(encoder);
                break;
            }
            default: break;
        }
    };

    auto warn_empty_launch = [&]() noexcept {
#ifndef NDEBUG
        LUISA_WARNING_WITH_LOCATION(
            "Empty launch detected. "
            "This might be caused by a shader dispatch command with all dispatch sizes set to zero. "
            "The command will be ignored.");
#endif
    };

    auto stage_argument_block = [&](size_t size) noexcept {
        return encoder.upload(argument_buffer.data(), size);
    };

    if (command->is_indirect()) {

        auto indirect = command->indirect_dispatch();
        if (indirect.max_dispatch_size == 0u) {
            warn_empty_launch();
            return;
        }

        auto indirect_buffer = reinterpret_cast<MetalIndirectDispatchBuffer *>(indirect.handle);
        auto indirect_binding = indirect_buffer->binding(indirect.offset, indirect.max_dispatch_size);

        auto argument_index = 0u;
        for (auto arg : _bound_arguments) {
            encode(arg, split_sampled_texture(argument_index++, arg));
        }
        for (auto arg : command->arguments()) {
            encode(arg, split_sampled_texture(argument_index++, arg));
        }
        auto argument_size = std::max(
            luisa::align(argument_offset, argument_alignment),
            static_cast<size_t>(argument_alignment));

        auto root_argument_address = stage_argument_block(argument_size);

        // update indirect command buffer
        {
            auto command_encoder = encoder.compute_encoder();
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
            auto table = encoder.argument_table(2u);
            table->setAddress(encoder.upload(&icb, sizeof(icb)), 0u);
            table->setAddress(root_argument_address, 1u);
            command_encoder->setArgumentTable(table);
            encoder.use_resource(_prepare_indirect);
            encoder.use_resource(indirect_buffer->dispatch_buffer());
            encoder.use_resource(indirect_buffer->command_buffer());
            constexpr auto block_size = MetalDevice::prepare_indirect_dispatches_block_size;
            auto block_count = (indirect_binding.capacity - indirect_binding.offset + block_size - 1u) / block_size;
            command_encoder->dispatchThreadgroups(MTL::Size{block_count, 1u, 1u}, MTL::Size{block_size, 1u, 1u});
            command_encoder->endEncoding();
        }

        // dispatch indirect
        auto compute_encoder = encoder.compute_encoder();
        {
            std::scoped_lock lock{_name_mutex};
            if (_name) { compute_encoder->setLabel(_name); }
        }
        compute_encoder->executeCommandsInBuffer(indirect_buffer->command_buffer(),
                                                 NS::Range::Make(indirect_binding.offset,
                                                                 indirect_binding.capacity - indirect_binding.offset));
        argument_index = 0u;
        encoder.use_resource(indirect_buffer->command_buffer());
        encoder.use_resource(_handle.indirect_entry.get());
        for (auto arg : _bound_arguments) { mark_usage(arg, argument_index++); }
        for (auto arg : command->arguments()) { mark_usage(arg, argument_index++); }
        compute_encoder->endEncoding();

    } else {

        auto single_dispatch_size = make_uint3(0u);
        luisa::span<const uint3> dispatch_sizes;
        if (command->is_multiple_dispatch()) {
            dispatch_sizes = command->dispatch_sizes();
            if (std::all_of(dispatch_sizes.begin(), dispatch_sizes.end(),
                            [](auto size) noexcept { return any(size == make_uint3(0u)); })) {
                warn_empty_launch();
                return;
            }
        } else {
            single_dispatch_size = command->dispatch_size();
            dispatch_sizes = luisa::span{&single_dispatch_size, 1u};
            if (any(single_dispatch_size == make_uint3(0u))) {
                warn_empty_launch();
                return;
            }
        }

        encoder.use_resource(_handle.entry.get());

        auto argument_index = 0u;
        for (auto arg : _bound_arguments) {
            encode(arg, split_sampled_texture(argument_index, arg));
            mark_usage(arg, argument_index++);
        }
        for (auto arg : command->arguments()) {
            encode(arg, split_sampled_texture(argument_index, arg));
            mark_usage(arg, argument_index++);
        }

        auto size = std::max(
            luisa::align(argument_offset, argument_alignment),
            static_cast<size_t>(argument_alignment));
        auto root_argument_address = stage_argument_block(size);

        for (auto dispatch_size : dispatch_sizes) {
            if (any(dispatch_size == make_uint3(0u))) { continue; }
            auto compute_encoder = encoder.compute_encoder();
            {
                std::scoped_lock lock{_name_mutex};
                if (_name) { compute_encoder->setLabel(_name); }
            }
            compute_encoder->setComputePipelineState(_handle.entry.get());
            auto table = encoder.argument_table(2u);
            table->setAddress(root_argument_address, 0u);
            table->setAddress(
                encoder.upload(&dispatch_size, sizeof(dispatch_size)), 1u);
            compute_encoder->setArgumentTable(table);
            auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
            auto blocks = (dispatch_size + block_size - 1u) / block_size;
            compute_encoder->dispatchThreadgroups(
                MTL::Size{blocks.x, blocks.y, blocks.z},
                MTL::Size{block_size.x, block_size.y, block_size.z});
            compute_encoder->endEncoding();
        }
    }
}

}// namespace luisa::compute::metal
