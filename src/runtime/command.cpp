//
// Created by Mike Smith on 2021/3/3.
//

#include <core/logging.h>
#include <runtime/command.h>

namespace luisa::compute {

std::span<const Command::Resource> Command::resources() const noexcept {
    return {_resource_slots.data(), _resource_count};
}

inline void Command::_use_resource(
    uint64_t handle, Command::Resource::Tag tag,
    Command::Resource::Usage usage) noexcept {

    if (_resource_count == max_resource_count) {
        LUISA_ERROR_WITH_LOCATION(
            "Number of resources in command exceeded limit {}.",
            max_resource_count);
    }
    if (std::find_if(_resource_slots.cbegin(),
                     _resource_slots.cbegin() + _resource_count,
                     [handle, tag](auto b) noexcept { return b.tag == tag && b.handle == handle; })
        != _resource_slots.cbegin() + _resource_count) {
        LUISA_ERROR_WITH_LOCATION(
            "Aliasing in {} resource with handle {}.",
            tag == Resource::Tag::BUFFER ? "buffer" : "texture",
            handle);
    }
    _resource_slots[_resource_count++] = {handle, tag, usage};
}

void Command::_buffer_read_only(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::BUFFER, Resource::Usage::READ);
}

void Command::_buffer_write_only(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::BUFFER, Resource::Usage::WRITE);
}

void Command::_buffer_read_write(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::BUFFER, Resource::Usage::READ_WRITE);
}

void Command::_texture_read_only(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::TEXTURE, Resource::Usage::READ);
}

void Command::_texture_write_only(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::TEXTURE, Resource::Usage::WRITE);
}

void Command::_texture_read_write(uint64_t handle) noexcept {
    _use_resource(handle, Resource::Tag::TEXTURE, Resource::Usage::READ_WRITE);
}

void KernelLaunchCommand::encode_buffer(
    uint64_t handle,
    size_t offset,
    Command::Resource::Usage usage) noexcept {

    BufferArgument argument{};
    argument.tag = Argument::Tag::BUFFER;
    argument.handle = handle;
    argument.offset = offset;
    argument.usage = usage;
    if (_argument_buffer_size + sizeof(BufferArgument) > _argument_buffer.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode buffer. "
            "Kernel argument buffer exceeded size limit {}.",
            _argument_buffer.size());
    }
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        &argument, sizeof(BufferArgument));
    _use_resource(handle, Resource::Tag::BUFFER, usage);
    _argument_buffer_size += sizeof(BufferArgument);
    _argument_count++;
}

void KernelLaunchCommand::encode_uniform(const void *data, size_t size, size_t alignment) noexcept {
    UniformArgument argument{};
    argument.tag = Argument::Tag::UNIFORM;
    argument.size = size;
    argument.alignment = alignment;
    if (_argument_buffer_size + sizeof(UniformArgument) + size > _argument_buffer.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode argument with size {}. "
            "Kernel argument buffer exceeded size limit {}.",
            size, _argument_buffer.size());
    }
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        &argument, sizeof(UniformArgument));
    _argument_buffer_size += sizeof(UniformArgument);
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        data, size);
    _argument_buffer_size += size;
    _argument_count++;
}

void KernelLaunchCommand::set_launch_size(uint3 dispatch_size, uint3 block_size) noexcept {
    _dispatch_size = dispatch_size;
    _block_size = block_size;
}

namespace detail {

#define LUISA_MAKE_COMMAND_POOL_IMPL(Cmd)       \
    Pool<Cmd> &pool_##Cmd() noexcept {          \
        static Pool<Cmd> pool{Arena::global()}; \
        return pool;                            \
    }
LUISA_MAKE_COMMAND_POOL_IMPL(BufferCopyCommand)
LUISA_MAKE_COMMAND_POOL_IMPL(BufferUploadCommand)
LUISA_MAKE_COMMAND_POOL_IMPL(BufferDownloadCommand)
LUISA_MAKE_COMMAND_POOL_IMPL(KernelLaunchCommand)
#undef LUISA_MAKE_COMMAND_POOL_IMPL

void CommandRecycle::operator()(Command *command) noexcept {
    command->accept(*this);
}

#define LUISA_MAKE_COMMAND_RECYCLE_VISIT(Cmd)                 \
    void CommandRecycle::visit(const Cmd *command) noexcept { \
        using Recycle = typename Pool<Cmd>::ObjectRecycle;    \
        Recycle{&pool_##Cmd()}(                               \
            const_cast<Cmd *>(                                \
                static_cast<const volatile Cmd *>(command))); \
    }
LUISA_MAKE_COMMAND_RECYCLE_VISIT(BufferCopyCommand)
LUISA_MAKE_COMMAND_RECYCLE_VISIT(BufferUploadCommand)
LUISA_MAKE_COMMAND_RECYCLE_VISIT(BufferDownloadCommand)
LUISA_MAKE_COMMAND_RECYCLE_VISIT(KernelLaunchCommand)
#undef LUISA_MAKE_COMMAND_RECYCLE_VISIT

}// namespace detail

}// namespace luisa::compute
