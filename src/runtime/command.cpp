//
// Created by Mike Smith on 2021/3/3.
//

#include <core/logging.h>
#include <runtime/command.h>

namespace luisa::compute {

std::span<const Command::Binding> Command::resources() const noexcept {
    return _resource_slots;
}

inline void Command::_use_resource(
    uint64_t handle, Command::Binding::Tag tag, Usage usage) noexcept {

    if (auto iter = std::find_if(
            _resource_slots.cbegin(),
            _resource_slots.cend(),
            [handle, tag, usage](auto b) noexcept {
                return b.tag == tag && b.handle == handle;
            });
        iter == _resource_slots.cend()) [[likely]] {
        _resource_slots.emplace_back(handle, tag, usage);
    } else [[unlikely]] {
        if ((to_underlying(iter->usage) & to_underlying(Usage::WRITE)) != 0u ||
            (to_underlying(usage) & to_underlying(Usage::WRITE)) != 0u) {
            auto t = [tag] {
                using namespace std::string_view_literals;
                switch (tag) {
                    case Binding::Tag::BUFFER: return "buffer"sv;
                    case Binding::Tag::TEXTURE: return "texture"sv;
                    case Binding::Tag::BINDLESS_ARRAY: return "bindless array"sv;
                    case Binding::Tag::ACCEL: return "accel"sv;
                    default: return "unknown"sv;
                }
            }();
            LUISA_WARNING_WITH_LOCATION(
                "Aliasing in {} with handle {}.",
                t, handle);
        }
    }
}

void Command::_buffer_read_only(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::BUFFER, Usage::READ);
}

void Command::_buffer_write_only(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::BUFFER, Usage::WRITE);
}

void Command::_buffer_read_write(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::BUFFER, Usage::READ_WRITE);
}

void Command::_texture_read_only(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::TEXTURE, Usage::READ);
}

void Command::_texture_write_only(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::TEXTURE, Usage::WRITE);
}

void Command::_texture_read_write(uint64_t handle) noexcept {
    _use_resource(handle, Binding::Tag::TEXTURE, Usage::READ_WRITE);
}

void Command::recycle() {
    _resource_slots.clear();
    _recycle();
}

void ShaderDispatchCommand::encode_buffer(
    uint32_t variable_uid,
    uint64_t handle,
    size_t offset,
    Usage usage) noexcept {

    if (_argument_buffer_size + sizeof(BufferArgument) > _argument_buffer.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode buffer. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer.size());
    }

    BufferArgument argument{variable_uid, handle, offset};
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        &argument, sizeof(BufferArgument));
    _use_resource(handle, Binding::Tag::BUFFER, usage);
    _argument_buffer_size += sizeof(BufferArgument);
    _argument_count++;
}

void ShaderDispatchCommand::encode_texture(
    uint32_t variable_uid,
    uint64_t handle,
    uint32_t level,
    Usage usage) noexcept {

    if (_argument_buffer_size + sizeof(TextureArgument) > _argument_buffer.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode texture. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer.size());
    }

    TextureArgument argument{variable_uid, handle, level};
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        &argument, sizeof(TextureArgument));
    _use_resource(handle, Binding::Tag::TEXTURE, usage);
    _argument_buffer_size += sizeof(TextureArgument);
    _argument_count++;
}

void ShaderDispatchCommand::encode_uniform(
    uint32_t variable_uid,
    const void *data,
    size_t size,
    size_t alignment) noexcept {

    if (_argument_buffer_size + sizeof(UniformArgument) + size > _argument_buffer.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode argument with size {}. "
            "Shader argument buffer exceeded size limit {}.",
            size, _argument_buffer.size());
    }

    UniformArgument argument{variable_uid, size, alignment};
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

void ShaderDispatchCommand::set_dispatch_size(uint3 launch_size) noexcept {
    _dispatch_size[0] = launch_size.x;
    _dispatch_size[1] = launch_size.y;
    _dispatch_size[2] = launch_size.z;
}

ShaderDispatchCommand::ShaderDispatchCommand(uint64_t handle, Function kernel) noexcept
    : _handle{handle},
      _kernel{kernel} {}

void ShaderDispatchCommand::encode_bindless_array(uint32_t variable_uid, uint64_t handle) noexcept {
    if (_argument_buffer_size + sizeof(BindlessArrayArgument) > _argument_buffer.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode texture heap. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer.size());
    }
    BindlessArrayArgument argument{variable_uid, handle};
    std::memcpy(
        _argument_buffer.data() + _argument_buffer_size,
        &argument, sizeof(BindlessArrayArgument));
    _use_resource(handle, Binding::Tag::BINDLESS_ARRAY, Usage::READ);
    _argument_buffer_size += sizeof(BindlessArrayArgument);
    _argument_count++;
}

void ShaderDispatchCommand::encode_accel(uint32_t variable_uid, uint64_t handle) noexcept {
    constexpr auto size = sizeof(AccelArgument);
    if (_argument_buffer_size + size > _argument_buffer.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode accel. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer.size());
    }
    AccelArgument argument{variable_uid, handle};
    std::memcpy(_argument_buffer.data() + _argument_buffer_size, &argument, size);
    _use_resource(handle, Binding::Tag::ACCEL, Usage::READ);
    _argument_buffer_size += size;
    _argument_count++;
}

namespace detail {

#define LUISA_MAKE_COMMAND_POOL_IMPL(Cmd) \
    Pool<Cmd> &pool_##Cmd() noexcept {    \
        static Pool<Cmd> pool;            \
        return pool;                      \
    }
LUISA_MAP(LUISA_MAKE_COMMAND_POOL_IMPL, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_POOL_IMPL

}// namespace detail

}// namespace luisa::compute
