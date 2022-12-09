//
// Created by Mike Smith on 2021/3/3.
//

#include <core/logging.h>
#include <runtime/command.h>
#include <ast/variable.h>
#include <ast/function_builder.h>

namespace luisa::compute {

inline void ShaderDispatchCommand::_encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    if (_argument_count >= _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer argument at index {}.",
            _argument_count);
    }
    if (_argument_buffer_size + sizeof(BufferArgument) > _argument_buffer->size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode buffer. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer->size());
    }
    if (auto t = _kernel.arguments()[_argument_count].type();
        !t->is_buffer()) {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got buffer for argument {}.",
            t->description(), _argument_count);
    }
    auto variable_uid = _kernel.arguments()[_argument_count].uid();
    auto usage = _kernel.variable_usage(variable_uid);
    BufferArgument argument{variable_uid, handle, offset, size};
    std::memcpy(
        _argument_buffer->data() + _argument_buffer_size,
        &argument, sizeof(BufferArgument));
    _argument_buffer_size += sizeof(BufferArgument);
    _argument_count++;
}

inline void ShaderDispatchCommand::_encode_texture(uint64_t handle, uint32_t level) noexcept {
    if (_argument_count >= _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture argument at index {}.",
            _argument_count);
    }
    if (_argument_buffer_size + sizeof(TextureArgument) > _argument_buffer->size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode texture. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer->size());
    }
    if (auto t = _kernel.arguments()[_argument_count].type();
        !t->is_texture()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got image for argument {}.",
            t->description(), _argument_count);
    }
    auto variable_uid = _kernel.arguments()[_argument_count].uid();
    auto usage = _kernel.variable_usage(variable_uid);
    TextureArgument argument{variable_uid, handle, level};
    std::memcpy(
        _argument_buffer->data() + _argument_buffer_size,
        &argument, sizeof(TextureArgument));
    _argument_buffer_size += sizeof(TextureArgument);
    _argument_count++;
}

inline void ShaderDispatchCommand::_encode_uniform(const void *data, size_t size) noexcept {
    if (_argument_count >= _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid uniform argument at index {}.",
            _argument_count);
    }
    if (_argument_buffer_size + sizeof(UniformArgument) + size > _argument_buffer->size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode argument with size {}. "
            "Shader argument buffer exceeded size limit {}.",
            size, _argument_buffer->size());
    }
    if (auto t = _kernel.arguments()[_argument_count].type();
        (!t->is_basic() && !t->is_structure() && !t->is_array()) ||
        t->size() != size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid uniform (size = {}) at index {}, "
            "expected {} (size = {}).",
            size, _argument_count,
            t->description(), t->size());
    }
    auto variable_uid = _kernel.arguments()[_argument_count].uid();
    auto arg_ptr = _argument_buffer->data() + _argument_buffer_size;
    auto data_ptr = arg_ptr + sizeof(UniformArgument);
    UniformArgument argument{variable_uid, data_ptr, size};
    std::memcpy(arg_ptr, &argument, sizeof(UniformArgument));
    std::memcpy(data_ptr, data, size);
    _argument_buffer_size += sizeof(UniformArgument) + size;
    _argument_count++;
}

void ShaderDispatchCommand::set_dispatch_size(uint3 launch_size) noexcept {
    if (_argument_count != _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Not all arguments are encoded (expected {}, got {}).",
            _kernel.arguments().size(), _argument_count);
    }
    _dispatch_size[0] = launch_size.x;
    _dispatch_size[1] = launch_size.y;
    _dispatch_size[2] = launch_size.z;
}

ShaderDispatchCommand::ShaderDispatchCommand(uint64_t handle, Function kernel) noexcept
    : Command{Command::Tag::EShaderDispatchCommand}, _handle{handle}, _kernel{kernel},
      _argument_buffer{luisa::make_unique<ArgumentBuffer>()} { _encode_pending_bindings(); }

inline void ShaderDispatchCommand::_encode_bindless_array(uint64_t handle) noexcept {
    if (_argument_count >= _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid bindless array argument at index {}.",
            _argument_count);
    }
    if (_argument_buffer_size + sizeof(BindlessArrayArgument) > _argument_buffer->size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode bindless array. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer->size());
    }
    if (auto t = _kernel.arguments()[_argument_count].type();
        !t->is_bindless_array()) {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got bindless array for argument {}.",
            t->description(), _argument_count);
    }
    auto v = _kernel.arguments()[_argument_count].uid();
    BindlessArrayArgument argument{v, handle};
    std::memcpy(
        _argument_buffer->data() + _argument_buffer_size,
        &argument, sizeof(BindlessArrayArgument));
    _argument_buffer_size += sizeof(BindlessArrayArgument);
    _argument_count++;
}

inline void ShaderDispatchCommand::_encode_accel(uint64_t handle) noexcept {
    if (_argument_count >= _kernel.arguments().size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid accel argument at index {}.",
            _argument_count);
    }
    constexpr auto size = sizeof(AccelArgument);
    if (_argument_buffer_size + size > _argument_buffer->size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to encode accel. "
            "Shader argument buffer exceeded size limit {}.",
            _argument_buffer->size());
    }
    if (auto t = _kernel.arguments()[_argument_count].type();
        !t->is_accel()) {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got accel for argument {}.",
            t->description(), _argument_count);
    }
    auto v = _kernel.arguments()[_argument_count].uid();
    AccelArgument argument{v, handle};
    std::memcpy(_argument_buffer->data() + _argument_buffer_size, &argument, size);
    _argument_buffer_size += size;
    _argument_count++;
}

inline void ShaderDispatchCommand::_encode_pending_bindings() noexcept {
    auto bindings = _kernel.builder()->argument_bindings();
    while (_argument_count < _kernel.arguments().size() &&
           !luisa::holds_alternative<luisa::monostate>(bindings[_argument_count])) {
        luisa::visit(
            [&, arg = _kernel.arguments()[_argument_count]]<typename T>(T binding) noexcept {
                if constexpr (std::is_same_v<T, detail::FunctionBuilder::BufferBinding>) {
                    _encode_buffer(binding.handle, binding.offset_bytes, binding.size_bytes);
                } else if constexpr (std::is_same_v<T, detail::FunctionBuilder::TextureBinding>) {
                    _encode_texture(binding.handle, binding.level);
                } else if constexpr (std::is_same_v<T, detail::FunctionBuilder::BindlessArrayBinding>) {
                    _encode_bindless_array(binding.handle);
                } else if constexpr (std::is_same_v<T, detail::FunctionBuilder::AccelBinding>) {
                    _encode_accel(binding.handle);
                } else {
                    LUISA_ERROR_WITH_LOCATION("Invalid argument binding type.");
                }
            },
            bindings[_argument_count]);
    }
}

void ShaderDispatchCommand::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    _encode_buffer(handle, offset, size);
    _encode_pending_bindings();
}

void ShaderDispatchCommand::encode_texture(uint64_t handle, uint32_t level) noexcept {
    _encode_texture(handle, level);
    _encode_pending_bindings();
}

void ShaderDispatchCommand::encode_uniform(const void *data, size_t size) noexcept {
    _encode_uniform(data, size);
    _encode_pending_bindings();
}

void ShaderDispatchCommand::encode_bindless_array(uint64_t handle) noexcept {
    _encode_bindless_array(handle);
    _encode_pending_bindings();
}

void ShaderDispatchCommand::encode_accel(uint64_t handle) noexcept {
    _encode_accel(handle);
    _encode_pending_bindings();
}// namespace detail

void AccelBuildCommand::Modification::set_transform(float4x4 m) noexcept {
    affine[0] = m[0][0];
    affine[1] = m[1][0];
    affine[2] = m[2][0];
    affine[3] = m[3][0];
    affine[4] = m[0][1];
    affine[5] = m[1][1];
    affine[6] = m[2][1];
    affine[7] = m[3][1];
    affine[8] = m[0][2];
    affine[9] = m[1][2];
    affine[10] = m[2][2];
    affine[11] = m[3][2];
    flags |= flag_transform;
}

void AccelBuildCommand::Modification::set_visibility(bool vis) noexcept {
    flags &= ~flag_visibility;// clear old visibility flags
    flags |= vis ? flag_visibility_on : flag_visibility_off;
}

void AccelBuildCommand::Modification::set_mesh(uint64_t handle) noexcept {
    mesh = handle;
    flags |= flag_mesh;
}

}// namespace luisa::compute
