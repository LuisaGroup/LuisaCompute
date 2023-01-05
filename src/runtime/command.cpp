//
// Created by Mike Smith on 2021/3/3.
//

#include <core/logging.h>
#include <runtime/command.h>
#include <raster/raster_scene.h>
#include <runtime/custom_struct.h>

namespace luisa::compute {

std::byte *ShaderDispatchCommandBase::_make_space(size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.resize(offset + size);
    return _argument_buffer.data() + offset;
}

void ShaderDispatchCommandBase::_encode_buffer(Function kernel, uint64_t handle, size_t offset, size_t size) noexcept {
#ifndef NDEBUG
    if (kernel) {
        if (_argument_count >= kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid buffer argument at index {}.",
                _argument_count);
        }
        if (auto t = kernel.arguments()[_argument_count].type();
            !t->is_buffer()) {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got buffer for argument {}.",
                t->description(), _argument_count);
        }
    }
#endif
    _encode_argument(BufferArgument{handle, offset, size});
}

void ShaderDispatchCommandBase::_encode_texture(Function kernel, uint64_t handle, uint32_t level) noexcept {
#ifndef NDEBUG
    if (kernel) {
        if (_argument_count >= kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid texture argument at index {}.",
                _argument_count);
        }
        if (auto t = kernel.arguments()[_argument_count].type();
            !t->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got image for argument {}.",
                t->description(), _argument_count);
        }
    }
#endif
    _encode_argument(TextureArgument{handle, level});
}

void ShaderDispatchCommandBase::_encode_uniform(Function kernel, const void *data, size_t size) noexcept {
#ifndef NDEBUG
    if (kernel) {
        if (_argument_count >= kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid uniform argument at index {}.",
                _argument_count);
        }
        if (auto t = kernel.arguments()[_argument_count].type();
            (!t->is_basic() && !t->is_structure() && !t->is_array()) ||
            t->size() != size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid uniform (size = {}) at index {}, "
                "expected {} (size = {}).",
                size, _argument_count,
                t->description(), t->size());
        }
    }
#endif
    _encode_argument(UniformArgumentHead{size});
    auto p = _make_space(size);
    std::memcpy(p, data, size);
}

void ShaderDispatchCommand::set_dispatch_size(uint3 launch_size) noexcept {
#ifndef NDEBUG
    if (_kernel) {
        if (_argument_count != _kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Not all arguments are encoded (expected {}, got {}).",
                _kernel.arguments().size(), _argument_count);
        }
    }
#endif
    _dispatch_size = launch_size;
}

void ShaderDispatchCommand::set_dispatch_size(IndirectArg indirect_arg) noexcept {
#ifndef NDEBUG
    if (_kernel) {
        if (_argument_count != _kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Not all arguments are encoded (expected {}, got {}).",
                _kernel.arguments().size(), _argument_count);
        }
    }
#endif
    _dispatch_size = indirect_arg;
}

void ShaderDispatchCommandBase::_encode_bindless_array(Function kernel, uint64_t handle) noexcept {
#ifndef NDEBUG
    if (kernel) {
        if (_argument_count >= kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid bindless array argument at index {}.",
                _argument_count);
        }
        if (auto t = kernel.arguments()[_argument_count].type();
            !t->is_bindless_array()) {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got bindless array for argument {}.",
                t->description(), _argument_count);
        }
    }
#endif
    _encode_argument(BindlessArrayArgument{handle});
}

void ShaderDispatchCommandBase::_encode_accel(Function kernel, uint64_t handle) noexcept {
#ifndef NDEBUG
    if (kernel) {
        if (_argument_count >= kernel.arguments().size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid accel argument at index {}.",
                _argument_count);
        }
        if (auto t = kernel.arguments()[_argument_count].type();
            !t->is_accel()) {
            LUISA_ERROR_WITH_LOCATION(
                "Expected {} but got accel for argument {}.",
                t->description(), _argument_count);
        }
    }
#endif
    _encode_argument(AccelArgument{handle});
}

void ShaderDispatchCommandBase::_encode_pending_bindings(Function kernel) noexcept {
    if (kernel) {
        auto bindings = kernel.builder()->argument_bindings();
        while (_argument_count < kernel.arguments().size() &&
               !luisa::holds_alternative<luisa::monostate>(bindings[_argument_count])) {
            luisa::visit(
                [&, arg = kernel.arguments()[_argument_count]]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        _encode_buffer(kernel, binding.handle, binding.offset_bytes, binding.size_bytes);
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        _encode_texture(kernel, binding.handle, binding.level);
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        _encode_bindless_array(kernel, binding.handle);
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        _encode_accel(kernel, binding.handle);
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid argument binding type.");
                    }
                },
                bindings[_argument_count]);
        }
    }
}

ShaderDispatchCommand::ShaderDispatchCommand(uint64_t handle, Function kernel) noexcept
    : ShaderDispatchCommandBase{Command::Tag::EShaderDispatchCommand}, _handle{handle}, _kernel(kernel) {
    _argument_buffer.reserve(256u);
    _encode_pending_bindings(_kernel);
}

void ShaderDispatchCommand::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    _encode_buffer(_kernel, handle, offset, size);
    _encode_pending_bindings(_kernel);
}

void ShaderDispatchCommand::encode_texture(uint64_t handle, uint32_t level) noexcept {
    _encode_texture(_kernel, handle, level);
    _encode_pending_bindings(_kernel);
}

void ShaderDispatchCommand::encode_uniform(const void *data, size_t size) noexcept {
    _encode_uniform(_kernel, data, size);
    _encode_pending_bindings(_kernel);
}

void ShaderDispatchCommand::encode_bindless_array(uint64_t handle) noexcept {
    _encode_bindless_array(_kernel, handle);
    _encode_pending_bindings(_kernel);
}

void ShaderDispatchCommand::encode_accel(uint64_t handle) noexcept {
    _encode_accel(_kernel, handle);
    _encode_pending_bindings(_kernel);
}
void DrawRasterSceneCommand::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    auto kernel = arg_kernel();
    _encode_buffer(kernel, handle, offset, size);
    _encode_pending_bindings(kernel);
}
void DrawRasterSceneCommand::encode_texture(uint64_t handle, uint32_t level) noexcept {
    auto kernel = arg_kernel();
    _encode_texture(kernel, handle, level);
    _encode_pending_bindings(kernel);
}
void DrawRasterSceneCommand::encode_uniform(const void *data, size_t size) noexcept {
    auto kernel = arg_kernel();
    _encode_uniform(kernel, data, size);
    _encode_pending_bindings(kernel);
}
void DrawRasterSceneCommand::encode_bindless_array(uint64_t handle) noexcept {
    auto kernel = arg_kernel();
    _encode_bindless_array(kernel, handle);
    _encode_pending_bindings(kernel);
}
void DrawRasterSceneCommand::encode_accel(uint64_t handle) noexcept {
    auto kernel = arg_kernel();
    _encode_accel(kernel, handle);
    _encode_pending_bindings(kernel);
}

Function DrawRasterSceneCommand::arg_kernel() {
    if (_vertex_func.builder() == nullptr) return Function{};
    if (_argument_count >= _vertex_func.arguments().size()) {
        _default_func = _pixel_func;
        _argument_count = 1;
    }
    return _default_func;
}

DrawRasterSceneCommand::DrawRasterSceneCommand(uint64_t handle,
                                               Function vertex_func,
                                               Function pixel_func) noexcept
    : ShaderDispatchCommandBase{Command::Tag::EDrawRasterSceneCommand},
      _handle{handle}, _vertex_func{vertex_func},
      _pixel_func{pixel_func} { _default_func = _vertex_func; }

namespace detail {
void log_cmd_invalidargs() {
    LUISA_ERROR_WITH_LOCATION("Invalid argument.");
}
}// namespace detail

}// namespace luisa::compute
