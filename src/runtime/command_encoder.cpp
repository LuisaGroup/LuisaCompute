#include <core/logging.h>
#include <runtime/command.h>
#include <runtime/command_encoder.h>
#include <runtime/raster/raster_scene.h>

namespace luisa::compute {

std::byte *ShaderDispatchCmdEncoder::_make_space(size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.resize(offset + size);
    return _argument_buffer.data() + offset;
}

ShaderDispatchCmdEncoder::ShaderDispatchCmdEncoder(size_t arg_count) : _argument_count(arg_count) {
    size_t size = arg_count * sizeof(Argument);
    _argument_buffer.reserve(size + 256);
    _argument_buffer.push_back_uninitialized(size);
}

ShaderDispatchCmdEncoder::Argument &ShaderDispatchCmdEncoder::create_arg() {
    auto idx = _argument_idx;
    _argument_idx++;
    return *std::launder(reinterpret_cast<Argument *>(_argument_buffer.data()) + idx);
}

void ShaderDispatchCmdEncoder::_encode_buffer(Function kernel, uint64_t handle, size_t offset, size_t size) noexcept {
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
    auto &&arg = create_arg();
    arg.tag = Argument::Tag::BUFFER;
    arg.buffer = ShaderDispatchCommandBase::Argument::Buffer{handle, offset, size};
}

void ShaderDispatchCmdEncoder::_encode_texture(Function kernel, uint64_t handle, uint32_t level) noexcept {
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
    auto &&arg = create_arg();
    arg.tag = Argument::Tag::TEXTURE;
    arg.texture = ShaderDispatchCommandBase::Argument::Texture{handle, level};
}

void ShaderDispatchCmdEncoder::_encode_uniform(Function kernel, const void *data, size_t size) noexcept {
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
    auto offset = _argument_buffer.size();
    _argument_buffer.push_back_uninitialized(size);
    std::memcpy(_argument_buffer.data() + offset, data, size);
    auto &&arg = create_arg();
    arg.tag = Argument::Tag::UNIFORM;
    arg.uniform.offset = offset;
    arg.uniform.size = size;
}

void ComputeDispatchCmdEncoder::set_dispatch_size(uint3 launch_size) noexcept {
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

void ComputeDispatchCmdEncoder::set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept {
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

void ShaderDispatchCmdEncoder::_encode_bindless_array(Function kernel, uint64_t handle) noexcept {
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
    auto &&arg = create_arg();
    arg.tag = Argument::Tag::BINDLESS_ARRAY;
    arg.bindless_array = Argument::BindlessArray{handle};
}

void ShaderDispatchCmdEncoder::_encode_accel(Function kernel, uint64_t handle) noexcept {
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
    auto &&arg = create_arg();
    arg.tag = Argument::Tag::ACCEL;
    arg.accel = Argument::Accel{handle};
}

void ShaderDispatchCmdEncoder::_encode_pending_bindings(Function kernel) noexcept {
    if (kernel) {
        auto bindings = kernel.builder()->argument_bindings();
        while (_argument_idx < _argument_count &&
               !luisa::holds_alternative<luisa::monostate>(bindings[_argument_idx])) {
            luisa::visit(
                [&, arg = kernel.arguments()[_argument_idx]]<typename T>(T binding) noexcept {
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
                bindings[_argument_idx]);
        }
    }
}

ComputeDispatchCmdEncoder::ComputeDispatchCmdEncoder(size_t arg_size, uint64_t handle, Function kernel) noexcept
    : ShaderDispatchCmdEncoder{arg_size}, _handle{handle}, _kernel{kernel} {
    _argument_buffer.reserve(256u);
    _encode_pending_bindings(_kernel);
}

void ComputeDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    _encode_buffer(_kernel, handle, offset, size);
    _encode_pending_bindings(_kernel);
}

void ComputeDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    _encode_texture(_kernel, handle, level);
    _encode_pending_bindings(_kernel);
}

void ComputeDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    _encode_uniform(_kernel, data, size);
    _encode_pending_bindings(_kernel);
}

void ComputeDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    _encode_bindless_array(_kernel, handle);
    _encode_pending_bindings(_kernel);
}

void ComputeDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    _encode_accel(_kernel, handle);
    _encode_pending_bindings(_kernel);
}

void RasterDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    auto kernel = arg_kernel();
    _encode_buffer(kernel, handle, offset, size);
    _encode_pending_bindings(kernel);
}

void RasterDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    auto kernel = arg_kernel();
    _encode_texture(kernel, handle, level);
    _encode_pending_bindings(kernel);
}

void RasterDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    auto kernel = arg_kernel();
    _encode_uniform(kernel, data, size);
    _encode_pending_bindings(kernel);
}

void RasterDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    auto kernel = arg_kernel();
    _encode_bindless_array(kernel, handle);
    _encode_pending_bindings(kernel);
}

void RasterDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    auto kernel = arg_kernel();
    _encode_accel(kernel, handle);
    _encode_pending_bindings(kernel);
}

Function RasterDispatchCmdEncoder::arg_kernel() {
    if (_vertex_func.builder() == nullptr) return Function{};
    if (_argument_count >= _vertex_func.arguments().size()) {
        _default_func = _pixel_func;
        _argument_count = 1;
    }
    return _default_func;
}

RasterDispatchCmdEncoder::~RasterDispatchCmdEncoder() noexcept = default;
RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(RasterDispatchCmdEncoder &&) noexcept = default;
RasterDispatchCmdEncoder &RasterDispatchCmdEncoder::operator=(RasterDispatchCmdEncoder &&) noexcept = default;

RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(
    size_t arg_size,
    uint64_t handle,
    Function vertex_func,
    Function pixel_func) noexcept
    : ShaderDispatchCmdEncoder{arg_size}, _handle{handle}, _vertex_func{vertex_func},
      _pixel_func{pixel_func} { _default_func = _vertex_func; }

luisa::unique_ptr<ShaderDispatchCommand> ComputeDispatchCmdEncoder::build() &&noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}, Actual argument count {}.", _argument_count, _argument_idx);
    }
    auto args = luisa::span{std::launder(reinterpret_cast<const Argument *>(_argument_buffer.data())), _argument_count};
    luisa::unique_ptr<ShaderDispatchCommand> cmd{
        new (luisa::detail::allocator_allocate(sizeof(ShaderDispatchCommand), alignof(ShaderDispatchCommand))) ShaderDispatchCommand{
            _handle,
            std::move(_argument_buffer),
            _argument_count}};
    cmd->_dispatch_size = _dispatch_size;
    return cmd;
}

luisa::unique_ptr<DrawRasterSceneCommand> RasterDispatchCmdEncoder::build() &&noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}, Actual argument count {}.", _argument_count, _argument_idx);
    }
    // friend class
    luisa::unique_ptr<DrawRasterSceneCommand> cmd{
        new (luisa::detail::allocator_allocate(sizeof(DrawRasterSceneCommand), alignof(DrawRasterSceneCommand))) DrawRasterSceneCommand{
            _handle,
            std::move(_argument_buffer),
            _argument_count}};
    memcpy(cmd->_rtv_texs, _rtv_texs, sizeof(Argument::Texture) * _rtv_count);
    cmd->_rtv_count = _rtv_count;
    cmd->_dsv_tex = _dsv_tex;
    cmd->_scene = std::move(scene);
    cmd->_viewport = viewport;
    return cmd;
}

DrawRasterSceneCommand::~DrawRasterSceneCommand() noexcept {}
DrawRasterSceneCommand::DrawRasterSceneCommand(uint64_t shader_handle, luisa::vector<std::byte> &&argument_buffer, size_t argument_count) noexcept
    : ShaderDispatchCommandBase{Tag::EDrawRasterSceneCommand, shader_handle, std::move(argument_buffer), argument_count} {}
luisa::span<const RasterMesh> DrawRasterSceneCommand::scene() const noexcept { return luisa::span{_scene}; }
DrawRasterSceneCommand::DrawRasterSceneCommand(DrawRasterSceneCommand &&) = default;

}// namespace luisa::compute
