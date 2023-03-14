#include <core/logging.h>
#include <ast/function_builder.h>
#include <runtime/rhi/command.h>
#include <runtime/command_encoder.h>
#include <runtime/raster/raster_scene.h>

namespace luisa::compute {

std::byte *ShaderDispatchCmdEncoder::_make_space(size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.resize(offset + size);
    return _argument_buffer.data() + offset;
}

ShaderDispatchCmdEncoder::ShaderDispatchCmdEncoder(
    uint64_t handle,
    size_t arg_count,
    size_t uniform_size,
    luisa::span<const Function::Binding> bindings) noexcept
    : _handle{handle}, _argument_count{arg_count}, _bindings{bindings} {
    if (auto arg_size_bytes = arg_count * sizeof(Argument)) {
        _argument_buffer.reserve(arg_size_bytes + uniform_size);
        _argument_buffer.resize_uninitialized(arg_size_bytes);
    }
}

ShaderDispatchCmdEncoder::Argument &ShaderDispatchCmdEncoder::_create_argument() noexcept {
    auto idx = _argument_idx;
    _argument_idx++;
    return *std::launder(reinterpret_cast<Argument *>(_argument_buffer.data()) + idx);
}

void ShaderDispatchCmdEncoder::_encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BUFFER;
    arg.buffer = ShaderDispatchCommandBase::Argument::Buffer{handle, offset, size};
}

void ShaderDispatchCmdEncoder::_encode_texture(uint64_t handle, uint32_t level) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::TEXTURE;
    arg.texture = ShaderDispatchCommandBase::Argument::Texture{handle, level};
}

void ShaderDispatchCmdEncoder::_encode_uniform(const void *data, size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.push_back_uninitialized(size);
    std::memcpy(_argument_buffer.data() + offset, data, size);
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::UNIFORM;
    arg.uniform.offset = offset;
    arg.uniform.size = size;
}

void ComputeDispatchCmdEncoder::set_dispatch_size(uint3 launch_size) noexcept {
    _dispatch_size = launch_size;
}

void ComputeDispatchCmdEncoder::set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept {
    _dispatch_size = indirect_arg;
}

void ShaderDispatchCmdEncoder::_encode_bindless_array(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BINDLESS_ARRAY;
    arg.bindless_array = Argument::BindlessArray{handle};
}

void ShaderDispatchCmdEncoder::_encode_accel(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::ACCEL;
    arg.accel = Argument::Accel{handle};
}

void ShaderDispatchCmdEncoder::_encode_pending_bindings() noexcept {
    while (_argument_idx < _bindings.size() &&
           !luisa::holds_alternative<luisa::monostate>(_bindings[_argument_idx])) {
        auto &&binding = _bindings[_argument_idx];
        luisa::visit(
            [&]<typename T>(T binding) noexcept {
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    _encode_buffer(binding.handle, binding.offset_bytes, binding.size_bytes);
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    _encode_texture(binding.handle, binding.level);
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    _encode_bindless_array(binding.handle);
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    _encode_accel(binding.handle);
                } else {
                    LUISA_ERROR_WITH_LOCATION("Invalid argument binding type.");
                }
            },
            binding);
    }
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Variable> arguments) noexcept {
    return std::accumulate(
        arguments.cbegin(), arguments.cend(),
        static_cast<size_t>(0u), [](auto size, auto arg) noexcept {
            return size + arg.type()->size();
        });
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Type *const> arg_types) noexcept {
    return std::accumulate(
        arg_types.cbegin(), arg_types.cend(),
        static_cast<size_t>(0u), [](auto size, auto arg) noexcept {
            return size + arg->size();
        });
}

ComputeDispatchCmdEncoder::ComputeDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size,
                                                     luisa::span<const Function::Binding> bindings) noexcept
    : ShaderDispatchCmdEncoder{handle, arg_count, uniform_size, bindings} { _encode_pending_bindings(); }

void ComputeDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    _encode_buffer(handle, offset, size);
    _encode_pending_bindings();
}

void ComputeDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    _encode_texture(handle, level);
    _encode_pending_bindings();
}

void ComputeDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    _encode_uniform(data, size);
    _encode_pending_bindings();
}

void ComputeDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    _encode_bindless_array(handle);
    _encode_pending_bindings();
}

void ComputeDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    _encode_accel(handle);
    _encode_pending_bindings();
}

void RasterDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    _encode_buffer(handle, offset, size);
    _encode_pending_bindings();
}

void RasterDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    _encode_texture(handle, level);
    _encode_pending_bindings();
}

void RasterDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    _encode_uniform(data, size);
    _encode_pending_bindings();
}

void RasterDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    _encode_bindless_array(handle);
    _encode_pending_bindings();
}

void RasterDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    _encode_accel(handle);
    _encode_pending_bindings();
}

RasterDispatchCmdEncoder::~RasterDispatchCmdEncoder() noexcept = default;
RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(RasterDispatchCmdEncoder &&) noexcept = default;
RasterDispatchCmdEncoder &RasterDispatchCmdEncoder::operator=(RasterDispatchCmdEncoder &&) noexcept = default;

RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size,
                                                   luisa::span<const Function::Binding> bindings) noexcept
    : ShaderDispatchCmdEncoder{handle, arg_count, uniform_size, bindings} {
}

luisa::unique_ptr<ShaderDispatchCommand> ComputeDispatchCmdEncoder::build() &&noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}. "
                    "Actual argument count {}.",
                    _argument_count, _argument_idx);
    }
    return luisa::make_unique<ShaderDispatchCommand>(
        _handle, std::move(_argument_buffer),
        _argument_count, _dispatch_size);
}

luisa::unique_ptr<DrawRasterSceneCommand> RasterDispatchCmdEncoder::build() &&noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}. "
                    "Actual argument count {}.",
                    _argument_count, _argument_idx);
    }
    return luisa::make_unique<DrawRasterSceneCommand>(
        _handle, std::move(_argument_buffer),
        _argument_count, _rtv_texs, _rtv_count,
        _dsv_tex, std::move(_scene), _viewport);
}

void RasterDispatchCmdEncoder::set_rtv_texs(luisa::span<const ShaderDispatchCommandBase::Argument::Texture> tex) noexcept {
    LUISA_ASSERT(tex.size() <= 8, "Too many render targets: {}.", tex.size());
    _rtv_count = tex.size();
    memcpy(_rtv_texs.data(), tex.data(), tex.size_bytes());
}

void RasterDispatchCmdEncoder::set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture tex) noexcept {
    _dsv_tex = tex;
}

void RasterDispatchCmdEncoder::set_scene(luisa::vector<RasterMesh> &&scene) noexcept {
    _scene = std::move(scene);
}

void RasterDispatchCmdEncoder::set_viewport(Viewport viewport) noexcept {
    _viewport = viewport;
}

}// namespace luisa::compute
