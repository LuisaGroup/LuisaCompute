#include <luisa/core/logging.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <numeric>
namespace luisa::compute {

std::byte *ShaderDispatchCmdEncoder::_make_space(size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.resize(offset + size);
    return _argument_buffer.data() + offset;
}

ShaderDispatchCmdEncoder::ShaderDispatchCmdEncoder(
    uint64_t handle,
    size_t arg_count,
    size_t uniform_size) noexcept
    : _handle{handle}, _argument_count{arg_count} {
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

void ShaderDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BUFFER;
    arg.buffer = ShaderDispatchCommandBase::Argument::Buffer{handle, offset, size};
}

void ShaderDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::TEXTURE;
    arg.texture = ShaderDispatchCommandBase::Argument::Texture{handle, level};
}

void ShaderDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.push_back_uninitialized(size);
    std::memcpy(_argument_buffer.data() + offset, data, size);
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::UNIFORM;
    arg.uniform.offset = offset;
    arg.uniform.size = size;
}
luisa::unique_ptr<ShaderDispatchCommand> ComputeDispatchCmdEncoder::build() && noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}. "
                    "Actual argument count {}.",
                    _argument_count, _argument_idx);
    }
    return luisa::make_unique<ShaderDispatchCommand>(
        _handle, std::move(_argument_buffer),
        _argument_count, _dispatch_size);
}
void ComputeDispatchCmdEncoder::set_dispatch_size(uint3 launch_size) noexcept {
    _dispatch_size = launch_size;
}
void ComputeDispatchCmdEncoder::set_dispatch_sizes(luisa::span<const uint3> sizes) noexcept {
    luisa::vector<uint3> vec;
    vec.push_back_uninitialized(sizes.size());
    std::memcpy(vec.data(), sizes.data(), sizes.size_bytes());
    _dispatch_size = std::move(vec);
}

void ComputeDispatchCmdEncoder::set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept {
    _dispatch_size = indirect_arg;
}

void ShaderDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BINDLESS_ARRAY;
    arg.bindless_array = Argument::BindlessArray{handle};
}

void ShaderDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::ACCEL;
    arg.accel = Argument::Accel{handle};
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Variable> arguments) noexcept {
    return std::accumulate(
        arguments.cbegin(), arguments.cend(),
        static_cast<size_t>(0u), [](auto size, auto arg) noexcept {
            auto arg_type = arg.type();
            // Do not allocate redundant uniform buffer
            return size + (arg_type->is_resource() ? 0u : arg_type->size());
        });
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Type *const> arg_types) noexcept {
    return std::accumulate(
        arg_types.cbegin(), arg_types.cend(),
        static_cast<size_t>(0u), [](auto size, auto arg_type) noexcept {
            LUISA_ASSERT(arg_type != nullptr, "Invalid argument type.");
            // Do not allocate redundant uniform buffer
            return size + (arg_type->is_resource() ? 0u : arg_type->size());
        });
}
ComputeDispatchCmdEncoder::ComputeDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
    : ShaderDispatchCmdEncoder{handle, arg_count, uniform_size} {}

}// namespace luisa::compute
