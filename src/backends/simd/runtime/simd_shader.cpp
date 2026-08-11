#include "simd_shader.h"

#include <cstring>

#include <luisa/core/logging.h>

#include "simd_buffer.h"
#include "simd_texture.h"

namespace luisa::compute::simd {

namespace {

[[nodiscard]] constexpr size_t align_up(
    size_t value, size_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

}// namespace

SIMDShader::SIMDShader(
    const ShaderOption &option, Function kernel,
    uint32_t warp_width) noexcept
    : _block_size{kernel.block_size()} {
    static_cast<void>(option);
    if (auto allowed = kernel.allowed_warp_size();
        allowed && *allowed != warp_width) {
        LUISA_ERROR_WITH_LOCATION(
            "SIMD kernel requests warp width {}, but the device was created "
            "with width {}.",
            *allowed, warp_width);
    }
    auto block_threads = static_cast<uint64_t>(_block_size.x) *
                         _block_size.y * _block_size.z;
    LUISA_ASSERT(
        warp_width != 0u && block_threads % warp_width == 0u,
        "SIMD thread block size {} must be a multiple of warp width {}.",
        block_threads, warp_width);
    _compiled = compile_simd_kernel(
        kernel, warp_width,
        kernel.name().empty() ? "simd_runtime_kernel" : kernel.name());
    if (!_compiled.succeeded()) {
        luisa::string diagnostics;
        for (auto &&message : _compiled.diagnostics) {
            if (!diagnostics.empty()) { diagnostics += '\n'; }
            diagnostics += message;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Failed to compile SIMD kernel (warp width {}):\n{}",
            warp_width, diagnostics);
    }
    _entry = reinterpret_cast<Entry *>(_compiled.entry);
    _build_bound_arguments(kernel.bound_arguments());
    _argument_usages.reserve(kernel.arguments().size());
    for (auto argument : kernel.arguments()) {
        _argument_usages.emplace_back(
            kernel.variable_usage(argument.uid()));
    }
}

void SIMDShader::_build_bound_arguments(
    luisa::span<const Function::Binding> bindings) noexcept {
    _bound_arguments.reserve(bindings.size());
    for (auto &&binding : bindings) {
        luisa::visit(
            [&]<typename T>(T value) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = Argument::Tag::BUFFER;
                    argument.buffer = value;
                } else if constexpr (
                    std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = Argument::Tag::TEXTURE;
                    argument.texture = value;
                } else if constexpr (
                    std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array = value;
                } else if constexpr (
                    std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag = Argument::Tag::ACCEL;
                    argument.accel = value;
                } else {
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid bound SIMD shader argument.");
                }
                _bound_arguments.emplace_back(argument);
            },
            binding);
    }
}

void SIMDShader::_dispatch_once(
    const void *argument_buffer, uint3 dispatch_size) const noexcept {
    auto block_size = _block_size;
    LUISA_ASSERT(
        block_size.x != 0u && block_size.y != 0u && block_size.z != 0u,
        "SIMD kernel block size must be nonzero.");
    auto ceil_div = [](uint32_t n, uint32_t d) noexcept {
        return (n + d - 1u) / d;
    };
    auto grid_size = make_uint3(
        ceil_div(dispatch_size.x, block_size.x),
        ceil_div(dispatch_size.y, block_size.y),
        ceil_div(dispatch_size.z, block_size.z));
    auto threads_per_block =
        block_size.x * block_size.y * block_size.z;
    LUISA_ASSERT(
        threads_per_block % _compiled.warp_width == 0u,
        "SIMD thread block size {} must be a multiple of warp width {}.",
        threads_per_block, _compiled.warp_width);
    auto warps_per_block =
        threads_per_block / _compiled.warp_width;
    for (auto bz = uint32_t{0u}; bz < grid_size.z; bz++) {
        for (auto by = uint32_t{0u}; by < grid_size.y; by++) {
            for (auto bx = uint32_t{0u}; bx < grid_size.x; bx++) {
                SIMDPacketLaunchConfig config{};
                config.block_id[0u] = bx;
                config.block_id[1u] = by;
                config.block_id[2u] = bz;
                config.dispatch_size[0u] = dispatch_size.x;
                config.dispatch_size[1u] = dispatch_size.y;
                config.dispatch_size[2u] = dispatch_size.z;
                config.block_size[0u] = block_size.x;
                config.block_size[1u] = block_size.y;
                config.block_size[2u] = block_size.z;
                for (auto warp = uint32_t{0u};
                     warp < warps_per_block; warp++) {
                    config.thread_index =
                        warp * _compiled.warp_width;
                    _entry(
                        argument_buffer, nullptr, &config,
                        _compiled.warp_width);
                }
            }
        }
    }
}

void SIMDShader::dispatch(
    luisa::unique_ptr<ShaderDispatchCommand> command) const noexcept {
    luisa::vector<std::byte> argument_buffer(
        _compiled.argument_buffer_size, std::byte{});
    auto offset = size_t{0u};
    auto allocate = [&](size_t size) noexcept {
        offset = align_up(offset, 16u);
        LUISA_ASSERT(
            offset <= argument_buffer.size() &&
                size <= argument_buffer.size() - offset,
            "SIMD shader argument buffer overflow.");
        auto *result = argument_buffer.data() + offset;
        offset = align_up(offset + size, 16u);
        return result;
    };
    auto encode = [&](const Argument &argument) noexcept {
        switch (argument.tag) {
            case Argument::Tag::BUFFER: {
                auto *buffer = reinterpret_cast<SIMDBuffer *>(
                    argument.buffer.handle);
                auto view = buffer->view(
                    argument.buffer.offset, argument.buffer.size);
                std::memcpy(
                    allocate(sizeof(view)), &view, sizeof(view));
                break;
            }
            case Argument::Tag::UNIFORM: {
                auto uniform = command->uniform(argument.uniform);
                std::memcpy(
                    allocate(uniform.size_bytes()), uniform.data(),
                    uniform.size_bytes());
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto *texture = reinterpret_cast<SIMDTexture *>(
                    argument.texture.handle);
                auto view = texture->host_view(argument.texture.level);
                std::memcpy(
                    allocate(sizeof(view)), &view, sizeof(view));
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY:
            case Argument::Tag::ACCEL:
                LUISA_ERROR_WITH_LOCATION(
                    "This SIMD runtime checkpoint does not support bindless "
                    "or acceleration-structure shader arguments yet.");
        }
    };
    for (auto &&argument : _bound_arguments) { encode(argument); }
    for (auto &&argument : command->arguments()) { encode(argument); }
    LUISA_ASSERT(
        _bound_arguments.size() + command->arguments().size() ==
            _argument_usages.size(),
        "SIMD shader argument count mismatch.");

    auto *arguments = argument_buffer.empty() ? nullptr :
                                              argument_buffer.data();
    if (command->is_indirect()) {
        LUISA_ERROR_WITH_LOCATION(
            "Indirect SIMD dispatch is not implemented yet.");
    }
    if (command->is_multiple_dispatch()) {
        for (auto dispatch_size : command->dispatch_sizes()) {
            _dispatch_once(arguments, dispatch_size);
        }
    } else {
        _dispatch_once(arguments, command->dispatch_size());
    }
}

Usage SIMDShader::argument_usage(size_t index) const noexcept {
    LUISA_ASSERT(
        index < _argument_usages.size(),
        "SIMD shader argument index out of range.");
    return _argument_usages[index];
}

}// namespace luisa::compute::simd
