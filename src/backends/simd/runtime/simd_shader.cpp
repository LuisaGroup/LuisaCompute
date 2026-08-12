#include "simd_shader.h"

#include <cstring>

#include <luisa/core/logging.h>

#include "../../common/env_flag.h"
#include "simd_bindless_array.h"
#include "simd_accel.h"
#include "simd_buffer.h"
#include "simd_thread_pool.h"
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
        kernel.name().empty() ? "simd_runtime_kernel" : kernel.name(),
        option.enable_fast_math);
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
    if (detail::env_flag(
            "LUISA_SIMD_REPORT_OPTIMIZATIONS")) {
        LUISA_INFO(
            "SIMD optimization report [{} W{}]: predicated_diamonds={}, "
            "factored_selects={}, unswitched_loops={}, cloned_blocks={}, "
            "cloned_instructions={}, merged_live_outs={}, "
            "direct_control_flow={}, "
            "uniform_buffer_broadcasts={}, contiguous_buffer_reads={}, "
            "contiguous_buffer_writes={}.",
            kernel.name().empty() ? "simd_runtime_kernel" : kernel.name(),
            warp_width, _compiled.predicated_diamond_count,
            _compiled.factored_select_count,
            _compiled.unswitched_loop_count,
            _compiled.unswitched_cloned_block_count,
            _compiled.unswitched_cloned_instruction_count,
            _compiled.unswitched_live_out_count,
            _compiled.direct_control_flow,
            _compiled.uniform_buffer_broadcast_count,
            _compiled.contiguous_buffer_read_count,
            _compiled.contiguous_buffer_write_count);
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
    SIMDThreadPool &thread_pool,
    const void *argument_buffer, uint3 dispatch_size) const noexcept {
    auto block_size = _block_size;
    LUISA_ASSERT(
        block_size.x != 0u && block_size.y != 0u && block_size.z != 0u,
        "SIMD kernel block size must be nonzero.");
    auto ceil_div = [](uint32_t n, uint32_t d) noexcept {
        return n / d + static_cast<uint32_t>(n % d != 0u);
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
    auto grid_xy = static_cast<uint64_t>(grid_size.x) * grid_size.y;
    auto grid_count = grid_xy * grid_size.z;
    constexpr auto target_chunks_per_worker = uint64_t{32u};
    auto target_chunks = static_cast<uint64_t>(
                             thread_pool.worker_count()) *
                         target_chunks_per_worker;
    auto grain_size = grid_count == 0u ?
                          uint64_t{1u} :
                          (grid_count - 1u) / target_chunks + 1u;
    thread_pool.parallel_for(
        grid_count, grain_size,
        [&](uint64_t begin, uint64_t end) noexcept {
            for (auto block = begin; block < end; block++) {
                auto bx = static_cast<uint32_t>(block % grid_size.x);
                auto by = static_cast<uint32_t>(
                    (block / grid_size.x) % grid_size.y);
                auto bz = static_cast<uint32_t>(block / grid_xy);
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
        });
}

void SIMDShader::dispatch(
    SIMDThreadPool &thread_pool,
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
            case Argument::Tag::BINDLESS_ARRAY: {
                auto *array = reinterpret_cast<SIMDBindlessArray *>(
                    argument.bindless_array.handle);
                auto view = array->host_view();
                std::memcpy(
                    allocate(sizeof(view)), &view, sizeof(view));
                break;
            }
            case Argument::Tag::ACCEL: {
                auto *accel = reinterpret_cast<SIMDAccel *>(
                    argument.accel.handle);
                auto view = accel->host_view();
                std::memcpy(
                    allocate(sizeof(view)), &view, sizeof(view));
                break;
            }
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
            _dispatch_once(thread_pool, arguments, dispatch_size);
        }
    } else {
        _dispatch_once(
            thread_pool, arguments, command->dispatch_size());
    }
}

Usage SIMDShader::argument_usage(size_t index) const noexcept {
    LUISA_ASSERT(
        index < _argument_usages.size(),
        "SIMD shader argument index out of range.");
    return _argument_usages[index];
}

}// namespace luisa::compute::simd
