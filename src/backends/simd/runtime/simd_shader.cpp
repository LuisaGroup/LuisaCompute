#include "simd_shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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

struct AssemblyStats {
    size_t instructions{0u};
    size_t vector_instructions{0u};
    size_t branches{0u};
    size_t calls{0u};
    size_t stack_references{0u};
    size_t stack_allocation_bytes{0u};
    size_t scalar_math_calls{0u};
};

[[nodiscard]] AssemblyStats inspect_assembly(
    std::string_view assembly) noexcept {
    AssemblyStats stats;
    for (auto line_begin = size_t{0u}; line_begin < assembly.size();) {
        auto line_end = assembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = assembly.size();
        }
        auto line = assembly.substr(line_begin, line_end - line_begin);
        auto first = line.find_first_not_of(" \t");
        if (first != std::string_view::npos) {
            line.remove_prefix(first);
        }
        if (!line.empty() && line.front() != '.' &&
            line.front() != '#' && line.back() != ':') {
            auto mnemonic_end = line.find_first_of(" \t");
            auto mnemonic = line.substr(0u, mnemonic_end);
            stats.instructions++;
            stats.vector_instructions +=
                !mnemonic.empty() &&
                (mnemonic.front() == 'v' || mnemonic.front() == 'k');
            stats.branches +=
                (!mnemonic.empty() && mnemonic.front() == 'j') ||
                mnemonic.starts_with("loop");
            auto call = mnemonic.starts_with("call");
            stats.calls += call;
            stats.stack_references +=
                line.find("%rsp") != std::string_view::npos ||
                line.find("%rbp") != std::string_view::npos;
            if (call) {
                constexpr std::array scalar_math_symbols{
                    "sinf", "cosf", "tanf", "asinf", "acosf",
                    "atanf", "atan2f", "expf", "exp2f", "exp10f",
                    "logf", "log2f", "log10f", "powf"};
                stats.scalar_math_calls += std::any_of(
                    scalar_math_symbols.begin(),
                    scalar_math_symbols.end(),
                    [&](std::string_view symbol) noexcept {
                        return line.find(symbol) !=
                               std::string_view::npos;
                    });
            }
            if (mnemonic == "subq" &&
                line.find("%rsp") != std::string_view::npos) {
                auto dollar = line.find('$');
                auto comma = line.find(',', dollar);
                if (dollar != std::string_view::npos &&
                    comma != std::string_view::npos) {
                    auto immediate = line.substr(
                        dollar + 1u, comma - dollar - 1u);
                    auto bytes = size_t{0u};
                    auto [end, error] = std::from_chars(
                        immediate.data(),
                        immediate.data() + immediate.size(), bytes);
                    if (error == std::errc{} &&
                        end == immediate.data() + immediate.size()) {
                        stats.stack_allocation_bytes =
                            std::max(
                                stats.stack_allocation_bytes, bytes);
                    }
                }
            }
        }
        line_begin = line_end + (line_end < assembly.size() ? 1u : 0u);
    }
    return stats;
}

void dump_compilation_artifacts(
    std::string_view directory, std::string_view kernel_name,
    uint32_t width, std::string_view assembly,
    std::string_view object) noexcept {
    static std::atomic_uint64_t sequence{0u};
    std::error_code error;
    auto path = std::filesystem::path{directory};
    std::filesystem::create_directories(path, error);
    if (error) {
        LUISA_WARNING(
            "Failed to create SIMD assembly directory '{}': {}.",
            directory, error.message());
        return;
    }
    auto safe_name = std::string{kernel_name};
    for (auto &character : safe_name) {
        auto safe = (character >= 'a' && character <= 'z') ||
                    (character >= 'A' && character <= 'Z') ||
                    (character >= '0' && character <= '9') ||
                    character == '_' || character == '-';
        if (!safe) { character = '_'; }
    }
    auto timestamp = std::chrono::system_clock::now()
                         .time_since_epoch()
                         .count();
    auto process_id =
#ifdef _WIN32
        static_cast<uint64_t>(_getpid());
#else
        static_cast<uint64_t>(getpid());
#endif
    auto index = sequence.fetch_add(1u, std::memory_order_relaxed);
    path /= safe_name + "_w" + std::to_string(width) + "_" +
            std::to_string(process_id) + "_" +
            std::to_string(timestamp) + "_" +
            std::to_string(index);
    auto assembly_path = path;
    assembly_path += ".s";
    std::ofstream assembly_stream{assembly_path, std::ios::binary};
    assembly_stream.write(
        assembly.data(), static_cast<std::streamsize>(assembly.size()));
    if (!assembly_stream) {
        LUISA_WARNING(
            "Failed to write SIMD assembly '{}'.",
            assembly_path.string());
        return;
    }
    LUISA_INFO(
        "SIMD assembly written to '{}'.", assembly_path.string());
    if (object.empty()) {
        LUISA_WARNING(
            "SIMD JIT object capture was empty for '{}'.",
            assembly_path.string());
        return;
    }
    auto object_path = path;
    object_path += ".o";
    std::ofstream object_stream{object_path, std::ios::binary};
    object_stream.write(
        object.data(), static_cast<std::streamsize>(object.size()));
    if (!object_stream) {
        LUISA_WARNING(
            "Failed to write SIMD JIT object '{}'.",
            object_path.string());
        return;
    }
    LUISA_INFO(
        "SIMD JIT object written to '{}'.", object_path.string());
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
    auto *assembly_directory =
        std::getenv("LUISA_SIMD_DUMP_ASSEMBLY_DIR");
    auto capture_assembly =
        detail::env_flag("LUISA_SIMD_REPORT_ASSEMBLY") ||
        assembly_directory != nullptr;
    _compiled = compile_simd_kernel(
        kernel, warp_width,
        kernel.name().empty() ? "simd_runtime_kernel" : kernel.name(),
        option.enable_fast_math, capture_assembly);
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
            "SIMD optimization report [{} W{}]: aggregate_allocas={}, "
            "aggregate_leaf_allocas={}, predicated_diamonds={}, "
            "predicated_refinement_rounds={}, predicated_forwarded_phis={}, "
            "predicated_forwarding_blocks={}, "
            "predicated_widened_update_diamonds={}, "
            "predicated_wide_select_ladder_diamonds={}, "
            "predicated_memory_diamonds={}, "
            "predicated_memory_instructions={}, "
            "factored_selects={}, unswitched_loops={}, cloned_blocks={}, "
            "cloned_instructions={}, merged_live_outs={}, "
            "coherent_mask_reuses={}, "
            "convergence_token_guards={}, "
            "direct_divergent_children={}, "
            "direct_control_flow={}, "
            "schedule_blocks={}, convergence_points={}, "
            "scalar_frame_metadata={}, "
            "state_slots={}, coalesced_state_slots={}, "
            "instruction_spills={}, cold_slots={}, "
            "stack_pinned_slots={}, "
            "ray_queries={}, ray_query_scratch_slots={}, "
            "ray_query_scratch_bytes={}, ray_query_status_slots={}, "
            "ray_query_state_handle_slots={}, "
            "uniform_buffer_broadcasts={}, contiguous_buffer_reads={}, "
            "contiguous_buffer_writes={}, paired_leaf_gathers={}.",
            kernel.name().empty() ? "simd_runtime_kernel" : kernel.name(),
            warp_width, _compiled.decomposed_aggregate_alloca_count,
            _compiled.inserted_aggregate_leaf_alloca_count,
            _compiled.predicated_diamond_count,
            _compiled.predicated_refinement_round_count,
            _compiled.predicated_forwarded_phi_count,
            _compiled.predicated_forwarding_block_count,
            _compiled.predicated_widened_update_diamond_count,
            _compiled.predicated_wide_select_ladder_diamond_count,
            _compiled.predicated_memory_diamond_count,
            _compiled.predicated_memory_instruction_count,
            _compiled.factored_select_count,
            _compiled.unswitched_loop_count,
            _compiled.unswitched_cloned_block_count,
            _compiled.unswitched_cloned_instruction_count,
            _compiled.unswitched_live_out_count,
            _compiled.coherent_mask_reuse_count,
            _compiled.convergence_token_guard_count,
            _compiled.direct_divergent_child_count,
            _compiled.direct_control_flow,
            _compiled.schedule_block_count,
            _compiled.convergence_point_count,
            _compiled.scalar_frame_metadata,
            _compiled.state_slot_count,
            _compiled.coalesced_state_slot_count,
            _compiled.spilled_instruction_count,
            _compiled.cold_state_slot_count,
            _compiled.stack_pinned_state_slot_count,
            _compiled.ray_query_count,
            _compiled.ray_query_scratch_slot_count,
            _compiled.ray_query_scratch_bytes,
            _compiled.ray_query_status_slot_count,
            _compiled.ray_query_state_handle_slot_count,
            _compiled.uniform_buffer_broadcast_count,
            _compiled.contiguous_buffer_read_count,
            _compiled.contiguous_buffer_write_count,
            _compiled.paired_leaf_gather_count);
    }
    if (capture_assembly) {
        auto stats = inspect_assembly(_compiled.assembly);
        LUISA_INFO(
            "SIMD assembly report [{} W{}]: bytes={}, instructions={}, "
            "vector_instructions={}, branches={}, calls={}, "
            "stack_references={}, stack_allocation_bytes={}, "
            "scalar_math_calls={}.",
            kernel.name().empty() ? "simd_runtime_kernel" : kernel.name(),
            warp_width, _compiled.assembly.size(), stats.instructions,
            stats.vector_instructions, stats.branches, stats.calls,
            stats.stack_references, stats.stack_allocation_bytes,
            stats.scalar_math_calls);
        if (assembly_directory != nullptr) {
            dump_compilation_artifacts(
                assembly_directory,
                kernel.name().empty() ? "simd_runtime_kernel" :
                                        kernel.name(),
                warp_width, _compiled.assembly,
                _compiled.jit->object());
        }
    }
    _entry = reinterpret_cast<Entry *>(_compiled.entry);
    if (detail::env_flag("LUISA_SIMD_REPORT_JIT_ADDRESS")) {
        LUISA_INFO(
            "SIMD JIT entry [{} W{}]: {}.",
            kernel.name().empty() ? "simd_runtime_kernel" :
                                    kernel.name(),
            warp_width, _compiled.entry);
    }
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
