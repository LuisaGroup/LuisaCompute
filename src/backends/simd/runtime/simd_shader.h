#pragma once

#include <cstdint>

#include <luisa/ast/function.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/resource.h>

#include "../simd_compiler.h"
#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute {
class ShaderPrintFormatter;
}// namespace luisa::compute

namespace luisa::compute::simd {

class SIMDThreadPool;

class SIMDShader {

public:
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    // Same physical ABI as Entry; the fourth argument is the number of full
    // logical packets to issue from launch_config.thread_index. The wrapper
    // advances that field in place while issuing the packets.
    using PacketBatchEntry = void(
        const void *, void *, SIMDPacketLaunchConfig *, uint32_t);
    // Same physical ABI as PacketBatchEntry; the fourth argument is the
    // number of consecutive flattened blocks beginning at block_id.
    using BlockBatchEntry = void(
        const void *, void *, SIMDPacketLaunchConfig *, uint32_t);

private:
    SIMDCompiledKernel _compiled;
    Entry *_entry{nullptr};
    PacketBatchEntry *_packet_batch_entry{nullptr};
    BlockBatchEntry *_block_batch_entry{nullptr};
    bool _enable_packet_batch_entry{false};
    bool _enable_block_batch_entry{false};
    bool _enable_predicated_acyclic_surface_filter{true};
    uint3 _block_size{1u, 1u, 1u};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    luisa::vector<Usage> _argument_usages;
    luisa::vector<luisa::unique_ptr<ShaderPrintFormatter>>
        _print_formatters;

private:
    void _build_bound_arguments(
        luisa::span<const Function::Binding> bindings) noexcept;
    void _dispatch_once(
        SIMDThreadPool &thread_pool,
        const void *argument_buffer, uint3 dispatch_size,
        const DeviceInterface::StreamLogCallback &log_callback,
        uint32_t kernel_id = 0u) const noexcept;

public:
    SIMDShader(
        const ShaderOption &option, Function kernel,
        uint32_t warp_width, uint32_t dispatch_worker_count) noexcept;
    ~SIMDShader() noexcept;

    void dispatch(
        SIMDThreadPool &thread_pool,
        const DeviceInterface::StreamLogCallback &log_callback,
        luisa::unique_ptr<ShaderDispatchCommand> command) const noexcept;
    [[nodiscard]] Usage argument_usage(size_t index) const noexcept;
    [[nodiscard]] auto native_handle() const noexcept {
        if (_compiled.block_batch_entry != nullptr) {
            return _compiled.block_batch_entry;
        }
        return _compiled.packet_batch_entry != nullptr ?
                   _compiled.packet_batch_entry :
                   _compiled.entry;
    }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto warp_width() const noexcept {
        return _compiled.warp_width;
    }
};

}// namespace luisa::compute::simd
