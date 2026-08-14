#pragma once

#include <cstdint>

#include <luisa/ast/function.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>

#include "../simd_compiler.h"
#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute::simd {

class SIMDThreadPool;

class SIMDShader {

public:
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);

private:
    SIMDCompiledKernel _compiled;
    Entry *_entry{nullptr};
    uint3 _block_size{1u, 1u, 1u};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    luisa::vector<Usage> _argument_usages;

private:
    void _build_bound_arguments(
        luisa::span<const Function::Binding> bindings) noexcept;
    void _dispatch_once(
        SIMDThreadPool &thread_pool,
        const void *argument_buffer, uint3 dispatch_size) const noexcept;

public:
    SIMDShader(
        const ShaderOption &option, Function kernel,
        uint32_t warp_width, uint32_t dispatch_worker_count) noexcept;
    ~SIMDShader() noexcept = default;

    void dispatch(
        SIMDThreadPool &thread_pool,
        luisa::unique_ptr<ShaderDispatchCommand> command) const noexcept;
    [[nodiscard]] Usage argument_usage(size_t index) const noexcept;
    [[nodiscard]] auto native_handle() const noexcept { return _entry; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto warp_width() const noexcept {
        return _compiled.warp_width;
    }
};

}// namespace luisa::compute::simd
