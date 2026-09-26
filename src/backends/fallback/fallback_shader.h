//
// Created by swfly on 2024/11/21.
//

#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/core/stl/memory.h>

namespace llvm {
class LLVMContext;
class Module;
class ExecutionEngine;
class TargetMachine;
}// namespace llvm

namespace llvm::orc {
class LLJIT;
}// namespace llvm::orc

namespace luisa::compute {
class ShaderPrintFormatter;
}// namespace luisa::compute
namespace luisa::compute::fallback {

class FallbackDevice;
class FallbackCommandQueue;

class FallbackShader {

public:
    using kernel_entry_t = void(const void *, const void *);

private:
    luisa::string _name;
    luisa::unordered_map<uint, size_t> _argument_offsets{};
    kernel_entry_t *_kernel_entry{nullptr};
    size_t _argument_buffer_size{};
    size_t _shared_memory_size{};
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments{};
    luisa::vector<luisa::unique_ptr<ShaderPrintFormatter>> _print_formatters;

    uint3 _block_size;
    luisa::unique_ptr<::llvm::orc::LLJIT> _jit{};
    luisa::unique_ptr<::llvm::TargetMachine> _target_machine{};

private:
    void _initialize_target_machine_jit(const ShaderOption &option) noexcept;
    void _build_bound_arguments(luisa::span<const Function::Binding> bindings) noexcept;

public:
    // void dispatch(ThreadPool &pool, const ShaderDispatchCommand *command) const noexcept;
    void dispatch(FallbackCommandQueue *queue, luisa::unique_ptr<ShaderDispatchCommand> command) noexcept;
    FallbackShader(FallbackDevice *device, const ShaderOption &option, Function kernel) noexcept;
    ~FallbackShader() noexcept;

    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    [[nodiscard]] auto shared_memory_size() const noexcept { return _shared_memory_size; }
    [[nodiscard]] auto native_handle() const noexcept { return _kernel_entry; }
    [[nodiscard]] auto print_formatter(size_t i) const noexcept -> const ShaderPrintFormatter * { return _print_formatters[i].get(); }
};

}// namespace luisa::compute::fallback
