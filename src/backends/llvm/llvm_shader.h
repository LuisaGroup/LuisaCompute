//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <ast/function.h>

namespace llvm {
class LLVMContext;
class Module;
class ExecutionEngine;
}// namespace llvm

namespace luisa::compute::llvm {

class LLVMDevice;

class LLVMShader {

public:
    using kernel_entry_t = void(const std::byte *, const uint3 *);

private:
    luisa::unique_ptr<::llvm::LLVMContext> _context;
    std::unique_ptr<::llvm::ExecutionEngine> _engine;
    luisa::unordered_map<uint, size_t> _argument_offsets;
    kernel_entry_t *_kernel_entry{nullptr};
    size_t _argument_buffer_size{};

public:
    LLVMShader(LLVMDevice *device, Function func) noexcept;
    ~LLVMShader() noexcept;
    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    [[nodiscard]] size_t argument_offset(uint uid) const noexcept;
    void invoke(const std::byte *args, uint3 block_id) const noexcept;
};

}// namespace luisa::compute::llvm
