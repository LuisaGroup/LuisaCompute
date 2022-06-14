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
    using kernel_entry_t = void(const std::byte *, uint, uint, uint, uint, uint, uint);

private:
    luisa::string _name;
    luisa::unique_ptr<::llvm::LLVMContext> _context;
    ::llvm::ExecutionEngine *_engine{nullptr};
    luisa::unordered_map<uint, size_t> _argument_offsets;
    kernel_entry_t *_kernel_entry{nullptr};
    size_t _argument_buffer_size{};

public:
    LLVMShader(LLVMDevice *device, Function func) noexcept;
    ~LLVMShader() noexcept;
    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    [[nodiscard]] size_t argument_offset(uint uid) const noexcept;
    void invoke(const std::byte *args, uint3 dispatch_size, uint3 block_id) const noexcept;
};

}// namespace luisa::compute::llvm
