//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <ast/function.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

namespace llvm {
class LLVMContext;
class Module;
class ExecutionEngine;
}// namespace llvm

namespace luisa::compute::llvm {

class LLVMDevice;

class LLVMShader {

public:
    using kernel_entry_t = void(const std::byte *, std::byte *, uint, uint, uint, uint, uint, uint);

private:
    luisa::string _name;
    luisa::unordered_map<uint, size_t> _argument_offsets;
    kernel_entry_t *_kernel_entry{nullptr};
    size_t _argument_buffer_size{};
<<<<<<< HEAD
    luisa::vector<FunctionBuilder::CpuCallback> _callbacks;
=======
    size_t _shared_memory_size{};
>>>>>>> c59c6d07052a2fbd1ba2eea94f8315c59faa51de

public:
    LLVMShader(LLVMDevice *device, Function func) noexcept;
    ~LLVMShader() noexcept;
    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    [[nodiscard]] auto shared_memory_size() const noexcept { return _shared_memory_size; }
    [[nodiscard]] size_t argument_offset(uint uid) const noexcept;
<<<<<<< HEAD
    [[nodiscard]] CpuCallback* callbacks() const noexcept { return _callbacks.data(); }
    void invoke(const std::byte *args, uint3 dispatch_size, uint3 block_id) const noexcept;
=======
    void invoke(const std::byte *args, std::byte *shared_memory,
                uint3 dispatch_size, uint3 block_id) const noexcept;
>>>>>>> c59c6d07052a2fbd1ba2eea94f8315c59faa51de
};

}// namespace luisa::compute::llvm
