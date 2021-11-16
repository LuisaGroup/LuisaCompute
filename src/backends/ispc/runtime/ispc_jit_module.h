//
// Created by Mike Smith on 2021/11/15.
//

#pragma once

#include <filesystem>
#include <core/allocator.h>

namespace llvm {
class LLVMContext;
class ExecutionEngine;
}

namespace lc::ispc {

class JITModule {

public:
    using function_type = void(
        uint32_t,// thd_cX
        uint32_t,// thd_cY
        uint32_t,// thd_cZ
        uint32_t,// thd_idX
        uint32_t,// thd_idY
        uint32_t,// thd_idZ
        uint64_t// arg
    );

private:
    luisa::unique_ptr<llvm::LLVMContext> _context;
    std::unique_ptr<llvm::ExecutionEngine> _engine;
    function_type *_run{nullptr};

private:
    JITModule(luisa::unique_ptr<llvm::LLVMContext> ctx,
              std::unique_ptr<llvm::ExecutionEngine> engine) noexcept;

public:
    JITModule() noexcept = default;
    ~JITModule() noexcept;
    JITModule(JITModule &&) noexcept = default;
    JITModule &operator=(JITModule &&) noexcept = default;
    [[nodiscard]] static JITModule load(const std::filesystem::path &ir_path) noexcept;
    void invoke(luisa::uint3 thread_count, luisa::uint3 thread_start, const void *args) const noexcept;
};

}
