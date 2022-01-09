//
// Created by Mike Smith on 2021/11/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <backends/ispc/runtime/ispc_module.h>

namespace llvm {
class LLVMContext;
class ExecutionEngine;
}

namespace lc::ispc {

class JITModule final : public Module {

private:
    luisa::unique_ptr<llvm::LLVMContext> _context;
    std::unique_ptr<llvm::ExecutionEngine> _engine;

private:
    JITModule(luisa::unique_ptr<llvm::LLVMContext> ctx,
              std::unique_ptr<llvm::ExecutionEngine> engine) noexcept;

public:
    ~JITModule() noexcept override;
    JITModule(JITModule &&) noexcept = default;
    JITModule &operator=(JITModule &&) noexcept = default;
    [[nodiscard]] static luisa::unique_ptr<Module> load(
        const Context &ctx, const std::filesystem::path &ir_path) noexcept;
};

}
