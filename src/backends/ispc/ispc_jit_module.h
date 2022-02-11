//
// Created by Mike Smith on 2021/11/15.
//

#pragma once

#include <filesystem>

#include <core/stl.h>
#include <backends/ispc/ispc_module.h>

namespace llvm {
class LLVMContext;
class ExecutionEngine;
}// namespace llvm

namespace luisa::compute {
class Context;
}

namespace luisa::compute::ispc {

using luisa::compute::Context;

class ISPCJITModule final : public ISPCModule {

private:
    luisa::unique_ptr<llvm::LLVMContext> _context;
    std::unique_ptr<llvm::ExecutionEngine> _engine;

private:
    ISPCJITModule(
        luisa::unique_ptr<llvm::LLVMContext> ctx,
        std::unique_ptr<llvm::ExecutionEngine> engine) noexcept;

public:
    ~ISPCJITModule() noexcept override;
    ISPCJITModule(ISPCJITModule &&) noexcept = default;
    ISPCJITModule &operator=(ISPCJITModule &&) noexcept = default;
    [[nodiscard]] static luisa::unique_ptr<ISPCModule> load(
        const Context &ctx, const std::filesystem::path &ir_path) noexcept;
};

}// namespace luisa::compute::ispc
