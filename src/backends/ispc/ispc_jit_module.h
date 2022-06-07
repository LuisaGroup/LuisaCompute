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
    llvm::ExecutionEngine *_engine{nullptr};

private:
    ISPCJITModule(luisa::unique_ptr<llvm::LLVMContext> ctx,
                  llvm::ExecutionEngine *engine) noexcept;

public:
    ~ISPCJITModule() noexcept override;
    ISPCJITModule(ISPCJITModule &&) noexcept = default;
    ISPCJITModule &operator=(ISPCJITModule &&) noexcept = default;
    [[nodiscard]] static luisa::shared_ptr<ISPCModule> load(
        const Context &ctx, const std::filesystem::path &ir_path) noexcept;
};

}// namespace luisa::compute::ispc
