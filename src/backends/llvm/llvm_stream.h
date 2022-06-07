//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <core/thread_pool.h>
#include <runtime/command_list.h>

namespace luisa::compute::llvm {

class LLVMEvent;

/**
 * @brief Stream of LLVM
 * 
 */
class LLVMStream final : public CommandVisitor {

private:
    ThreadPool _pool;

public:
    LLVMStream() noexcept = default;
    void synchronize() noexcept { _pool.synchronize(); }
    void dispatch(const CommandList &cmd_list) noexcept;
    void dispatch(luisa::move_only_function<void()> &&f) noexcept;
    void signal(LLVMEvent *event) noexcept;
    void wait(LLVMEvent *event) noexcept;

public:
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;
    void visit(const ShaderDispatchCommand *command) noexcept override;
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
    void visit(const BindlessArrayUpdateCommand *command) noexcept override;
};

}// namespace luisa::compute::llvm
