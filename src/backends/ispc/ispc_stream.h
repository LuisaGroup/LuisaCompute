//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <core/thread_pool.h>
#include <runtime/command_list.h>

namespace luisa::compute::ispc {

class ISPCEvent;

/**
 * @brief Stream of ISPC
 * 
 */
class ISPCStream final : public CommandVisitor {

private:
    ThreadPool _pool;

public:
    ISPCStream() noexcept = default;
    /**
     * @brief Synchronize
     * 
     */
    void synchronize() noexcept { _pool.synchronize(); }
    /**
     * @brief Dispatch list of commands
     * 
     * @param cmd_list list of commands
     */
    void dispatch(const CommandList &cmd_list) noexcept;
    /**
     * @brief Dispatch a host function
     *
     * @param f host function to dispatch
     */
    void dispatch(luisa::move_only_function<void()> &&f) noexcept;
    /**
     * @brief Signal event
     * 
     * @param event event
     */
    void signal(ISPCEvent *event) noexcept;
    /**
     * @brief Wait event
     * 
     * @param event event
     */
    void wait(ISPCEvent *event) noexcept;

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

}// namespace luisa::compute::ispc
