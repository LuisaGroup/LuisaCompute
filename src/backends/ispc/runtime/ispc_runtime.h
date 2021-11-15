#pragma once
#include <vstl/Common.h>
#include <runtime/command.h>
#include "ispc_shader.h"

using namespace luisa;
using namespace luisa::compute;
namespace lc::ispc {
class CommandExecutor : public CommandVisitor {
public:
    ThreadPool *tPool;
    vstd::vector<std::pair<ThreadTaskHandle, Shader::ArgVector>> handles;
    CommandExecutor(ThreadPool *tPool) : tPool(tPool) {}
    void visit(BufferUploadCommand const *cmd) noexcept override;
    void visit(BufferDownloadCommand const *cmd) noexcept override;
    void visit(BufferCopyCommand const *cmd) noexcept override;
    void visit(BufferToTextureCopyCommand const *cmd) noexcept override {}
    void visit(ShaderDispatchCommand const *cmd) noexcept override;
    void visit(TextureUploadCommand const *cmd) noexcept override;
    void visit(TextureDownloadCommand const *cmd) noexcept override;
    void visit(TextureCopyCommand const *cmd) noexcept override;
    void visit(TextureToBufferCopyCommand const *cmd) noexcept override;
    void visit(AccelUpdateCommand const *cmd) noexcept override;
    void visit(AccelBuildCommand const *cmd) noexcept override;
    void visit(MeshUpdateCommand const *cmd) noexcept override;
    void visit(MeshBuildCommand const *cmd) noexcept override;
    void visit(BindlessArrayUpdateCommand const *cmd) noexcept override;
};
}// namespace lc::ispc