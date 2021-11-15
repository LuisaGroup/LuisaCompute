#pragma vengine_package ispc_vsproject

#include "ispc_runtime.h"
namespace lc::ispc {
void CommandExecutor::visit(BufferUploadCommand const *cmd) noexcept {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(cmd->handle());
    memcpy(ptr + cmd->offset(), cmd->data(), cmd->size());
}
void CommandExecutor::visit(BufferDownloadCommand const *cmd) noexcept {
    uint8_t const *ptr = reinterpret_cast<uint8_t const *>(cmd->handle());
    memcpy(cmd->data(), ptr + cmd->offset(), cmd->size());
}
void CommandExecutor::visit(BufferCopyCommand const *cmd) noexcept {
    uint8_t const *src = reinterpret_cast<uint8_t const *>(cmd->src_handle());
    uint8_t *dst = reinterpret_cast<uint8_t *>(cmd->dst_handle());
    memcpy(dst + cmd->dst_offset(), src + cmd->src_offset(), cmd->size());
}
struct ShaderDispatcher {
    Function func;
    Shader::ArgVector &vec;
    Shader *sd;
    void operator()(uint, ShaderDispatchCommand::BufferArgument const &arg) {
        Shader::PackArg<float *>(vec, reinterpret_cast<float *>(arg.handle));
    }
    void operator()(uint, ShaderDispatchCommand::TextureArgument const &arg) {
    }
    void operator()(uint var_id, std::span<std::byte const> arg) {
        Shader::PackArr(vec, arg.data(), arg.size(), CodegenUtility::GetTypeAlign(*func.arguments()[sd->GetArgIndex(var_id)].type()));
    }
    void operator()(uint, ShaderDispatchCommand::BindlessArrayArgument const &arg) {}
    void operator()(uint, ShaderDispatchCommand::AccelArgument const &arg) {}
};
void CommandExecutor::visit(ShaderDispatchCommand const *cmd) noexcept {
    Shader::ArgVector vec;
    Shader *sd = reinterpret_cast<Shader *>(cmd->handle());
    ShaderDispatcher disp{cmd->kernel(), vec, sd};
    cmd->decode(disp);
    auto szzz = cmd->dispatch_size();
    handles.emplace_back(sd->dispatch(tPool, cmd->dispatch_size(), vec), std::move(vec));
}
void CommandExecutor::visit(TextureUploadCommand const *cmd) noexcept {}
void CommandExecutor::visit(TextureDownloadCommand const *cmd) noexcept {}
void CommandExecutor::visit(TextureCopyCommand const *cmd) noexcept {}
void CommandExecutor::visit(TextureToBufferCopyCommand const *cmd) noexcept {}
void CommandExecutor::visit(AccelUpdateCommand const *cmd) noexcept {}
void CommandExecutor::visit(AccelBuildCommand const *cmd) noexcept {}
void CommandExecutor::visit(MeshUpdateCommand const *cmd) noexcept {}
void CommandExecutor::visit(MeshBuildCommand const *cmd) noexcept {}
void CommandExecutor::visit(BindlessArrayUpdateCommand const *cmd) noexcept {}
}// namespace lc::ispc