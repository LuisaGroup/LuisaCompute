#pragma once

#include <vstl/Common.h>
#include <runtime/command.h>
#include "ispc_shader.h"
#include <vstl/LockFreeArrayQueue.h>
#include "ispc_event.h"
using namespace luisa;
using namespace luisa::compute;
namespace lc::ispc {
class CommandExecutor : public CommandVisitor {
public:
    struct Signal {
        Event *evt;
    };
    struct Wait {
        Event *evt;
    };
    using HandleType = vstd::variant<
        ThreadTaskHandle,
        BufferUploadCommand,
        BufferDownloadCommand,
        BufferCopyCommand,
        TextureUploadCommand,
        TextureDownloadCommand,
        Signal,
        Wait>;

private:
    size_t outsideTaskCount = 0;
    ThreadPool *tPool;
    std::atomic_size_t taskCount = 0;
    std::atomic_size_t executedTask = 0;
    bool enabled = true;
    std::thread dispatchThread;
    std::mutex dispMtx;
    std::condition_variable mainThdCv;
    std::condition_variable dispThdCv;
    vstd::LockFreeArrayQueue<HandleType> syncTasks;

public:
    template <typename T>
    void AddTask(T&& t) {
        syncTasks.Push(std::forward<T>(t));
        outsideTaskCount++;
    }
    CommandExecutor(ThreadPool *tPool);
    ~CommandExecutor();

    void ThreadExecute();
    void WaitThread();
    void ExecuteDispatch();
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