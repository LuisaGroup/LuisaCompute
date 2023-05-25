#pragma once
#include <dstorage/dstorage.h>
#include <DXRuntime/Device.h>
#include <vstl/lockfree_array_queue.h>
#include <DXRuntime/DxPtr.h>
#include <DXApi/CmdQueueBase.h>
#include <runtime/command_list.h>
namespace lc::dx {
class LCEvent;
class DStorageFileImpl : public vstd::IOperatorNewBase {
public:
    ComPtr<IDStorageFile> file;
    size_t size_bytes;
    DStorageFileImpl(ComPtr<IDStorageFile> &&file, size_t size_bytes) : file{std::move(file)}, size_bytes{size_bytes} {}
};
class DStorageCommandQueue : public CmdQueueBase{
    struct WaitQueueHandle {
        HANDLE handles[2];
    };
    struct CallbackEvent {
        using Variant = vstd::variant<
            WaitQueueHandle,
            vstd::vector<vstd::function<void()>>,
            LCEvent const *>;
        Variant evt;
        uint64_t fence;
        bool wakeupThread;
        template<typename Arg>
            requires(std::is_constructible_v<Variant, Arg &&>)
        CallbackEvent(Arg &&arg,
                      uint64_t fence,
                      bool wakeupThread)
            : evt{std::forward<Arg>(arg)}, fence{fence}, wakeupThread{wakeupThread} {}
    };
    IDStorageFactory *factory;
    std::mutex mtx;
    std::mutex exec_mtx;
    std::thread thd;
    std::condition_variable waitCv;
    std::condition_variable mainCv;
    uint64 executedFrame = 0;
    std::atomic_uint64_t lastFrame = 0;
    bool enabled = true;
    ComPtr<IDStorageQueue2> fileQueue;
    ComPtr<IDStorageQueue2> memQueue;
    vstd::LockFreeArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    void Signal(ID3D12Fence *fence, UINT64 value);
    uint64 LastFrame() const { return lastFrame; }
    DStorageCommandQueue(IDStorageFactory *factory, Device *device);
    void AddEvent(LCEvent const *evt);
    uint64 Execute(luisa::compute::CommandList &&list);
    void Complete(uint64 fence);
    void Complete();
    KILL_MOVE_CONSTRUCT(DStorageCommandQueue)
    KILL_COPY_CONSTRUCT(DStorageCommandQueue)
    ~DStorageCommandQueue();
};
}// namespace lc::dx