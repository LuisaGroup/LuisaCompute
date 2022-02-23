#pragma once
#include <DXRuntime/Device.h>
#include <vstl/LockFreeArrayQueue.h>
namespace toolhub::directx {
class CommandBuffer;
class CommandAllocator;
class IGpuAllocator;
class LCEvent;
class CommandQueue : vstd::IOperatorNewBase {
public:
    using AllocatorPtr = vstd::unique_ptr<CommandAllocator>;

private:
    using CallbackEvent = vstd::variant<
        AllocatorPtr,
        std::pair<LCEvent const*, uint64>>;
    Device *device;
    IGpuAllocator *resourceAllocator;
    D3D12_COMMAND_LIST_TYPE type;
    std::mutex mtx;
    std::thread thd;
    std::condition_variable waitCv;
    std::condition_variable mainCv;
    std::atomic_uint64_t executedFrame = 0;
    uint64 lastFrame = 0;
    bool enabled = true;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> cmdFence;
    vstd::LockFreeArrayQueue<AllocatorPtr> allocatorPool;
    vstd::LockFreeArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    ID3D12CommandQueue *Queue() const { return queue.Get(); }
    CommandQueue(
        Device *device,
        IGpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    ~CommandQueue();
    AllocatorPtr CreateAllocator(size_t maxAllocCount);
    void AddEvent(LCEvent const *evt);
    uint64 Execute(AllocatorPtr &&alloc);
    void Complete(uint64 fence);
    KILL_MOVE_CONSTRUCT(CommandQueue)
    KILL_COPY_CONSTRUCT(CommandQueue)
};
}// namespace toolhub::directx