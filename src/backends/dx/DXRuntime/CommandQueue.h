#pragma once
#include <DXRuntime/Device.h>
#include <vstl/LockFreeArrayQueue.h>
namespace toolhub::directx {
class CommandBuffer;
class CommandAllocator;
class CommandAllocatorBase;
class IGpuAllocator;
class LCEvent;
class CommandQueue : vstd::IOperatorNewBase {
public:
    using AllocatorPtr = vstd::unique_ptr<CommandAllocator>;

private:
    using CallbackEvent = vstd::variant<
        std::pair<AllocatorPtr, uint64>,
        vstd::move_only_func<void()>,
        std::pair<LCEvent const *, uint64>>;
    Device *device;
    IGpuAllocator *resourceAllocator;
    D3D12_COMMAND_LIST_TYPE type;
    std::mutex mtx;
    std::thread thd;
    std::condition_variable waitCv;
    std::condition_variable mainCv;
    uint64 executedFrame = 0;
    uint64 lastFrame = 0;
    bool enabled = true;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> cmdFence;
    vstd::LockFreeArrayQueue<AllocatorPtr> allocatorPool;
    vstd::LockFreeArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    void ExecuteDuringWaiting();
    uint64 LastFrame() const { return lastFrame; }
    ID3D12CommandQueue *Queue() const { return queue.Get(); }
    CommandQueue(
        Device *device,
        IGpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    ~CommandQueue();
    AllocatorPtr CreateAllocator(size_t maxAllocCount);
    void Callback(vstd::move_only_func<void()> &&f);
    void AddEvent(LCEvent const *evt);
    uint64 Execute(AllocatorPtr &&alloc);
    void ExecuteEmpty(AllocatorPtr &&alloc);
    uint64 ExecuteAndPresent(AllocatorPtr &&alloc, IDXGISwapChain3 *swapChain);
    void Complete(uint64 fence);
    void Complete();
    void ForceSync(
        AllocatorPtr  &alloc,
        CommandBuffer &cb);
    KILL_MOVE_CONSTRUCT(CommandQueue)
    KILL_COPY_CONSTRUCT(CommandQueue)
};
}// namespace toolhub::directx