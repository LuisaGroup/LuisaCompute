#pragma once
#include <DXRuntime/Device.h>
#include <vstl/lockfree_array_queue.h>
#include <DxRuntime/DxPtr.h>
namespace toolhub::directx {
class CommandBuffer;
class CommandAllocator;
class CommandAllocatorBase;
class GpuAllocator;
class LCEvent;
class CommandQueue : vstd::IOperatorNewBase {
public:
    using AllocatorPtr = vstd::unique_ptr<CommandAllocator>;

private:
    using CallbackEvent = vstd::variant<
        std::pair<AllocatorPtr, uint64>,
        std::pair<vstd::function<void()>, uint64>,
        std::pair<vstd::vector<vstd::function<void()>>, uint64>,
        std::pair<LCEvent const *, uint64>>;
    Device *device;
    GpuAllocator *resourceAllocator;
    D3D12_COMMAND_LIST_TYPE type;
    std::mutex mtx;
    std::thread thd;
    std::condition_variable waitCv;
    std::condition_variable mainCv;
    uint64 executedFrame = 0;
    std::atomic_uint64_t lastFrame = 0;
    bool enabled = true;
    DxPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> cmdFence;
    vstd::LockFreeArrayQueue<AllocatorPtr> allocatorPool;
    vstd::LockFreeArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();
	template <typename Func>
    uint64 _Execute(AllocatorPtr &&alloc, Func &&callback);


public:
    void WaitFrame(uint64 lastFrame);
    uint64 LastFrame() const { return lastFrame; }
    ID3D12CommandQueue *Queue() const { return queue; }
    CommandQueue(
        Device *device,
        GpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    ~CommandQueue();
    AllocatorPtr CreateAllocator(size_t maxAllocCount);
    void Callback(vstd::function<void()> &&f);
    void AddEvent(LCEvent const *evt);
    uint64 Execute(AllocatorPtr &&alloc);
    uint64 ExecuteCallback(AllocatorPtr &&alloc, vstd::function<void()> &&callback);
    uint64 ExecuteCallbacks(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks);
	void ExecuteEmpty(AllocatorPtr &&alloc);
    void ExecuteEmptyCallbacks(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks);
    uint64 ExecuteAndPresent(AllocatorPtr &&alloc, IDXGISwapChain3 *swapChain, bool vsync);
    void Complete(uint64 fence);
    void Complete();
    void ForceSync(
        AllocatorPtr &alloc,
        CommandBuffer &cb);
    KILL_MOVE_CONSTRUCT(CommandQueue)
    KILL_COPY_CONSTRUCT(CommandQueue)
};
}// namespace toolhub::directx