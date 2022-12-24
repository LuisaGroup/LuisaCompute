#pragma once
#include <vstl/functional.h>
#include <DXRuntime/CommandBuffer.h>
#include <vstl/stack_allocator.h>
#include <Resource/UploadBuffer.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <vstl/lockfree_array_queue.h>
namespace toolhub::directx {
class CommandAllocatorBase : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBuffer;

protected:
    Device *device;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    mutable vstd::optional<CommandBuffer> cbuffer;
    D3D12_COMMAND_LIST_TYPE type;
    GpuAllocator *resourceAllocator;
    CommandAllocatorBase(Device *device, GpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);
    vstd::LockFreeArrayQueue<vstd::function<void()>> executeAfterComplete;
    // Embeded with external queue
    void WaitExternQueue(ID3D12Fence *fence, uint64 fenceIndex);

public:
    template<typename Func>
        requires(std::is_constructible_v<vstd::function<void()>, Func &&>)
    void ExecuteAfterComplete(Func &&func) {
        executeAfterComplete.Push(std::forward<Func>(func));
    }
    virtual ~CommandAllocatorBase();
    ID3D12CommandAllocator *Allocator() const { return allocator.Get(); }
    D3D12_COMMAND_LIST_TYPE Type() const { return type; }
    virtual void Reset(CommandQueue *queue);
    CommandBuffer *GetBuffer() const;
    void Execute(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    void ExecuteAndPresent(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, IDXGISwapChain3 *swapchain, bool vsync);
    void Complete(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
};
}// namespace toolhub::directx