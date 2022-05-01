#pragma once
#include <vstl/functional.h>
#include <DXRuntime/CommandBuffer.h>
#include <vstl/StackAllocator.h>
#include <Resource/UploadBuffer.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <vstl/LockFreeArrayQueue.h>
namespace toolhub::directx {
class CommandAllocatorBase : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBuffer;

protected:
    Device *device;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    mutable vstd::optional<CommandBuffer> cbuffer;
    D3D12_COMMAND_LIST_TYPE type;
    IGpuAllocator *resourceAllocator;
    CommandAllocatorBase(Device *device, IGpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);
    vstd::LockFreeArrayQueue<vstd::move_only_func<void()>> executeAfterComplete;

public:
    template<typename Func>
        requires(std::is_constructible_v<vstd::move_only_func<void()>, Func &&>)
    void ExecuteAfterComplete(Func &&func) {
        executeAfterComplete.Push(std::forward<Func>(func));
    }
    virtual ~CommandAllocatorBase();
    ID3D12CommandAllocator *Allocator() const { return allocator.Get(); }
    D3D12_COMMAND_LIST_TYPE Type() const { return type; }
    virtual void Reset(CommandQueue *queue);
    CommandBuffer *GetBuffer() const;
    void Execute(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    void ExecuteAndPresent(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, IDXGISwapChain3 *swapchain);
    void Complete(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    void Complete_Async(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
};
}// namespace toolhub::directx