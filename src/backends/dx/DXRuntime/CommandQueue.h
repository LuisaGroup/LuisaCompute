#pragma once
#include <DXRuntime/Device.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <DXRuntime/DxPtr.h>
#include <dxgi1_4.h>
namespace lc::dx {
class CommandBuffer;
class CommandAllocator;
class GpuAllocator;
class LCEvent;
class CommandQueue : vstd::IOperatorNewBase {
public:
    using AllocatorPtr = vstd::unique_ptr<CommandAllocator>;

private:
    struct WaitFence {
    };
    struct CallbackEvent {
        using Variant = vstd::variant<
            AllocatorPtr,
            vstd::vector<vstd::function<void()>>,
            LCEvent const *,
            WaitFence>;
        Variant evt;
        uint64_t fence;
        bool wakeupThread;
        template<typename Arg>
            requires(luisa::is_constructible_v<Variant, Arg &&>)
        CallbackEvent(Arg &&arg,
                      uint64_t fence,
                      bool wakeupThread)
            : evt{std::forward<Arg>(arg)}, fence{fence}, wakeupThread{wakeupThread} {}
    };
    std::atomic_bool _enabled = true;
    std::atomic_uint64_t _executed_frame = 0;
    std::atomic_uint64_t _completed_allocator_frame = 0;
    std::atomic_uint64_t _last_frame = 0;
    Device *_device;
    GpuAllocator *_resource_allocator;
    D3D12_COMMAND_LIST_TYPE _type;
    luisa::spin_mutex _mtx;
    DxPtr<ID3D12CommandQueue> _queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> _cmd_fence;
    vstd::LockFreeArrayQueue<AllocatorPtr> _allocator_pool;
    vstd::SingleThreadArrayQueue<CallbackEvent> _executed_allocators;
    void _execute_thread();

public:
    void wait_frame(uint64 lastFrame);
    uint64 last_frame() const { return _last_frame; }
    [[nodiscard]] uint64_t completed_allocator_frame() const noexcept {
        return _completed_allocator_frame.load(std::memory_order_acquire);
    }
    void wait_allocator_frame(uint64_t fence) const noexcept;
    ID3D12CommandQueue *queue() const { return _queue; }
    CommandQueue(
        Device *device,
        GpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    ~CommandQueue();
    AllocatorPtr create_allocator(size_t maxAllocCount);
    void add_event(LCEvent const *evt, uint64_t fenceIdx);
    void signal();
    uint64_t execute(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks, luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty);
    [[nodiscard]] D3D12_COMMAND_LIST_TYPE Type() const noexcept { return _type; }
    void complete(uint64 fence);
    void complete();
    void force_sync(
        AllocatorPtr &alloc,
        CommandBuffer &cb);
    KILL_MOVE_CONSTRUCT(CommandQueue)
    KILL_COPY_CONSTRUCT(CommandQueue)
private:
    // make sure thread always construct after all members
    std::thread _thd;
};
}// namespace lc::dx
