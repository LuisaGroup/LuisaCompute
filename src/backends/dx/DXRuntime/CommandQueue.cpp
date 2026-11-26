#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/GpuAllocator.h>
#include <DXApi/LCEvent.h>
#include <luisa/core/intrin.h>
namespace lc::dx {
CommandQueue::CommandQueue(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : _device(device),
      _resource_allocator(resourceAllocator),
      _type(type),
      _thd([this] { _execute_thread(); }) {
    auto CreateQueue = [&] {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = type;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
        switch (type) {
            case D3D12_COMMAND_LIST_TYPE_DIRECT:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
                break;
            default:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
                break;
        }
        ThrowIfFailed(device->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(_queue.GetAddressOf())));
    };
    if (device->device_settings) {
        _queue = {device->device_settings->CreateQueue(type), false};
        if (!_queue) [[unlikely]] {
            CreateQueue();
        }
    } else {
        CreateQueue();
    }
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&_cmd_fence)));
}
CommandQueue::AllocatorPtr CommandQueue::create_allocator(size_t maxAllocCount) {
    if (maxAllocCount != std::numeric_limits<uint64_t>::max()) {
        if (_last_frame > maxAllocCount)
            complete(_last_frame - maxAllocCount);
    }
    auto newPtr = _allocator_pool.dequeue();
    if (newPtr) {
        (*newPtr)->get_buffer()->update_command_buffer(_device);
        return std::move(*newPtr);
    }
    return AllocatorPtr(new CommandAllocator(_device, _resource_allocator, _type));
}

void CommandQueue::add_event(LCEvent const *evt, uint64_t fenceIdx) {
    ++_last_frame;
    {
        std::lock_guard lck{_mtx};
        _executed_allocators.enqueue(evt, fenceIdx, true);
        _work_epoch.fetch_add(1u, std::memory_order_release);
    }
    _work_epoch.notify_one();
}

void CommandQueue::_execute_thread() {
    auto drain_queue = [&] {
        uint64_t fence = 0;
        bool wakeupThread = false;
        auto Weakup = [&] {
            if (wakeupThread) {
                uint64_t prev_value = _executed_frame.load(std::memory_order_relaxed);
                while (prev_value < fence &&
                       !_executed_frame.compare_exchange_weak(
                           prev_value, fence,
                           std::memory_order_release,
                           std::memory_order_relaxed)) {
                }
                _executed_frame.notify_all();
            }
        };
        auto ExecuteAllocator = [&](AllocatorPtr &b) {
            b->complete(this, _cmd_fence.Get(), fence);
            {
                uint64_t prev_alloc = _completed_allocator_frame.load(std::memory_order_relaxed);
                while (prev_alloc < fence &&
                       !_completed_allocator_frame.compare_exchange_weak(
                           prev_alloc, fence,
                           std::memory_order_release,
                           std::memory_order_relaxed)) {
                }
                _completed_allocator_frame.notify_all();
            }
            // Wake Sync()/complete() waiters once GPU completion is visible,
            // overlapping allocator/command-list reset with host scheduling.
            Weakup();
            b->reset(this);
            _allocator_pool.enqueue(std::move(b));
        };
        auto ExecuteCallbacks = [&](vstd::vector<vstd::function<void()>> &vec) {
            for (auto &&i : vec) {
                i();
            }
            Weakup();
        };

        auto ExecuteEvent = [&](LCEvent const *evt) {
            _device->wait_fence(evt->fence(), fence);
            {
                std::lock_guard lck(evt->event_mtx);
                const auto next_val = std::max<uint64_t>(
                    fence, evt->finished_event.load(std::memory_order_relaxed));
                evt->finished_event.store(next_val, std::memory_order_release);
            }
            evt->finished_event.notify_all();
            if (wakeupThread) {
                _executed_frame.fetch_add(1u, std::memory_order_release);
                _executed_frame.notify_all();
            }
        };
        auto ExecuteHandle = [&](WaitFence) {
            _device->wait_fence(_cmd_fence.Get(), fence);
            Weakup();
        };
        while (true) {
            vstd::optional<CallbackEvent> b;
            {
                std::lock_guard lck{_mtx};
                b = _executed_allocators.dequeue();
            }
            if (!b) break;
            fence = b->fence;
            wakeupThread = b->wakeupThread;
            b->evt.multi_visit(
                ExecuteAllocator,
                ExecuteCallbacks,
                ExecuteEvent,
                ExecuteHandle);
        }
    };
    while (_enabled.load(std::memory_order_acquire)) {
        const uint64_t epoch_before = _work_epoch.load(std::memory_order_acquire);
        drain_queue();
        for (uint32_t spin = 0u; spin < 8192u; ++spin) {
            if (_work_epoch.load(std::memory_order_acquire) != epoch_before ||
                !_enabled.load(std::memory_order_acquire)) {
                break;
            }
            LUISA_INTRIN_PAUSE();
        }
        if (_enabled.load(std::memory_order_acquire) &&
            _work_epoch.load(std::memory_order_acquire) == epoch_before) {
            _work_epoch.wait(epoch_before, std::memory_order_acquire);
        }
    }
    drain_queue();
}
void CommandQueue::force_sync(
    AllocatorPtr &alloc,
    CommandBuffer &cb) {
    cb._close();
    complete();
    auto curFrame = ++_last_frame;
    alloc->execute(this, _cmd_fence.Get(), curFrame, {}, false);
    alloc->complete(this, _cmd_fence.Get(), curFrame);
    {
        uint64_t prev_alloc = _completed_allocator_frame.load(std::memory_order_relaxed);
        while (prev_alloc < curFrame &&
               !_completed_allocator_frame.compare_exchange_weak(
                   prev_alloc, curFrame,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
        }
        _completed_allocator_frame.notify_all();
    }
    alloc->reset(this);
    _executed_frame.store(curFrame, std::memory_order_release);
    _executed_frame.notify_all();

    cb._reset();
}
CommandQueue::~CommandQueue() {
    {
        std::lock_guard lck(_mtx);
        _enabled.store(false, std::memory_order_release);
        _work_epoch.fetch_add(1u, std::memory_order_release);
    }
    _work_epoch.notify_one();
    _thd.join();
}
void CommandQueue::wait_allocator_frame(uint64_t fence) const noexcept {
    for (;;) {
        if (_completed_allocator_frame.load(std::memory_order_acquire) >= fence) {
            return;
        }
        for (uint32_t spin = 0u; spin < 4096u; ++spin) {
            if (_completed_allocator_frame.load(std::memory_order_acquire) >= fence) {
                return;
            }
            LUISA_INTRIN_PAUSE();
        }
        const auto cur = _completed_allocator_frame.load(std::memory_order_acquire);
        if (cur >= fence) {
            return;
        }
        _completed_allocator_frame.wait(cur, std::memory_order_acquire);
    }
}
void CommandQueue::wait_frame(uint64_t lastFrame) {
    if (lastFrame > 0)
        _queue->Wait(_cmd_fence.Get(), lastFrame);
}
void CommandQueue::signal() {
    auto curFrame = ++_last_frame;
    ThrowIfFailed(_queue->Signal(_cmd_fence.Get(), curFrame));
    {
        std::lock_guard lck{_mtx};
        _executed_allocators.enqueue(WaitFence{}, curFrame, true);
        _work_epoch.fetch_add(1u, std::memory_order_release);
    }
    _work_epoch.notify_one();
}
uint64_t CommandQueue::execute(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks, luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty) {
    auto curFrame = ++_last_frame;
    alloc->execute(this, _cmd_fence.Get(), curFrame, swapChains, cmdlist_is_empty);
    if (cmdlist_is_empty) {
        _allocator_pool.enqueue(std::move(alloc));
        if (!callbacks.empty()) {
            {
                std::lock_guard lck{_mtx};
                _executed_allocators.enqueue(std::move(callbacks), curFrame, true);
                _work_epoch.fetch_add(1u, std::memory_order_release);
            }
            _work_epoch.notify_one();
        }
    } else {
        {
            std::lock_guard lck{_mtx};
            _executed_allocators.enqueue(std::move(alloc), curFrame, callbacks.empty());
            if (!callbacks.empty())
                _executed_allocators.enqueue(std::move(callbacks), curFrame, true);
            _work_epoch.fetch_add(1u, std::memory_order_release);
        }
        _work_epoch.notify_one();
    }
    return curFrame;
}

void CommandQueue::complete(uint64_t fence) {
    for (;;) {
        if (_executed_frame.load(std::memory_order_acquire) >= fence) {
            return;
        }
        for (uint32_t spin = 0u; spin < 4096u; ++spin) {
            if (_executed_frame.load(std::memory_order_acquire) >= fence) {
                return;
            }
            LUISA_INTRIN_PAUSE();
        }
        const auto cur = _executed_frame.load(std::memory_order_acquire);
        if (cur >= fence) {
            return;
        }
        _executed_frame.wait(cur, std::memory_order_acquire);
    }
}
void CommandQueue::complete() {
    complete(_last_frame);
}

}// namespace lc::dx
