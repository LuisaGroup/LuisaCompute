#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandQueue.h>
namespace lc::dx {
template<typename Pack>
uint64 CommandAllocator::Visitor<Pack>::allocate(uint64 size) {
    auto packPtr = new Pack(
        self->_device,
        size,
        self->_resource_allocator);
    return reinterpret_cast<uint64>(packPtr);
}
template<typename Pack>
vstd::unique_ptr<Pack> CommandAllocator::Visitor<Pack>::create(uint64 size) {
    return vstd::make_unique<Pack>(
        self->_device,
        size,
        self->_resource_allocator);
}

template<typename Pack>
void CommandAllocator::Visitor<Pack>::deallocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
}
template<typename T>
void CommandAllocator::BufferAllocator<T>::clear() {
    _large_buffers.clear();
    _alloc.dispose();
}
template<typename T>
CommandAllocator::BufferAllocator<T>::BufferAllocator(size_t initCapacity)
    : _alloc(initCapacity, &visitor) {
}
template<typename T>
CommandAllocator::BufferAllocator<T>::~BufferAllocator() = default;
template<typename T>
BufferView CommandAllocator::BufferAllocator<T>::allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = _alloc.allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = _large_buffers.emplace_back(visitor.create(size));
        return BufferView(v.get(), 0, size);
    }
}
template<typename T>
BufferView CommandAllocator::BufferAllocator<T>::allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = _alloc.allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = _large_buffers.emplace_back(visitor.create(size));
        return BufferView(v.get(), 0, size);
    }
}
// void CommandAllocator::WaitExternQueue(ID3D12Fence *fence, uint64 fenceIndex) {
//     if (device->device_settings) {
//         auto after_queue = device->device_settings->GetQueue();
//         if (after_queue) {
//             after_queue->Wait(fence, fenceIndex);
//         }
//     }
// }
void CommandAllocator::execute(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex,
    luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty) {
    if (cmdlist_is_empty && swapChains.empty()) return;
    auto present = [&](IDXGISwapChain *swapchain, bool vsync) {
        if (!swapchain) return;
        HRESULT present_hresult;
        if (vsync) {
            present_hresult = swapchain->Present(1, 0);
        } else {
            present_hresult = swapchain->Present(0, DXGI_PRESENT_ALLOW_TEARING | DXGI_PRESENT_DO_NOT_WAIT);
        }
        // Present may return a non-S_OK success status when an occluded or
        // do-not-wait frame is skipped. Only FAILED values are actual errors.
        if (FAILED(present_hresult)) {
            LUISA_WARNING("Present failed with HRESULT {:#x}.",
                          static_cast<uint32_t>(present_hresult));

            if (present_hresult == DXGI_ERROR_DEVICE_REMOVED || present_hresult == DXGI_ERROR_DEVICE_HUNG || present_hresult == DXGI_ERROR_DEVICE_RESET) {
                ThrowIfFailed(present_hresult);
            }
        }
    };
    ID3D12CommandList *cmdList = _cbuffer->cmd_list();
    auto cmdQueue = queue->queue();
    if (!_device->device_settings) {
        if (!cmdlist_is_empty) {
            cmdQueue->ExecuteCommandLists(
                1,
                &cmdList);
        }
        for (auto &i : swapChains)
            present(i.first, i.second);
        ThrowIfFailed(cmdQueue->Signal(fence, fenceIndex));
    } else {
        if (!cmdlist_is_empty) {
            if (!_device->device_settings->ExecuteCommandList(cmdQueue, static_cast<ID3D12GraphicsCommandList *>(cmdList)))
                cmdQueue->ExecuteCommandLists(
                    1,
                    &cmdList);
        }
        for (auto &i : swapChains)
            present(i.first, i.second);
        if (!_device->device_settings->SignalFence(cmdQueue, fence, fenceIndex)) {
            ThrowIfFailed(cmdQueue->Signal(fence, fenceIndex));
        }
    }
}
void CommandAllocator::complete(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    _device->wait_fence(fence, fenceIndex);
    while (auto evt = _execute_after_complete.dequeue()) {
        (*evt)();
    }
    _res_dispose_list_mtx.lock();
    auto vec = std::move(_res_dispose_list);
    _res_dispose_list_mtx.unlock();
    (void)vec;
}

CommandBuffer *CommandAllocator::get_buffer() const {
    return _cbuffer;
}
static size_t TEMP_SIZE = 1024ull * 1024ull;
CommandAllocator::CommandAllocator(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : _device(device),
      _type(type),
      _resource_allocator(resourceAllocator),
      _rtv_visitor(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
      _dsv_visitor(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV),
      _upload_allocator(TEMP_SIZE),
      _default_allocator(TEMP_SIZE),
      _readback_allocator(TEMP_SIZE),
      rtv_allocator(64, &_rtv_visitor),
      dsv_allocator(64, &_dsv_visitor) {

    _cbuffer.create(
        device,
        this);
    _cbuffer->_reset();
    _upload_allocator.visitor.self = this;
    _default_allocator.visitor.self = this;
    _readback_allocator.visitor.self = this;
}
ID3D12CommandAllocator *CommandAllocator::allocator() {
    if (!_allocator) {
        ThrowIfFailed(
            _device->device->CreateCommandAllocator(_type, IID_PPV_ARGS(_allocator.GetAddressOf())));
        ThrowIfFailed(
            _allocator->Reset());
    }
    return _allocator.Get();
}

CommandAllocator::~CommandAllocator() {
    _cbuffer.destroy();
}
void CommandAllocator::reset(CommandQueue *queue) {
    _readback_allocator.clear();
    _upload_allocator.clear();
    _default_allocator.clear();
    rtv_allocator.clear();
    dsv_allocator.clear();
    if (_allocator)
        ThrowIfFailed(
            _allocator->Reset());
    _cbuffer->_reset();
}

DefaultBuffer const *CommandAllocator::allocate_scratch_buffer(size_t targetSize) {
    if (_scratch_buffer) {
        if (_scratch_buffer->GetByteSize() < targetSize) {
            size_t allocSize = _scratch_buffer->GetByteSize();
            while (allocSize < targetSize) {
                allocSize = std::max<size_t>(allocSize + 1, (allocSize * 3) / 2);
            }
            dispose_after_complete(std::move(_scratch_buffer));
            allocSize = CalcAlign(allocSize, 65536);
            _scratch_buffer = vstd::create_unique(new DefaultBuffer(_device, allocSize, _device->default_allocator.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        return _scratch_buffer.get();
    } else {
        targetSize = CalcAlign(targetSize, 65536);
        _scratch_buffer = vstd::create_unique(new DefaultBuffer(_device, targetSize, _device->default_allocator.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        return _scratch_buffer.get();
    }
}

BufferView CommandAllocator::get_temp_readback_buffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return _readback_allocator.allocate(size);
    } else {
        return _readback_allocator.allocate(size, align);
    }
}

BufferView CommandAllocator::get_temp_upload_buffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return _upload_allocator.allocate(size);
    } else {
        return _upload_allocator.allocate(size, align);
    }
}
BufferView CommandAllocator::get_temp_default_buffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return _default_allocator.allocate(size);
    } else {
        return _default_allocator.allocate(size, align);
    }
}

uint64 CommandAllocator::DescHeapVisitor::allocate(uint64 size) {
    return reinterpret_cast<uint64>(new DescriptorHeap(
        device,
        type,
        size, false));
}

void CommandAllocator::DescHeapVisitor::deallocate(uint64 handle) {
    delete reinterpret_cast<DescriptorHeap *>(handle);
}
}// namespace lc::dx
