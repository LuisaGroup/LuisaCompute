#pragma once

#include <luisa/runtime/context.h>
#ifdef byte
#undef byte
#endif
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/runtime/rhi/device_interface.h>

struct IDxcCompiler3;
struct IDxcLibrary;
struct IDxcUtils;
namespace luisa::compute {
struct DirectXHeap {
    uint64_t handle;
    ID3D12Heap *heap;
    size_t offset;
};
class DirectXFuncTable {
public:
    [[nodiscard]] virtual DirectXHeap AllocateBufferHeap(
        luisa::string_view name,
        uint64_t sizeBytes,
        D3D12_HEAP_TYPE heapType,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    [[nodiscard]] virtual DirectXHeap AllocateTextureHeap(
        luisa::string_view name,
        size_t sizeBytes,
        bool isRenderTexture,
        D3D12_HEAP_FLAGS extraFlags) const noexcept = 0;
    virtual void DeAllocateHeap(uint64_t handle) const noexcept = 0;
};
struct DirectXDeviceConfigExt : public DeviceConfigExt {

    struct ExternalDevice {
        ID3D12Device *device;
        IDXGIAdapter1 *adapter;
        IDXGIFactory2 *factory;
    };

    [[nodiscard]] virtual bool callback_thread_use_fiber() const noexcept { return false; }

    virtual luisa::optional<ExternalDevice> CreateExternalDevice() noexcept { return {}; }
    // Called during create_device
    virtual void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory,
        DirectXFuncTable const *funcTable,
        luisa::BinaryIO const *shaderIo,
        IDxcCompiler3 *dxcCompiler,
        IDxcLibrary *dxcLibrary,
        IDxcUtils *dxcUtils,
        ID3D12DescriptorHeap *shaderDescriptor,
        ID3D12DescriptorHeap *samplerDescriptor) noexcept {}

    // plugin resources
    virtual ID3D12CommandQueue *CreateQueue(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    virtual ID3D12GraphicsCommandList *BorrowCommandList(D3D12_COMMAND_LIST_TYPE type) noexcept { return nullptr; }

    // Custom callback
    // return true if this callback is implemented
    virtual bool ExecuteCommandList(
        ID3D12CommandQueue *queue,
        ID3D12GraphicsCommandList *cmdList) noexcept { return false; }

    virtual bool SignalFence(
        ID3D12CommandQueue *queue,
        ID3D12Fence *fence, uint64_t fenceIndex) noexcept { return false; }

    virtual ~DirectXDeviceConfigExt() noexcept override = default;
};

}// namespace luisa::compute
#ifdef byte
#undef byte
#endif