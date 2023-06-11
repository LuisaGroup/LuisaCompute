#pragma once
#include <luisa/runtime/context.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/device_interface.h>
namespace luisa::compute {
struct DirectXDeviceConfigExt : public DeviceConfigExt, public vstd::IOperatorNewBase {
    virtual ID3D12Device *GetDevice() noexcept = 0;
    virtual IDXGIAdapter1 *GetAdapter() noexcept = 0;
    virtual IDXGIFactory2 *GetDXGIFactory() noexcept = 0;
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
};
}// namespace luisa::compute
