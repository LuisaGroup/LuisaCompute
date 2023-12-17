#pragma once

#include <luisa/runtime/context.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

struct DirectXDeviceConfigExt : public DeviceConfigExt, public vstd::IOperatorNewBase {

    struct ExternalDevice {
        ID3D12Device *device;
        IDXGIAdapter1 *adapter;
        IDXGIFactory2 *factory;
    };

    virtual vstd::optional<ExternalDevice> CreateExternalDevice() noexcept { return {}; }

    virtual void ReadbackDX12Device(
        ID3D12Device *device,
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *factory) noexcept {}

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

    ~DirectXDeviceConfigExt() noexcept override = default;
};

}// namespace luisa::compute
