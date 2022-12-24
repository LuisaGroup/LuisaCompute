#pragma once
#include <runtime/context.h>
#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_4.h>
#include <vstl/common.h>
namespace luisa::compute {
struct DirectXExternalRuntime : public vstd::IOperatorNewBase {
    virtual ID3D12Device *GetDevice() = 0;
    virtual IDXGIAdapter1 *GetAdapter() = 0;
    virtual IDXGIFactory4 *GetDXGIFactory() = 0;
    // queue is nullable
    virtual ID3D12CommandQueue *GetQueue() = 0;
};
struct DirectXDeviceSettings : public DeviceConfig, public vstd::IOperatorNewBase {
    inline static Hash128 kHash{"eb8674d890a44168959dd0b8ede27132"};
    // external runtime is nullable
    virtual luisa::unique_ptr<DirectXExternalRuntime> CreateExternalRuntime() const = 0;
    DirectXDeviceSettings() {
        hash = kHash;
    }
};
}// namespace luisa::compute