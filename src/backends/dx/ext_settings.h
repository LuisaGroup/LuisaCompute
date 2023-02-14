#pragma once
#include <runtime/context.h>
#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_4.h>
#include <vstl/common.h>
namespace luisa::compute {
struct DirectXDeviceConfigExt : public DeviceConfigExt, public vstd::IOperatorNewBase {
    virtual ID3D12Device *GetDevice() = 0;
    virtual IDXGIAdapter1 *GetAdapter() = 0;
    virtual IDXGIFactory4 *GetDXGIFactory() = 0;
    // queue is nullable
    virtual ID3D12CommandQueue *CreateQueue(D3D12_COMMAND_LIST_TYPE type) = 0;
};
}// namespace luisa::compute