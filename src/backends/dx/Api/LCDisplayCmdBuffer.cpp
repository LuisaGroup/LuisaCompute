#include <Api/LCDisplayCmdBuffer.h>
#include <dxgi1_2.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
namespace toolhub::directx {
LCDisplayCmdBuffer::LCDisplayCmdBuffer(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type,
    HWND windowHandle,
    uint width,
    uint height)
    : LCCmdBuffer(device, resourceAllocator, type) {
    for (auto &&i : m_renderTargets) {
        i.New(device);
    }
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = LCDevice::maxAllocatorCount;
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;
    {
        ComPtr<IDXGISwapChain1> localSwap;
        ThrowIfFailed(device->dxgiFactory->CreateSwapChainForHwnd(
            queue.Queue(),
            windowHandle,
            &swapChainDesc,
            nullptr,
            nullptr,
            &localSwap));
        ThrowIfFailed(localSwap.As(&swapChain));
    }
    for (uint32_t n = 0; n < LCDevice::maxAllocatorCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n]->rt)));
    }
}
void LCDisplayCmdBuffer::Present(RenderTexture *rt) {
    auto alloc = queue.CreateAllocator(LCDevice::maxAllocatorCount);
    {
        frameIndex = swapChain->GetCurrentBackBufferIndex();
        auto cb = alloc->GetBuffer();
        auto bd = cb->Build();
        auto cmdList = bd.CmdList();
        tracker.RecordState(
            rt, D3D12_RESOURCE_STATE_COPY_SOURCE);
        tracker.RecordState(
            m_renderTargets[frameIndex], D3D12_RESOURCE_STATE_COPY_DEST);
        tracker.UpdateState(bd);
        D3D12_TEXTURE_COPY_LOCATION sourceLocation;
        sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        sourceLocation.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION destLocation;
        destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destLocation.SubresourceIndex = 0;
        sourceLocation.pResource = rt->GetResource();
        destLocation.pResource = m_renderTargets[frameIndex]->GetResource();
        cmdList->CopyTextureRegion(
            &destLocation,
            0, 0, 0,
            &sourceLocation,
            nullptr);
        tracker.RestoreState(bd);
    }
    lastFence = queue.ExecuteAndPresent(std::move(alloc), swapChain.Get());
}
}// namespace toolhub::directx