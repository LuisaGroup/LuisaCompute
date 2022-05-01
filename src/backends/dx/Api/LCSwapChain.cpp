#include <Api/LCSwapChain.h>
#include <dxgi1_2.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
namespace toolhub::directx {
LCSwapChain::LCSwapChain(
    Device *device,
    CommandQueue *queue,
    IGpuAllocator *resourceAllocator,
    HWND windowHandle,
    uint width,
    uint height,
    bool allowHDR,
    uint backBufferCount) {
    auto frameCount = backBufferCount + 1;
    m_renderTargets.push_back_func(
        [&] { return device; },
        frameCount);
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = frameCount;
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;
    {
        ComPtr<IDXGISwapChain1> localSwap;
        ThrowIfFailed(device->dxgiFactory->CreateSwapChainForHwnd(
            queue->Queue(),
            windowHandle,
            &swapChainDesc,
            nullptr,
            nullptr,
            &localSwap));
        ThrowIfFailed(localSwap.As(&swapChain));
    }
    for (uint32_t n = 0; n < frameCount; n++) {
        ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n].rt)));
    }
}

}// namespace toolhub::directx