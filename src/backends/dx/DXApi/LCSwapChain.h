#pragma once
#include <DXApi/LCCmdBuffer.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCDevice.h>
#include <Resource/SwapChain.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
class LCSwapChain : public vstd::IOperatorNewBase {
public:
    vstd::vector<SwapChain> m_renderTargets;
    ComPtr<IDXGISwapChain3> swapChain;
    uint64 frameIndex = 0;
    bool vsync;
    LCSwapChain(
        Device *device,
        CommandQueue *queue,
        GpuAllocator *resourceAllocator,
        HWND windowHandle,
        uint width,
        uint height,
        bool allowHDR,
        bool vsync,
        uint backBufferCount);
};
}// namespace toolhub::directx