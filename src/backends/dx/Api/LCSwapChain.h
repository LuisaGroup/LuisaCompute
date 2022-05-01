#pragma once
#include <Api/LCCmdBuffer.h>
#include <Resource/RenderTexture.h>
#include <Api/LCDevice.h>
#include <Resource/SwapChain.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
class LCSwapChain : public vstd::IOperatorNewBase {
public:
    vstd::vector<SwapChain> m_renderTargets;
    ComPtr<IDXGISwapChain3> swapChain;
    uint64 frameIndex = 0;
    LCSwapChain(
        Device *device,
        CommandQueue *queue,
        IGpuAllocator *resourceAllocator,
        HWND windowHandle,
        uint width,
        uint height,
        bool allowHDR,
        uint backBufferCount);
};
}// namespace toolhub::directx