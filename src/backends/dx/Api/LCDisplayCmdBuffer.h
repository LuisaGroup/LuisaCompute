#pragma once
#include <Api/LCCmdBuffer.h>
#include <Resource/RenderTexture.h>
#include <Api/LCDevice.h>
#include <Resource/SwapChain.h>
namespace toolhub::directx {
class LCDisplayCmdBuffer : public LCCmdBuffer {
    vstd::optional<SwapChain> m_renderTargets[LCDevice::maxAllocatorCount];
    ComPtr<IDXGISwapChain3> swapChain;
    uint frameIndex = 0;

public:
    LCDisplayCmdBuffer(
        Device *device,
        IGpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type,
        HWND windowHandle,
        uint width,
        uint height);
    void Present(RenderTexture *rt);
};
}// namespace toolhub::directx