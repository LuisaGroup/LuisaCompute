#pragma once
#include <DXApi/LCCmdBuffer.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCDevice.h>
#include <Resource/SwapChain.h>
#include <DXRuntime/CommandQueue.h>
namespace lc::dx {
class LCSwapChain : public Resource {
public:
    vstd::vector<SwapChain> m_renderTargets;
    ComPtr<IDXGISwapChain3> swapChain;
    uint64 frameIndex = 0;
    bool vsync;
    Tag GetTag() const override { return Tag::SwapChain; }
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
}// namespace lc::dx
