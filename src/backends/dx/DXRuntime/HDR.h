#pragma once
#include <d3dx12.h>
#include <dxgi1_4.h>
#include <luisa/core/basic_types.h>
namespace lc::dx {
class HDR {
private:
    bool supportHdr;

public:
    enum class HDRSupport : uint {
        None,
        RGB10,// DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
        F16,  // DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
    };
    bool SupportHdr() const { return supportHdr; }
    HDR(IDXGIFactory2 *factory, IDXGIAdapter1 *adapter);
    std::pair<HDRSupport, DXGI_COLOR_SPACE_TYPE> CheckSwapChainSupport(IDXGISwapChain3* swapChain);
    ~HDR();
};
}// namespace lc::dx
