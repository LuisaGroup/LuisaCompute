#include "HDR.h"
#include <dxgi1_6.h>
namespace lc::dx {
using Microsoft::WRL::ComPtr;
inline int ComputeIntersectionArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
    return std::max(0, std::min(ax2, bx2) - std::max(ax1, bx1)) * std::max(0, std::min(ay2, by2) - std::max(ay1, by1));
}
HDR::HDR(IDXGIFactory2 *factory, IDXGIAdapter1 *adapter) {
    UINT i = 0;
    ComPtr<IDXGIOutput> currentOutput;
    // float bestIntersectArea = -1;
    supportHdr = false;
    while (adapter->EnumOutputs(i, &currentOutput) != DXGI_ERROR_NOT_FOUND) {
        ComPtr<IDXGIOutput6> output6;
        ThrowIfFailed(currentOutput.As(&output6));

        DXGI_OUTPUT_DESC1 desc1;
        ThrowIfFailed(output6->GetDesc1(&desc1));

        if (desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
            supportHdr = true;
            break;
        }
        ++i;
    }
}
std::pair<HDR::HDRSupport, DXGI_COLOR_SPACE_TYPE> HDR::CheckSwapChainSupport(IDXGISwapChain3 *swapChain) {
    DXGI_COLOR_SPACE_TYPE currentColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    HDRSupport result = HDRSupport::None;
    auto CheckSupport = [&](DXGI_COLOR_SPACE_TYPE colorSpace, HDRSupport support) {
        UINT colorSpaceSupport = 0;
        if (SUCCEEDED(swapChain->CheckColorSpaceSupport(colorSpace, &colorSpaceSupport)) &&
            ((colorSpaceSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) == DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
            currentColorSpace = colorSpace;
            result = support;
            return true;
        }
        return false;
    };
    if (supportHdr) {
        if (CheckSupport(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, HDRSupport::F16)) return {result, currentColorSpace};
        if (CheckSupport(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, HDRSupport::RGB10)) return {result, currentColorSpace};
    }
    CheckSupport(supportHdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, HDRSupport::None);
    return {result, currentColorSpace};
}
HDR::~HDR() {
}
}// namespace lc::dx
