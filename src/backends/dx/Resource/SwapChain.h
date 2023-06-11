#pragma once
#include <Resource/Resource.h>
namespace lc::dx {
class SwapChain : public Resource {
public:
    ComPtr<ID3D12Resource> rt;
    SwapChain(SwapChain &&) = default;
    SwapChain(Device *device)
        : Resource(device) {}
    Tag GetTag() const override { return Tag::SwapChain; }
    ~SwapChain() = default;
    ID3D12Resource *GetResource() const override { return rt.Get(); }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return D3D12_RESOURCE_STATE_PRESENT;
    }
};
}// namespace lc::dx
