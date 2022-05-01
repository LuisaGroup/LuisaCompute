#pragma once
#include <d3dx12.h>
#include <runtime/sampler.h>
using namespace luisa::compute;
namespace toolhub::directx {
class GlobalSamplers {
public:
    static vstd::span<D3D12_SAMPLER_DESC> GetSamplers();
    static size_t GetIndex(
        Sampler const &sampler);
};
}// namespace toolhub::directx