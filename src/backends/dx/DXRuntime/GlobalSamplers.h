#pragma once
#include <d3dx12.h>
#include <luisa/runtime/rhi/sampler.h>
namespace lc::dx {
using namespace luisa::compute;
class GlobalSamplers {
public:
    static vstd::span<D3D12_SAMPLER_DESC> GetSamplers();
    static size_t GetIndex(
        Sampler const &sampler);
};
}// namespace lc::dx
