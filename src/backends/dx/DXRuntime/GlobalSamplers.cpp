#pragma vengine_package vengine_directx
#include <DXRuntime/GlobalSamplers.h>
namespace vstd {
template<>
struct compare<Sampler> {
    int32 operator()(Sampler const &a, Sampler const &b) const {
        if ((uint)a.filter() > (uint)b.filter())
            return 1;
        if ((uint)a.filter() < (uint)b.filter())
            return -1;
        if ((uint)a.address() > (uint)b.address())
            return 1;
        if ((uint)a.address() < (uint)b.address())
            return -1;
        return 0;
    }
};
}// namespace vstd
namespace toolhub::directx {
struct GlobalSampleData {
    std::array<D3D12_STATIC_SAMPLER_DESC, 16> arr;
    vstd::HashMap<
        Sampler,
        size_t>
        searchMap;
    GlobalSampleData() {
        memset(arr.data(), 0, sizeof(D3D12_STATIC_SAMPLER_DESC) * arr.size());
        size_t idx = 0;
        for (auto x : vstd::range(4))
            for (auto y : vstd::range(4)) {
                auto d = vstd::create_disposer([&] { ++idx; });
                auto &&v = arr[idx];
                switch ((Sampler::Filter)y) {
                    case Sampler::Filter::POINT:
                        v.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                        break;
                    case Sampler::Filter::LINEAR_POINT:
                        v.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                        break;
                    case Sampler::Filter::LINEAR_LINEAR:
                        v.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                        break;
                    case Sampler::Filter::ANISOTROPIC:
                        v.Filter = D3D12_FILTER_ANISOTROPIC;
                        break;
                }
                D3D12_TEXTURE_ADDRESS_MODE address = [&] {
                    switch ((Sampler::Address)x) {
                        case Sampler::Address::EDGE:
                            return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                        case Sampler::Address::REPEAT:
                            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                        case Sampler::Address::MIRROR:
                            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
                        default:
                            v.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
                            return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
                    }
                }();
                v.AddressU = address;
                v.AddressV = address;
                v.AddressW = address;
                v.MipLODBias = 0;
                v.MaxAnisotropy = 16;
                v.MinLOD = 0;
                v.MaxLOD = 16;
                v.ShaderRegister = idx;
                v.RegisterSpace = 0;
                v.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                searchMap.Emplace(
                    Sampler(
                        (Sampler::Filter)y,
                        (Sampler::Address)x),
                    idx);
            }
    }
};
static GlobalSampleData sampleData;

vstd::span<D3D12_STATIC_SAMPLER_DESC> GlobalSamplers::GetSamplers() {
    return {sampleData.arr.data(), sampleData.arr.size()};
}
size_t GlobalSamplers::GetIndex(
    Sampler const &sampler) {
    auto ite = sampleData.searchMap.Find(sampler);
    if (!ite) return std::numeric_limits<size_t>::max();
    return ite.Value();
}
}// namespace toolhub::directx