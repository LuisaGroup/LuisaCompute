
#include <DXRuntime/GlobalSamplers.h>
namespace toolhub::directx {
struct SamplerHash{
    size_t operator()(Sampler const& s) const {
        return luisa::hash64(&s, sizeof(Sampler), luisa::Hash64::default_seed);
    }
};
struct GlobalSampleData {
    std::array<D3D12_SAMPLER_DESC, 16> arr;
    vstd::unordered_map<
        Sampler,
        size_t,
        SamplerHash>
        searchMap;
    GlobalSampleData() {
        memset(arr.data(), 0, sizeof(D3D12_SAMPLER_DESC) * arr.size());
        size_t idx = 0;
        for (auto x : vstd::range(4))
            for (auto y : vstd::range(4)) {
                auto d = vstd::scope_exit([&] { ++idx; });
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
                    default: assert(false); break;
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
                            memset(v.BorderColor, 0, 16);
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
                searchMap.try_emplace(
                    Sampler(
                        (Sampler::Filter)y,
                        (Sampler::Address)x),
                    idx);
            }
    }
};
static GlobalSampleData sampleData;

vstd::span<D3D12_SAMPLER_DESC> GlobalSamplers::GetSamplers() {
    return {sampleData.arr.data(), sampleData.arr.size()};
}
size_t GlobalSamplers::GetIndex(
    Sampler const &sampler) {
    auto ite = sampleData.searchMap.find(sampler);
    if (ite == sampleData.searchMap.end())
        return std::numeric_limits<size_t>::max();
    return ite->second;
}
}// namespace toolhub::directx