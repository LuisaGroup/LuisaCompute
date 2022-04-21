#pragma once
#include <DXRuntime/Device.h>
#include <vstl/VGuid.h>
namespace toolhub::directx {
class PipelineLibrary : public vstd::IOperatorNewBase {
private:
    ComPtr<ID3D12PipelineLibrary> pipeLib;
    Device *device;
    mutable std::mutex mtx;

public:
    PipelineLibrary(
        Device *device,
        vstd::span<ComputeShader const *> computes);
    bool Deserialize(
        vstd::span<vbyte const> data);
    void Serialize(
        vstd::function<void *(size_t)> const &allocFunc);
    ~PipelineLibrary();
    ComPtr<ID3D12PipelineState> GetPipelineState(
        vstd::Guid md5,
        D3D12_COMPUTE_PIPELINE_STATE_DESC const &computeDesc) const;
};
}// namespace toolhub::directx