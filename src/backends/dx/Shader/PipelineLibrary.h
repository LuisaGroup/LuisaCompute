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
    enum class DeserResult : vbyte {
        Success,
        FileNotFound,
        FileUncompatible
    };
    PipelineLibrary(
        Device *device);
    DeserResult Deserialize(vstd::string const &path);
    void Serialize(
        vstd::vector<vbyte> &result);
    ~PipelineLibrary();
    ComPtr<ID3D12PipelineState> GetPipelineState(
        vstd::Guid md5,
        D3D12_COMPUTE_PIPELINE_STATE_DESC const &computeDesc) const;
};
}// namespace toolhub::directx