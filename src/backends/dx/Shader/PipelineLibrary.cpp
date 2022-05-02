
#include <Shader/PipelineLibrary.h>
#include <vstl/BinaryReader.h>
#include <Shader/ComputeShader.h>
namespace toolhub::directx {
bool PipelineLibrary::Deserialize(vstd::span<vbyte const> data) {
    return (device->device->CreatePipelineLibrary(
                data.data(),
                data.size(),
                IID_PPV_ARGS(&pipeLib)) == S_OK);
}
void PipelineLibrary::Serialize(
    vstd::function<void *(size_t)> const &allocFunc) {
    auto sz = pipeLib->GetSerializedSize();
    auto ptr = allocFunc(sz);
    ThrowIfFailed(
        pipeLib->Serialize(
            ptr,
            sz));
}
PipelineLibrary::PipelineLibrary(
    Device *device,
    vstd::span<ComputeShader const *> computes)
    : device(device) {
    if (computes.empty()) {
        ThrowIfFailed(device->device->CreatePipelineLibrary(
            nullptr,
            0,
            IID_PPV_ARGS(&pipeLib)));
    } else {
        ThrowIfFailed(device->device->CreatePipelineLibrary(
            nullptr,
            0,
            IID_PPV_ARGS(&pipeLib)));
        for (auto &&cs : computes) {
            char base64Chars[24];
            WCHAR base64WChar[23];
            cs->GetGuid().ToBase64(base64Chars);
            for (auto i : vstd::range(22)) {
                base64WChar[i] = base64Chars[i];
            }
            base64WChar[22] = 0;
            ThrowIfFailed(pipeLib->StorePipeline(
                base64WChar,
                cs->Pso()));
        }
    }
}

ComPtr<ID3D12PipelineState> PipelineLibrary::GetPipelineState(
    vstd::Guid md5,
    D3D12_COMPUTE_PIPELINE_STATE_DESC const &computeDesc) const {
    char base64Chars[24];
    WCHAR base64WChar[23];
    md5.ToBase64(base64Chars);
    for (auto i : vstd::range(22)) {
        base64WChar[i] = base64Chars[i];
    }
    base64WChar[22] = 0;
    ComPtr<ID3D12PipelineState> result;
    auto loadResult = pipeLib->LoadComputePipeline(base64WChar, &computeDesc, IID_PPV_ARGS(&result));
    if (loadResult == S_OK) {
        return result;
    } else {
        return nullptr;
    }
}
PipelineLibrary::~PipelineLibrary() {
}
}// namespace toolhub::directx