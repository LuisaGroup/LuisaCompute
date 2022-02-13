#pragma vengine_package vengine_directx
#include <Shader/PipelineLibrary.h>
#include <vstl/BinaryReader.h>
#include <Shader/ComputeShader.h>
namespace toolhub::directx {
PipelineLibrary::PipelineLibrary(
    Device *device)
    : device(device) {
}
PipelineLibrary::DeserResult PipelineLibrary::Deserialize(vstd::string const &path) {
    {
        BinaryReader reader(path);
        if (reader) {
            auto vec = reader.Read();
            if (device->device->CreatePipelineLibrary(
                    vec.data(),
                    vec.size(),
                    IID_PPV_ARGS(&pipeLib)) == S_OK) {
                return DeserResult::Success;
            } else {
                return DeserResult::FileUncompatible;
            }
        } else {
            return DeserResult::FileNotFound;
        }
    }
}
void PipelineLibrary::Serialize(
    vstd::vector<vbyte> &vec) {
    ThrowIfFailed(device->device->CreatePipelineLibrary(
        nullptr,
        0,
        IID_PPV_ARGS(&pipeLib)));
    device->IteratePipeline([&](ComputeShader const *cs, vstd::Guid guid) {
        char base64Chars[24];
        WCHAR base64WChar[23];
        guid.ToBase64(base64Chars);
        for (auto i : vstd::range(22)) {
            base64WChar[i] = base64Chars[i];
        }
        base64WChar[22] = 0;
        pipeLib->StorePipeline(
            base64WChar,
            cs->Pso());
        return true;
    });
    auto lastSize = vec.size();
    auto serSize = pipeLib->GetSerializedSize();
    vec.resize(lastSize + serSize);
    auto ptr = vec.data() + lastSize;
    ThrowIfFailed(pipeLib->Serialize(
        ptr,
        serSize));
}

ComPtr<ID3D12PipelineState> PipelineLibrary::GetPipelineState(
    vstd::Guid md5,
    D3D12_COMPUTE_PIPELINE_STATE_DESC const &computeDesc) const {
    auto GetNewPipeState = [&] {
        ComPtr<ID3D12PipelineState> result;
        ThrowIfFailed(device->device->CreateComputePipelineState(
            &computeDesc,
            IID_PPV_ARGS(&result)));
        return result;
    };
    if (!pipeLib) return GetNewPipeState();
    char base64Chars[24];
    WCHAR base64WChar[23];
    md5.ToBase64(base64Chars);
    for (auto i : vstd::range(22)) {
        base64WChar[i] = base64Chars[i];
    }
    base64WChar[22] = 0;
    ComPtr<ID3D12PipelineState> result;
    auto loadResult = pipeLib->LoadComputePipeline(base64WChar, &computeDesc, IID_PPV_ARGS(result.GetAddressOf()));
    if (loadResult == S_OK) {
        return result;
    } else {
        return GetNewPipeState();
    }
}
PipelineLibrary::~PipelineLibrary() {
}
}// namespace toolhub::directx