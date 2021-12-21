#pragma vengine_package vengine_directx
#include <Shader/RTShader.h>
#include <Runtime/CommandBuffer.h>
namespace toolhub::directx {
wchar_t const *RTShader::GetClosestHitFuncName() {
    return L"closest_hit";
}

wchar_t const *RTShader::GetRayGenFuncName() {
    return L"raygen";
}
wchar_t const *RTShader::GetIntersectFuncName() {
    return L"intersect_hit";
}
wchar_t const *RTShader::GetMissFuncName() {
    return L"miss_hit";
}
wchar_t const *RTShader::GetAnyHitFuncName() {
    return L"any_hit";
}
void RTShader::Update(CommandBufferBuilder &builder) const {
    auto cmd = builder.CmdList();
    BufferView raygenView(&identityBuffer, 0, raygenIdentifier.size());
    BufferView missView(&identityBuffer, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, missIdentifier.size());
    BufferView hitGroupView(&identityBuffer, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT * 2, identifier.size());
    D3D12_RESOURCE_BARRIER transBarrier;
    transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    transBarrier.Transition.pResource = identityBuffer.GetResource();
    transBarrier.Transition.StateBefore = identityBuffer.GetInitState();
    transBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    cmd->ResourceBarrier(1, &transBarrier);
    auto d = vstd::create_disposer([&] {
        std::swap(transBarrier.Transition.StateBefore, transBarrier.Transition.StateAfter);
        cmd->ResourceBarrier(1, &transBarrier);
    });

    builder.Upload(
        raygenView,
        raygenIdentifier.data());
    builder.Upload(
        missView,
        missIdentifier.data());
    builder.Upload(
        hitGroupView,
        identifier.data());
}
void RTShader::DispatchRays(
    CommandBufferBuilder &originCmdList,
    uint width,
    uint height,
    uint depth) const {
    if (!finishedUpdate.test_and_set()) {
        Update(originCmdList);
    }
    ID3D12GraphicsCommandList4 *cmdList = originCmdList.CmdList();
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    auto hitGroups = GetHitGroup();
    dispatchDesc.Depth = depth;
    dispatchDesc.Height = height;
    dispatchDesc.Width = width;
    dispatchDesc.HitGroupTable.StartAddress = hitGroups.hitGroupVAddress;
    dispatchDesc.HitGroupTable.SizeInBytes = hitGroups.hitGroupSize;
    dispatchDesc.HitGroupTable.StrideInBytes = hitGroups.hitGroupSize;
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = hitGroups.rayGenSize;
    dispatchDesc.RayGenerationShaderRecord.StartAddress = hitGroups.rayGenVAddress;
    dispatchDesc.MissShaderTable.SizeInBytes = hitGroups.missSize;
    dispatchDesc.MissShaderTable.StrideInBytes = hitGroups.missSize;
    dispatchDesc.MissShaderTable.StartAddress = hitGroups.missVAddress;
    cmdList->SetPipelineState1(stateObj.Get());
    cmdList->DispatchRays(
        &dispatchDesc);
}

DXRHitGroup RTShader::GetHitGroup() const {
    return {
        D3D12_HIT_GROUP_TYPE_TRIANGLES,
        identityBuffer.GetAddress() + D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT,
        D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
        identityBuffer.GetAddress(),
        D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
        identityBuffer.GetAddress() + D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT * 2,
        D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
}

RTShader::RTShader(
    bool closestHit,
    bool anyHit,
    bool intersectHit,
    std::span<std::pair<vstd::string_view, Property>> properties,
    vstd::span<vbyte> binData,
    Device *device)
    : Shader(properties, device->device.Get()),
      identityBuffer(
          device,
          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT * 3,
          device->defaultAllocator) {
    finishedUpdate.clear();
    using Microsoft::WRL::ComPtr;
    CD3DX12_STATE_OBJECT_DESC raytracingPipeline{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE};
    auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(binData.data(), binData.size());
    lib->SetDXILLibrary(&libdxil);

    auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    if (closestHit) {
        hitGroup->SetClosestHitShaderImport(GetClosestHitFuncName());
    }
    if (anyHit) {
        hitGroup->SetAnyHitShaderImport(GetAnyHitFuncName());
    }
    if (intersectHit) {
        hitGroup->SetIntersectionShaderImport(GetIntersectFuncName());
    }

    //hitGroup->SetHitGroupExport(vstd::Guid(true).ToString());
    hitGroup->SetHitGroupExport(L"hh");
    hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(sizeof(float4), sizeof(float2));
    auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRootSignature->SetRootSignature(rootSig.Get());
    auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    pipelineConfig->Config(1);
    ThrowIfFailed(device->device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&stateObj)));
    ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
    ThrowIfFailed(stateObj.As(&stateObjectProperties));
    auto BindIdentifier = [&](wchar_t const *name, ByteVector &bufferMem) {
        void *ptr = stateObjectProperties->GetShaderIdentifier(name);
        assert(ptr);
        bufferMem.resize(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        memcpy(bufferMem.data(),
               ptr,
               D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    };
    BindIdentifier(GetRayGenFuncName(), raygenIdentifier);
    BindIdentifier(GetMissFuncName(), missIdentifier);
    BindIdentifier(L"hh", identifier);
    //TODO: identifier buffer
}
RTShader::~RTShader() {}
}// namespace toolhub::directx