#pragma vengine_package vengine_directx
#include <Resource/BottomAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/Mesh.h>
#include <DXRuntime/ResourceStateTracker.h>
namespace toolhub::directx {
namespace detail {
void GetStaticTriangleGeometryDesc(D3D12_RAYTRACING_GEOMETRY_DESC &geometryDesc, Mesh const *mesh) {
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometryDesc.Triangles.IndexFormat = (DXGI_FORMAT)GFXFormat_R32_UInt;
    geometryDesc.Triangles.Transform3x4 = 0;
    geometryDesc.Triangles.VertexFormat = (DXGI_FORMAT)GFXFormat_R32G32B32_Float;
    geometryDesc.Triangles.VertexBuffer.StrideInBytes = mesh->vStride;
    geometryDesc.Triangles.IndexBuffer = mesh->iHandle->GetAddress() + mesh->iOffset;
    geometryDesc.Triangles.IndexCount = mesh->iCount;
    geometryDesc.Triangles.VertexBuffer.StartAddress = mesh->vHandle->GetAddress() + mesh->vOffset;
    geometryDesc.Triangles.VertexCount = mesh->vCount;
}
}// namespace detail
BottomAccel::BottomAccel(
    Device *device,
    Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vCount,
    Buffer const *iHandle, size_t iOffset, size_t iCount,
    luisa::compute::AccelBuildHint hint)
    : device(device),
      mesh(
          device,
          vHandle, vOffset, vStride, vCount,
          iHandle, iOffset, iCount) {
    auto GetPreset = [&] {
        switch (hint) {
            case AccelBuildHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelBuildHint::FAST_REBUILD:
            case AccelBuildHint::FAST_UPDATE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
    };
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomInput;
    bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomInput.Flags = GetPreset();
    this->hint = bottomInput.Flags;
    bottomInput.NumDescs = 1;
    bottomInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottomInput.pGeometryDescs = &geometryDesc;
    detail::GetStaticTriangleGeometryDesc(
        geometryDesc,
        &mesh);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(
        &bottomInput,
        &bottomLevelPrebuildInfo);
    accelBuffer.New(
        device,
        bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes,
        device->defaultAllocator,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
}
BottomAccel::~BottomAccel() {
}
size_t BottomAccel::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    bool update,
    BottomAccelData &bottomData) {
    mesh.Build(tracker);
    auto &&bottomStruct = bottomData.bottomStruct;
    auto &&geometryDesc = bottomData.geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomInput = bottomStruct.Inputs;
    bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomInput.Flags = hint;
    bottomInput.NumDescs = 1;
    bottomInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottomInput.pGeometryDescs = &geometryDesc;
    detail::GetStaticTriangleGeometryDesc(
        geometryDesc,
        &mesh);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(
        &bottomInput,
        &bottomLevelPrebuildInfo);

    bottomStruct.DestAccelerationStructureData = accelBuffer->GetAddress();
    if (update) {
        bottomStruct.SourceAccelerationStructureData = bottomStruct.DestAccelerationStructureData;
        bottomStruct.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    } else {
        bottomStruct.SourceAccelerationStructureData = 0;
        bottomStruct.Inputs.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)bottomStruct.Inputs.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }
    return update ? bottomLevelPrebuildInfo.UpdateScratchDataSizeInBytes : bottomLevelPrebuildInfo.ScratchDataSizeInBytes;
}
void BottomAccel::UpdateStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    BufferView const &scratchBuffer,
    BottomAccelData &accelData) {
    accelData.bottomStruct.ScratchAccelerationStructureData = scratchBuffer.buffer->GetAddress() + scratchBuffer.offset;
    accelData.bottomStruct.Inputs.pGeometryDescs = &accelData.geometryDesc;
    builder.CmdList()->BuildRaytracingAccelerationStructure(
        &accelData.bottomStruct,
        0,
        nullptr);
}
}// namespace toolhub::directx