#pragma vengine_package vengine_directx
#include <Resource/BottomAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/Mesh.h>
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
    Buffer const *iHandle, size_t iOffset, size_t iCount)
    : device(device),
      mesh(
          device,
          vHandle, vOffset, vStride, vCount,
          iHandle, iOffset, iCount) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomInput = bottomStruct.Inputs;
    bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomInput.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
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
        CalcAlign(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, 65536),
        device->defaultAllocator,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
}
BottomAccel::~BottomAccel() {
}
void BottomAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder) const {
    mesh.Build(tracker, builder);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomInput = bottomStruct.Inputs;
    bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomInput.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
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

    auto scratchBuffer = builder.GetCB()->GetAlloc()->AllocateScratchBuffer(bottomLevelPrebuildInfo.ScratchDataSizeInBytes);
    bottomStruct.SourceAccelerationStructureData = 0;
    bottomStruct.DestAccelerationStructureData = accelBuffer->GetAddress();
    bottomStruct.ScratchAccelerationStructureData = scratchBuffer->GetAddress();
    builder.CmdList()->BuildRaytracingAccelerationStructure(
        &bottomStruct,
        0,
        nullptr);
    D3D12_RESOURCE_BARRIER uavBarrier;
    D3D12_RESOURCE_BARRIER uavBarriers[2];
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uavBarrier.UAV.pResource = accelBuffer->GetResource();
    uavBarriers[0] = uavBarrier;
    uavBarrier.UAV.pResource = scratchBuffer->GetResource();
    uavBarriers[1] = uavBarrier;
    builder.CmdList()->ResourceBarrier(
        vstd::array_count(uavBarriers), uavBarriers);
}
}// namespace toolhub::directx