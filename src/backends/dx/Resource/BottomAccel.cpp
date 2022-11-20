
#include <Resource/BottomAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/Mesh.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <Resource/TopAccel.h>
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
bool BottomAccel::RequireCompact() const {
    return (((uint)hint & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
BottomAccel::BottomAccel(
    Device *device,
    Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vCount,
    Buffer const *iHandle, size_t iOffset, size_t iCount,
    luisa::compute::AccelUsageHint hint,
    bool allow_compact, bool allow_update)
    : device(device),
      mesh(
          device,
          vHandle, vOffset, vStride, vCount,
          iHandle, iOffset, iCount) {
    auto GetPreset = [&] {
        switch (hint) {
            case AccelUsageHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            default:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
    };
    this->hint = GetPreset();
    if (allow_compact) {
        this->hint |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    if (allow_update) {
        this->hint |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
}
BottomAccel::~BottomAccel() {
}
void BottomAccel::SyncTopAccel() const {
    for (auto &&k : *TopAccel::TopAccels()) {
        k.first->UpdateMesh(this);
    }
}

size_t BottomAccel::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    bool update,
    Buffer const *vHandle,
    Buffer const *iHandle,
    BottomAccelData &bottomData) {
    auto refreshUpdate = vstd::create_disposer([&] { this->update = update; });
    if ((uint)(hint & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) == 0)
        update = false;

    if (mesh.vHandle != vHandle || mesh.iHandle != iHandle) {
        update = false;
        mesh.vHandle = vHandle;
        mesh.iHandle = iHandle;
    }
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
    auto SetAccelBuffer = [&] {
        accelBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            CalcAlign(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, 65536),
            device->defaultAllocator,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE));
    };
    if (!accelBuffer) {
        update = false;
        SetAccelBuffer();
    } else if (accelBuffer->GetByteSize() < bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes) {
        update = false;
        builder.GetCB()->GetAlloc()->ExecuteAfterComplete([v = std::move(accelBuffer)] {});
        SetAccelBuffer();
        SyncTopAccel();
    }
    bottomStruct.DestAccelerationStructureData = accelBuffer->GetAddress();
    if (update) {
        bottomStruct.SourceAccelerationStructureData = bottomStruct.DestAccelerationStructureData;
        bottomStruct.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    } else {
        bottomStruct.SourceAccelerationStructureData = 0;
        bottomStruct.Inputs.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)bottomStruct.Inputs.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }
    bottomStruct.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

    return (update ? bottomLevelPrebuildInfo.UpdateScratchDataSizeInBytes : bottomLevelPrebuildInfo.ScratchDataSizeInBytes) + 8;
}
bool BottomAccel::CheckAccel(
    CommandBufferBuilder &builder) {
    auto disp = vstd::create_disposer([&] { compactSize = 0; });
    if (compactSize == 0)
        return false;
    auto &&alloc = builder.GetCB()->GetAlloc();
    auto newAccelBuffer = vstd::create_unique(new DefaultBuffer(
        device,
        CalcAlign(compactSize, 65536),
        device->defaultAllocator,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE));
    builder.CmdList()->CopyRaytracingAccelerationStructure(
        newAccelBuffer->GetAddress(),
        accelBuffer->GetAddress(),
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    alloc->ExecuteAfterComplete([b = std::move(accelBuffer)] {});
    accelBuffer = std::move(newAccelBuffer);
    SyncTopAccel();
    return true;
}
void BottomAccel::UpdateStates(
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer,
    BottomAccelData &accelData) {
    accelData.bottomStruct.ScratchAccelerationStructureData = scratchBuffer.buffer->GetAddress() + scratchBuffer.offset;
    accelData.bottomStruct.Inputs.pGeometryDescs = &accelData.geometryDesc;
    if (RequireCompact()) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo;
        postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - 8;
        postInfo.DestBuffer = scratchBuffer.buffer->GetAddress() + compactOffset;
        builder.CmdList()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            1,
            &postInfo);
    } else {
        builder.CmdList()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            0,
            nullptr);
    }
}
void BottomAccel::FinalCopy(
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer) {
    auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - 8;
    auto &&alloc = builder.GetCB()->GetAlloc();
    auto readback = alloc->GetTempReadbackBuffer(8);
    builder.CopyBuffer(
        scratchBuffer.buffer,
        readback.buffer,
        compactOffset,
        readback.offset,
        8);
    alloc->ExecuteAfterComplete([readback, this] {
        static_cast<ReadbackBuffer const *>(readback.buffer)->CopyData(readback.offset, {(vbyte *)&compactSize, 8});
    });
}
}// namespace toolhub::directx