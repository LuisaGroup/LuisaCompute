#include <Resource/BottomAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <Resource/TopAccel.h>
namespace toolhub::directx {
namespace detail {
void MeshPreprocess(
    Buffer const *vHandle,
    Buffer const *iHandle,
    ResourceStateTracker &tracker) {
    tracker.RecordState(
        vHandle,
        tracker.BufferReadState());
    tracker.RecordState(
        iHandle,
        tracker.BufferReadState());
}
void AABBPreprocess(
    Buffer const *aabbHandle,
    ResourceStateTracker &tracker) {
    tracker.RecordState(
        aabbHandle,
        tracker.BufferReadState());
}
void GetStaticTriangleGeometryDesc(
    D3D12_RAYTRACING_GEOMETRY_DESC &geometryDesc,
    Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vSize,
    Buffer const *iHandle, size_t iOffset, size_t iSize) {
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometryDesc.Triangles.IndexFormat = (DXGI_FORMAT)GFXFormat_R32_UInt;
    geometryDesc.Triangles.Transform3x4 = 0;
    geometryDesc.Triangles.VertexFormat = (DXGI_FORMAT)GFXFormat_R32G32B32_Float;
    geometryDesc.Triangles.VertexBuffer.StrideInBytes = vStride;
    geometryDesc.Triangles.IndexBuffer = iHandle->GetAddress() + iOffset;
    geometryDesc.Triangles.IndexCount = iSize / sizeof(uint);
    geometryDesc.Triangles.VertexBuffer.StartAddress = vHandle->GetAddress() + vOffset;
    geometryDesc.Triangles.VertexCount = vSize / vStride;
}
void GetStaticAABBGeometryDesc(
    D3D12_RAYTRACING_GEOMETRY_DESC &geometryDesc,
    Buffer const *aabbBuffer, size_t aabbObjectOffset, size_t aabbObjectCount) {
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometryDesc.AABBs.AABBCount = aabbObjectCount;
    geometryDesc.AABBs.AABBs.StartAddress = aabbBuffer->GetAddress() + aabbObjectOffset * 32;
    geometryDesc.AABBs.AABBs.StrideInBytes = 32;
}
}// namespace detail
bool BottomAccel::RequireCompact() const {
    return (((uint)hint & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
BottomAccel::BottomAccel(
    Device *device,
    AccelOption const &option)
    : device(device) {
    auto GetPreset = [&] {
        switch (option.hint) {
            case AccelOption::UsageHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelOption::UsageHint::FAST_BUILD:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
    };
    this->hint = GetPreset();
    if (option.allow_compaction) {
        this->hint |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    if (option.allow_update) {
        this->hint |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
}
BottomAccel::~BottomAccel() {
    for (auto &&i : handles) {
        i->accel->allInstance[i->accelIndex].handle = nullptr;
        MeshHandle::DestroyHandle(i);
    }
}
void BottomAccel::SyncTopAccel() {
    std::lock_guard lck(handleMtx);
    for (auto &&i : handles) {
        assert(i->mesh == this);
        i->accel->UpdateMesh(i);
    }
}

size_t BottomAccel::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    bool update,
    vstd::variant<MeshOptions, AABBOptions> const &options,
    BottomAccelData &bottomData) {
    auto refreshUpdate = vstd::scope_exit([&] { this->update = update; });
    if ((uint)(hint & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) == 0)
        update = false;
    auto &&bottomStruct = bottomData.bottomStruct;
    auto &&geometryDesc = bottomData.geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomInput = bottomStruct.Inputs;
    bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomInput.Flags = hint;
    bottomInput.NumDescs = 1;
    bottomInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottomInput.pGeometryDescs = &geometryDesc;
    if (options.index() == 0) {
        auto &meshOption = options.get<0>();
        detail::MeshPreprocess(meshOption.vHandle, meshOption.iHandle, tracker);
        detail::GetStaticTriangleGeometryDesc(
            geometryDesc,
            meshOption.vHandle, meshOption.vOffset, meshOption.vStride, meshOption.vSize, meshOption.iHandle, meshOption.iOffset, meshOption.iSize);
    } else {
        auto &aabbOption = options.get<1>();
        detail::AABBPreprocess(aabbOption.aabbBuffer, tracker);
        detail::GetStaticAABBGeometryDesc(
            geometryDesc,
            aabbOption.aabbBuffer, aabbOption.offset, aabbOption.count);
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(
        &bottomInput,
        &bottomLevelPrebuildInfo);
    auto SetAccelBuffer = [&] {
        accelBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            CalcAlign(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, 65536),
            device->defaultAllocator.get(),
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

    return (update ? bottomLevelPrebuildInfo.UpdateScratchDataSizeInBytes : bottomLevelPrebuildInfo.ScratchDataSizeInBytes) + sizeof(size_t);
}
bool BottomAccel::CheckAccel(
    CommandBufferBuilder &builder) {
    auto disp = vstd::scope_exit([&] { compactSize = 0; });
    if (compactSize == 0)
        return false;
    auto &&alloc = builder.GetCB()->GetAlloc();
    auto newAccelBuffer = vstd::create_unique(new DefaultBuffer(
        device,
        CalcAlign(compactSize, 65536),
        device->defaultAllocator.get(),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE));
    builder.GetCB()->CmdList()->CopyRaytracingAccelerationStructure(
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
        auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - sizeof(size_t);
        postInfo.DestBuffer = scratchBuffer.buffer->GetAddress() + compactOffset;
        builder.GetCB()->CmdList()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            1,
            &postInfo);
    } else {
        builder.GetCB()->CmdList()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            0,
            nullptr);
    }
}
void BottomAccel::FinalCopy(
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer) {
    auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - sizeof(size_t);
    auto &&alloc = builder.GetCB()->GetAlloc();
    auto readback = alloc->GetTempReadbackBuffer(sizeof(size_t));
    builder.CopyBuffer(
        scratchBuffer.buffer,
        readback.buffer,
        compactOffset,
        readback.offset,
        sizeof(size_t));
    alloc->ExecuteAfterComplete([readback, this] {
        static_cast<ReadbackBuffer const *>(readback.buffer)->CopyData(readback.offset, {(uint8_t *)&compactSize, sizeof(size_t)});
    });
}
MeshHandle *BottomAccel::AddAccelRef(TopAccel *accel, uint index) {
    auto meshHandle = MeshHandle::AllocateHandle();
    meshHandle->mesh = this;
    meshHandle->accel = accel;
    meshHandle->accelIndex = index;
    {
        std::lock_guard lck(handleMtx);
        meshHandle->meshIndex = handles.size();
        handles.emplace_back(meshHandle);
    }
    return meshHandle;
}
void BottomAccel::RemoveAccelRef(MeshHandle *handle) {
    assert(handle->mesh == this);
    {
        std::lock_guard lck(handleMtx);
        auto last = handles.back();
        handles.pop_back();
        if (last != handle) {
            last->meshIndex = handle->meshIndex;
            handles[handle->meshIndex] = last;
        }
    }
    MeshHandle::DestroyHandle(handle);
}
namespace detail {
static vstd::Pool<MeshHandle> meshHandlePool(32, false);
static vstd::spin_mutex meshHandleMtx;
}// namespace detail
MeshHandle *MeshHandle::AllocateHandle() {
    using namespace detail;
    return meshHandlePool.New_Lock(meshHandleMtx);
}
void MeshHandle::DestroyHandle(MeshHandle *handle) {
    using namespace detail;
    meshHandlePool.Delete_Lock(meshHandleMtx, handle);
}
}// namespace toolhub::directx