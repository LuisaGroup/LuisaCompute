#include <Resource/BottomAccel.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/TopAccel.h>
#include <luisa/runtime/rtx/aabb.h>
#include <luisa/core/logging.h>
namespace lc::dx {
using namespace luisa::compute;
namespace detail {
void MeshPreprocess(
    Buffer const *vHandle,
    uint64 vert_offset, uint64 vert_size,
    Buffer const *iHandle,
    uint64 idx_offset, uint64 idx_size,
    EnhancedBarrierTracker &tracker) {
    tracker.Record(
        BufferView(vHandle, vert_offset, vert_size),
        EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
    tracker.Record(
        BufferView(iHandle, idx_offset, idx_size),
        EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
}
void AABBPreprocess(
    Buffer const *aabbHandle,
    uint64 vert_offset, uint64 vert_size,
    EnhancedBarrierTracker &tracker) {
    tracker.Record(
        BufferView(aabbHandle, vert_offset, vert_size),
        EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
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
    Buffer const *aabbBuffer, size_t aabbObjectOffset, size_t aabbObjectSize) {
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    geometryDesc.AABBs.AABBCount = aabbObjectSize / sizeof(AABB);
    geometryDesc.AABBs.AABBs.StartAddress = aabbBuffer->GetAddress() + aabbObjectOffset;
    geometryDesc.AABBs.AABBs.StrideInBytes = sizeof(AABB);
}
}// namespace detail
bool BottomAccel::RequireCompact() const {
    return (((uint)hint & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
Buffer const *BottomAccel::GetAccelBuffer() const {
    if (!accelBuffer) [[unlikely]] {
        LUISA_ERROR("BLAS not initialized.");
    }
    return accelBuffer.get();
}
BottomAccel::BottomAccel(
    Device *device,
    AccelOption const &option)
    : Resource(device), compactSize(0) {
    if (!device->feature_check.raytracing_supported()) [[unlikely]] {
        LUISA_ERROR("RayTracing not supported on this device.");
    }
    auto GetPreset = [&] {
        switch (option.hint) {
            case AccelOption::UsageHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelOption::UsageHint::FAST_BUILD:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
        LUISA_ERROR_WITH_LOCATION("Unreachable.");
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
        auto accel = i->accel;
        accel->allInstance[i->accelIndex].handle = nullptr;
        // A mesh-refresh entry queued by the last BLAS recreate (SyncTopAccel)
        // may still reference this handle; drop it so a later TLAS build never
        // dereferences the destroyed handle.
        if (auto ite = accel->setMap.find(i->accelIndex);
            ite != accel->setMap.end() && ite->second == i) {
            accel->setMap.erase(ite);
        }
        MeshHandle::DestroyHandle(i);
    }
}
void BottomAccel::SyncTopAccel() {
    std::lock_guard lck(handleMtx);
    for (auto &&i : handles) {
        LUISA_ASSUME(i->mesh == this);
        i->accel->UpdateMesh(i);
    }
}

size_t BottomAccel::PreProcessStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
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
        detail::MeshPreprocess(meshOption.vHandle, meshOption.vOffset, meshOption.vSize, meshOption.iHandle, meshOption.iOffset, meshOption.iSize, tracker);
        detail::GetStaticTriangleGeometryDesc(
            geometryDesc,
            meshOption.vHandle, meshOption.vOffset, meshOption.vStride, meshOption.vSize, meshOption.iHandle, meshOption.iOffset, meshOption.iSize);
    } else {
        auto &aabbOption = options.get<1>();
        detail::AABBPreprocess(aabbOption.aabbBuffer, aabbOption.offset, aabbOption.size, tracker);
        detail::GetStaticAABBGeometryDesc(
            geometryDesc,
            aabbOption.aabbBuffer, aabbOption.offset, aabbOption.size);
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(
        &bottomInput,
        &bottomLevelPrebuildInfo);
    // workaround driver bug
    if (RequireCompact() && device->gpu_type == Device::GpuType::NVIDIA) {
        bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes += 65536;
    }
    auto SetAccelBuffer = [&] {
        accelBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            CalcAlign(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, 65536),
            device->default_allocator.get(),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            false,
            "blas-accel-buffer"));
    };
    if (!accelBuffer) {
        update = false;
        SetAccelBuffer();
    } else if (accelBuffer->GetByteSize() < bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes) {
        update = false;
        builder.get_cb()->get_alloc()->dispose_after_complete(std::move(accelBuffer));
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
    tracker.Record(GetAccelBuffer(), EnhancedBarrierTracker::Range(), EnhancedBarrierTracker::Usage::BuildAccel);
    return (update ? bottomLevelPrebuildInfo.UpdateScratchDataSizeInBytes : bottomLevelPrebuildInfo.ScratchDataSizeInBytes) + sizeof(size_t);
}
bool BottomAccel::CheckAccel(
    CommandBufferBuilder &builder) {
    auto disp = vstd::scope_exit([&] { compactSize = 0; });
    if (compactSize == 0)
        return false;
    auto &&alloc = builder.get_cb()->get_alloc();
    auto newAccelBuffer = vstd::create_unique(new DefaultBuffer(
        device,
        CalcAlign(compactSize, 65536),
        device->default_allocator.get(),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        false,
        "blas-accel-buffer"));
    builder.get_cb()->cmd_list()->CopyRaytracingAccelerationStructure(
        newAccelBuffer->GetAddress(),
        accelBuffer->GetAddress(),
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    alloc->dispose_after_complete(std::move(accelBuffer));
    accelBuffer = std::move(newAccelBuffer);
    SyncTopAccel();
    return true;
}
void BottomAccel::UpdateStates(
    EnhancedBarrierTracker &tracker,
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer,
    BottomAccelData &accelData) const {
    accelData.bottomStruct.ScratchAccelerationStructureData = scratchBuffer.buffer->GetAddress() + scratchBuffer.offset;
    accelData.bottomStruct.Inputs.pGeometryDescs = &accelData.geometryDesc;
    if (RequireCompact()) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo;
        postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - sizeof(size_t);
        postInfo.DestBuffer = scratchBuffer.buffer->GetAddress() + compactOffset;
        builder.get_cb()->cmd_list()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            1,
            &postInfo);
    } else {
        builder.get_cb()->cmd_list()->BuildRaytracingAccelerationStructure(
            &accelData.bottomStruct,
            0,
            nullptr);
    }
}
void BottomAccel::FinalCopy(
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer) {
    auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - sizeof(size_t);
    auto &&alloc = builder.get_cb()->get_alloc();
    auto readback = alloc->get_temp_readback_buffer(sizeof(size_t));
    builder.copy_buffer(
        scratchBuffer.buffer,
        readback.buffer,
        compactOffset,
        readback.offset,
        sizeof(size_t));
    alloc->execute_after_complete([readback, this] {
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
    LUISA_ASSUME(handle->mesh == this);
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
static vstd::Pool<MeshHandle> meshHandlePool(256, false);
static vstd::spin_mutex meshHandleMtx;
}// namespace detail
MeshHandle *MeshHandle::AllocateHandle() {
    using namespace detail;
    return meshHandlePool.create_lock(meshHandleMtx);
}
void MeshHandle::DestroyHandle(MeshHandle *handle) {
    using namespace detail;
    meshHandlePool.destroy_lock(meshHandleMtx, handle);
}
}// namespace lc::dx
