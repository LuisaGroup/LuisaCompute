#pragma vengine_package vengine_directx
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Runtime/CommandAllocator.h>
#include <Runtime/ResourceStateTracker.h>
#include <Runtime/CommandBuffer.h>
#include <Resource/BottomAccel.h>
#include <Resource/Mesh.h>
namespace toolhub::directx {
namespace detail {
void GetRayTransform(D3D12_RAYTRACING_INSTANCE_DESC &inst, float4x4 const &tr) {
    auto GetRow = [&](uint row) {
        return float4(
            tr.cols[0][row],
            tr.cols[1][row],
            tr.cols[2][row],
            tr.cols[3][row]);
    };
    float4 *x = (float4 *)(&inst.Transform[0][0]);
    float4 *y = (float4 *)(&inst.Transform[1][0]);
    float4 *z = (float4 *)(&inst.Transform[2][0]);
    *x = GetRow(0);
    *y = GetRow(1);
    *z = GetRow(2);
}
}// namespace detail
TopAccel::TopAccel(Device *device)
    : device(device) {
    memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
    memset(&topLevelPrebuildInfo, 0, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topLevelBuildDesc.Inputs.NumDescs = 0;
}
bool TopAccel::IsBufferInAccel(Buffer const *buffer) const {
    std::lock_guard lck(mtx);
    return resourceMap.Find(buffer);
}
bool TopAccel::IsMeshInAccel(Mesh const *mesh) const {
    return IsBufferInAccel(mesh->vHandle);
}
void TopAccel::UpdateBottomAccel(uint idx, BottomAccel const *c) {
    auto &&oldC = accelMap[idx];
    if (oldC) {
        auto m = oldC->GetMesh();
        auto v = m->vHandle;
        auto i = m->iHandle;
        resourceMap.Remove(v);
        resourceMap.Remove(i);
    }
    oldC = c;
    {
        auto m = oldC->GetMesh();
        auto v = m->vHandle;
        auto i = m->iHandle;
        resourceMap.Emplace(v);
        resourceMap.Emplace(i);
    }
}
bool TopAccel::Update(
    uint idx,
    BottomAccel const *accel,
    uint mask,
    float4x4 const &localToWorld) {
    std::lock_guard lck(mtx);
    UpdateBottomAccel(idx, accel);
    if (idx >= Length()) {
        return false;
    }
    D3D12_RAYTRACING_INSTANCE_DESC ist;
    detail::GetRayTransform(ist, localToWorld);
    ist.InstanceID = accel->GetMesh()->GetMeshInstIdx();
    ist.InstanceMask = mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = accel->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(
        UpdateCommand{
            .buffer = BufferView(instBuffer, sizeof(ist) * idx, sizeof(ist)),
            .ist = ist});
    return true;
}
void TopAccel::Emplace(
    BottomAccel const *accel,
    uint mask,
    float4x4 const &localToWorld) {
    auto &&len = topLevelBuildDesc.Inputs.NumDescs;
    uint tarIdx = len;
    len++;
    accelMap.emplace_back(nullptr);
    if (capacity < len) {
        auto newCapa = capacity;
        do {
            newCapa = newCapa * 1.5 + 8;
        } while (newCapa < len);
        Reserve(newCapa);
    }
    Update(tarIdx, accel, mask, localToWorld);
}
TopAccel::~TopAccel() {
}
void TopAccel::Reserve(
    size_t newCapacity) {
    std::lock_guard lck(mtx);
    if (newCapacity <= capacity) return;
    capacity = newCapacity;
    auto &&input = topLevelBuildDesc.Inputs;
    auto defaultCount = input.NumDescs;
    input.NumDescs = newCapacity;
    auto d = vstd::create_disposer([&] { input.NumDescs = defaultCount; });
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    auto RebuildBuffer = [&](
                             vstd::ObjectPtr<DefaultBuffer> &buffer,
                             size_t tarSize,
                             D3D12_RESOURCE_STATES initState,
                             bool copy) {
        if (buffer && buffer->GetByteSize() >= tarSize) return false;
        DefaultBuffer const *b = buffer ? (DefaultBuffer const *)buffer : (DefaultBuffer const *)nullptr;

        auto newBuffer = vstd::MakeObjectPtr(new DefaultBuffer(
            device,
            CalcAlign(tarSize, 65536),
            device->defaultAllocator,
            initState));
        if (copy && b) {
            delayCommands.emplace_back(
                CopyCommand{
                    .srcBuffer = buffer,
                    .dstBuffer = newBuffer});
        }
        buffer = std::move(newBuffer);
        return true;
    };
    if (RebuildBuffer(instBuffer, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * newCapacity, D3D12_RESOURCE_STATE_COMMON, true)) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
    if (RebuildBuffer(accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false)) {
        topLevelBuildDesc.SourceAccelerationStructureData = 0;
        topLevelBuildDesc.DestAccelerationStructureData = accelBuffer->GetAddress();
        topLevelBuildDesc.Inputs.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)topLevelBuildDesc.Inputs.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }
}
void TopAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder) {
    std::lock_guard lck(mtx);
    auto alloc = builder.GetCB()->GetAlloc();
    for (auto &&i : delayCommands) {
        i.multi_visit(
            [&](CopyCommand &cpyCmd) {
                DefaultBuffer *srcBuffer = cpyCmd.srcBuffer;
                DefaultBuffer *dstBuffer = cpyCmd.dstBuffer;
                tracker.RecordState(
                    srcBuffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                tracker.RecordState(
                    dstBuffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                tracker.UpdateState(builder);
                builder.CopyBuffer(
                    srcBuffer,
                    dstBuffer,
                    0,
                    0,
                    srcBuffer->GetByteSize());
            },
            [&](UpdateCommand &update) {
                tracker.RecordState(
                    update.buffer.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                tracker.UpdateState(builder);
                auto d = vstd::create_disposer([&] { tracker.RecordState(update.buffer.buffer); });
                builder.Upload(
                    update.buffer,
                    &update.ist);
                tracker.RecordState(
                    update.buffer.buffer);
            });
    }
    delayCommands.clear();
    if (Length() == 0) return;
    auto scratchBuffer = alloc->AllocateScratchBuffer(
        topLevelBuildDesc.SourceAccelerationStructureData == 0 ? topLevelPrebuildInfo.ScratchDataSizeInBytes : topLevelPrebuildInfo.UpdateScratchDataSizeInBytes);
    topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer->GetAddress();
    tracker.UpdateState(builder);
    builder.CmdList()->BuildRaytracingAccelerationStructure(
        &topLevelBuildDesc,
        0,
        nullptr);
    topLevelBuildDesc.SourceAccelerationStructureData = topLevelBuildDesc.DestAccelerationStructureData;
    topLevelBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
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