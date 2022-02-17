#pragma vengine_package vengine_directx
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/BottomAccel.h>
#include <Resource/Mesh.h>
namespace toolhub::directx {
namespace detail {
void GetRayTransform(D3D12_RAYTRACING_INSTANCE_DESC &inst, MeshInstance &meshInst, float4x4 const &tr) {
    float *x[3] = {inst.Transform[0],
                   inst.Transform[1],
                   inst.Transform[2]};
    size_t vv = 0;
    for (auto i : vstd::range(4))
        for (auto j : vstd::range(3)) {
            auto ptr = reinterpret_cast<float const *>(&tr.cols[j]);
            x[j][i] = ptr[i];
            float v = ((float *)&tr[i])[j];
            meshInst.mat[vv] = v;
            vv++;
        }

    memcpy(meshInst.mat, &tr, sizeof(meshInst.mat));
}
}// namespace detail
TopAccel::TopAccel(Device *device, luisa::compute::AccelBuildHint hint)
    : device(device) {
    auto GetPreset = [&] {
        switch (hint) {
            case AccelBuildHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelBuildHint::FAST_REBUILD:
            case AccelBuildHint::FAST_UPDATE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
    };
    memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
    memset(&topLevelPrebuildInfo, 0, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = GetPreset() |
                           D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topLevelBuildDesc.Inputs.NumDescs = 0;
}
bool TopAccel::IsBufferInAccel(Buffer const *buffer) const {
    std::lock_guard lck(mtx);
    return resourceRefMap.Find(buffer);
}
void TopAccel::IncreRef(Buffer const *bf) {
    auto ite = resourceRefMap.Emplace(bf, 0);
    ite.Value()++;
}
void TopAccel::DecreRef(Buffer const *bf) {
    auto ite = resourceRefMap.Find(bf);
    if (!ite) return;
    auto &&v = ite.Value();
    --v;
    if (v == 0)
        resourceRefMap.Remove(ite);
}
bool TopAccel::IsMeshInAccel(Mesh const *mesh) const {
    return IsBufferInAccel(mesh->vHandle);
}
void TopAccel::UpdateBottomAccel(uint idx, BottomAccel const *c) {
    auto &&oldC = accelMap[idx];
    if (oldC.mesh) {
        auto m = oldC.mesh->GetMesh();
        auto v = m->vHandle;
        auto i = m->iHandle;
        DecreRef(v);
        DecreRef(i);
    }
    oldC.mesh = c;
    {
        auto m = oldC.mesh->GetMesh();
        auto v = m->vHandle;
        auto i = m->iHandle;
        IncreRef(v);
        IncreRef(i);
    }
}
bool TopAccel::Update(
    uint idx,
    float4x4 const &localToWorld) {
    std::lock_guard lck(mtx);
    if (idx >= Length()) {
        return false;
    }
    auto &&c = accelMap[idx];
    c.transform = localToWorld;
    D3D12_RAYTRACING_INSTANCE_DESC ist;
    MeshInstance meshInst;
    detail::GetRayTransform(ist, meshInst, localToWorld);
    ist.InstanceID = idx;
    ist.InstanceMask = c.mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = c.mesh->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(
        UpdateCommand{
            .ist = ist,
            .meshInst = meshInst,
            .buffer = BufferView(instBuffer.get(), sizeof(ist) * idx, sizeof(ist)),
            .customBuffer = BufferView(customInstBuffer.get(), sizeof(meshInst) * idx, sizeof(meshInst))});
    return true;
}
bool TopAccel::Update(
    uint idx,
    BottomAccel const *accel,
    uint mask,
    float4x4 const &localToWorld) {
    std::lock_guard lck(mtx);
    if (idx >= Length()) {
        return false;
    }
    UpdateBottomAccel(idx, accel);
    {
        auto &&c = accelMap[idx];
        c.mask = mask;
        c.transform = localToWorld;
    }
    D3D12_RAYTRACING_INSTANCE_DESC ist;
    MeshInstance meshInst;
    detail::GetRayTransform(ist, meshInst, localToWorld);
    ist.InstanceID = idx;
    ist.InstanceMask = mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = accel->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(
        UpdateCommand{
            .ist = ist,
            .meshInst = meshInst,
            .buffer = BufferView(instBuffer.get(), sizeof(ist) * idx, sizeof(ist)),
            .customBuffer = BufferView(customInstBuffer.get(), sizeof(meshInst) * idx, sizeof(meshInst))});
    return true;
}
bool TopAccel::Update(
    uint idx,
    uint mask) {
    std::lock_guard lck(mtx);
    if (idx >= Length()) {
        return false;
    }
    auto &&c = accelMap[idx];
    D3D12_RAYTRACING_INSTANCE_DESC ist;
    MeshInstance meshInst;
    detail::GetRayTransform(ist, meshInst, c.transform);
    ist.InstanceID = idx;
    ist.InstanceMask = c.mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = c.mesh->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(
        UpdateCommand{
            .ist = ist,
            .meshInst = meshInst,
            .buffer = BufferView(instBuffer.get(), sizeof(ist) * idx, sizeof(ist)),
            .customBuffer = BufferView(customInstBuffer.get(), sizeof(meshInst) * idx, sizeof(meshInst))});
    return true;
}
void TopAccel::Emplace(
    BottomAccel const *accel,
    uint mask,
    float4x4 const &localToWorld) {
    auto &&len = topLevelBuildDesc.Inputs.NumDescs;
    uint tarIdx = len;
    len++;
    accelMap.emplace_back(Element{});
    if (capacity < len) {
        auto newCapa = capacity;
        do {
            newCapa = newCapa * 1.5 + 8;
        } while (newCapa < len);
        Reserve(newCapa);
    }
    Update(tarIdx, accel, mask, localToWorld);
}
void TopAccel::PopBack() {
    auto &&len = topLevelBuildDesc.Inputs.NumDescs;
    if (len == 0) return;
    len--;
    accelMap.erase_last();
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
                             vstd::unique_ptr<DefaultBuffer> &buffer,
                             size_t tarSize,
                             D3D12_RESOURCE_STATES initState,
                             bool copy) {
        if (buffer && buffer->GetByteSize() >= tarSize) return false;
        DefaultBuffer const *b = buffer ? (DefaultBuffer const *)buffer.get() : (DefaultBuffer const *)nullptr;

        auto newBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            CalcAlign(tarSize, 65536),
            device->defaultAllocator,
            initState));
        if (copy && b) {
            delayCommands.emplace_back(
                CopyCommand{
                    .srcBuffer = std::move(buffer),
                    .dstBuffer = newBuffer.get()});
        }
        buffer = std::move(newBuffer);
        return true;
    };
    if (RebuildBuffer(instBuffer, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * newCapacity, D3D12_RESOURCE_STATE_COMMON, true)) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
    RebuildBuffer(customInstBuffer, sizeof(MeshInstance) * newCapacity, D3D12_RESOURCE_STATE_COMMON, true);
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
    vstd::vector<UpdateCommand> cmdCache;
    auto alloc = builder.GetCB()->GetAlloc();
    for (auto &&i : delayCommands) {
        i.multi_visit(
            [&](CopyCommand &cpyCmd) {
                DefaultBuffer const *srcBuffer = cpyCmd.srcBuffer.get();
                DefaultBuffer const *dstBuffer = cpyCmd.dstBuffer;
                tracker.RecordState(
                    srcBuffer,
                    VEngineShaderResourceState);
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
                tracker.RecordState(
                    srcBuffer);
                tracker.RecordState(
                    dstBuffer);
                tracker.UpdateState(builder);
            },
            [&](UpdateCommand &update) {
                tracker.RecordState(
                    update.buffer.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                tracker.RecordState(
                    update.customBuffer.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                cmdCache.emplace_back(update);
            });
    }
    tracker.UpdateState(builder);
    for (auto &&i : cmdCache) {
        auto d = vstd::create_disposer([&] {
            tracker.RecordState(i.buffer.buffer);
            tracker.RecordState(i.customBuffer.buffer);
        });
        builder.Upload(
            i.buffer,
            &i.ist);
        builder.Upload(
            i.customBuffer,
            &i.meshInst);
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
    tracker.RecordState(
        scratchBuffer,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    tracker.UpdateState(builder);
}
}// namespace toolhub::directx