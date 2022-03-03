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
void GetRayTransform(D3D12_RAYTRACING_INSTANCE_DESC &inst, float4x4 const &tr) {
    float *x[3] = {inst.Transform[0],
                   inst.Transform[1],
                   inst.Transform[2]};
    for (auto i : vstd::range(4))
        for (auto j : vstd::range(3)) {
            auto ptr = reinterpret_cast<float const *>(&tr.cols[i]);
            x[j][i] = ptr[j];
        }
}
}// namespace detail
TopAccel::MeshInstance::MeshInstance(float4x4 const &m) {
    size_t idx = 0;
    for (auto &&i : m.cols)
        for (auto j : vstd::range(3)) {
            v[idx] = i[j];
            ++idx;
        }
}

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
    D3D12_RAYTRACING_INSTANCE_DESC &ist = accelMap[idx].inst;
    detail::GetRayTransform(ist, localToWorld);
    ist.InstanceID = idx;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    delayCommands.emplace_back(idx);
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
    D3D12_RAYTRACING_INSTANCE_DESC &ist = accelMap[idx].inst;
    detail::GetRayTransform(ist, localToWorld);
    ist.InstanceID = idx;
    ist.InstanceMask = mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = accel->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(idx);
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
    D3D12_RAYTRACING_INSTANCE_DESC &ist = c.inst;
    ist.InstanceID = idx;
    ist.InstanceMask = mask;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = c.mesh->GetAccelBuffer()->GetAddress();
    delayCommands.emplace_back(idx);
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
        capacity = len;
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
void TopAccel::PreProcess(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder) {
    std::lock_guard lck(mtx);
    struct Buffers {
        vstd::unique_ptr<DefaultBuffer> v;
        Buffers(vstd::unique_ptr<DefaultBuffer> &&a)
            : v(std::move(a)) {}
        void operator()() const {}
    };
    auto GenerateNewBuffer = [&](vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state) {
        if (!oldBuffer) {
            newSize = CalcAlign(newSize, 65536);
            oldBuffer = vstd::create_unique(new DefaultBuffer(
                device,
                newSize,
                device->defaultAllocator,
                state));
            return true;
        } else {
            if (newSize <= oldBuffer->GetByteSize()) return false;
            newSize = CalcAlign(newSize, 65536);
            auto newBuffer = new DefaultBuffer(
                device,
                newSize,
                device->defaultAllocator,
                state);
            if (needCopy) {
                tracker.RecordState(
                    oldBuffer.get(),
                    VEngineShaderResourceState);
                tracker.RecordState(
                    newBuffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                tracker.UpdateState(builder);
                builder.CopyBuffer(
                    oldBuffer.get(),
                    newBuffer,
                    0,
                    0,
                    oldBuffer->GetByteSize());
                tracker.RecordState(
                    oldBuffer.get());
                tracker.RecordState(
                    newBuffer);
            }
            builder.GetCB()->GetAlloc()->ExecuteAfterComplete(Buffers{std::move(oldBuffer)});
            oldBuffer = vstd::create_unique(newBuffer);
            return true;
        }
    };
    size_t instanceByteCount = capacity * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto &&input = topLevelBuildDesc.Inputs;
    auto defaultCount = input.NumDescs;
    input.NumDescs = capacity;
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    if (GenerateNewBuffer(instBuffer, instanceByteCount, true, VEngineShaderResourceState)) {
        topLevelBuildDesc.Inputs.InstanceDescs = instBuffer->GetAddress();
    }
    if (GenerateNewBuffer(accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, false, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
        topLevelBuildDesc.DestAccelerationStructureData = accelBuffer->GetAddress();
    }
    input.NumDescs = defaultCount;
    if (!delayCommands.empty()) {
        tracker.RecordState(
            instBuffer.get(),
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
}
void TopAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    bool update) {
    std::lock_guard lck(mtx);
    if (Length() == 0) return;
    auto alloc = builder.GetCB()->GetAlloc();
    if (!delayCommands.empty()) {
        for (auto &&i : delayCommands) {
            builder.Upload(
                BufferView(instBuffer.get(), sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * i, sizeof(D3D12_RAYTRACING_INSTANCE_DESC)),
                &accelMap[i].inst);
        }
        delayCommands.clear();
        tracker.RecordState(
            instBuffer.get(),
            VEngineShaderResourceState);
    }
    tracker.UpdateState(builder);
    if (update) {
        topLevelBuildDesc.SourceAccelerationStructureData = topLevelBuildDesc.DestAccelerationStructureData;
        topLevelBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    } else {
        topLevelBuildDesc.SourceAccelerationStructureData = 0;
        topLevelBuildDesc.Inputs.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)topLevelBuildDesc.Inputs.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }

    auto scratchBuffer = alloc->AllocateScratchBuffer(
        topLevelBuildDesc.SourceAccelerationStructureData == 0 ? topLevelPrebuildInfo.ScratchDataSizeInBytes : topLevelPrebuildInfo.UpdateScratchDataSizeInBytes);
    topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer->GetAddress();
    builder.CmdList()->BuildRaytracingAccelerationStructure(
        &topLevelBuildDesc,
        0,
        nullptr);
    tracker.RecordState(
        scratchBuffer,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
}// namespace toolhub::directx