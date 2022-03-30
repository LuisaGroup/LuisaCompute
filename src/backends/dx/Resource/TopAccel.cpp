
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
    return resourceRefMap.Find(buffer);
}
void TopAccel::IncreRef(BottomAccel const *a) {
    auto Inc = [&](auto &&bf) {
        auto ite = resourceRefMap.Emplace(bf, 0);
        ite.Value()++;
    };
    auto mesh = static_cast<BottomAccel const *>(a);
    auto v = mesh->GetMesh()->vHandle;
    auto i = mesh->GetMesh()->iHandle;
    Inc(v);
    Inc(i);
}
void TopAccel::DecreRef(BottomAccel const *a) {
    auto Dec = [&](auto &&bf) {
        auto ite = resourceRefMap.Find(bf);
        if (!ite) return;
        auto &&v = ite.Value();
        --v;
        if (v == 0)
            resourceRefMap.Remove(ite);
    };
    auto mesh = static_cast<BottomAccel const *>(a);
    Dec(mesh->GetMesh()->vHandle);
    Dec(mesh->GetMesh()->iHandle);
}
bool TopAccel::IsMeshInAccel(Mesh const *mesh) const {
    return IsBufferInAccel(mesh->vHandle);
}

void TopAccel::Update(
    uint idx,
    BottomAccel const *accel) {
    auto &&oldC = accelMesh[idx];
    DecreRef(oldC);
    IncreRef(accel);
    oldC = accel;
    auto &&s = setDesc.emplace_back();
    std::pair<D3D12_GPU_VIRTUAL_ADDRESS, uint> value(accel->GetAccelBuffer()->GetAddress(), idx);
    *reinterpret_cast<SetPlaceHolder *>(&s) =
        *reinterpret_cast<SetPlaceHolder *>(&value);
}
void TopAccel::Emplace(
    BottomAccel const *mesh,
    luisa::float4x4 transform,
    bool visible) {
    auto &&len = topLevelBuildDesc.Inputs.NumDescs;
    uint tarIdx = len;
    len += 1;
    auto &&ist = newInstanceDesc.emplace_back();
    detail::GetRayTransform(ist, transform);
    ist.InstanceID = tarIdx;
    ist.InstanceMask = visible ? 255u : 0u;
    ist.InstanceContributionToHitGroupIndex = 0;
    ist.Flags = 0;
    ist.AccelerationStructure = mesh->GetAccelBuffer()->GetAddress();
    accelMesh.emplace_back(mesh);
    IncreRef(mesh);
}
void TopAccel::PopBack() {
    auto &&len = topLevelBuildDesc.Inputs.NumDescs;
    if (len == 0) return;
    len--;
    auto last = accelMesh.erase_last();
    DecreRef(last);
    if (!newInstanceDesc.empty()) {
        newInstanceDesc.erase_last();
    }
}

TopAccel::~TopAccel() {
}
size_t TopAccel::PreProcess(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    bool update) {
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
    size_t instanceByteCount = Length() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto &&input = topLevelBuildDesc.Inputs;
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    if (GenerateNewBuffer(instBuffer, instanceByteCount, true, VEngineShaderResourceState)) {
        topLevelBuildDesc.Inputs.InstanceDescs = instBuffer->GetAddress();
    }
    if (GenerateNewBuffer(accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, false, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
        topLevelBuildDesc.DestAccelerationStructureData = accelBuffer->GetAddress();
    }
    if (update) {
        topLevelBuildDesc.SourceAccelerationStructureData = topLevelBuildDesc.DestAccelerationStructureData;
        topLevelBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    } else {
        topLevelBuildDesc.SourceAccelerationStructureData = 0;
        topLevelBuildDesc.Inputs.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)topLevelBuildDesc.Inputs.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }

    if (!newInstanceDesc.empty()) {
        tracker.RecordState(
            instBuffer.get(),
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    return update ? topLevelPrebuildInfo.UpdateScratchDataSizeInBytes : topLevelPrebuildInfo.ScratchDataSizeInBytes;
}
void TopAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer) {
    if (Length() == 0) return;
    auto alloc = builder.GetCB()->GetAlloc();
    // Emplace new
    bool needUpdate = false;
    if (!newInstanceDesc.empty()) {
        size_t lastSize = accelMesh.size() - newInstanceDesc.size();
        builder.Upload(
            BufferView(instBuffer.get(), lastSize * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), newInstanceDesc.byte_size()),
            newInstanceDesc.data());
        newInstanceDesc.clear();
        needUpdate = true;
    }
    // Update
    if (!setDesc.empty()) {
        tracker.RecordState(
            instBuffer.get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        tracker.UpdateState(builder);
        auto cs = device->setAccelKernel;
        auto setBuffer = alloc->GetTempUploadBuffer(setDesc.byte_size());
        auto cbuffer = alloc->GetTempUploadBuffer(8, 256);
        struct CBuffer {
            uint dsp;
            uint count;
        };
        CBuffer cbValue;
        cbValue.dsp = setDesc.size();
        cbValue.count = Length();
        reinterpret_cast<UploadBuffer const *>(cbuffer.buffer)
            ->CopyData(cbuffer.offset,
                       {reinterpret_cast<vbyte const *>(&cbValue), sizeof(CBuffer)});
        reinterpret_cast<UploadBuffer const *>(setBuffer.buffer)
            ->CopyData(setBuffer.offset,
                       {reinterpret_cast<vbyte const *>(setDesc.data()), setDesc.byte_size()});
        BindProperty properties[3];
        properties[0].name = "_Global"sv;
        properties[0].prop = cbuffer;
        properties[1].name = "_SetBuffer"sv;
        properties[1].prop = setBuffer;
        properties[2].name = "_InstBuffer"sv;
        properties[2].prop = BufferView(instBuffer.get());
        //TODO
        builder.DispatchCompute(
            cs,
            uint3(setDesc.size(), 1, 1),
            properties);
        needUpdate = true;
        setDesc.clear();
    }
    if (needUpdate) {
        tracker.RecordState(
            instBuffer.get());
        tracker.UpdateState(builder);
    }
    topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer.buffer->GetAddress() + scratchBuffer.offset;
    builder.CmdList()->BuildRaytracingAccelerationStructure(
        &topLevelBuildDesc,
        0,
        nullptr);
}
}// namespace toolhub::directx