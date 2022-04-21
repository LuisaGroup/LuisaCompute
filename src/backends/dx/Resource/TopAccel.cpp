
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/BottomAccel.h>
#include <Resource/Mesh.h>
namespace toolhub::directx {
vstd::HashMap<TopAccel *> *TopAccel::TopAccels() {
    static vstd::HashMap<TopAccel *> topAccel;
    return &topAccel;
}

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
TopAccel::TopAccel(Device *device, luisa::compute::AccelUsageHint hint,
                   bool allow_compact, bool allow_update)
    : device(device) {
    //TODO: allow_compact not supported
    allow_compact = false;
    auto GetPreset = [&] {
        switch (hint) {
            case AccelUsageHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelUsageHint::FAST_BUILD:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
    };
    memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
    memset(&topLevelPrebuildInfo, 0, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = GetPreset();
    if (allow_compact) {
        topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }
    if (allow_update) {
        topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topLevelBuildDesc.Inputs.NumDescs = 0;
    TopAccels()->Emplace(this);
}
void TopAccel::UpdateMesh(
    BottomAccel const *mesh) {
    auto ite = meshMap.Find(mesh);
    if (!ite) return;
    for (auto &&kv : ite.Value()) {
        auto &&ist = allInstance[kv.first];
        ist.mesh = mesh->GetAccelBuffer()->GetAddress();
        setMap.ForceEmplace(kv.first, ist);
    }
    requireBuild = true;
}
void TopAccel::RemoveMesh(BottomAccel const *mesh, uint64 index) {
    auto idices = meshMap.Find(mesh);
    if (!idices) return;
    auto &&map = idices.Value();
    auto ite = map.Find(index);
    if (!ite) return;
    map.Remove(ite);
    if (map.size() == 0) meshMap.Remove(idices);
}
void TopAccel::AddMesh(BottomAccel const *mesh, uint64 index) {
    meshMap.Emplace(mesh).Value().Emplace(index);
}

TopAccel::~TopAccel() {
    TopAccels()->Remove(this);
}
size_t TopAccel::PreProcess(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    uint64 size,
    vstd::span<AccelBuildCommand::Modification const> const &modifications,
    bool update) {
    auto refreshUpdate = vstd::create_disposer([&] { this->update = update; });
    if ((uint)(topLevelBuildDesc.Inputs.Flags &
               D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) == 0 ||
        topLevelBuildDesc.Inputs.NumDescs != size) update = false;
    topLevelBuildDesc.Inputs.NumDescs = size;
    allInstance.resize(size);
    setDesc.clear();

    if (!modifications.empty()) {
        for (auto m : modifications) {
            auto ite = setMap.Find(m.index);
            if (ite) setMap.Remove(ite);
            if (m.mesh != 0) {
                auto mesh = reinterpret_cast<BottomAccel *>(m.mesh);
                auto oldMesh = reinterpret_cast<BottomAccel *>(allInstance[m.index].mesh);
                if (oldMesh != mesh) {
                    RemoveMesh(oldMesh, m.index);
                    AddMesh(mesh, m.index);
                }
                m.mesh = mesh->GetAccelBuffer()->GetAddress();
                update = false;
            } else {
                m.mesh = allInstance[m.index].mesh;
            }
            allInstance[m.index] = m;
            setDesc.emplace_back(m);
        }
    }
    if (requireBuild) {
        requireBuild = false;
        update = false;
    }
    if (setMap.size() != 0) {
        setDesc.reserve(setMap.size());
        for (auto &&i : setMap) {
            setDesc.emplace_back(i.second);
        }
        setMap.Clear();
    }
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
    size_t instanceByteCount = size * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto &&input = topLevelBuildDesc.Inputs;
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    if (GenerateNewBuffer(instBuffer, instanceByteCount, true, VEngineShaderResourceState)) {
        topLevelBuildDesc.Inputs.InstanceDescs = instBuffer->GetAddress();
    }
    if (GenerateNewBuffer(accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, false, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
        update = false;
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

    return (update ? topLevelPrebuildInfo.UpdateScratchDataSizeInBytes : topLevelPrebuildInfo.ScratchDataSizeInBytes) + 8;
}
void TopAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    BufferView const &scratchBuffer) {
    if (Length() == 0) return;
    auto alloc = builder.GetCB()->GetAlloc();
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
            {properties, 3});
        setDesc.clear();
        tracker.RecordState(
            instBuffer.get());
        tracker.UpdateState(builder);
    }
    topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer.buffer->GetAddress() + scratchBuffer.offset;
    if (RequireCompact()) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo;
        postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        auto compactOffset = scratchBuffer.offset + scratchBuffer.byteSize - 8;
        postInfo.DestBuffer = scratchBuffer.buffer->GetAddress() + compactOffset;
        builder.CmdList()->BuildRaytracingAccelerationStructure(
            &topLevelBuildDesc,
            1,
            &postInfo);
    } else {
        builder.CmdList()->BuildRaytracingAccelerationStructure(
            &topLevelBuildDesc,
            0,
            nullptr);
    }
}
void TopAccel::FinalCopy(
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
bool TopAccel::RequireCompact() const {
    return (((uint)topLevelBuildDesc.Inputs.Flags & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
bool TopAccel::CheckAccel(
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
    return true;
}
}// namespace toolhub::directx