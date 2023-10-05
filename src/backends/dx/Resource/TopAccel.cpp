#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/BottomAccel.h>
#include <luisa/core/logging.h>

namespace lc::dx {

TopAccel::TopAccel(Device *device, AccelOption const &option)
    : Resource(device) {
    //TODO: allow_compact not supported
    // option.allow_compaction = false;
    auto GetPreset = [&] {
        switch (option.hint) {
            case AccelOption::UsageHint::FAST_TRACE:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            case AccelOption::UsageHint::FAST_BUILD:
                return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
        LUISA_ERROR_WITH_LOCATION("Unreachable.");
    };
    memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
    memset(&topLevelPrebuildInfo, 0, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = GetPreset();
    // if (option.allow_compaction) {
    //     topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    // }
    if (option.allow_update) {
        topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topLevelBuildDesc.Inputs.NumDescs = 0;
}
void TopAccel::UpdateMesh(
    MeshHandle *handle) {
    auto instIndex = handle->accelIndex;
    assert(allInstance[instIndex].handle == handle);
    setMap.force_emplace(instIndex, handle);
    requireBuild = true;
}
void TopAccel::SetMesh(BottomAccel *mesh, uint64 index) {
    auto &&inst = allInstance[index].handle;
    if (inst != nullptr) {
        if (inst->mesh == mesh) return;
        inst->mesh->RemoveAccelRef(inst);
    }
    inst = mesh->AddAccelRef(this, index);
    inst->accelIndex = index;
}
TopAccel::~TopAccel() {
    for (auto &&i : allInstance) {
        auto mesh = i.handle;
        if (mesh)
            mesh->mesh->RemoveAccelRef(mesh);
    }
}
bool TopAccel::GenerateNewBuffer(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state) {
    if (!oldBuffer) {
        newSize = CalcAlign(newSize, 65536);
        oldBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            newSize,
            device->defaultAllocator.get(),
            state));
        return true;
    } else {
        if (newSize <= oldBuffer->GetByteSize()) return false;
        newSize = CalcAlign(newSize, 65536);
        auto newBuffer = new DefaultBuffer(
            device,
            newSize,
            device->defaultAllocator.get(),
            state);
        if (needCopy) {
            tracker.RecordState(
                oldBuffer.get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE);
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
                oldBuffer.get(), oldBuffer->GetInitState());
            tracker.RecordState(
                newBuffer, oldBuffer->GetInitState());
        }
        builder.GetCB()->GetAlloc()->DisposeAfterComplete(std::move(oldBuffer));
        oldBuffer = vstd::create_unique(newBuffer);
        return true;
    }
}

void TopAccel::PreProcessInst(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    uint64 size,
    vstd::span<AccelBuildCommand::Modification const> const &modifications) {
    auto &&input = topLevelBuildDesc.Inputs;
    if (input.NumDescs != size) update = false;
    input.NumDescs = size;
    allInstance.resize(size);
    setDesc.clear();
    vstd::push_back_all(setDesc, modifications);
#ifndef NDEBUG
    for (auto &&m : modifications) {
        if (m.flags & m.flag_user_id) {
            if (m.user_id >= (1u << 24u)) [[unlikely]] {
                LUISA_ERROR("DirectX can-not support user_id larger than {}", (1u << 24u) - 1);
            }
        }
    }
#endif
    for (auto &&m : setDesc) {
        auto ite = setMap.find(m.index);
        bool updateMesh = (m.flags & m.flag_primitive);
        if (ite != setMap.end()) {
            if (!updateMesh) {
                m.primitive = ite->second->mesh->GetAccelBuffer()->GetAddress();
                m.flags |= m.flag_primitive;
            }
            setMap.erase(ite);
        }
        if (updateMesh) {
            auto mesh = reinterpret_cast<BottomAccel *>(m.primitive);
            SetMesh(mesh, m.index);
            m.primitive = mesh->GetAccelBuffer()->GetAddress();
            update = false;
        }
    }
    if (requireBuild) {
        requireBuild = false;
        update = false;
    }
    if (setMap.size() != 0) {
        update = false;
        setDesc.reserve(setMap.size());
        for (auto &&i : setMap) {
            auto &mod = setDesc.emplace_back(i.first);
            mod.flags = mod.flag_primitive;
            mod.primitive = i.second->mesh->GetAccelBuffer()->GetAddress();
        }
        setMap.clear();
    }

    size_t instanceByteCount = size * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    if (GenerateNewBuffer(
            tracker, builder, instBuffer, instanceByteCount, true, tracker.ReadState(ResourceReadUsage::AccelBuildSrc))) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
}

size_t TopAccel::PreProcess(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    uint64 size,
    vstd::span<AccelBuildCommand::Modification const> const &modifications,
    bool update) {
    update &= this->update;
    auto refreshUpdate = vstd::scope_exit([&] { this->update &= update; });
    auto &&input = topLevelBuildDesc.Inputs;
    if ((uint)(input.Flags &
               D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) == 0 ||
        input.NumDescs != size) update = false;
    input.NumDescs = size;
    allInstance.resize(size);
    vstd::span<AccelBuildCommand::Modification> mutable_mod{
        const_cast<AccelBuildCommand::Modification *>(modifications.data()),
        modifications.size()};
    for (auto &&m : mutable_mod) {
        auto ite = setMap.find(m.index);
#ifndef NDEBUG
        if (m.flags & m.flag_user_id) {
            if (m.user_id >= (1u << 24u)) [[unlikely]] {
                LUISA_ERROR("DirectX can-not support user_id larger than {}", (1u << 24u) - 1);
            }
        }
#endif
        bool updateMesh = (m.flags & m.flag_primitive);
        if (ite != setMap.end()) {
            if (!updateMesh) {
                m.primitive = ite->second->mesh->GetAccelBuffer()->GetAddress();
                m.flags |= m.flag_primitive;
            }
            setMap.erase(ite);
        }
        if (updateMesh) {
            auto mesh = reinterpret_cast<BottomAccel *>(m.primitive);
            SetMesh(mesh, m.index);
            m.primitive = mesh->GetAccelBuffer()->GetAddress();
            update = false;
        }
    }
    if (requireBuild) {
        requireBuild = false;
        update = false;
    }
    setDesc.clear();
    if (setMap.size() != 0) {
        update = false;
        setDesc.reserve(setMap.size());
        for (auto &&i : setMap) {
            auto &mod = setDesc.emplace_back(i.first);
            mod.flags = mod.flag_primitive;
            mod.primitive = i.second->mesh->GetAccelBuffer()->GetAddress();
        }
        setMap.clear();
    }

    size_t instanceByteCount = size * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    if (GenerateNewBuffer(
            tracker, builder, instBuffer, instanceByteCount, true, tracker.ReadState(ResourceReadUsage::AccelBuildSrc))) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    if (GenerateNewBuffer(tracker, builder, accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, false, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
        update = false;
        topLevelBuildDesc.DestAccelerationStructureData = accelBuffer->GetAddress();
    }
    if (update) {
        topLevelBuildDesc.SourceAccelerationStructureData = topLevelBuildDesc.DestAccelerationStructureData;
        input.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    } else {
        topLevelBuildDesc.SourceAccelerationStructureData = 0;
        input.Flags =
            (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)input.Flags) & (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
    }
    tracker.RecordState(
        GetAccelBuffer(),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    if (!modifications.empty() || !setDesc.empty()) {
        tracker.RecordState(
            instBuffer.get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    return (update ? topLevelPrebuildInfo.UpdateScratchDataSizeInBytes : topLevelPrebuildInfo.ScratchDataSizeInBytes) + sizeof(size_t);
}
void TopAccel::Build(
    ResourceStateTracker &tracker,
    CommandBufferBuilder &builder,
    vstd::span<AccelBuildCommand::Modification const> const &modifications,
    BufferView const *scratchBuffer) {
    if (Length() == 0) return;
    auto alloc = builder.GetCB()->GetAlloc();
    // Update
    if (!modifications.empty() || !setDesc.empty()) {
        auto cs = device->setAccelKernel.Get(device);
        auto size = modifications.size() + setDesc.size();
        auto size_bytes = size * sizeof(AccelBuildCommand::Modification);
        auto setBuffer = alloc->GetTempUploadBuffer(size_bytes);
        auto cbuffer = alloc->GetTempUploadBuffer(sizeof(size_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        struct CBuffer {
            uint dsp;
            uint count;
        };
        CBuffer cbValue;
        cbValue.dsp = size;
        cbValue.count = Length();
        static_cast<UploadBuffer const *>(cbuffer.buffer)
            ->CopyData(cbuffer.offset,
                       {reinterpret_cast<uint8_t const *>(&cbValue), sizeof(CBuffer)});
        auto dataBuffer = static_cast<UploadBuffer const *>(setBuffer.buffer);
        if (!setDesc.empty()) {
            dataBuffer->CopyData(setBuffer.offset, {reinterpret_cast<uint8_t const *>(setDesc.data()), setDesc.size_bytes()});
        }
        if (!modifications.empty()) {
            dataBuffer->CopyData(setBuffer.offset + setDesc.size_bytes(), {reinterpret_cast<uint8_t const *>(modifications.data()), modifications.size_bytes()});
        }
        BindProperty properties[3];
        properties[0] = cbuffer;
        properties[1] = setBuffer;
        properties[2] = BufferView(instBuffer.get());
        builder.DispatchCompute(
            cs,
            uint3(size, 1, 1),
            properties);
        setDesc.clear();
    }
    if (scratchBuffer) {
        auto readState = tracker.ReadState(ResourceReadUsage::AccelBuildSrc);
        if ((eastl::to_underlying(tracker.GetState(instBuffer.get())) & eastl::to_underlying(readState)) == 0) {
            tracker.RecordState(instBuffer.get(), readState);
            tracker.UpdateState(builder);
        }
        topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer->buffer->GetAddress() + scratchBuffer->offset;
        if (RequireCompact()) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo;
            postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
            auto compactOffset = scratchBuffer->offset + scratchBuffer->byteSize - sizeof(size_t);
            postInfo.DestBuffer = scratchBuffer->buffer->GetAddress() + compactOffset;
            builder.GetCB()->CmdList()->BuildRaytracingAccelerationStructure(
                &topLevelBuildDesc,
                1,
                &postInfo);
        } else {
            builder.GetCB()->CmdList()->BuildRaytracingAccelerationStructure(
                &topLevelBuildDesc,
                0,
                nullptr);
        }
        update = true;
    }
}
void TopAccel::FinalCopy(
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
bool TopAccel::RequireCompact() const {
    return (((uint)topLevelBuildDesc.Inputs.Flags & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
bool TopAccel::CheckAccel(
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
    alloc->DisposeAfterComplete(std::move(accelBuffer));
    accelBuffer = std::move(newAccelBuffer);
    return true;
}
}// namespace lc::dx
