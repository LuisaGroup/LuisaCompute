#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandBuffer.h>
#include <Resource/BottomAccel.h>
#include <luisa/core/logging.h>

namespace lc::dx {

TopAccel::TopAccel(Device *device, AccelOption const &option)
    : Resource(device) {
    if (!device->feature_check.raytracing_supported()) [[unlikely]] {
        LUISA_ERROR("RayTracing not supported on this device.");
    }
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
    std::memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
    std::memset(&topLevelPrebuildInfo, 0, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));
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
    LUISA_ASSUME(allInstance[instIndex].handle == handle);
    setMap[instIndex] = handle;
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
    char const *name,
    EnhancedBarrierTracker &tracker,
    CommandBufferBuilder &builder,
    vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state) {
    if (!oldBuffer) {
        newSize = CalcAlign(newSize, 65536);
        oldBuffer = vstd::create_unique(new DefaultBuffer(
            device,
            newSize,
            device->default_allocator.get(),
            state,
            false,
            name));
        return true;
    } else {
        if (newSize <= oldBuffer->GetByteSize()) return false;
        newSize = CalcAlign(newSize, 65536);
        auto newBuffer = new DefaultBuffer(
            device,
            newSize,
            device->default_allocator.get(),
            state);
        if (needCopy) {
            tracker.Record(
                BufferView(oldBuffer.get()),
                EnhancedBarrierTracker::Usage::CopySource);
            tracker.Record(
                BufferView(newBuffer),
                EnhancedBarrierTracker::Usage::CopyDest);
            GraphicsCmdlistBarrierCallback callback(builder);
            tracker.UpdateState(&callback);
            builder.copy_buffer(
                oldBuffer.get(),
                newBuffer,
                0,
                0,
                oldBuffer->GetByteSize());
        }
        builder.get_cb()->get_alloc()->dispose_after_complete(std::move(oldBuffer));
        oldBuffer = vstd::create_unique(newBuffer);
        return true;
    }
}
void TopAccel::ResizeAllInstance(size_t size) {
    if (size < allInstance.size()) {
        for (auto &i : vstd::ptr_range(allInstance.data() + size, allInstance.data() + allInstance.size())) {
            if (!i.handle) continue;
            i.handle->mesh->RemoveAccelRef(i.handle);
        }
        // Mesh-refresh entries queued by BLAS re-creation (SyncTopAccel) may
        // still reference the destroyed (pooled) handles of the removed slots;
        // drop them so a later build never dereferences a recycled handle and
        // binds an instance to the wrong BLAS.
        for (auto ite = setMap.begin(); ite != setMap.end();) {
            if (ite->first >= size) {
                ite = setMap.erase(ite);
            } else {
                ++ite;
            }
        }
    }
    allInstance.resize(size);
}

void TopAccel::PreProcessInst(
    EnhancedBarrierTracker &tracker,
    CommandBufferBuilder &builder,
    uint64 size,
    vstd::span<AccelBuildCommand::Modification const> const &modifications) {
    auto &&input = topLevelBuildDesc.Inputs;
    if (input.NumDescs != size) update = false;
    input.NumDescs = size;
    ResizeAllInstance(size);
    InitSetDesc(modifications);
    ProcessSetDesc(tracker);
    if (requireBuild) {
        requireBuild = false;
        update = false;
    }
    ProcessSetMap();
    size_t instanceByteCount = size * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    if (GenerateNewBuffer(
            "tlas-instance-buffer",
            tracker, builder, instBuffer, instanceByteCount, true,
            D3D12_RESOURCE_STATE_COMMON)) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
    if (!setDesc.empty()) {
        tracker.Record(
            BufferView(instBuffer.get(), 0, instBuffer->GetByteSize()),
            EnhancedBarrierTracker::Usage::ComputeUAV);
    }
}
void TopAccel::ProcessSetMap() {
    if (setMap.size() != 0) {
        update = false;
        setDesc.reserve(setDesc.size() + setMap.size());
        for (auto &&i : setMap) {
            if (i.first >= allInstance.size()) continue;
            auto &mod = setDesc.emplace_back();
            std::memset(&mod, 0, sizeof(PackedModifier));
            mod.index = i.first;
            mod.flags = AccelBuildCommand::Modification::flag_primitive;
            mod.primitive = i.second->mesh->GetAccelBuffer()->GetAddress();
        }
        setMap.clear();
    }
}

void TopAccel::ProcessSetDesc(EnhancedBarrierTracker &tracker) {

    for (auto &&m : setDesc) {
        auto ite = setMap.find(m.index);
#ifndef NDEBUG
        if (m.flags & AccelBuildCommand::Modification::flag_user_id) {
            if (m.user_id >= (1u << 24u)) [[unlikely]] {
                LUISA_ERROR("DirectX can-not support user_id larger than {}", (1u << 24u) - 1);
            }
        }
        if (m.index >= (1u << 24u)) [[unlikely]] {
            LUISA_ERROR("DirectX can-not support instance_id larger than {}", (1u << 24u) - 1);
        }
#endif
        bool updateMesh = (m.flags & AccelBuildCommand::Modification::flag_primitive);

        if (ite != setMap.end()) {
            if (!updateMesh) {
                m.primitive = reinterpret_cast<uint64_t>(ite->second->mesh);
                m.flags |= AccelBuildCommand::Modification::flag_primitive;
                updateMesh = true;
            }
            setMap.erase(ite);
        }
        if (updateMesh) {
            auto mesh = reinterpret_cast<BottomAccel *>(m.primitive);
            tracker.Record(mesh->GetAccelBuffer(), EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
            SetMesh(mesh, m.index);
            m.primitive = mesh->GetAccelBuffer()->GetAddress();
            update = false;
        }
        // TODO: motion vector support
        // m.motion_transform_buffer
    }
}
void TopAccel::InitSetDesc(vstd::span<AccelBuildCommand::Modification const> const &modifications) {
    setDesc.clear();
    luisa::vector_resize(setDesc, modifications.size());
    {
        auto iter = setDesc.data();
        for (auto &i : modifications) {
            std::memcpy(iter->affine, i.affine, sizeof(iter->affine));
            iter->primitive = i.primitive;
            iter->index = i.index;
            iter->vis_mask = i.vis_mask;
            iter->user_id = i.user_id;
            iter->flags = i.flags;
            iter++;
        }
    }
#ifndef NDEBUG
    for (auto &&m : modifications) {
        if (m.flags & AccelBuildCommand::Modification::flag_user_id) {
            if (m.user_id >= (1u << 24u)) [[unlikely]] {
                LUISA_ERROR("DirectX can-not support user_id larger than {}", (1u << 24u) - 1);
            }
        }
        if (m.index >= (1u << 24u)) [[unlikely]] {
            LUISA_ERROR("DirectX can-not support instance_id larger than {}", (1u << 24u) - 1);
        }
    }
#endif
}

size_t TopAccel::PreProcess(
    EnhancedBarrierTracker &tracker,
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
    ResizeAllInstance(size);
    InitSetDesc(modifications);
    ProcessSetDesc(tracker);
    if (requireBuild) {
        requireBuild = false;
        update = false;
    }
    ProcessSetMap();

    size_t instanceByteCount = size * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    if (GenerateNewBuffer(
            "tlas-instance-buffer",
            tracker, builder, instBuffer, instanceByteCount, true, D3D12_RESOURCE_STATE_COMMON)) {
        input.InstanceDescs = instBuffer->GetAddress();
    }
    device->device->GetRaytracingAccelerationStructurePrebuildInfo(&input, &topLevelPrebuildInfo);
    if (GenerateNewBuffer("tlas-accel-buffer", tracker, builder, accelBuffer, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, false, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)) {
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
    tracker.Record(
        BufferView(GetAccelBuffer(), 0, GetAccelBuffer()->GetByteSize()),
        EnhancedBarrierTracker::Usage::BuildAccel);
    if (!setDesc.empty()) {
        tracker.Record(
            BufferView(instBuffer.get(), 0, instBuffer->GetByteSize()),
            EnhancedBarrierTracker::Usage::ComputeUAV);
    }
    return (update ? topLevelPrebuildInfo.UpdateScratchDataSizeInBytes : topLevelPrebuildInfo.ScratchDataSizeInBytes) + sizeof(size_t);
}
void TopAccel::Build(
    EnhancedBarrierTracker &tracker,
    CommandBufferBuilder &builder,
    BufferView const *scratchBuffer) {
    if (Length() == 0) return;
    auto alloc = builder.get_cb()->get_alloc();
    // Update
    if (!setDesc.empty()) {
        auto cs = device->set_accel_kernel.get(device);
        auto size = setDesc.size();
        auto size_bytes = luisa::size_bytes(setDesc);
        auto setBuffer = alloc->get_temp_upload_buffer(size_bytes, 16);
        auto cbuffer = alloc->get_temp_upload_buffer(sizeof(size_t), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
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
            dataBuffer->CopyData(setBuffer.offset, {reinterpret_cast<uint8_t const *>(setDesc.data()), size_bytes});
        }
        BindProperty properties[3];
        properties[0] = cbuffer;
        properties[1] = setBuffer;
        properties[2] = BufferView(instBuffer.get());
        builder.dispatch_compute(
            cs,
            uint3(size, 1, 1),
            properties);
    }
    if (scratchBuffer) {
        tracker.Record(
            BufferView(instBuffer.get(), 0, instBuffer->GetByteSize()),
            EnhancedBarrierTracker::Usage::AccelInstanceBuffer);
        GraphicsCmdlistBarrierCallback callback(builder);
        tracker.UpdateState(&callback);
        topLevelBuildDesc.ScratchAccelerationStructureData = scratchBuffer->buffer->GetAddress() + scratchBuffer->offset;
        if (RequireCompact()) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postInfo;
            postInfo.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
            auto compactOffset = scratchBuffer->offset + scratchBuffer->byteSize - sizeof(size_t);
            postInfo.DestBuffer = scratchBuffer->buffer->GetAddress() + compactOffset;
            builder.get_cb()->cmd_list()->BuildRaytracingAccelerationStructure(
                &topLevelBuildDesc,
                1,
                &postInfo);
        } else {
            builder.get_cb()->cmd_list()->BuildRaytracingAccelerationStructure(
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
bool TopAccel::RequireCompact() const {
    return (((uint)topLevelBuildDesc.Inputs.Flags & (uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0) && !update;
}
bool TopAccel::CheckAccel(
    CommandBufferBuilder &builder) {
    auto disp = vstd::scope_exit([&] { compactSize = 0; });
    if (compactSize == 0)
        return false;
    auto &&alloc = builder.get_cb()->get_alloc();
    auto newAccelBuffer = vstd::create_unique(new DefaultBuffer(
        device,
        CalcAlign(compactSize, 65536),
        device->default_allocator.get(),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE));

    builder.get_cb()->cmd_list()->CopyRaytracingAccelerationStructure(
        newAccelBuffer->GetAddress(),
        accelBuffer->GetAddress(),
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    alloc->dispose_after_complete(std::move(accelBuffer));
    accelBuffer = std::move(newAccelBuffer);
    return true;
}
}// namespace lc::dx
