#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandAllocator.h>
#include <luisa/core/logging.h>

namespace lc::dx {

BindlessArray::BindlessArray(
    Device *device, uint arraySize)
    : Resource(device),
      buffer(device, arraySize * sizeof(BindlessStruct), device->defaultAllocator.get()) {
    binded.resize(arraySize);
}
BindlessArray::~BindlessArray() {
    auto Return = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->globalHeap->ReturnIndex(i);
        }
    };
    auto ReturnTex = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->globalHeap->ReturnIndex(i & BindlessStruct::mask);
        }
    };
    for (auto &&i : binded) {
        Return(i.first.buffer);
        ReturnTex(i.first.tex2D);
        ReturnTex(i.first.tex3D);
    }
    for (auto &&i : freeQueue) {
        device->globalHeap->ReturnIndex(i);
    }
}
void BindlessArray::TryReturnIndexTex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(originValue & BindlessStruct::mask);
        originValue = BindlessStruct::n_pos;
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.value();
        v--;
        if (v == 0) {
            ptrMap.remove(index);
        }
    }
    index = {};
}
void BindlessArray::TryReturnIndex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(originValue);
        originValue = BindlessStruct::n_pos;
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.value();
        v--;
        if (v == 0) {
            ptrMap.remove(index);
        }
    }
    index = {};
}
BindlessArray::MapIndex BindlessArray::AddIndex(size_t ptr) {
    auto ite = ptrMap.emplace(ptr, 0);
    ite.value()++;
    return ite;
}
void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::Modification> mods) {
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    auto EmplaceTex = [&]<bool isTex2D>(BindlessStruct &bindGrp, MapIndicies &indices, uint64_t handle, TextureBase const *tex, Sampler const &samp) {
        if constexpr (isTex2D)
            TryReturnIndexTex(indices.tex2D, bindGrp.tex2D);
        else
            TryReturnIndexTex(indices.tex3D, bindGrp.tex3D);
        auto texIdx = device->globalHeap->AllocateIndex();
        device->globalHeap->CreateSRV(
            tex->GetResource(),
            tex->GetColorSrvDesc(),
            texIdx);
        auto smpIdx = GlobalSamplers::GetIndex(samp);
        if constexpr (isTex2D) {
            indices.tex2D = AddIndex(handle);
            bindGrp.write_samp2d(texIdx, smpIdx);
        } else {
            indices.tex3D = AddIndex(handle);
            bindGrp.write_samp3d(texIdx, smpIdx);
        }
    };
    for (auto &&mod : mods) {
        auto &bindGrp = binded[mod.slot].first;
        auto &indices = binded[mod.slot].second;
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        switch (mod.buffer.op) {
            case Ope::REMOVE:
                TryReturnIndex(indices.buffer, bindGrp.buffer);
                break;
            case Ope::EMPLACE: {
                TryReturnIndex(indices.buffer, bindGrp.buffer);
                BufferView v{reinterpret_cast<Buffer *>(mod.buffer.handle), mod.buffer.offset_bytes};
                auto newIdx = device->globalHeap->AllocateIndex();
                auto desc = v.buffer->GetColorSrvDesc(
                    v.offset,
                    v.byteSize);
#ifndef NDEBUG
                if (!desc) {
                    LUISA_ERROR("illagel buffer");
                }
#endif
                device->globalHeap->CreateSRV(
                    v.buffer->GetResource(),
                    *desc,
                    newIdx);
                bindGrp.buffer = newIdx;
                indices.buffer = AddIndex(mod.buffer.handle);
                break;
            }
            default: break;
        }
        switch (mod.tex2d.op) {
            case Ope::REMOVE:
                TryReturnIndexTex(indices.tex2D, bindGrp.tex2D);
                break;
            case Ope::EMPLACE:
                EmplaceTex.operator()<true>(bindGrp, indices, mod.tex2d.handle, reinterpret_cast<TextureBase *>(mod.tex2d.handle), mod.tex2d.sampler);
                break;
            default: break;
        }
        switch (mod.tex3d.op) {
            case Ope::REMOVE:
                TryReturnIndexTex(indices.tex3D, bindGrp.tex3D);
                break;
            case Ope::EMPLACE:
                EmplaceTex.operator()<false>(bindGrp, indices, mod.tex3d.handle, reinterpret_cast<TextureBase *>(mod.tex3d.handle), mod.tex3d.sampler);
                break;
            default: break;
        }
    }
}
void BindlessArray::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    tracker.RecordState(
        &buffer,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    std::lock_guard lck{mtx};
    struct BindlessElement{
        uint idx;
        BindlessStruct e;   
    };
    if (!mods.empty()) {
        auto alloc = builder.GetCB()->GetAlloc();
        auto tempBuffer = alloc->GetTempUploadBuffer(sizeof(BindlessElement) * mods.size(), 16);
        auto ubuffer = static_cast<UploadBuffer const *>(tempBuffer.buffer);
        auto offset = tempBuffer.offset;
        for (auto &&mod : mods) {
            BindlessElement e;
            e.idx = mod.slot;
            e.e = binded[mod.slot].first;
            ubuffer->CopyData(offset, {reinterpret_cast<uint8_t const *>(&e), sizeof(BindlessElement)});
            offset += sizeof(BindlessElement);
        }
        auto cs = device->setBindlessKernel.Get(device);
        auto cbuffer = alloc->GetTempUploadBuffer(sizeof(uint), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        struct CBuffer {
            uint dsp;
        };
        CBuffer cbValue;
        cbValue.dsp = mods.size();
        static_cast<UploadBuffer const *>(cbuffer.buffer)
            ->CopyData(cbuffer.offset,
                       {reinterpret_cast<uint8_t const *>(&cbValue), sizeof(CBuffer)});
        BindProperty properties[3];
        properties[0] = cbuffer;
        properties[1] = tempBuffer;
        properties[2] = BufferView(&buffer);
        builder.DispatchCompute(
            cs,
            uint3(mods.size(), 1, 1),
            properties);

        tracker.RecordState(
            &buffer,
            tracker.ReadState(ResourceReadUsage::Srv));
    }
    if (!freeQueue.empty()) {
        builder.GetCB()->GetAlloc()->ExecuteAfterComplete(
            [vec = std::move(freeQueue),
             device = device] {
                for (auto &&i : vec) {
                    device->globalHeap->ReturnIndex(i);
                }
            });
    }
}
}// namespace lc::dx
