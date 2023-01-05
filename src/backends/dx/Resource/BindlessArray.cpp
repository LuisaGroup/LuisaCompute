
#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/Buffer.h>
namespace toolhub::directx {

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
    for (auto &&i : binded) {
        Return(i.first.buffer);
        Return(i.first.tex2D);
        Return(i.first.tex3D);
    }
    while (auto i = freeQueue.Pop()) {
        device->globalHeap->ReturnIndex(*i);
    }
}
void BindlessArray::TryReturnIndex(MapIndex &index, uint32_t &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.Push(originValue);
        originValue = BindlessStruct::n_pos;
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.Value();
        v--;
        if (v == 0) {
            ptrMap.Remove(index);
        }
    }
    index = {};
}
BindlessArray::MapIndex BindlessArray::AddIndex(size_t ptr) {
    auto ite = ptrMap.Emplace(ptr, 0);
    ite.Value()++;
    return ite;
}

void BindlessArray::Bind(size_t handle, Property const &prop, uint index) {
    auto &bindGrp = binded[index].first;
    auto &indices = binded[index].second;
    auto dsp = vstd::scope_exit([&] {
        updateMap.ForceEmplace(
            index,
            bindGrp);
    });
    prop.multi_visit(
        [&](BufferView const &v) {
            TryReturnIndex(indices.buffer, bindGrp.buffer);
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
            indices.buffer = AddIndex(handle);
        },
        [&](std::pair<TextureBase const *, Sampler> const &v) {
            bool isTex2D = (v.first->Dimension() == TextureDimension::Tex2D);
            if (isTex2D)
                TryReturnIndex(indices.tex2D, bindGrp.tex2D);
            else
                TryReturnIndex(indices.tex3D, bindGrp.tex3D);
            auto texIdx = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateSRV(
                v.first->GetResource(),
                v.first->GetColorSrvDesc(),
                texIdx);
            auto smpIdx = GlobalSamplers::GetIndex(v.second);
            if (isTex2D) {
                indices.tex2D = AddIndex(handle);

                bindGrp.tex2D = texIdx;
                bindGrp.tex2DX = v.first->Width();
                bindGrp.tex2DY = v.first->Height();
                bindGrp.samp2D = smpIdx;
            } else {
                indices.tex3D = AddIndex(handle);
                bindGrp.tex3D = texIdx;
                bindGrp.tex3DX = v.first->Width();
                bindGrp.tex3DY = v.first->Height();
                bindGrp.tex3DZ = v.first->Depth();
                bindGrp.samp3D = smpIdx;
            }
        });
}

void BindlessArray::UnBind(BindTag tag, uint index) {

    auto &bindGrp = binded[index].first;
    auto &indices = binded[index].second;
    switch (tag) {
        case BindTag::Buffer:
            TryReturnIndex(indices.buffer, bindGrp.buffer);
            break;
        case BindTag::Tex2D:
            TryReturnIndex(indices.tex2D, bindGrp.tex2D);
            break;
        case BindTag::Tex3D:
            TryReturnIndex(indices.tex3D, bindGrp.tex3D);
            break;
    }
}

void BindlessArray::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker) const {

    if (updateMap.size() > 0) {
        tracker.RecordState(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
}
void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::Modification> mods) {
    if (mods.empty()) return;
    auto EmplaceTex = [&]<bool isTex2D>(BindlessStruct &bindGrp, MapIndicies &indices, uint64_t handle, TextureBase const *tex, Sampler const &samp) {
        if constexpr (isTex2D)
            TryReturnIndex(indices.tex2D, bindGrp.tex2D);
        else
            TryReturnIndex(indices.tex3D, bindGrp.tex3D);
        auto texIdx = device->globalHeap->AllocateIndex();
        device->globalHeap->CreateSRV(
            tex->GetResource(),
            tex->GetColorSrvDesc(),
            texIdx);
        auto smpIdx = GlobalSamplers::GetIndex(samp);
        if constexpr (isTex2D) {
            indices.tex2D = AddIndex(handle);
            bindGrp.tex2D = texIdx;
            bindGrp.tex2DX = tex->Width();
            bindGrp.tex2DY = tex->Height();
            bindGrp.samp2D = smpIdx;
        } else {
            indices.tex3D = AddIndex(handle);
            bindGrp.tex3D = texIdx;
            bindGrp.tex3DX = tex->Width();
            bindGrp.tex3DY = tex->Height();
            bindGrp.tex3DZ = tex->Depth();
            bindGrp.samp3D = smpIdx;
        }
    };
    for (auto &&mod : mods) {
        auto &bindGrp = binded[mod.index].first;
        auto &indices = binded[mod.index].second;
        using Cmd = BindlessArrayUpdateCommand::ModificationCmd;
        switch (mod.cmd) {
            case Cmd::EmplaceBuffer: {
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
            } break;
            case Cmd::EmplaceTex2D: {
                EmplaceTex.operator()<true>(
                    bindGrp, indices,
                    mod.image.handle,
                    reinterpret_cast<TextureBase const *>(mod.image.handle),
                    mod.image.sampler);
            } break;
            case Cmd::EmplaceTex3D: {
                EmplaceTex.operator()<false>(
                    bindGrp, indices,
                    mod.image.handle,
                    reinterpret_cast<TextureBase const *>(mod.image.handle),
                    mod.image.sampler);
            } break;
            case Cmd::RemoveBuffer:
                TryReturnIndex(indices.buffer, bindGrp.buffer);
                break;
            case Cmd::RemoveTex2D:
                TryReturnIndex(indices.tex2D, bindGrp.tex2D);
                break;
            case Cmd::RemoveTex3D:
                TryReturnIndex(indices.tex3D, bindGrp.tex3D);
                break;
        }
    }
}
void BindlessArray::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    if (mods.empty()) return;
    tracker.RecordState(
        &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    if (!mods.empty()) {
        for (auto &&mod : mods) {
            builder.Upload(
                BufferView{
                    &buffer,
                    sizeof(BindlessStruct) * mod.index,
                    sizeof(BindlessStruct)},
                &binded[mod.index].first);
        }
        tracker.RecordState(
            &buffer);
    }
    vstd::vector<uint> needReturnIdx;
    while (auto i = freeQueue.Pop()) {
        needReturnIdx.push_back(i);
    }
    if (!needReturnIdx.empty()) {
        builder.GetCB()->GetAlloc()->ExecuteAfterComplete(
            [vec = std::move(needReturnIdx),
             device = device] {
                for (auto &&i : vec) {
                    device->globalHeap->ReturnIndex(i);
                }
            });
    }
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker) const {

    if (updateMap.size() > 0) {
        for (auto &&kv : updateMap) {
            builder.Upload(
                BufferView{
                    &buffer,
                    sizeof(BindlessStruct) * kv.first,
                    sizeof(BindlessStruct)},
                &kv.second);
        }
        updateMap.Clear();
        tracker.RecordState(
            &buffer);
    }
    vstd::vector<uint> needReturnIdx;
    while (auto i = freeQueue.Pop()) {
        needReturnIdx.push_back(i);
    }
    if (!needReturnIdx.empty()) {
        builder.GetCB()->GetAlloc()->ExecuteAfterComplete(
            [vec = std::move(needReturnIdx),
             device = device] {
                for (auto &&i : vec) {
                    device->globalHeap->ReturnIndex(i);
                }
            });
    }
}
}// namespace toolhub::directx