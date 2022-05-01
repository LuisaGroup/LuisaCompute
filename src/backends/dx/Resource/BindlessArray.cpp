
#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandAllocator.h>
namespace toolhub::directx {

BindlessArray::BindlessArray(
    Device *device, uint arraySize)
    : Resource(device),
      buffer(device, arraySize * sizeof(BindlessStruct), device->defaultAllocator, VEngineShaderResourceState) {
    binded.resize(arraySize);
}
uint BindlessArray::GetNewIndex() {
    return device->globalHeap->AllocateIndex();
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
void BindlessArray::TryReturnIndex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.Push(originValue);
        originValue = BindlessStruct::n_pos;
        index = {};
        // device->globalHeap->ReturnIndex(originValue);
        auto &&v = index.Value();
        v--;
        if (v == 0) {
            ptrMap.Remove(index);
        }
    }
}
BindlessArray::MapIndex BindlessArray::AddIndex(size_t ptr) {
    auto ite = ptrMap.Emplace(ptr, 0);
    ite.Value()++;
    return ite;
}

void BindlessArray::Bind(Property const &prop, uint index) {
    auto &bindGrp = binded[index].first;
    auto &indices = binded[index].second;
    auto dsp = vstd::create_disposer([&] {
        updateMap.ForceEmplace(
            index,
            bindGrp);
    });
    prop.multi_visit(
        [&](BufferView const &v) {
            TryReturnIndex(indices.buffer, bindGrp.buffer);
            uint newIdx = GetNewIndex();
            auto desc = v.buffer->GetColorSrvDesc(
                v.offset,
                v.byteSize);
#ifdef _DEBUG
            if (!desc) {
                VEngine_Log("illagel buffer");
                VENGINE_EXIT;
            }
#endif
            device->globalHeap->CreateSRV(
                v.buffer->GetResource(),
                *desc,
                newIdx);
            bindGrp.buffer = newIdx;
            indices.buffer = AddIndex(reinterpret_cast<size_t>(v.buffer));
        },
        [&](std::pair<TextureBase const *, Sampler> const &v) {
            bool isTex2D = (v.first->Dimension() == TextureDimension::Tex2D);
            if (isTex2D)
                TryReturnIndex(indices.tex2D, bindGrp.tex2D);
            else
                TryReturnIndex(indices.tex3D, bindGrp.tex3D);
            uint texIdx = GetNewIndex();
            device->globalHeap->CreateSRV(
                v.first->GetResource(),
                v.first->GetColorSrvDesc(),
                texIdx);
            auto smpIdx = GlobalSamplers::GetIndex(v.second);
            if (isTex2D) {
                indices.tex2D = AddIndex(reinterpret_cast<size_t>(v.first));
                
                bindGrp.tex2D = texIdx;
                bindGrp.tex2DX = v.first->Width();
                bindGrp.tex2DY = v.first->Height();
                bindGrp.samp2D = smpIdx;
            } else {
                indices.tex3D = AddIndex(reinterpret_cast<size_t>(v.first));
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
bool BindlessArray::IsPtrInBindless(size_t ptr) const {

    return ptrMap.Find(ptr);
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
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker) const {
    {

        if (updateMap.size() > 0) {
            for (auto &&kv : updateMap) {
                auto &&sb = kv.second;
                builder.Upload(
                    BufferView(
                        &buffer,
                        sizeof(BindlessStruct) * kv.first,
                        sizeof(BindlessStruct)),
                    &kv.second);
            }
            updateMap.Clear();
            tracker.RecordState(
                &buffer);
        }
    }
    vstd::vector<uint> needReturnIdx;
    while (auto i = freeQueue.Pop()) {
        needReturnIdx.push_back(i);
    }
    builder.GetCB()->GetAlloc()->ExecuteAfterComplete(
        [vec = std::move(needReturnIdx),
         device = device] {
            for (auto &&i : vec) {
                device->globalHeap->ReturnIndex(i);
            }
        });
}
}// namespace toolhub::directx