#pragma vengine_package vengine_directx
#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <DXRuntime/CommandAllocator.h>
namespace toolhub::directx {
static void GenTex2DSize(BindlessArray::BindlessStruct &s, uint2 size) {
    s.tex2DSize = size.y;
    s.tex2DSize <<= 16;
    s.tex2DSize |= size.x;
}
static void GenTex3DSize(BindlessArray::BindlessStruct &s, uint3 size) {
    s.tex3DSizeXY = size.y;
    s.tex3DSizeXY <<= 16;
    s.tex3DSizeXY |= size.x;
    s.tex3DSizeZSamp &= 65535;
    s.tex3DSizeZSamp |= (size.z << 16);
}
static void GenSampler2D(BindlessArray::BindlessStruct &s, uint samp2D) {
    s.tex3DSizeZSamp &= 4294967040u;
    s.tex3DSizeZSamp |= samp2D;
}
static void GenSampler3D(BindlessArray::BindlessStruct &s, uint samp3D) {
    s.tex3DSizeZSamp &= 4294902015u;
    samp3D <<= 8;
    s.tex3DSizeZSamp |= samp3D;
}

BindlessArray::BindlessArray(
    Device *device, uint arraySize)
    : Resource(device),
      buffer(device, arraySize * sizeof(BindlessStruct), device->defaultAllocator, VEngineShaderResourceState) {
    binded.resize(arraySize);
    memset(binded.data(), std::numeric_limits<int>::max(), binded.byte_size());
}
void BindlessArray::AddDepend(uint idx, BindTag tag, size_t ptr) {
    auto ite = ptrMap.Emplace(ptr, 0);
    ite.Value()++;
    indexMap.Emplace(std::pair<uint, BindTag>(idx, tag), ite);
}
void BindlessArray::RemoveDepend(uint idx, BindTag tag) {
    auto ite = indexMap.Find(std::pair<uint, BindTag>(idx, tag));
    if (!ite) return;
    auto &&v = ite.Value();
    auto &&refCount = v.Value();
    refCount--;
    if (refCount == 0) {
        ptrMap.Remove(v);
    }
    indexMap.Remove(ite);
}
uint BindlessArray::GetNewIndex() {
    return device->globalHeap->AllocateIndex();
}

BindlessArray::~BindlessArray() {
}
void BindlessArray::TryReturnIndex(uint originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.Push(originValue);
        // device->globalHeap->ReturnIndex(originValue);
    }
}
void BindlessArray::Bind(Property const &prop, uint index) {
    BindlessStruct &bindGrp = binded[index];
    std::lock_guard lck(globalMtx);
    auto dsp = vstd::create_disposer([&] {
        updateMap.ForceEmplace(
            index,
            bindGrp);
    });
    prop.multi_visit(
        [&](BufferView const &v) {
            AddDepend(index, BindTag::Buffer, reinterpret_cast<size_t>(v.buffer));
            TryReturnIndex(bindGrp.buffer);
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
                index);
            bindGrp.buffer = newIdx;
        },
        [&](std::pair<TextureBase const *, Sampler> const &v) {
            bool isTex2D = (v.first->Dimension() == TextureDimension::Tex2D);
            if (isTex2D)
                TryReturnIndex(bindGrp.tex2D);
            else
                TryReturnIndex(bindGrp.tex3D);
            uint texIdx = GetNewIndex();
            device->globalHeap->CreateSRV(
                v.first->GetResource(),
                v.first->GetColorSrvDesc(),
                texIdx);
            auto smpIdx = GlobalSamplers::GetIndex(v.second);
            if (isTex2D) {
                AddDepend(index, BindTag::Tex2D, reinterpret_cast<size_t>(v.first));
                bindGrp.tex2D = texIdx;
                GenTex2DSize(bindGrp, uint2(v.first->Width(), v.first->Height()));
                GenSampler2D(bindGrp, smpIdx);
            } else {
                AddDepend(index, BindTag::Tex3D, reinterpret_cast<size_t>(v.first));
                bindGrp.tex3D = texIdx;
                GenTex3DSize(bindGrp, uint3(v.first->Width(), v.first->Height(), v.first->Depth()));
                GenSampler3D(bindGrp, smpIdx);
            }
        });
}

void BindlessArray::UnBind(BindTag tag, uint index) {
    std::lock_guard lck(globalMtx);
    auto &&bindGrp = binded[index];
    RemoveDepend(index, tag);
    switch (tag) {
        case BindTag::Buffer:
            TryReturnIndex(bindGrp.buffer);
            break;
        case BindTag::Tex2D:
            TryReturnIndex(bindGrp.tex2D);
            break;
        case BindTag::Tex3D:
            TryReturnIndex(bindGrp.tex3D);
            break;
    }
}
bool BindlessArray::IsPtrInBindless(size_t ptr) const {
    std::lock_guard lck(globalMtx);
    return ptrMap.Find(ptr);
}
void BindlessArray::PreProcessStates(
    CommandBufferBuilder &builder,
    ResourceStateTracker &tracker) const {
    std::lock_guard lck(globalMtx);
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
        std::lock_guard lck(globalMtx);
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
    while (auto i = freeQueue.Pop()) {
        device->globalHeap->ReturnIndex(*i);
    }
}
}// namespace toolhub::directx