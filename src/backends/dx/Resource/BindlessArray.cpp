#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <DXRuntime/CommandAllocator.h>
#include <luisa/core/logging.h>

#include "../../common/bindless_update_contract.h"

namespace lc::dx {

namespace {

template<typename T, typename F>
void validate_bindless_modifications(
    vstd::span<const T> modifications,
    size_t slot_count, F &&operations) {
    for (auto &&modification : modifications) {
        LUISA_ASSERT(
            lc::bindless_update_detail::slot_in_bounds(
                modification.slot, slot_count),
            "DX bindless update slot {} is outside [0, {}).",
            modification.slot, slot_count);
        operations(modification, [](auto operation) {
            LUISA_ASSERT(
                lc::bindless_update_detail::valid_operation(operation),
                "DX bindless update contains an invalid operation value {}.",
                static_cast<uint32_t>(operation));
        });
    }
}

[[nodiscard]] BufferView bindless_buffer_view(
    const BindlessArrayUpdateCommand::ModifiedBuffer &modified) {
    auto buffer = reinterpret_cast<Buffer *>(modified.handle);
    LUISA_ASSERT(modified.offset_bytes <= buffer->GetByteSize(),
                 "Bindless buffer offset {} exceeds buffer size {}.",
                 modified.offset_bytes, buffer->GetByteSize());
    auto remaining_size = buffer->GetByteSize() - modified.offset_bytes;
    auto size = modified.size_bytes ==
                        BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                    remaining_size :
                    modified.size_bytes;
    LUISA_ASSERT(size > 0u && size <= remaining_size,
                 "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                 modified.offset_bytes, modified.offset_bytes + size,
                 buffer->GetByteSize());
    return BufferView{buffer, modified.offset_bytes, size};
}

}// namespace

BindlessArray::BindlessArray(
    Device *device, uint arraySize,
    BindlessSlotType type)
    : Resource(device),
      buffer(device, type != BindlessSlotType::MULTIPLE ? sizeof(uint) : (arraySize * sizeof(BindlessStruct)), device->default_allocator.get()) {
    if (!device->feature_check.bindless_binding_supported()) [[unlikely]] {
        LUISA_ERROR("Current device not support bindless.");
    }
    switch (type) {
        case BindlessSlotType::MULTIPLE:
            typed_binded.reset_as<vstd::vector<std::pair<BindlessStruct, MapIndicies>>>(arraySize);
            break;
        default: {
            _buffer_node = device->global_heap->SubAllocate(arraySize);
            typed_binded.reset_as<vstd::vector<MapIndex>>(arraySize);
        } break;
    }
}
BindlessArray::~BindlessArray() {
    if (_buffer_node) {
        device->global_heap->DeAllocate(_buffer_node);
    }
    auto Return = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->global_heap->ReturnIndex(i);
        }
    };
    auto ReturnTex = [&](auto &&i) {
        if (i != BindlessStruct::n_pos) {
            device->global_heap->ReturnIndex(i & BindlessStruct::mask);
        }
    };
    if (auto binded = typed_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndicies>>>()) {
        for (auto &&i : *binded) {
            Return(i.first.buffer);
            ReturnTex(i.first.tex2D);
            ReturnTex(i.first.tex3D);
        }
    }
    for (auto &&i : freeQueue) {
        device->global_heap->ReturnIndex(i);
    }
}
void BindlessArray::Deref(MapIndex &index) {
    if (!index) return;
    auto &&v = index.value();
    v--;
    if (v == 0) {
        ptrMap.remove(index);
    }
    index = {};
}
void BindlessArray::TryReturnIndexTex(MapIndex &index, uint &originValue) {
    if (originValue != BindlessStruct::n_pos) {
        freeQueue.push_back(originValue & BindlessStruct::mask);
        originValue = BindlessStruct::n_pos;
        // device->global_heap->ReturnIndex(originValue);
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
        // device->global_heap->ReturnIndex(originValue);
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
void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) {
    auto bind_ptr = typed_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(bind_ptr && _buffer_node);
    auto &binded = *bind_ptr;
    validate_bindless_modifications(
        mods, binded.size(), [](auto &&mod, auto &&validate) {
            validate(mod.buffer.op);
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Operation;
        if (mod.buffer.op == Ope::NONE) { continue; }
        auto &indices = binded[mod.slot];
        Deref(indices);
        if (mod.buffer.op == Ope::EMPLACE) {
            auto v = bindless_buffer_view(mod.buffer);
            auto newIdx = device->global_heap->GetSubAllocOffset(_buffer_node) + mod.slot;
            auto desc = v.buffer->GetColorSrvDesc(
                v.offset,
                v.byteSize);
#ifndef NDEBUG
            if (!desc) {
                LUISA_ERROR("illagel buffer");
            }
#endif
            device->global_heap->CreateSRV(
                v.buffer->GetResource(),
                *desc,
                newIdx);
            indices = AddIndex(mod.buffer.handle);
        }
    }
}

template<typename T>
void BindlessArray::_BindTexture(vstd::span<const T> mods) {
    static_assert(
        std::is_same_v<T, BindlessArrayUpdateCommand::Texture2DModification> ||
        std::is_same_v<T, BindlessArrayUpdateCommand::Texture3DModification>);
    auto bind_ptr = typed_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(bind_ptr && _buffer_node);
    auto &binded = *bind_ptr;
    validate_bindless_modifications(
        mods, binded.size(), [](auto &&mod, auto &&validate) {
            if constexpr (std::is_same_v<
                              T,
                              BindlessArrayUpdateCommand::Texture2DModification>) {
                validate(mod.tex2d.op);
            } else {
                validate(mod.tex3d.op);
            }
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    for (auto &&mod : mods) {
        auto &texture = [&]() -> const BindlessArrayUpdateCommand::ModifiedTexture & {
            if constexpr (std::is_same_v<
                              T,
                              BindlessArrayUpdateCommand::Texture2DModification>) {
                return mod.tex2d;
            } else {
                return mod.tex3d;
            }
        }();
        if (texture.op == Ope::NONE) { continue; }
        auto &indices = binded[mod.slot];
        Deref(indices);
        if (texture.op == Ope::EMPLACE) {
            auto newIdx =
                device->global_heap->GetSubAllocOffset(_buffer_node) +
                mod.slot;
            auto tex = reinterpret_cast<TextureBase const *>(texture.handle);
            device->global_heap->CreateSRV(
                tex->GetResource(),
                tex->GetColorSrvDesc(),
                newIdx);
            indices = AddIndex(texture.handle);
        }
    }
}

void BindlessArray::Bind(
    vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) {
    _BindTexture(mods);
}

void BindlessArray::Bind(
    vstd::span<const BindlessArrayUpdateCommand::Texture3DModification> mods) {
    _BindTexture(mods);
}

void BindlessArray::Bind(vstd::span<const BindlessArrayUpdateCommand::Modification> mods) {
    auto binded_ptr = typed_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndicies>>>();
    LUISA_DEBUG_ASSERT(binded_ptr);
    auto &binded = *binded_ptr;
    validate_bindless_modifications(
        mods, binded.size(), [](auto &&mod, auto &&validate) {
            validate(mod.buffer.op);
            validate(mod.tex2d.op);
            validate(mod.tex3d.op);
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    auto EmplaceTex = [&]<bool isTex2D>(BindlessStruct &bindGrp, MapIndicies &indices, uint64_t handle, TextureBase const *tex, Sampler const &samp) {
        if constexpr (isTex2D)
            TryReturnIndexTex(indices.tex2D, bindGrp.tex2D);
        else
            TryReturnIndexTex(indices.tex3D, bindGrp.tex3D);
        auto texIdx = device->global_heap->AllocateIndex();
        device->global_heap->CreateSRV(
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
                auto v = bindless_buffer_view(mod.buffer);
                auto newIdx = device->global_heap->AllocateIndex();
                auto desc = v.buffer->GetColorSrvDesc(
                    v.offset,
                    v.byteSize);
#ifndef NDEBUG
                if (!desc) {
                    LUISA_ERROR("illagel buffer");
                }
#endif
                device->global_heap->CreateSRV(
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
    EnhancedBarrierTracker &tracker) const {
    std::lock_guard lck{mtx};
    if (offset_setted && _buffer_node) return;
    tracker.Record(
        BufferView(&buffer),
        _buffer_node ? EnhancedBarrierTracker::Usage::CopyDest : EnhancedBarrierTracker::Usage::ComputeUAV);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Modification> mods) const {
    auto binded_ptr = typed_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndicies>>>();
    LUISA_DEBUG_ASSERT(binded_ptr);
    auto &binded = *binded_ptr;
    std::lock_guard lck{mtx};
    struct BindlessElement {
        uint idx;
        BindlessStruct e;
    };
    if (!mods.empty()) {
        auto alloc = builder.get_cb()->get_alloc();
        auto tempBuffer = alloc->get_temp_upload_buffer(sizeof(BindlessElement) * mods.size(), 16);
        auto ubuffer = static_cast<UploadBuffer const *>(tempBuffer.buffer);
        auto offset = tempBuffer.offset;
        for (auto &&mod : mods) {
            BindlessElement e;
            e.idx = mod.slot;
            e.e = binded[mod.slot].first;
            ubuffer->CopyData(offset, {reinterpret_cast<uint8_t const *>(&e), sizeof(BindlessElement)});
            offset += sizeof(BindlessElement);
        }
        auto cs = device->set_bindless_kernel.get(device);
        auto cbuffer = alloc->get_temp_upload_buffer(sizeof(uint), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
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
        builder.dispatch_compute(
            cs,
            uint3(mods.size(), 1, 1),
            properties);
    }
    if (!freeQueue.empty()) {
        builder.get_cb()->get_alloc()->execute_after_complete(
            [vec = std::move(freeQueue),
             device = device] {
                for (auto &&i : vec) {
                    device->global_heap->ReturnIndex(i);
                }
            });
    }
}
template<typename T>
void BindlessArray::_UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const T> mods) const {
    LUISA_DEBUG_ASSERT(_buffer_node);
    std::lock_guard lck{mtx};
    if (offset_setted) return;
    offset_setted = true;
    auto alloc = builder.get_cb()->get_alloc();
    auto cbuffer = alloc->get_temp_upload_buffer(sizeof(uint), 16);
    uint value = device->global_heap->GetSubAllocOffset(_buffer_node);
    static_cast<UploadBuffer const *>(cbuffer.buffer)
        ->CopyData(cbuffer.offset,
                   {reinterpret_cast<uint8_t const *>(&value), sizeof(uint)});
    builder.copy_buffer(
        cbuffer.buffer,
        &buffer,
        cbuffer.offset,
        0,
        sizeof(uint));
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) const {
    _UpdateStates<BindlessArrayUpdateCommand::Texture2DModification>(builder, tracker, mods);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::Texture3DModification> mods) const {
    _UpdateStates<BindlessArrayUpdateCommand::Texture3DModification>(builder, tracker, mods);
}
void BindlessArray::UpdateStates(
    CommandBufferBuilder &builder,
    EnhancedBarrierTracker &tracker,
    vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) const {
    _UpdateStates<BindlessArrayUpdateCommand::BufferModification>(builder, tracker, mods);
}
}// namespace lc::dx
