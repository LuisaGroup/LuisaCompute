#include "bindless_array.h"
#include "compute_shader.h"
#include "upload_buffer.h"
#include "device.h"
#include "resource_barrier.h"
#include "sampler_anisotropy.h"
#include "stream.h"
#include <luisa/core/stl/unordered_map.h>
#include <limits>
#include <volk.h>
#include "log.h"
namespace lc::vk {
namespace bdls_detail {
using BindlessOperation = BindlessArrayUpdateCommand::Operation;

void validate_modified_buffer(
    const BindlessArrayUpdateCommand::ModifiedBuffer &modified,
    size_t slot) {
    switch (modified.op) {
        case BindlessOperation::NONE:
        case BindlessOperation::REMOVE:
            LUISA_ASSERT(
                modified.handle == 0u && modified.offset_bytes == 0u &&
                    modified.size_bytes == 0u,
                "Vulkan bindless buffer slot {} has non-canonical {} payload.",
                slot, modified.op == BindlessOperation::NONE ? "NONE" : "REMOVE");
            break;
        case BindlessOperation::EMPLACE:
            LUISA_ASSERT(modified.handle != 0u,
                         "Vulkan bindless buffer slot {} has a null EMPLACE handle.",
                         slot);
            LUISA_ASSERT(modified.size_bytes != 0u,
                         "Vulkan bindless buffer slot {} has an empty view.",
                         slot);
            break;
        default:
            LUISA_ERROR("Vulkan bindless buffer slot {} has invalid operation {}.",
                        slot, luisa::to_underlying(modified.op));
    }
}

void validate_modified_texture(
    const BindlessArrayUpdateCommand::ModifiedTexture &modified,
    size_t slot, const char *dimension,
    bool sampler_anisotropy_enabled) {
    switch (modified.op) {
        case BindlessOperation::NONE:
        case BindlessOperation::REMOVE:
            LUISA_ASSERT(
                modified.handle == 0u,
                "Vulkan bindless {} texture slot {} has non-canonical {} payload.",
                dimension, slot,
                modified.op == BindlessOperation::NONE ? "NONE" : "REMOVE");
            break;
        case BindlessOperation::EMPLACE:
            LUISA_ASSERT(
                modified.handle != 0u,
                "Vulkan bindless {} texture slot {} has a null EMPLACE handle.",
                dimension, slot);
            LUISA_ASSERT(
                luisa::to_underlying(modified.sampler.filter()) <
                        lc::vk::detail::sampler_filter_count &&
                    luisa::to_underlying(modified.sampler.address()) <
                        lc::vk::detail::sampler_address_count,
                "Vulkan bindless {} texture slot {} has an invalid sampler.",
                dimension, slot);
            LUISA_ASSERT(
                modified.sampler.filter() !=
                        Sampler::Filter::ANISOTROPIC ||
                    sampler_anisotropy_enabled,
                "Vulkan bindless {} texture slot {} requests ANISOTROPIC "
                "filtering, but samplerAnisotropy is not enabled on this "
                "logical device.",
                dimension, slot);
            break;
        default:
            LUISA_ERROR(
                "Vulkan bindless {} texture slot {} has invalid operation {}.",
                dimension, slot, luisa::to_underlying(modified.op));
    }
}

template<typename Modification, typename Validator>
void validate_modifications(
    luisa::span<const Modification> modifications,
    size_t slot_count, Validator &&validate) {
    luisa::unordered_set<size_t> modified_slots;
    modified_slots.reserve(modifications.size());
    for (auto &&modification : modifications) {
        LUISA_ASSERT(
            modification.slot < slot_count,
            "Vulkan bindless slot {} exceeds the array's {} slots.",
            modification.slot, slot_count);
        auto inserted = modified_slots.emplace(modification.slot).second;
        LUISA_ASSERT(inserted,
                     "Vulkan bindless update contains duplicate slot {}.",
                     modification.slot);
        validate(modification);
    }
}

[[nodiscard]] size_t checked_slot_storage_size(
    size_t slot_count, size_t slot_size, const char *storage_name) {
    LUISA_ASSERT(slot_count > 0u,
                 "Vulkan bindless arrays must contain at least one slot.");
    LUISA_ASSERT(slot_count <=
                     std::numeric_limits<size_t>::max() / slot_size,
                 "Vulkan bindless {} size overflows for {} slots of {} bytes.",
                 storage_name, slot_count, slot_size);
    return slot_count * slot_size;
}

[[nodiscard]] size_t index_record_size(BindlessSlotType type) {
    switch (type) {
        case BindlessSlotType::MULTIPLE:
            return sizeof(BindlessArray::BindlessStruct);
        case BindlessSlotType::BUFFER_ONLY:
            return sizeof(uint4);
        case BindlessSlotType::TEXTURE2D_ONLY:
        case BindlessSlotType::TEXTURE3D_ONLY:
            return sizeof(uint);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid Vulkan bindless slot type.");
}

[[nodiscard]] auto make_buffer_metadata(
    Device *device, BindlessSlotType type, size_t slot_count) {
    if (type != BindlessSlotType::MULTIPLE &&
        type != BindlessSlotType::BUFFER_ONLY) {
        return luisa::unique_ptr<DefaultBuffer>{};
    }
    return luisa::make_unique<DefaultBuffer>(
        device, checked_slot_storage_size(
                    slot_count, sizeof(StorageBufferMetadata),
                    "buffer metadata"));
}

Device::HeapAlloc &get_alloc(Device &device, BindlessSlotType type) {
    switch (type) {
        case BindlessSlotType::BUFFER_ONLY:
            return device.buffer_heap_pool;
        case BindlessSlotType::TEXTURE2D_ONLY:
            return device.tex2d_heap_pool;
        case BindlessSlotType::TEXTURE3D_ONLY:
            return device.tex3d_heap_pool;
        default:
            LUISA_ERROR("Bad bindless type.");
    }
}

[[nodiscard]] StorageBufferDescriptorRange buffer_descriptor_range(
    const BindlessArrayUpdateCommand::ModifiedBuffer &modified) {
    LUISA_ASSERT(modified.handle != 0u,
                 "Cannot bind a null Vulkan buffer handle.");
    auto buffer = reinterpret_cast<const Buffer *>(modified.handle);
    LUISA_ASSERT(modified.offset_bytes <= buffer->byte_size(),
                 "Bindless buffer offset {} exceeds buffer size {}.",
                 modified.offset_bytes, buffer->byte_size());
    auto remaining_size = buffer->byte_size() - modified.offset_bytes;
    auto logical_size = modified.size_bytes ==
                                BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                            remaining_size :
                            modified.size_bytes;
    LUISA_ASSERT(logical_size > 0u && logical_size <= remaining_size,
                 "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                 modified.offset_bytes,
                 modified.offset_bytes + logical_size,
                 buffer->byte_size());
    return storage_buffer_descriptor_range(
        buffer, modified.offset_bytes, logical_size, 0u,
        buffer->device_address_capable());
}

template<typename Modification>
void upload_buffer_metadata(
    CommandBuffer *cmdbuffer, DefaultBuffer *metadata_buffer,
    luisa::span<const Modification> modifications) {
    if (metadata_buffer == nullptr) { return; }
    using Operation = BindlessArrayUpdateCommand::Operation;
    luisa::vector<StorageBufferMetadata> records;
    luisa::vector<size_t> slots;
    records.reserve(modifications.size());
    slots.reserve(modifications.size());
    for (auto &&modification : modifications) {
        auto &&modified = modification.buffer;
        if (modified.op == Operation::NONE) { continue; }
        slots.emplace_back(modification.slot);
        records.emplace_back(
            modified.op == Operation::EMPLACE ?
                buffer_descriptor_range(modified).metadata :
                StorageBufferMetadata{});
    }
    if (records.empty()) { return; }

    auto upload = cmdbuffer->states()->upload_alloc.allocate(
        luisa::size_bytes(records), alignof(StorageBufferMetadata));
    static_cast<const UploadBuffer *>(upload.buffer)->copy_from(records.data(), upload.offset, luisa::size_bytes(records));

    luisa::vector<VkBufferCopy2> regions;
    regions.reserve(records.size());
    for (auto i = 0u; i < records.size(); ++i) {
        auto destination_offset =
            slots[i] * sizeof(StorageBufferMetadata);
        LUISA_ASSERT(destination_offset <= metadata_buffer->byte_size() &&
                         sizeof(StorageBufferMetadata) <=
                             metadata_buffer->byte_size() - destination_offset,
                     "Bindless metadata slot {} exceeds metadata buffer size {}.",
                     slots[i], metadata_buffer->byte_size());
        regions.emplace_back(VkBufferCopy2{
            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            nullptr,
            upload.offset + i * sizeof(StorageBufferMetadata),
            destination_offset,
            sizeof(StorageBufferMetadata)});
    }
    VkCopyBufferInfo2 copy_info{
        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        nullptr,
        upload.buffer->vk_buffer(),
        metadata_buffer->vk_buffer(),
        static_cast<uint32_t>(regions.size()),
        regions.data()};
    vkCmdCopyBuffer2(cmdbuffer->cmdbuffer(), &copy_info);
}

template<typename Record, typename Modification>
void upload_slot_records(
    CommandBuffer *cmdbuffer, DefaultBuffer &destination,
    luisa::span<const Modification> modifications,
    luisa::span<const Record> records) {
    if (modifications.empty()) { return; }
    vstd::vector<Record> changed_records;
    changed_records.reserve(modifications.size());
    for (auto &&modification : modifications) {
        LUISA_ASSERT(modification.slot < records.size(),
                     "Vulkan bindless slot {} exceeds {} encoded records.",
                     modification.slot, records.size());
        changed_records.emplace_back(records[modification.slot]);
    }
    auto upload = cmdbuffer->states()->upload_alloc.allocate(
        luisa::size_bytes(changed_records), alignof(Record));
    static_cast<const UploadBuffer *>(upload.buffer)->copy_from(changed_records.data(), upload.offset, luisa::size_bytes(changed_records));
    vstd::vector<VkBufferCopy2> regions;
    regions.reserve(modifications.size());
    for (auto i = 0u; i < modifications.size(); ++i) {
        regions.emplace_back(VkBufferCopy2{
            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            nullptr,
            upload.offset + i * sizeof(Record),
            modifications[i].slot * sizeof(Record),
            sizeof(Record)});
    }
    VkCopyBufferInfo2 copy_info{
        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        nullptr,
        upload.buffer->vk_buffer(),
        destination.vk_buffer(),
        static_cast<uint32_t>(regions.size()),
        regions.data()};
    vkCmdCopyBuffer2(cmdbuffer->cmdbuffer(), &copy_info);
}
}// namespace bdls_detail
BindlessArray::BindlessArray(Device *device, BindlessSlotType type, size_t size)
    : Resource(device),
      _indices_buffer(
          device,
          bdls_detail::checked_slot_storage_size(
              size, bdls_detail::index_record_size(type), "index")),
      _buffer_metadata{
          bdls_detail::make_buffer_metadata(device, type, size)},
      _type(type), _slot_count(size) {
    if (!device->enable_bindless()) [[unlikely]] {
        LUISA_ERROR("Bindless not enabled, Bindless-Array can not be loaded.");
    }
    LUISA_ASSERT(size > 0u && size <= device->bindless_heap_capacity(),
                 "Vulkan bindless array size {} exceeds the device heap "
                 "capacity {}.",
                 size, device->bindless_heap_capacity());
    switch (type) {
        case BindlessSlotType::MULTIPLE:
            _typed_binded.reset_as<vstd::vector<std::pair<BindlessStruct, MapIndices>>>(size);
            _encoded_binded.reset_as<vstd::vector<std::pair<BindlessStruct, MapIndices>>>(size);
            break;
        default: {
            _typed_binded.reset_as<vstd::vector<MapIndex>>(size);
            _encoded_binded.reset_as<vstd::vector<MapIndex>>(size);
            _typed_descriptor_indices.resize(
                size, BindlessStruct::kInvalidPos);
            if (type == BindlessSlotType::BUFFER_ONLY) {
                _typed_buffer_bindings.resize(size);
            }
        } break;
    }
}
BindlessArray::~BindlessArray() {
    if (auto binded = _encoded_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndices>>>()) {
        for (auto &idx : *binded) {
            auto &i = idx.first;
            if (i.buffer != BindlessStruct::kInvalidPos) {
                device()->buffer_heap_pool.dealloc(i.buffer);
            }
            if (i.tex_2d != BindlessStruct::kInvalidPos) {
                device()->tex2d_heap_pool.dealloc(
                    i.tex_2d & BindlessStruct::kMask);
            }
            if (i.tex_3d != BindlessStruct::kInvalidPos) {
                device()->tex3d_heap_pool.dealloc(
                    i.tex_3d & BindlessStruct::kMask);
            }
        }
    } else {
        auto &alloc = bdls_detail::get_alloc(*device(), _type);
        for (auto descriptor_index : _typed_descriptor_indices) {
            if (descriptor_index != BindlessStruct::kInvalidPos) {
                alloc.dealloc(descriptor_index);
            }
        }
    }
}
void BindlessArray::pre_update(ResourceBarrier *barrier) {
    barrier->record(
        BufferView{&_indices_buffer},
        _type == BindlessSlotType::MULTIPLE ?
            ResourceBarrier::Usage::kComputeUAV :
            ResourceBarrier::Usage::kCopyDest);
    if (_buffer_metadata != nullptr) {
        barrier->record(
            BufferView{_buffer_metadata.get()},
            ResourceBarrier::Usage::kCopyDest);
    }
}
void BindlessArray::_return_value(
    Map &resource_map, vstd::vector<FreeValue> &retired_slots,
    MapIndex &index, uint type, uint &origin_value) {
    LUISA_ASSERT(type <= 2u,
                 "Invalid Vulkan bindless descriptor class {}.", type);
    auto has_descriptor = origin_value != BindlessStruct::kInvalidPos;
    LUISA_ASSERT(
        static_cast<bool>(index) == has_descriptor,
        "Vulkan bindless descriptor/resource tracking state diverged.");
    if (has_descriptor) {
        retired_slots.push_back(FreeValue{
            ._type = type,
            ._index = type == 0u ?
                          origin_value :
                          origin_value & BindlessStruct::kMask});
        origin_value = BindlessStruct::kInvalidPos;
        auto &&v = index.value();
        LUISA_ASSERT(v > 0u,
                     "Vulkan bindless resource reference count underflow.");
        v--;
        if (v == 0) {
            resource_map.remove(index);
        }
    }
    index = {};
}
void BindlessArray::_deref(Map &resource_map, Map::Index &index) {
    if (!index) return;
    auto &&v = index.value();
    LUISA_ASSERT(v > 0u,
                 "Vulkan bindless resource reference count underflow.");
    v--;
    if (v == 0) {
        resource_map.remove(index);
    }
    index = {};
}
void BindlessArray::bind(vstd::span<const BindlessArrayUpdateCommand::Texture2DModification> mods) {
    auto bind_ptr = _typed_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(bind_ptr);
    auto &binded = *bind_ptr;
    bdls_detail::validate_modifications(
        mods, binded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_texture(
                modification.tex2d, modification.slot, "2D",
                device()->enable_sampler_anisotropy());
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.tex2d.op != Ope::NONE) {
            auto &indices = binded[mod.slot];
            _deref(_ptr_map, indices);
            if (mod.tex2d.op == Ope::EMPLACE) {
                indices = _add_index(_ptr_map, mod.tex2d.handle);
            }
        }
    }
}
void BindlessArray::bind(vstd::span<const BindlessArrayUpdateCommand::Texture3DModification> mods) {
    auto bind_ptr = _typed_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(bind_ptr);
    auto &binded = *bind_ptr;
    bdls_detail::validate_modifications(
        mods, binded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_texture(
                modification.tex3d, modification.slot, "3D",
                device()->enable_sampler_anisotropy());
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.tex3d.op != Ope::NONE) {
            auto &indices = binded[mod.slot];
            _deref(_ptr_map, indices);
            if (mod.tex3d.op == Ope::EMPLACE) {
                indices = _add_index(_ptr_map, mod.tex3d.handle);
            }
        }
    }
}
void BindlessArray::bind(vstd::span<const BindlessArrayUpdateCommand::BufferModification> mods) {
    auto bind_ptr = _typed_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(bind_ptr);
    auto &binded = *bind_ptr;
    bdls_detail::validate_modifications(
        mods, binded.size(), [](auto &&modification) {
            bdls_detail::validate_modified_buffer(
                modification.buffer, modification.slot);
        });
    std::lock_guard lck{mtx};
    if (mods.empty()) return;
    for (auto &&mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.buffer.op != Ope::NONE) {
            auto &indices = binded[mod.slot];
            _deref(_ptr_map, indices);
            if (mod.buffer.op == Ope::EMPLACE) {
                indices = _add_index(_ptr_map, mod.buffer.handle);
            }
        }
    }
}
auto BindlessArray::_add_index(Map &resource_map, size_t ptr) -> Map::Index {
    auto ite = resource_map.emplace(ptr, 0);
    ite.value()++;
    return ite;
}

bool BindlessArray::contains_buffer_alias(
    const Buffer *source) const noexcept {
    LUISA_ASSERT(source != nullptr,
                 "Cannot query a null Vulkan bindless buffer alias.");
    std::lock_guard lock{mtx};
    for (auto iter = _ptr_map.begin(); iter != _ptr_map.end(); ++iter) {
        auto *resource = reinterpret_cast<const Resource *>(iter->first);
        if (resource != nullptr &&
            resource->tag() == Resource::Tag::kBuffer) {
            auto *buffer = static_cast<const Buffer *>(resource);
            if (buffer == source ||
                buffer->vk_buffer() == source->vk_buffer()) {
                return true;
            }
        }
    }
    return false;
}

bool BindlessArray::encoded_buffers_support_device_address() const noexcept {
    std::lock_guard lock{mtx};
    auto supported = true;
    traverse_encoded_resources([&](uint64_t handle) noexcept {
        auto *resource = reinterpret_cast<const Resource *>(handle);
        LUISA_ASSERT(
            resource != nullptr &&
                (resource->tag() == Resource::Tag::kBuffer ||
                 resource->tag() == Resource::Tag::kTexture),
            "Vulkan bindless device-address validation encountered an "
            "invalid encoded resource.");
        if (resource->tag() == Resource::Tag::kBuffer) {
            supported &= static_cast<const Buffer *>(resource)
                             ->device_address_capable();
        }
    });
    return supported;
}

void BindlessArray::bind(luisa::span<BindlessArrayUpdateCommand::Modification const> mods) {
    auto binded_ptr = _typed_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndices>>>();
    LUISA_DEBUG_ASSERT(binded_ptr);
    auto &binded = *binded_ptr;
    bdls_detail::validate_modifications(
        mods, binded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_buffer(
                modification.buffer, modification.slot);
            bdls_detail::validate_modified_texture(
                modification.tex2d, modification.slot, "2D",
                device()->enable_sampler_anisotropy());
            bdls_detail::validate_modified_texture(
                modification.tex3d, modification.slot, "3D",
                device()->enable_sampler_anisotropy());
        });
    std::lock_guard lck{mtx};

    for (auto &&mod : mods) {
        auto &indices = binded[mod.slot].second;
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        if (mod.buffer.op != Ope::NONE) {
            _deref(_ptr_map, indices.buffer);
            if (mod.buffer.op == Ope::EMPLACE) {
                indices.buffer = _add_index(_ptr_map, mod.buffer.handle);
            }
        }
        if (mod.tex2d.op != Ope::NONE) {
            _deref(_ptr_map, indices.tex_2d);
            if (mod.tex2d.op == Ope::EMPLACE) {
                indices.tex_2d = _add_index(_ptr_map, mod.tex2d.handle);
            }
        }
        if (mod.tex3d.op != Ope::NONE) {
            _deref(_ptr_map, indices.tex_3d);
            if (mod.tex3d.op == Ope::EMPLACE) {
                indices.tex_3d = _add_index(_ptr_map, mod.tex3d.handle);
            }
        }
    }
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::BufferModification const> mods) {
    std::lock_guard lck{mtx};
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    auto encoded_ptr = _encoded_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(encoded_ptr);
    auto &encoded = *encoded_ptr;
    bdls_detail::validate_modifications(
        mods, encoded.size(), [](auto &&modification) {
            bdls_detail::validate_modified_buffer(
                modification.buffer, modification.slot);
        });
    bdls_detail::upload_buffer_metadata(
        cmdbuffer, _buffer_metadata.get(), mods);
    vstd::vector<FreeValue> retired_slots;
    for (auto &mod : mods) {
        auto &resource_index = encoded[mod.slot];
        if (mod.buffer.op != Ope::NONE) {
            _return_value(
                _encoded_ptr_map, retired_slots, resource_index, 0u,
                _typed_descriptor_indices[mod.slot]);
            if (mod.buffer.op == Ope::EMPLACE) {
                _typed_descriptor_indices[mod.slot] =
                    device()->buffer_heap_pool.alloc();
                resource_index = _add_index(
                    _encoded_ptr_map, mod.buffer.handle);
            }
        }
        if (mod.buffer.op == Ope::EMPLACE) {
            auto idx = _typed_descriptor_indices[mod.slot];
            auto buffer = reinterpret_cast<Buffer *>(mod.buffer.handle);
            auto buffer_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            auto descriptor =
                bdls_detail::buffer_descriptor_range(mod.buffer);
            LUISA_ASSERT(
                descriptor.metadata.descriptor_bias_bytes <=
                        std::numeric_limits<uint>::max() &&
                    descriptor.metadata.logical_size_bytes <=
                        std::numeric_limits<uint>::max(),
                "Vulkan typed bindless buffer metadata exceeds its 32-bit "
                "shader ABI (bias {}, size {}).",
                descriptor.metadata.descriptor_bias_bytes,
                descriptor.metadata.logical_size_bytes);
            _typed_buffer_bindings[mod.slot] = TypedBufferBinding{
                .descriptor_index = idx,
                .descriptor_bias_bytes = static_cast<uint>(
                    descriptor.metadata.descriptor_bias_bytes),
                .logical_size_bytes = static_cast<uint>(
                    descriptor.metadata.logical_size_bytes)};
            *buffer_info = VkDescriptorBufferInfo{
                buffer->vk_buffer(),
                descriptor.offset,
                descriptor.range};
            write_desc_sets.emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                device()->bdls_buffer_set(),
                0,
                idx,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_info,
                nullptr});
        } else if (mod.buffer.op == Ope::REMOVE) {
            _typed_buffer_bindings[mod.slot] = {};
        }
    }
    bdls_detail::upload_slot_records(
        cmdbuffer, _indices_buffer, mods,
        luisa::span<const TypedBufferBinding>{_typed_buffer_bindings});
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
    _defer_descriptor_recycling(
        cmdbuffer, std::move(retired_slots));
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Texture2DModification const> mods) {
    std::lock_guard lck{mtx};
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    auto encoded_ptr = _encoded_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(encoded_ptr);
    auto &encoded = *encoded_ptr;
    bdls_detail::validate_modifications(
        mods, encoded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_texture(
                modification.tex2d, modification.slot, "2D",
                device()->enable_sampler_anisotropy());
        });
    vstd::vector<FreeValue> retired_slots;
    for (auto &mod : mods) {
        auto &resource_index = encoded[mod.slot];
        if (mod.tex2d.op != Ope::NONE) {
            _return_value(
                _encoded_ptr_map, retired_slots, resource_index, 1u,
                _typed_descriptor_indices[mod.slot]);
            if (mod.tex2d.op == Ope::EMPLACE) {
                _typed_descriptor_indices[mod.slot] =
                    device()->tex2d_heap_pool.alloc();
                resource_index = _add_index(
                    _encoded_ptr_map, mod.tex2d.handle);
            }
        }
        if (mod.tex2d.op == Ope::EMPLACE) {
            auto idx = _typed_descriptor_indices[mod.slot];
            auto img_view = &device()->tex2d_bindless_imgview[idx];
            _emplace_tex(*img_view, cmdbuffer, write_desc_sets, device()->bdls_tex2d_set(), idx, reinterpret_cast<Texture *>(mod.tex2d.handle));
        }
    }
    bdls_detail::upload_slot_records(
        cmdbuffer, _indices_buffer, mods,
        luisa::span<const uint>{_typed_descriptor_indices});
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
    _defer_descriptor_recycling(
        cmdbuffer, std::move(retired_slots));
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Texture3DModification const> mods) {
    std::lock_guard lck{mtx};
    using Ope = BindlessArrayUpdateCommand::Modification::Operation;
    auto encoded_ptr = _encoded_binded.try_get<vstd::vector<MapIndex>>();
    LUISA_DEBUG_ASSERT(encoded_ptr);
    auto &encoded = *encoded_ptr;
    bdls_detail::validate_modifications(
        mods, encoded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_texture(
                modification.tex3d, modification.slot, "3D",
                device()->enable_sampler_anisotropy());
        });
    vstd::vector<FreeValue> retired_slots;
    for (auto &mod : mods) {
        auto &resource_index = encoded[mod.slot];
        if (mod.tex3d.op != Ope::NONE) {
            _return_value(
                _encoded_ptr_map, retired_slots, resource_index, 2u,
                _typed_descriptor_indices[mod.slot]);
            if (mod.tex3d.op == Ope::EMPLACE) {
                _typed_descriptor_indices[mod.slot] =
                    device()->tex3d_heap_pool.alloc();
                resource_index = _add_index(
                    _encoded_ptr_map, mod.tex3d.handle);
            }
        }
        if (mod.tex3d.op == Ope::EMPLACE) {
            auto idx = _typed_descriptor_indices[mod.slot];
            auto img_view = &device()->tex3d_bindless_imgview[idx];
            _emplace_tex(*img_view, cmdbuffer, write_desc_sets, device()->bdls_tex3d_set(), idx, reinterpret_cast<Texture *>(mod.tex3d.handle));
        }
    }
    bdls_detail::upload_slot_records(
        cmdbuffer, _indices_buffer, mods,
        luisa::span<const uint>{_typed_descriptor_indices});
    if (!write_desc_sets.empty()) {
        vkUpdateDescriptorSets(
            device()->logic_device(),
            write_desc_sets.size(),
            write_desc_sets.data(),
            0,
            nullptr);
        write_desc_sets.clear();
    }
    _defer_descriptor_recycling(
        cmdbuffer, std::move(retired_slots));
}
void BindlessArray::_emplace_tex(
    VkImageView &img_view,
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    VkDescriptorSet tex_set,
    uint tex_idx,
    Texture const *tex) const {
    // Descriptor writes declare the layout expected when the descriptor is
    // consumed; they do not transition the image. The consuming dispatch's
    // exact bindless-resource snapshot is responsible for ordering prior
    // producers and transitioning every mip to GENERAL.
    LUISA_ASSERT(tex != nullptr,
                 "Cannot bind a null Vulkan texture.");
    auto expected_dimension = [&] {
        if (tex_set == device()->bdls_tex2d_set()) { return 2u; }
        if (tex_set == device()->bdls_tex3d_set()) { return 3u; }
        LUISA_ERROR("Invalid Vulkan bindless texture descriptor set.");
    }();
    LUISA_ASSERT(
        tex->dimension() == expected_dimension,
        "Cannot bind a {}D Vulkan texture to a {}D bindless array.",
        tex->dimension(), expected_dimension);
    LUISA_ASSERT(tex_idx < device()->bindless_heap_capacity(),
                 "Vulkan bindless texture descriptor {} exceeds capacity {}.",
                 tex_idx, device()->bindless_heap_capacity());
    if (img_view) {
        vkDestroyImageView(device()->logic_device(), img_view, Device::alloc_callbacks());
        img_view = nullptr;
    }
    auto image_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorImageInfo>();
    VkImageViewCreateInfo img_view_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .image = tex->vk_image(),
        .viewType = expected_dimension == 2u ?
                        VK_IMAGE_VIEW_TYPE_2D :
                        VK_IMAGE_VIEW_TYPE_3D,
        .format = Texture::to_vk_format(tex->format()),
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = tex->mip(), .baseArrayLayer = 0, .layerCount = 1}};
    VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &img_view_create_info, Device::alloc_callbacks(), &img_view));

    *image_info = VkDescriptorImageInfo{
        nullptr,
        img_view,
        // Bindless textures use a persistent GENERAL layout so descriptor
        // contents remain valid across submissions and command-buffer state
        // restoration. ResourceBarrier::process_bindless records the same
        // layout with read-only access masks.
        VK_IMAGE_LAYOUT_GENERAL};
    write_desc_sets.emplace_back(VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        tex_set,
        0,
        tex_idx,
        1,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        image_info,
        nullptr,
        nullptr});
}
void BindlessArray::_defer_descriptor_recycling(
    CommandBuffer *cmdbuffer,
    vstd::vector<FreeValue> &&retired_slots) const {
    if (retired_slots.empty()) { return; }
    cmdbuffer->states()->callbacks.emplace_back(
        [retired_slots = std::move(retired_slots), device = device()]() {
            for (auto &i : retired_slots) {
                switch (i._type) {
                    case 0:
                        device->buffer_heap_pool.dealloc(i._index);
                        break;
                    case 1:
                        device->tex2d_heap_pool.dealloc(i._index);
                        break;
                    case 2:
                        device->tex3d_heap_pool.dealloc(i._index);
                        break;
                    default:
                        LUISA_ERROR(
                            "Invalid deferred Vulkan bindless descriptor class {}.",
                            i._type);
                }
            }
        });
}
void BindlessArray::update(
    CommandBuffer *cmdbuffer,
    luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
    luisa::vector<uint4> &cache,
    luisa::span<BindlessArrayUpdateCommand::Modification const> mods) {
    auto binded_ptr = _encoded_binded.try_get<vstd::vector<std::pair<BindlessStruct, MapIndices>>>();
    LUISA_DEBUG_ASSERT(binded_ptr);
    auto &binded = *binded_ptr;
    std::lock_guard lck{mtx};
    bdls_detail::validate_modifications(
        mods, binded.size(), [this](auto &&modification) {
            bdls_detail::validate_modified_buffer(
                modification.buffer, modification.slot);
            bdls_detail::validate_modified_texture(
                modification.tex2d, modification.slot, "2D",
                device()->enable_sampler_anisotropy());
            bdls_detail::validate_modified_texture(
                modification.tex3d, modification.slot, "3D",
                device()->enable_sampler_anisotropy());
        });
    if (mods.empty()) { return; }
    vstd::vector<FreeValue> retired_slots;
    bdls_detail::upload_buffer_metadata(
        cmdbuffer, _buffer_metadata.get(), mods);
    auto dsc_buffer = cmdbuffer->states()->upload_alloc.allocate(16 * mods.size(), 16);
    auto shader = device()->set_bindless_kernel.get(device());
    cache.clear();
    cache.reserve(mods.size());
    auto _emplace_tex = [&]<bool IsTex2D>(
                            BindlessStruct &bind_grp,
                            MapIndices &resource_indices,
                            uint64_t handle,
                            Sampler sampler,
                            Texture const *tex) {
        VkDescriptorSet tex_set;
        uint tex_idx;
        VkImageView *img_view;
        if constexpr (IsTex2D) {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.tex_2d, 1u, bind_grp.tex_2d);
            tex_idx = device()->tex2d_heap_pool.alloc();
            tex_set = device()->bdls_tex2d_set();
            bind_grp.write_samp2d(
                tex_idx,
                lc::vk::detail::sampler_heap_index(
                    luisa::to_underlying(sampler.filter()),
                    luisa::to_underlying(sampler.address())));
            resource_indices.tex_2d = _add_index(
                _encoded_ptr_map, handle);
            img_view = &device()->tex2d_bindless_imgview[tex_idx];
        } else {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.tex_3d, 2u, bind_grp.tex_3d);
            tex_idx = device()->tex3d_heap_pool.alloc();
            tex_set = device()->bdls_tex3d_set();
            bind_grp.write_samp3d(
                tex_idx,
                lc::vk::detail::sampler_heap_index(
                    luisa::to_underlying(sampler.filter()),
                    luisa::to_underlying(sampler.address())));
            resource_indices.tex_3d = _add_index(
                _encoded_ptr_map, handle);
            img_view = &device()->tex3d_bindless_imgview[tex_idx];
        }
        this->_emplace_tex(
            *img_view,
            cmdbuffer,
            write_desc_sets,
            tex_set,
            tex_idx,
            tex);
    };
    for (auto &mod : mods) {
        using Ope = BindlessArrayUpdateCommand::Modification::Operation;
        auto &bind_grp = binded[mod.slot].first;
        auto &resource_indices = binded[mod.slot].second;
        if (mod.buffer.op == Ope::EMPLACE) {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.buffer, 0u, bind_grp.buffer);
            bind_grp.buffer = device()->buffer_heap_pool.alloc();
            resource_indices.buffer = _add_index(
                _encoded_ptr_map, mod.buffer.handle);
            auto buffer = reinterpret_cast<Buffer *>(mod.buffer.handle);
            auto buffer_info = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            auto descriptor =
                bdls_detail::buffer_descriptor_range(mod.buffer);
            *buffer_info = VkDescriptorBufferInfo{
                buffer->vk_buffer(),
                descriptor.offset,
                descriptor.range};
            write_desc_sets.emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                device()->bdls_buffer_set(),
                0,
                bind_grp.buffer,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_info,
                nullptr});
        } else if (mod.buffer.op == Ope::REMOVE) {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.buffer, 0u, bind_grp.buffer);
        }
        if (mod.tex2d.op == Ope::EMPLACE) {
            _emplace_tex.operator()<true>(
                bind_grp, resource_indices, mod.tex2d.handle,
                mod.tex2d.sampler,
                reinterpret_cast<Texture *>(mod.tex2d.handle));
        } else if (mod.tex2d.op == Ope::REMOVE) {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.tex_2d, 1u, bind_grp.tex_2d);
        }
        if (mod.tex3d.op == Ope::EMPLACE) {
            _emplace_tex.operator()<false>(
                bind_grp, resource_indices, mod.tex3d.handle,
                mod.tex3d.sampler,
                reinterpret_cast<Texture *>(mod.tex3d.handle));
        } else if (mod.tex3d.op == Ope::REMOVE) {
            _return_value(
                _encoded_ptr_map, retired_slots,
                resource_indices.tex_3d, 2u, bind_grp.tex_3d);
        }
        auto &v = cache.emplace_back();
        v.x = mod.slot;
        std::memcpy(&v.y, &bind_grp, sizeof(BindlessStruct));
        static_assert(sizeof(BindlessStruct) == 12);
    }
    //
    VkDescriptorSet desc_set;
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = cmdbuffer->states()->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = shader->desc_set_layout().data()};
    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(
            device()->logic_device(),
            &alloc_info,
            &desc_set));
    uint value = mods.size();
    vkCmdPushConstants(
        cmdbuffer->cmdbuffer(),
        shader->pipeline_layout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        4,
        &value);
    VkDescriptorBufferInfo arg_buffer_info{
        dsc_buffer.buffer->vk_buffer(),
        dsc_buffer.offset,
        dsc_buffer.size_bytes};
    VkDescriptorBufferInfo buffer_info{
        _indices_buffer.vk_buffer(),
        0,
        _indices_buffer.byte_size()};
    static_cast<UploadBuffer const *>(dsc_buffer.buffer)->copy_from(cache.data(), dsc_buffer.offset, luisa::size_bytes(cache));

    auto local_write_begin = write_desc_sets.size();
    write_desc_sets.emplace_back(VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        desc_set,
        0,
        0,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &arg_buffer_info,
        nullptr});
    write_desc_sets.emplace_back(VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        desc_set,
        1,
        0,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &buffer_info,
        nullptr});
    LUISA_ASSERT(
        write_desc_sets.size() - local_write_begin ==
            shader->local_descriptor_binding_count(),
        "Vulkan bindless-update kernel consumed {} local descriptor bindings "
        "but its validated interface requires {}.",
        write_desc_sets.size() - local_write_begin,
        shader->local_descriptor_binding_count());
    vkUpdateDescriptorSets(
        device()->logic_device(),
        write_desc_sets.size(),
        write_desc_sets.data(),
        0,
        nullptr);
    write_desc_sets.clear();

    vkCmdBindDescriptorSets(
        cmdbuffer->cmdbuffer(),
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader->pipeline_layout(),
        0,
        1,
        &desc_set,
        0,
        nullptr);
    vkCmdBindPipeline(cmdbuffer->cmdbuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
    vkCmdDispatch(cmdbuffer->cmdbuffer(), (mods.size() + 255) / 256, 1, 1);
    _defer_descriptor_recycling(
        cmdbuffer, std::move(retired_slots));
}
}// namespace lc::vk
