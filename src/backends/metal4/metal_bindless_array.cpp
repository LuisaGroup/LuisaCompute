#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_stage_buffer_pool.h"
#include "metal_command_encoder.h"
#include "metal_device.h"
#include "metal_bindless_array.h"

namespace luisa::compute::metal {

namespace {

// The Metal update kernel consumes the original compact, 64-byte wire record.
// Keep that device ABI explicit instead of memcpy'ing the evolving runtime
// command object into device memory.
struct alignas(16) DeviceBindlessSlotModification {
    using Operation = BindlessArrayUpdateCommand::Operation;
    using Texture = BindlessArrayUpdateCommand::ModifiedTexture;
    struct Buffer {
        uint64_t handle;
        size_t size_bytes;
        Operation op;
    };
    size_t slot;
    Buffer buffer;
    Texture tex2d;
    Texture tex3d;
};
static_assert(sizeof(DeviceBindlessSlotModification::Buffer) == 24u);
static_assert(sizeof(DeviceBindlessSlotModification::Texture) == 16u);
static_assert(sizeof(DeviceBindlessSlotModification) == 64u);

}// namespace

MetalBindlessArray::MetalBindlessArray(MetalDevice *device, size_t size) noexcept
    : _device{device},
      _array{device->handle()->newBuffer(
          size * sizeof(Slot), MTL::ResourceStorageModePrivate |
                                   MTL::ResourceHazardTrackingModeTracked)},
      _update{device->builtin_update_bindless_slots()},
      _buffer_tracker{size}, _texture_tracker{size} {
    _buffer_slots.resize(size);
    _tex2d_slots.resize(size);
    _tex3d_slots.resize(size);
    if (_device != nullptr && _device->residency_set() != nullptr) {
        _device->register_residency(_array);
    } else {
        auto residency_desc = NS::TransferPtr(
            MTL::ResidencySetDescriptor::alloc()->init());
        residency_desc->setInitialCapacity(std::min<size_t>(size + 1u, 1024u));
        residency_desc->setLabel(MTLSTR("LuisaCompute BindlessArray ResidencySet"));
        NS::Error *error = nullptr;
        _residency_set = device->handle()->newResidencySet(residency_desc.get(), &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to create Metal4 bindless residency set: {}.",
                error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(_residency_set != nullptr,
                     "Failed to create Metal4 bindless residency set.");
        _residency_set->addAllocation(_array);
        _residency_set->commit();
        _residency_set->requestResidency();
    }
}

MetalBindlessArray::~MetalBindlessArray() noexcept {
    luisa::vector<const MTL::Allocation *> tracked_allocations;
    _buffer_tracker.traverse([&](uint64_t handle) noexcept {
        tracked_allocations.emplace_back(reinterpret_cast<MTL::Buffer *>(handle));
    });
    _texture_tracker.traverse([&](uint64_t handle) noexcept {
        tracked_allocations.emplace_back(reinterpret_cast<MTL::Texture *>(handle));
    });
    if (_device != nullptr && _device->residency_set() != nullptr) {
        tracked_allocations.emplace_back(_array);
        _device->unregister_residency(tracked_allocations);
        tracked_allocations.pop_back();
    }
    if (_residency_set != nullptr) {
        _residency_set->endResidency();
        _residency_set->release();
        _residency_set = nullptr;
    }
    for (auto *alloc : tracked_allocations) {
        const_cast<MTL::Allocation *>(alloc)->release();
    }
    _array->release();
}

void MetalBindlessArray::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _array->setLabel(nullptr);
    } else {
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _array->setLabel(mtl_name);
        mtl_name->release();
    }
}

void MetalBindlessArray::update(MetalCommandEncoder &encoder,
                                BindlessArrayUpdateCommand *cmd) noexcept {

    std::scoped_lock lock{_mutex};

    auto residency_dirty = false;
    luisa::vector<const MTL::Allocation *> added_allocations;
    luisa::vector<const MTL::Allocation *> removed_allocations;
    auto retain_tracked_buffer = [&](MTL::Buffer *buffer) noexcept {
        auto handle = reinterpret_cast<uint64_t>(buffer);
        if (!_buffer_tracker.contains(handle)) {
            buffer->retain();
            added_allocations.emplace_back(buffer);
            residency_dirty = true;
        }
        _buffer_tracker.retain(handle);
    };
    auto retain_tracked_texture = [&](MTL::Texture *texture) noexcept {
        auto handle = reinterpret_cast<uint64_t>(texture);
        if (!_texture_tracker.contains(handle)) {
            texture->retain();
            added_allocations.emplace_back(texture);
            residency_dirty = true;
        }
        _texture_tracker.retain(handle);
    };

    using Mod = BindlessArrayUpdateCommand::Modification;
    auto mods = cmd->steal_modifications();
    luisa::vector<DeviceBindlessSlotModification> device_mods;
    device_mods.reserve(mods.size());
    for (auto &m : mods) {
        device_mods.emplace_back(DeviceBindlessSlotModification{
            .slot = m.slot,
            .buffer = DeviceBindlessSlotModification::Buffer{
                m.buffer.handle, m.buffer.size_bytes, m.buffer.op},
            .tex2d = m.tex2d,
            .tex3d = m.tex3d});
        auto &device_mod = device_mods.back();
        // update buffer slot
        if (m.buffer.op == Mod::Operation::EMPLACE) {
            auto buffer = reinterpret_cast<MetalBuffer *>(m.buffer.handle)->handle();
            auto buffer_length = static_cast<size_t>(buffer->length());
            LUISA_ASSERT(m.buffer.offset_bytes < buffer_length,
                         "Bindless buffer offset {} exceeds buffer size {}.",
                         m.buffer.offset_bytes, buffer_length);
            auto buffer_address = buffer->gpuAddress() + m.buffer.offset_bytes;
            auto remaining_size = buffer_length - m.buffer.offset_bytes;
            auto buffer_size =
                m.buffer.size_bytes ==
                        BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                    remaining_size :
                    m.buffer.size_bytes;
            LUISA_ASSERT(buffer_size > 0u && buffer_size <= remaining_size,
                         "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                         m.buffer.offset_bytes,
                         m.buffer.offset_bytes + buffer_size,
                         buffer_length);
            device_mod.buffer.handle = buffer_address;
            device_mod.buffer.size_bytes = buffer_size;
            if (auto old_buffer = _buffer_slots[m.slot];
                old_buffer != buffer) {
                if (old_buffer != nullptr) {
                    _buffer_tracker.release(reinterpret_cast<uint64_t>(old_buffer));
                }
                retain_tracked_buffer(buffer);
                _buffer_slots[m.slot] = buffer;
            }
        } else if (m.buffer.op == Mod::Operation::REMOVE) {
            if (auto old_buffer = _buffer_slots[m.slot]) {
                _buffer_tracker.release(reinterpret_cast<uint64_t>(old_buffer));
                _buffer_slots[m.slot] = nullptr;
            }
        }
        // update texture2d slot
        if (m.tex2d.op == Mod::Operation::EMPLACE) {
            auto texture = reinterpret_cast<MetalTextureBase *>(m.tex2d.handle)->handle();
            auto texture_id = texture->gpuResourceID();
            device_mod.tex2d.handle = luisa::bit_cast<uint64_t>(texture_id);
            if (auto old_texture = _tex2d_slots[m.slot];
                old_texture != texture) {
                if (old_texture != nullptr) {
                    _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                }
                retain_tracked_texture(texture);
                _tex2d_slots[m.slot] = texture;
            }
        } else if (m.tex2d.op == Mod::Operation::REMOVE) {
            if (auto old_texture = _tex2d_slots[m.slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                _tex2d_slots[m.slot] = nullptr;
            }
        }
        // update texture3d slot
        if (m.tex3d.op == Mod::Operation::EMPLACE) {
            auto texture_base = reinterpret_cast<MetalTextureBase *>(m.tex3d.handle);
            LUISA_ASSERT(texture_base->kind() == MetalTextureBase::Kind::TEXTURE,
                         "A Metal depth texture cannot be bound as a 3D texture.");
            auto texture = texture_base->handle();
            auto texture_id = texture->gpuResourceID();
            device_mod.tex3d.handle = luisa::bit_cast<uint64_t>(texture_id);
            if (auto old_texture = _tex3d_slots[m.slot];
                old_texture != texture) {
                if (old_texture != nullptr) {
                    _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                }
                retain_tracked_texture(texture);
                _tex3d_slots[m.slot] = texture;
            }
        } else if (m.tex3d.op == Mod::Operation::REMOVE) {
            if (auto old_texture = _tex3d_slots[m.slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                _tex3d_slots[m.slot] = nullptr;
            }
        }
    }
    _buffer_tracker.commit([&](uint64_t handle) noexcept {
        removed_allocations.emplace_back(reinterpret_cast<MTL::Buffer *>(handle));
        residency_dirty = true;
    });
    _texture_tracker.commit([&](uint64_t handle) noexcept {
        removed_allocations.emplace_back(reinterpret_cast<MTL::Texture *>(handle));
        residency_dirty = true;
    });
    if (residency_dirty) {
        if (_device != nullptr && _device->residency_set() != nullptr) {
            if (!added_allocations.empty()) {
                _device->register_residency(added_allocations);
            }
            if (!removed_allocations.empty()) {
                _device->unregister_residency(removed_allocations);
            }
        } else if (_residency_set != nullptr) {
            if (!added_allocations.empty()) {
                _residency_set->addAllocations(
                    added_allocations.data(), added_allocations.size());
            }
            if (!removed_allocations.empty()) {
                _residency_set->removeAllocations(
                    removed_allocations.data(), removed_allocations.size());
            }
            _residency_set->commit();
        }
        for (auto *alloc : removed_allocations) {
            const_cast<MTL::Allocation *>(alloc)->release();
        }
    }

    // update the buffer
    auto size_bytes = luisa::span{device_mods}.size_bytes();
    if (size_bytes == 0u) { return; }
    encoder.with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->data(), device_mods.data(), size_bytes);
        auto command_encoder = encoder.coalesced_compute_encoder(MTL::StageDispatch);
        command_encoder->setComputePipelineState(_update);
        auto n = static_cast<uint>(mods.size());
        auto table = encoder.argument_table(3u);
        table->setAddress(_array->gpuAddress(), 0u);
        table->setAddress(
            upload_buffer->buffer()->gpuAddress() + upload_buffer->offset(), 1u);
        table->setAddress(encoder.upload(&n, sizeof(n)), 2u);
        command_encoder->setArgumentTable(table);
        encoder.use_device_resource(_update);
        encoder.use_device_resource(_array);
        if (_residency_set != nullptr) { encoder.use_residency_set(_residency_set); }
        encoder.use_stage_allocation(upload_buffer);
        auto block_size = MetalDevice::update_bindless_slots_block_size;
        auto threadgroup_count = (n + block_size - 1u) / block_size;
        command_encoder->dispatchThreadgroups(MTL::Size{threadgroup_count, 1u, 1u},
                                              MTL::Size{block_size, 1u, 1u});
    });
}

void MetalBindlessArray::mark_resource_usages(
    MetalCommandEncoder &encoder) noexcept {
    if (encoder.stream()->has_device_residency() && _residency_set == nullptr) {
        return;
    }
    std::scoped_lock lock{_mutex};
    if (_residency_set != nullptr) {
        encoder.use_residency_set(_residency_set);
    } else {
        encoder.use_device_resource(_array);
    }
}

bool MetalBindlessArray::contains_allocation(
    const MTL::Allocation *allocation) noexcept {
    if (allocation == nullptr) { return false; }
    if (allocation == _array) { return true; }
    std::scoped_lock lock{_mutex};
    auto key = reinterpret_cast<uint64_t>(allocation);
    return _buffer_tracker.contains(key) || _texture_tracker.contains(key);
}

}// namespace luisa::compute::metal
