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
    : _array{device->handle()->newBuffer(
          size * sizeof(Slot), MTL::ResourceStorageModePrivate |
                                   MTL::ResourceHazardTrackingModeTracked)},
      _update{device->builtin_update_bindless_slots()},
      _buffer_tracker{size}, _texture_tracker{size} {
    _buffer_slots.resize(size);
    _tex2d_slots.resize(size);
    _tex3d_slots.resize(size);
}

MetalBindlessArray::~MetalBindlessArray() noexcept {
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
                _buffer_tracker.release(reinterpret_cast<uint64_t>(old_buffer));
                _buffer_tracker.retain(reinterpret_cast<uint64_t>(buffer));
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
            auto texture = reinterpret_cast<MetalTexture *>(m.tex2d.handle)->handle();
            auto texture_id = texture->gpuResourceID();
            device_mod.tex2d.handle = luisa::bit_cast<uint64_t>(texture_id);
            if (auto old_texture = _tex2d_slots[m.slot];
                old_texture != texture) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                _texture_tracker.retain(reinterpret_cast<uint64_t>(texture));
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
            auto texture = reinterpret_cast<MetalTexture *>(m.tex3d.handle)->handle();
            auto texture_id = texture->gpuResourceID();
            device_mod.tex3d.handle = luisa::bit_cast<uint64_t>(texture_id);
            if (auto old_texture = _tex3d_slots[m.slot];
                old_texture != texture) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                _texture_tracker.retain(reinterpret_cast<uint64_t>(texture));
                _tex3d_slots[m.slot] = texture;
            }
        } else if (m.tex3d.op == Mod::Operation::REMOVE) {
            if (auto old_texture = _tex3d_slots[m.slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(old_texture));
                _tex3d_slots[m.slot] = nullptr;
            }
        }
    }
    _buffer_tracker.commit();
    _texture_tracker.commit();

    // update the buffer
    auto size_bytes = luisa::span{device_mods}.size_bytes();
    encoder.with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->data(), device_mods.data(), size_bytes);
        auto command_encoder = encoder.command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        command_encoder->setComputePipelineState(_update);
        command_encoder->setBuffer(_array, 0u, 0u);
        command_encoder->setBuffer(upload_buffer->buffer(), upload_buffer->offset(), 1u);
        auto n = static_cast<uint>(mods.size());
        command_encoder->setBytes(&n, sizeof(uint), 2u);
        auto block_size = MetalDevice::update_bindless_slots_block_size;
        auto threadgroup_count = (n + block_size) / block_size;
        command_encoder->dispatchThreadgroups(MTL::Size{threadgroup_count, 1u, 1u},
                                              MTL::Size{block_size, 1u, 1u});
        command_encoder->endEncoding();
    });
}

void MetalBindlessArray::mark_resource_usages(MTL::ComputeCommandEncoder *encoder,
                                              MTL::ResourceUsage usage) noexcept {
    std::scoped_lock lock{_mutex};
    encoder->useResource(_array, MTL::ResourceUsageRead);
    _buffer_tracker.traverse([encoder, usage](auto resource) noexcept {
        encoder->useResource(reinterpret_cast<MTL::Buffer *>(resource), usage);
    });
    _texture_tracker.traverse([encoder](auto resource) noexcept {
        encoder->useResource(reinterpret_cast<MTL::Texture *>(resource),
                             MTL::ResourceUsageRead | MTL::ResourceUsageSample);
    });
}

}// namespace luisa::compute::metal
