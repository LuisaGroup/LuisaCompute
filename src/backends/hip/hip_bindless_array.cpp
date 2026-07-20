#include <mutex>

#include <luisa/core/logging.h>
#include <luisa/core/pool.h>
#include <luisa/runtime/bindless_array.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_bindless_array.h"
#include "hip_texture.h"

namespace luisa::compute::hip {

namespace {

[[nodiscard]] uint64_t pack_texture_size_xy(uint3 size) noexcept {
    return static_cast<uint64_t>(size.x) |
           (static_cast<uint64_t>(size.y) << 32u);
}

[[nodiscard]] uint64_t pack_texture_levels_sampler(
    size_t levels, Sampler sampler, PixelStorage storage) noexcept {
    LUISA_ASSERT(levels > 0u &&
                     levels <= HIPBindlessArray::texture_level_count_mask,
                 "Invalid HIP bindless texture level count {}.", levels);
    LUISA_ASSERT(sampler.code() <= HIPBindlessArray::texture_sampler_mask,
                 "Invalid HIP bindless texture sampler code {}.", sampler.code());
    LUISA_ASSERT(to_underlying(storage) <= HIPBindlessArray::texture_storage_mask,
                 "Invalid HIP bindless texture storage {}.", to_underlying(storage));
    return static_cast<uint64_t>(levels) |
           (static_cast<uint64_t>(sampler.code()) << HIPBindlessArray::texture_sampler_shift) |
           (static_cast<uint64_t>(to_underlying(storage)) << HIPBindlessArray::texture_storage_shift);
}

class TextureRecycleCallback : public HIPCallbackContext {

private:
    luisa::vector<hipDeviceptr_t> _tables;

    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<TextureRecycleCallback> pool;
        return pool;
    }

public:
    explicit TextureRecycleCallback(luisa::vector<hipDeviceptr_t> &&tables) noexcept
        : _tables{std::move(tables)} {}

    [[nodiscard]] static TextureRecycleCallback *create(luisa::vector<hipDeviceptr_t> &&tables) noexcept {
        return _pool().create(std::move(tables));
    }

    void recycle() noexcept override {
        for (auto table : _tables) {
            LUISA_CHECK_HIP(hipFree(table));
        }
        _pool().destroy(this);
    }
};

}// namespace

HIPBindlessArray::HIPBindlessArray(size_t capacity) noexcept
    : _capacity{capacity},
      _host_slots(capacity, Slot{}),
      _tex2d_slots(capacity),
      _tex3d_slots(capacity),
      _texture_handle_table_tracker{capacity} {
    LUISA_CHECK_HIP(hipMalloc(&_handle, capacity * sizeof(Slot)));
    LUISA_CHECK_HIP(hipMemset(_handle, 0, capacity * sizeof(Slot)));
}

HIPBindlessArray::~HIPBindlessArray() noexcept {
    if (_handle) {
        LUISA_CHECK_HIP(hipFree(_handle));
    }
    if (_sampler_table) {
        LUISA_CHECK_HIP(hipFree(_sampler_table));
    }
    _texture_handle_table_tracker.traverse([](auto table) noexcept {
        auto handle_table = reinterpret_cast<hipDeviceptr_t>(table);
        LUISA_CHECK_HIP(hipFree(handle_table));
    });
}

void HIPBindlessArray::update(HIPCommandEncoder &encoder,
                              BindlessArrayUpdateCommand *cmd) noexcept {

    std::scoped_lock lock{_mutex};

    using Mod = BindlessArrayUpdateCommand::Modification;
    using BufferMod = BindlessArrayUpdateCommand::BufferModification;
    using Tex2DMod = BindlessArrayUpdateCommand::Texture2DModification;
    using Tex3DMod = BindlessArrayUpdateCommand::Texture3DModification;
    using Op = BindlessArrayUpdateCommand::Operation;

    auto stream = encoder.stream()->handle();

    luisa::vector<size_t> dirty_slots;

    auto release_texture_slot = [&](auto &slot) noexcept {
        if (slot.handle_table) {
            _texture_handle_table_tracker.release(reinterpret_cast<uint64_t>(slot.handle_table));
            slot.handle_table = nullptr;
        }
    };

    auto emplace_texture_slot = [&](auto &slot, const HIPTexture *texture, Sampler sampler) noexcept {
        auto level_count = texture->levels();
        luisa::vector<HIPImageDescriptor> descriptors(level_count);
        texture->copy_image_descriptors(descriptors);
        auto table_bytes = level_count * sizeof(HIPImageDescriptor);
        LUISA_CHECK_HIP(hipMalloc(&slot.handle_table, table_bytes));
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(slot.handle_table, descriptors.data(), table_bytes, stream));
        if (_sampler_table == nullptr) {
            static constexpr auto sampler_count = 16u;
            std::array<HIPSamplerDescriptor, sampler_count> sampler_descriptors{};
            texture->copy_sampler_descriptors(sampler_descriptors);
            LUISA_CHECK_HIP(hipMalloc(&_sampler_table, sizeof(sampler_descriptors)));
            LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                _sampler_table, sampler_descriptors.data(),
                sizeof(sampler_descriptors), stream));
        }
        _texture_handle_table_tracker.retain(reinterpret_cast<uint64_t>(slot.handle_table));
        return HIPTextureObject{
            slot.handle_table,
            pack_texture_levels_sampler(
                level_count, sampler, texture->storage())};
    };

    auto process_buffer = [&](size_t slot, const auto &buf) noexcept {
        if (buf.op == Op::EMPLACE) {
            auto buffer = reinterpret_cast<const HIPBuffer *>(buf.handle);
            LUISA_ASSERT(buf.offset_bytes < buffer->size_bytes(),
                         "Offset {} exceeds buffer size {}.",
                         buf.offset_bytes, buffer->size_bytes());
            auto address = reinterpret_cast<uint64_t>(buffer->handle()) + buf.offset_bytes;
            auto size = buffer->size_bytes() - buf.offset_bytes;
            _host_slots[slot].buffer = address;
            _host_slots[slot].size = size;
            dirty_slots.emplace_back(slot);
        } else if (buf.op == Op::REMOVE) {
            _host_slots[slot].buffer = 0u;
            _host_slots[slot].size = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    auto process_tex2d = [&](size_t slot, const auto &tex) noexcept {
        if (tex.op == Op::EMPLACE) {
            release_texture_slot(_tex2d_slots[slot]);
            auto texture = reinterpret_cast<const HIPTexture *>(tex.handle);
            auto tex_object = emplace_texture_slot(_tex2d_slots[slot], texture, tex.sampler);
            auto size = texture->size();
            _host_slots[slot].tex2d = reinterpret_cast<uint64_t>(tex_object.handles);
            _host_slots[slot].tex2d_levels = tex_object.level_count;
            _host_slots[slot].tex2d_size = pack_texture_size_xy(size);
            dirty_slots.emplace_back(slot);
        } else if (tex.op == Op::REMOVE) {
            release_texture_slot(_tex2d_slots[slot]);
            _host_slots[slot].tex2d = 0u;
            _host_slots[slot].tex2d_levels = 0u;
            _host_slots[slot].tex2d_size = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    auto process_tex3d = [&](size_t slot, const auto &tex) noexcept {
        if (tex.op == Op::EMPLACE) {
            release_texture_slot(_tex3d_slots[slot]);
            auto texture = reinterpret_cast<const HIPTexture *>(tex.handle);
            auto tex_object = emplace_texture_slot(_tex3d_slots[slot], texture, tex.sampler);
            auto size = texture->size();
            _host_slots[slot].tex3d = reinterpret_cast<uint64_t>(tex_object.handles);
            _host_slots[slot].tex3d_levels = tex_object.level_count;
            _host_slots[slot].tex3d_size_xy = pack_texture_size_xy(size);
            _host_slots[slot].tex3d_size_z = size.z;
            dirty_slots.emplace_back(slot);
        } else if (tex.op == Op::REMOVE) {
            release_texture_slot(_tex3d_slots[slot]);
            _host_slots[slot].tex3d = 0u;
            _host_slots[slot].tex3d_levels = 0u;
            _host_slots[slot].tex3d_size_xy = 0u;
            _host_slots[slot].tex3d_size_z = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    cmd->visit_modifications([&](auto &mods) noexcept {
        using T = std::decay_t<decltype(mods)>;
        if constexpr (std::is_same_v<T, luisa::vector<Mod>>) {
            for (auto &m : mods) {
                process_buffer(m.slot, m.buffer);
                process_tex2d(m.slot, m.tex2d);
                process_tex3d(m.slot, m.tex3d);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<BufferMod>>) {
            for (auto &m : mods) {
                process_buffer(m.slot, m.buffer);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<Tex2DMod>>) {
            for (auto &m : mods) {
                process_tex2d(m.slot, m.tex2d);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<Tex3DMod>>) {
            for (auto &m : mods) {
                process_tex3d(m.slot, m.tex3d);
            }
        }
    });

    luisa::vector<hipDeviceptr_t> retired_tables;
    _texture_handle_table_tracker.commit([&](auto table) noexcept {
        retired_tables.emplace_back(reinterpret_cast<hipDeviceptr_t>(table));
    });
    if (!retired_tables.empty()) {
        encoder.add_callback(TextureRecycleCallback::create(std::move(retired_tables)));
    }

    if (dirty_slots.empty()) { return; }

    std::sort(dirty_slots.begin(), dirty_slots.end());
    dirty_slots.erase(std::unique(dirty_slots.begin(), dirty_slots.end()),
                      dirty_slots.end());

    for (auto slot : dirty_slots) {
        auto dst = static_cast<std::byte *>(_handle) + slot * sizeof(Slot);
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            dst, &_host_slots[slot], sizeof(Slot), stream));
    }
}

void HIPBindlessArray::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = std::move(name);
}

}// namespace luisa::compute::hip
