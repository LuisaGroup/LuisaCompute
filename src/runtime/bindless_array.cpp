#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

void ShaderInvokeBase::encode(ShaderDispatchCmdEncoder &encoder, const BindlessArray &array) noexcept {
    array._check_is_valid();
#ifndef NDEBUG
    if (array.dirty()) [[unlikely]] {
        LUISA_WARNING("Dispatching shader with a dirty bindless array.");
    }
#endif
    encoder.encode_bindless_array(array.handle());
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slot_count, BindlessSlotType type) noexcept {
    return _create<BindlessArray>(slot_count, type);
}

BindlessArray::BindlessArray(BindlessArray &&rhs) noexcept
    : Resource{std::move(rhs)},
      _size{rhs._size},
      _updates{std::move(rhs._updates)} {
    rhs._size = 0;
}

BindlessArray::BindlessArray(DeviceInterface *device, size_t size, BindlessSlotType type) noexcept
    : Resource{device, Tag::BINDLESS_ARRAY, [&] {
                   auto info = device->create_bindless_array(size, type);
#ifdef LUISA_ENABLE_SAFE_MODE
                   if (!info.valid()) {
                       LUISA_ERROR("Failed to create bindless array.");
                   }
#endif
                   return info;
               }()},
      _size{size} {
    switch (type) {
        case BindlessSlotType::MULTIPLE: _updates.emplace<ModSlotSet_MultiPurpose>(); break;
        case BindlessSlotType::BUFFER_ONLY: _updates.emplace<ModSlotSet_BufferOnly>(); break;
        case BindlessSlotType::TEXTURE2D_ONLY: _updates.emplace<ModSlotSet_Texture2DOnly>(); break;
        case BindlessSlotType::TEXTURE3D_ONLY: _updates.emplace<ModSlotSet_Texture3DOnly>(); break;
    }
}

void BindlessArray::emplace_buffer_handle_on_update(
    size_t index, uint64_t handle, size_t offset_bytes,
    size_t size_bytes) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [&]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &mods) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, BufferModification>) {
                auto [iter, _] = mods.emplace(index);
                const_cast<typename Mod::Buffer &>(iter->buffer) =
                    Modification::Buffer::emplace(
                        handle, offset_bytes, size_bytes);
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for emplace_buffer_handle_on_update.");
            }
        },
        _updates);
}

void BindlessArray::emplace_tex2d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [&]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &mods) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, Texture2DModification>) {
                auto [iter, _] = mods.emplace(index);
                const_cast<typename Mod::Texture &>(iter->tex2d) = Modification::Texture::emplace(handle, sampler);
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for emplace_tex2d_handle_on_update.");
            }
        },
        _updates);
}

void BindlessArray::emplace_tex3d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [&]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &mods) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, Texture3DModification>) {
                auto [iter, _] = mods.emplace(index);
                const_cast<typename Mod::Texture &>(iter->tex3d) = Modification::Texture::emplace(handle, sampler);
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for emplace_tex3d_handle_on_update.");
            }
        },
        _updates);
}

BindlessArray &BindlessArray::remove_buffer_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [index]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &updates) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, BufferModification>) {
                auto [iter, _] = updates.emplace(index);
                const_cast<typename Mod::Buffer &>(iter->buffer) = Modification::Buffer::remove();
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for remove_buffer_on_update.");
            }
        },
        _updates);
    return *this;
}

BindlessArray &BindlessArray::remove_tex2d_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [index]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &updates) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, Texture2DModification>) {
                auto [iter, _] = updates.emplace(index);
                const_cast<typename Mod::Texture &>(iter->tex2d) = Modification::Texture::remove();
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for remove_tex2d_on_update.");
            }
        },
        _updates);
    return *this;
}

BindlessArray &BindlessArray::remove_tex3d_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    luisa::visit(
        [index]<typename Mod>(luisa::unordered_set<Mod, ModSlotHash, ModSlotEqual> &updates) noexcept {
            if constexpr (std::is_same_v<Mod, Modification> || std::is_same_v<Mod, Texture3DModification>) {
                auto [iter, _] = updates.emplace(index);
                const_cast<typename Mod::Texture &>(iter->tex3d) = Modification::Texture::remove();
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid bindless slot type for remove_tex3d_on_update.");
            }
        },
        _updates);
    return *this;
}

luisa::unique_ptr<Command> BindlessArray::update() noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    return luisa::visit(
        [this]<typename T>(luisa::unordered_set<T, ModSlotHash, ModSlotEqual> &updates) noexcept -> luisa::unique_ptr<Command> {
            if (updates.empty()) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION("No update to bindless array.");
                return nullptr;
            }
            luisa::vector<T> mods;
            mods.reserve(updates.size());
            for (auto m : updates) { mods.emplace_back(m); }
            updates.clear();
            return luisa::make_unique<BindlessArrayUpdateCommand>(handle(), std::move(mods));
        },
        _updates);
}

BindlessArray::~BindlessArray() noexcept {
    if (!luisa::visit([](auto &&t) { return t.empty(); }, _updates)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Bindless array #{} destroyed with {} pending updates. "
            "Did you forget to call update()?",
            this->handle(),
            luisa::visit([](auto &&t) { return t.size(); }, _updates));
    }
    if (*this) {
        device()->destroy_bindless_array(handle());
    }
}

}// namespace luisa::compute
