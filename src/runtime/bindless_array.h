//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/stl/unordered_map.h>
#include <runtime/sampler.h>
#include <runtime/mipmap.h>
#include <runtime/resource.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/volume.h>

namespace luisa::compute {

class BindlessTexture2D;
class BindlessTexture3D;
class Command;
class Device;
class ManagedBindless;

namespace detail {
class BindlessArrayExprProxy;
}

template<typename T>
class BindlessBuffer;

template<typename T>
struct Expr;

class LC_RUNTIME_API BindlessArray final : public Resource {

public:
    using Modification = BindlessArrayUpdateCommand::Modification;

    struct ModSlotHash {
        using is_avalanching = void;
        [[nodiscard]] auto operator()(Modification m, uint64_t seed = hash64_default_seed) const noexcept {
            return hash_value(static_cast<size_t>(m.slot), seed);
        }
    };

    struct ModSlotEqual {
        [[nodiscard]] auto operator()(Modification lhs, Modification rhs) const noexcept {
            return lhs.slot == rhs.slot;
        }
    };

private:
    size_t _size{0u};
    luisa::unordered_set<Modification, ModSlotHash, ModSlotEqual> _updates;

private:
    friend class Device;
    friend class ManagedBindless;
    BindlessArray(DeviceInterface *device, size_t size) noexcept;

public:
    BindlessArray() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto size() const noexcept { return _size; }
    void emplace_buffer_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept;
    void emplace_tex2d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void emplace_tex3d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;
    BindlessArray &remove_buffer_on_update(size_t index) noexcept;
    BindlessArray &remove_tex2d_on_update(size_t index) noexcept;
    BindlessArray &remove_tex3d_on_update(size_t index) noexcept;

    template<typename T>
        requires is_buffer_or_view_v<std::remove_cvref_t<T>>
    auto &emplace_on_update(size_t index, T &&buffer, size_t offset = 0) noexcept {
        emplace_buffer_on_update(index, buffer.handle(), offset * buffer.stride());
        return *this;
    }

    auto &emplace_on_update(size_t index, const Image<float> &image, Sampler sampler) noexcept {
        emplace_tex2d_on_update(index, image.handle(), sampler);
        return *this;
    }

    auto &emplace_on_update(size_t index, const Volume<float> &volume, Sampler sampler) noexcept {
        emplace_tex3d_on_update(index, volume.handle(), sampler);
        return *this;
    }

    [[nodiscard]] luisa::unique_ptr<Command> update() noexcept;

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BindlessArrayExprProxy *>(this);
    }

    // see implementations in dsl/expr.h
    template<typename I>
    BindlessTexture2D tex2d(I &&index) const noexcept;

    template<typename I>
    BindlessTexture3D tex3d(I &&index) const noexcept;

    template<typename T, typename I>
    BindlessBuffer<T> buffer(I &&index) const noexcept;
};

}// namespace luisa::compute
