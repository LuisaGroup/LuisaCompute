//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <runtime/sampler.h>
#include <runtime/mipmap.h>
#include <runtime/resource.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/volume.h>

namespace luisa::compute {

class BindlessTexture2D;
class BindlessTexture3D;

template<typename T>
class BindlessBuffer;

template<typename T>
struct Expr;

class BindlessArray final : public Resource {

public:
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    size_t _size{0u};

private:
    friend class Device;
    BindlessArray(Device::Interface *device, size_t size) noexcept;

public:
    BindlessArray() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto size() const noexcept { return _size; }
    void emplace_buffer(size_t index, uint64_t handle, size_t offset_bytes) noexcept;
    void emplace_tex2d(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;

    template<typename T>
        requires is_buffer_or_view_v<std::remove_cvref_t<T>>
    void emplace(size_t index, T &&buffer) noexcept {
        BufferView view{std::forward<T>(buffer)};
        emplace_buffer(index, view.handle(), view.offset_bytes());
    }

    template<typename T>
    void emplace(size_t index, const Image<T> &image, Sampler sampler) noexcept {
        emplace_tex2d(index, image.handle(), sampler);
    }

    template<typename T>
    void emplace(size_t index, const Volume<T> &volume, Sampler sampler) noexcept {
        emplace_tex3d(index, volume.handle(), sampler);
    }

    // see implementations in dsl/expr.h
    template<typename I>
    BindlessTexture2D tex2d(I &&index) const noexcept;

    template<typename I>
    BindlessTexture2D tex3d(I &&index) const noexcept;

    template<typename T, typename I>
    BindlessBuffer<T> buffer(I &&index) const noexcept;
};

}// namespace luisa::compute
