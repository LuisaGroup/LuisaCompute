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

    void _emplace_buffer(size_t index, uint64_t handle, size_t offset_bytes) noexcept;
    void _emplace_tex2d(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void _emplace_tex3d(size_t index, uint64_t handle, Sampler sampler) noexcept;

public:
    BindlessArray() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto size() const noexcept { return _size; }
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;

    template<typename T>
        requires is_buffer_or_view_v<std::remove_cvref_t<T>>
    void emplace(size_t index, T &&buffer) noexcept {
        BufferView view{std::forward<T>(buffer)};
        _emplace_buffer(index, view.handle(), view.offset_bytes());
    }

    void emplace(size_t index, const Image<float> &image, Sampler sampler) noexcept {
        _emplace_tex2d(index, image.handle(), sampler);
    }

    void emplace(size_t index, const Volume<float> &volume, Sampler sampler) noexcept {
        _emplace_tex3d(index, volume.handle(), sampler);
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
