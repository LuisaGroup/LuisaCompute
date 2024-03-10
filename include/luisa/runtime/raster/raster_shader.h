#pragma once

#include <luisa/runtime/shader.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/backends/ext/raster_ext_interface.h>

namespace luisa::compute {

class Accel;
class BindlessArray;

namespace detail {

template<typename T>
struct PixelDst : public std::false_type {};

template<typename T>
struct PixelDst<Image<T>> : public std::true_type {
    static ShaderDispatchCommandBase::Argument::Texture get(Image<T> const &v) noexcept {
        return {v.handle(), 0};
    }
};

template<typename T>
struct PixelDst<ImageView<T>> : public std::true_type {
    static ShaderDispatchCommandBase::Argument::Texture get(ImageView<T> const &v) noexcept {
        return {v.handle(), v.level()};
    }
};

template<typename T, typename... Args>
static constexpr bool LegalDst() noexcept {
    constexpr bool r = PixelDst<T>::value;
    if constexpr (sizeof...(Args) == 0) {
        return r;
    } else if constexpr (!r) {
        return false;
    } else {
        return LegalDst<Args...>();
    }
}

}// namespace detail
namespace detail {
class RasterInvokeBase {
public:
    ShaderDispatchCmdEncoder encoder;
    RasterInvokeBase() noexcept = default;
    explicit RasterInvokeBase(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
        : encoder{handle, arg_count, uniform_size} {}

    RasterInvokeBase(RasterInvokeBase &&) noexcept = default;
    RasterInvokeBase(const RasterInvokeBase &) noexcept = delete;
    RasterInvokeBase &operator=(RasterInvokeBase &&) noexcept = default;
    RasterInvokeBase &operator=(const RasterInvokeBase &) noexcept = delete;

    template<typename T>
    RasterInvokeBase &operator<<(BufferView<T> buffer) noexcept {
        encoder.encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
        return *this;
    }

    template<typename T>
    RasterInvokeBase &operator<<(ImageView<T> image) noexcept {
        encoder.encode_texture(image.handle(), image.level());
        return *this;
    }

    template<typename T>
    RasterInvokeBase &operator<<(VolumeView<T> volume) noexcept {
        encoder.encode_texture(volume.handle(), volume.level());
        return *this;
    }

    template<typename T>
    RasterInvokeBase &operator<<(const Buffer<T> &buffer) noexcept {
        buffer._check_is_valid();
        return *this << buffer.view();
    }

    RasterInvokeBase &operator<<(const ByteBuffer &buffer) noexcept;

    template<typename T>
    RasterInvokeBase &operator<<(const Image<T> &image) noexcept {
        image._check_is_valid();
        return *this << image.view();
    }

    template<typename T>
    RasterInvokeBase &operator<<(const Volume<T> &volume) noexcept {
        volume._check_is_valid();
        return *this << volume.view();
    }

    template<typename T>
    RasterInvokeBase &operator<<(T data) noexcept {
        encoder.encode_uniform(&data, sizeof(T));
        return *this;
    }

    RasterInvokeBase &operator<<(const Accel &accel) noexcept {
        ShaderInvokeBase::encode(encoder, accel);
        return *this;
    }

    RasterInvokeBase &operator<<(const BindlessArray &array) noexcept {
        ShaderInvokeBase::encode(encoder, array);
        return *this;
    }

    RasterInvokeBase &operator<<(const IndirectDispatchBuffer &dispatch_buffer) noexcept {
        ShaderInvokeBase::encode(encoder, dispatch_buffer);
        return *this;
    }
};
}// namespace detail

template<typename... Args>
class RasterShader : public Resource {
    friend class RasterExt;
private:
    friend class Device;
    RasterExt *_raster_ext{};
    // JIT Shader
    // clang-format off

    RasterShader(DeviceInterface *device,
                 RasterExt* raster_ext,
                 Function vert,
                 Function pixel,
                 const ShaderOption &option)noexcept
        : Resource(
              device,
              Tag::RASTER_SHADER,
              raster_ext->create_raster_shader(
                  vert,
                  pixel,
                  option)),
                  _raster_ext{raster_ext}
        {
        }
    // AOT Shader
    RasterShader(
        DeviceInterface *device,
        RasterExt* raster_ext,
        luisa::string_view file_path)noexcept
        : Resource(
              device,
              Tag::RASTER_SHADER,
              // TODO
              raster_ext->load_raster_shader(
                detail::shader_argument_types<Args...>(),
                file_path)),
            _raster_ext{raster_ext}
        {
        }
    // clang-format on

public:
    RasterShader() noexcept = default;
    RasterShader(RasterShader &&) noexcept = default;
    RasterShader(RasterShader const &) noexcept = delete;
    RasterShader &operator=(RasterShader &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    RasterShader &operator=(RasterShader const &) noexcept = delete;
    ~RasterShader() noexcept override {
        if (*this) { _raster_ext->destroy_raster_shader(handle()); }
    }
    using Resource::operator bool;
    static size_t uniform_size() noexcept {
        return ShaderDispatchCmdEncoder::compute_uniform_size(detail::shader_argument_types<Args...>());
    }
};

}// namespace luisa::compute
