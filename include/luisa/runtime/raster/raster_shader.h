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

class LC_RUNTIME_API RasterShaderInvoke {

private:
    RasterDispatchCmdEncoder _command;
    luisa::span<const Function::Binding> _bindings;
public:
    explicit RasterShaderInvoke(
        size_t arg_size,
        size_t uniform_size,
        uint64_t handle,
        luisa::span<const Function::Binding> bindings) noexcept
        : _command{handle, arg_size, uniform_size, bindings} {
    }
    RasterShaderInvoke(RasterShaderInvoke &&) noexcept = default;
    RasterShaderInvoke(const RasterShaderInvoke &) noexcept = delete;
    RasterShaderInvoke &operator=(RasterShaderInvoke &&) noexcept = default;
    RasterShaderInvoke &operator=(const RasterShaderInvoke &) noexcept = delete;

    template<typename T>
    RasterShaderInvoke &operator<<(BufferView<T> buffer) noexcept {
        _command.encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
        return *this;
    }

    template<typename T>
    RasterShaderInvoke &operator<<(ImageView<T> image) noexcept {
        _command.encode_texture(image.handle(), image.level());
        return *this;
    }

    template<typename T>
    RasterShaderInvoke &operator<<(VolumeView<T> volume) noexcept {
        _command.encode_texture(volume.handle(), volume.level());
        return *this;
    }

    template<typename T>
    RasterShaderInvoke &operator<<(const Buffer<T> &buffer) noexcept {
        return *this << buffer.view();
    }

    template<typename T>
    RasterShaderInvoke &operator<<(const Image<T> &image) noexcept {
        return *this << image.view();
    }

    template<typename T>
    RasterShaderInvoke &operator<<(const Volume<T> &volume) noexcept {
        return *this << volume.view();
    }

    template<typename T>
    RasterShaderInvoke &operator<<(T data) noexcept {
        _command.encode_uniform(&data, sizeof(T));
        return *this;
    }

    RasterShaderInvoke &operator<<(const Accel &accel) noexcept {
        ShaderInvokeBase::encode(_command, accel);
        return *this;
    }

    RasterShaderInvoke &operator<<(const BindlessArray &array) noexcept {
        ShaderInvokeBase::encode(_command, array);
        return *this;
    }

    RasterShaderInvoke &operator<<(const IndirectDispatchBuffer &dispatch_buffer) noexcept {
        ShaderInvokeBase::encode(_command, dispatch_buffer);
        return *this;
    }

    template<typename... Rtv>
        requires(sizeof...(Rtv) == 0 || detail::LegalDst<Rtv...>())
    [[nodiscard]] auto draw(luisa::vector<RasterMesh> &&scene, MeshFormat const &mesh_format, Viewport viewport, const RasterState &raster_state, DepthBuffer const *dsv, Rtv const &...rtv) && noexcept {
        if (dsv) {
            _command.set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture{dsv->handle(), 0});
        } else {
            _command.set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture{invalid_resource_handle, 0});
        }
        if constexpr (sizeof...(Rtv) > 0) {
            auto tex_args = {detail::PixelDst<std::remove_cvref_t<Rtv>>::get(rtv)...};
            _command.set_rtv_texs(tex_args);
        }
        _command.set_scene(std::move(scene));
        _command.set_viewport(viewport);
        _command.set_raster_state(raster_state);
        _command.set_mesh_format(&mesh_format);
        return std::move(_command).build();
    }
};
}// namespace detail

template<typename... Args>
class RasterShader : public Resource {
    friend class RasterExt;
private:
    friend class Device;
    RasterExt *_raster_ext{};
    luisa::vector<Function::Binding> _bindings;
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
    [[nodiscard]] auto operator()(detail::prototype_to_shader_invocation_t<Args>... args) const noexcept {
        size_t arg_size;
        if (_bindings.empty()) {
            arg_size = (0u + ... + detail::shader_argument_encode_count<Args>::value);
        } else {
            arg_size = _bindings.size();
        }
        auto uniform_size = ShaderDispatchCmdEncoder::compute_uniform_size(
            detail::shader_argument_types<Args...>());
        detail::RasterShaderInvoke invoke(
            arg_size,
            uniform_size,
            handle(),
            _bindings);
        return std::move((invoke << ... << args));
    }
};

}// namespace luisa::compute
