//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/concepts.h>
#include <runtime/device.h>
#include <ast/expression.h>
#include <ast/function_builder.h>
#include <dsl/arg.h>
#include <dsl/expr.h>
#include <dsl/var.h>

namespace luisa::compute {

template<PixelFormat>
class ImageView;

namespace detail {

[[nodiscard]] auto valid_mipmap_levels(uint width, uint height, uint requested_levels) noexcept {
    auto rounded_size = next_pow2(std::min(width, height));
    auto max_levels = static_cast<uint>(std::log2(rounded_size));
    return requested_levels == 0u
               ? max_levels
               : std::min(requested_levels, max_levels);
}

}// namespace detail

// Images are textures without sampling.
template<PixelFormat format>
class Image : concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;
    uint2 _size;

public:
    Image(Device &device, uint2 size) noexcept
        : _device{&device},
          _handle{device.create_texture(
              format, 2u,
              size.x, size.y, 1u,
              1u, false)},
          _size{size} {}

    Image(Image &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._size} { another._device = nullptr; }

    ~Image() noexcept {
        if (_device != nullptr) {
            _device->dispose_texture(_handle);
        }
    }

    Image &operator=(Image &&rhs) noexcept {
        if (&rhs != this) {
            _device->dispose_texture(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._size;
            rhs._device = nullptr;
        }
        return *this;
    }

    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto view() const noexcept { return ImageView<format>{_handle, _size}; }

    template<typename UV>
    [[nodiscard]] decltype(auto) operator[](UV uv) const noexcept {
        return this->view()[std::forward<UV>(uv)];
    }

    [[nodiscard]] CommandHandle copy_to(void *data) const noexcept { return view().copy_to(data); }
    [[nodiscard]] CommandHandle copy_from(const void *data) const noexcept { view().copy_from(data); }
};

namespace detail {
template<PixelFormat>
class ImageAccess;
}

template<PixelFormat format>
class ImageView {

private:
    const RefExpr *_expression{nullptr};
    uint64_t _handle{};
    uint2 _size{};

private:
    friend class Image<format>;
    constexpr explicit ImageView(uint64_t handle, uint2 size) noexcept
        : _handle{handle}, _size{size} {}

public:
    ImageView(const Image<format> &image) noexcept : ImageView{image.view()} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, format,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, format,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] detail::ImageAccess<format> operator[](detail::Expr<uint2> uv) const noexcept {
        auto self = _expression ? _expression : FunctionBuilder::current()->image_binding(_handle);
        return {self, uv.expression()};
    }

    // for internal use
    explicit ImageView(detail::ArgumentCreation) noexcept
        : _expression{FunctionBuilder::current()->image()} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
};

namespace detail {

template<PixelFormat format>
class ImageAccess : public detail::Expr<float4> {

private:
    const Expression *_image;
    const Expression *_uv;

    [[nodiscard]] static auto _read(const Expression *image, const Expression *uv) noexcept {
        auto f = FunctionBuilder::current();
        auto expr = f->call(Type::of<float4>(), "texture_read", {image, uv});
        if constexpr (pixel_format_is_srgb(format)) {
            expr = f->call(Type::of<float4>(), "srgb_to_linear", {expr});
        }
        return expr;
    }

public:
    ImageAccess(const Expression *image, const Expression *uv) noexcept
        : Expr<float4>{_read(image, uv)},
          _image{image},
          _uv{uv} {}

    ImageAccess(ImageAccess &&) noexcept = default;
    ImageAccess(const ImageAccess &) noexcept = default;

    [[nodiscard]] operator Var<float4>() const noexcept {
        return Expr<float4>{*this};
    }

    void operator=(Expr<float4> rhs) noexcept {
        auto f = FunctionBuilder::current();
        auto value = rhs.expression();
        if constexpr (pixel_format_is_srgb(format)) {
            value = f->call(
                Type::of<float4>(),
                "linear_to_srgb",
                {value});
        }
        auto expr = f->call(
            nullptr, "texture_write",
            {_image, _uv, value});
        f->void_(expr);
    }

    void operator=(ImageAccess rhs) noexcept {// bypass colorspace conversion in this case
        auto f = FunctionBuilder::current();
        auto rhs_read = f->call(Type::of<float4>(), "texture_read", {rhs._image, rhs._uv});
        auto self_write = f->call(nullptr, "texture_write", {_image, _uv, rhs_read});
        f->void_(self_write);
    }

#define LUISA_MAKE_IMAGE_ASSIGN_OP(op)                             \
    template<typename T>                                           \
    void operator op##=(T &&rhs) noexcept {                        \
        *this = Expr<float4>{*this} op Expr{std::forward<T>(rhs)}; \
    }
    LUISA_MAKE_IMAGE_ASSIGN_OP(+)
    LUISA_MAKE_IMAGE_ASSIGN_OP(-)
    LUISA_MAKE_IMAGE_ASSIGN_OP(*)
    LUISA_MAKE_IMAGE_ASSIGN_OP(/)
#undef LUISA_MAKE_IMAGE_ASSIGN_OP
};

}// namespace detail

template<PixelFormat format>
Var(detail::ImageAccess<format>) -> Var<float4>;

}// namespace luisa::compute
