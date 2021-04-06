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

namespace luisa::compute::dsl {

class ImageView;

// Image's are textures without sampling.
class Image : concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;
    uint2 _size;
    PixelFormat _format;

private:
    friend class Device;

public:
    Image(Device &device, PixelFormat format, uint2 size) noexcept;
    Image(Image &&another) noexcept;
    Image &operator=(Image &&rhs) noexcept;
    ~Image() noexcept;

    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto format() const noexcept { return _format; }

    [[nodiscard]] ImageView view() const noexcept;

    template<typename UV>
    [[nodiscard]] float4 operator[](UV uv) const noexcept;
};

namespace detail {
class ImageAccess;
}

class ImageView {

private:
    const RefExpr *_expression{nullptr};
    uint64_t _handle{};
    uint2 _size{};
    PixelFormat _format{};

private:
    friend class Image;
    constexpr explicit ImageView(uint64_t handle, PixelFormat format, uint2 size) noexcept
        : _handle{handle},
          _size{size},
          _format{format} {}

public:
    ImageView(const Image &image) noexcept : ImageView{image.view()} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto format() const noexcept { return _format; }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _format,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _format,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] detail::ImageAccess operator[](detail::Expr<uint2> uv) const noexcept;

    // for internal use
    explicit ImageView(detail::ArgumentCreation) noexcept
        : _expression{FunctionBuilder::current()->image()} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
};

template<typename UV>
float4 Image::operator[](UV uv) const noexcept {
    return this->view()[std::forward<UV>(uv)];
}

namespace detail {

class ImageAccess : public detail::Expr<float4> {

private:
    const Expression *_image;
    const Expression *_uv;

public:
    ImageAccess(const Expression *image, const Expression *uv) noexcept
        : Expr<float4>{FunctionBuilder::current()->call(
            Type::of<float4>(), "builtin_texture_read", {image, uv})},
          _image{image}, _uv{uv} {}

    ImageAccess(ImageAccess &&) noexcept = default;
    ImageAccess(const ImageAccess &) noexcept = default;

    [[nodiscard]] operator Var<float4>() const noexcept {
        return Expr<float4>{*this};
    }

    void operator=(Expr<float4> rhs) noexcept {
        auto f = FunctionBuilder::current();
        auto expr = f->call(
            nullptr, "builtin_texture_write",
            {_image, _uv, rhs.expression()});
        f->void_(expr);
    }

    void operator=(ImageAccess rhs) noexcept {
        this->operator=(Expr<float4>{rhs});
    }

#define LUISA_MAKE_IMAGE_ASSIGN_OP(op)               \
    template<typename T>                             \
    void operator op##=(T &&rhs) noexcept {          \
        *this = *this op Expr{std::forward<T>(rhs)}; \
    }
    LUISA_MAKE_IMAGE_ASSIGN_OP(+)
    LUISA_MAKE_IMAGE_ASSIGN_OP(-)
    LUISA_MAKE_IMAGE_ASSIGN_OP(*)
    LUISA_MAKE_IMAGE_ASSIGN_OP(/)
#undef LUISA_MAKE_IMAGE_ASSIGN_OP
};

}// namespace detail

Var(detail::ImageAccess)->Var<float4>;

}// namespace luisa::compute::dsl
