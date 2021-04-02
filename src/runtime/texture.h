//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/concepts.h>
#include <runtime/pixel_format.h>

namespace luisa::compute {

class Device;

namespace dsl {
class TextureView;
}

class TextureSampler {

public:
    enum struct Wrap {
    
    };
    
    enum struct Filter {
    
    };
    
private:
    Wrap _wrap_u;
    Wrap _wrap_v;
    Filter _mag_filter;
    Filter _min_filter;

public:


};

class Texture : concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;
    PixelFormat _format;
    uint _width;
    uint _height;
    uint _mipmap_levels;

private:
    friend class Device;
    Texture(Device *device, PixelFormat format, uint width, uint height, uint mipmap_levels) noexcept;

public:
    Texture(Texture &&another) noexcept;
    Texture &operator=(Texture &&rhs) noexcept;
    ~Texture() noexcept;
    
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto width() const noexcept { return _width; }
    [[nodiscard]] auto height() const noexcept { return _height; }
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] auto mipmap_levels() const noexcept { return _mipmap_levels; }
    
    // following methods will be implemented in dsl/texture_view.h
    [[nodiscard]] dsl::TextureView view() const noexcept;
    
    template<typename UV, typename Level>
    [[nodiscard]] float4 sample(UV uv, Level mip_level) const noexcept;
    
    template<typename UV, typename Level>
    [[nodiscard]] float4 sample(TextureSampler sampler, UV uv, Level mip_level) const noexcept;
    
    template<typename UV>
    [[nodiscard]] float4 operator[](UV uv) const noexcept;
};

}// namespace luisa::compute
