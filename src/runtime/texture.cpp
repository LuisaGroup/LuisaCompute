//
// Created by Mike Smith on 2021/7/23.
//

#include <runtime/texture.h>

namespace luisa::compute {

Command *TextureView2D::load(const void *pixels, uint mip_level) noexcept {
    if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    return TextureUploadCommand::create(
        _handle, _storage, mip_level,
        make_uint3(0u),
        make_uint3(mipmap_size, 1u),
        pixels);
}

Command *TextureView2D::load(ImageView<float> image, uint mip_level) noexcept {
    if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    if (!all(mipmap_size == image.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) "
            "([{}, {}] vs. [{}, {}]).",
            image.handle(), _handle, mip_level,
            image.size().x, image.size().y, mipmap_size.x, mipmap_size.y);
        mipmap_size = min(mipmap_size, image.size());
    }
    return TextureCopyCommand::create(
        image.handle(), _handle,
        0u, mip_level,
        make_uint3(image.offset(), 0u), make_uint3(0u),
        make_uint3(mipmap_size, 1u));
}

Command *TextureView3D::load(const void *pixels, uint mip_level) noexcept {
    if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    return TextureUploadCommand::create(
        _handle, _storage, mip_level,
        make_uint3(0u), mipmap_size,
        pixels);
}

Command *TextureView3D::load(VolumeView<float> volume, uint mip_level) noexcept {
    if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    if (!all(mipmap_size == volume.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) "
            "([{}, {}, {}] vs. [{}, {}, {}]).",
            volume.handle(), _handle, mip_level,
            volume.size().x, volume.size().y, volume.size().z,
            mipmap_size.x, mipmap_size.y, mipmap_size.z);
        mipmap_size = min(mipmap_size, volume.size());
    }
    return TextureCopyCommand::create(
        volume.handle(), _handle, 0u, mip_level,
        volume.offset(), make_uint3(0u), mipmap_size);
}

}// namespace luisa::compute
