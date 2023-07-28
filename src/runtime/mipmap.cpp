#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/mipmap.h>

namespace luisa::compute::detail {

void MipmapView::_error_mipmap_copy_buffer_size_mismatch(size_t mip_size, size_t buffer_size) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "No enough data (required = {} bytes) in buffer (size = {} bytes).",
        mip_size, buffer_size);
}

MipmapView::MipmapView(uint64_t handle, uint3 size, uint32_t level, PixelStorage storage) noexcept
    : _handle{handle},
      _size{size},
      _level{level},
      _storage{storage} {
    LUISA_VERBOSE_WITH_LOCATION(
        "Mipmap: size = [{}, {}, {}], storage = {}, level = {}.",
        size.x, size.y, size.z, luisa::to_string(storage), level);
}

[[nodiscard]] luisa::unique_ptr<TextureCopyCommand> MipmapView::copy_from(MipmapView src) const noexcept {
    if (!all(_size == src._size)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "MipmapView sizes mismatch in copy command "
            "(src: [{}, {}], dest: [{}, {}]).",
            src._size.x, src._size.y, _size.x, _size.y);
    }
    if (src._storage != _storage) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "MipmapView storages mismatch "
            "(src = {}, dst = {})",
            to_underlying(src._storage),
            to_underlying(_storage));
    }
    return luisa::make_unique<TextureCopyCommand>(
        _storage, src._handle, _handle, src._level, _level, _size);
}

}// namespace luisa::compute::detail

