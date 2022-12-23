#include <runtime/mipmap.h>
#include <runtime/image.h>
#include <core/logging.h>
namespace luisa::compute ::detail {
MipmapView::MipmapView(uint64_t handle, uint3 size, uint32_t level, PixelStorage storage) noexcept
    : _handle{handle},
      _size{size},
      _level{level},
      _storage{storage} {
    LUISA_VERBOSE_WITH_LOCATION(
        "Mipmap: size = [{}, {}, {}], level = {}.",
        size.x, size.y, size.z, level);
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
    return TextureCopyCommand::create(
        _storage, src._handle, _handle, src._level, _level, _size);
}
LC_RUNTIME_API void log_invalid_mip(size_t level, size_t mip) {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid mipmap level {} for image with {} levels.",
        level, mip);
}
LC_RUNTIME_API void log_invalid_pixel_format(const char *name) {
    LUISA_ERROR_WITH_LOCATION("Invalid pixel storage for {} format.", name);
}
}// namespace luisa::compute::detail